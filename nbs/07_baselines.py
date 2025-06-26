# %% [markdown]
# # Baselines
# > This notebook is used to run the baselines for the GCN-ID 2024 dataset.

# %%
#| default_exp baselines

# %%
#| eval: false
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wildlife_tools.similarity import CosineSimilarity
from wildlife_datasets import analysis, datasets, splits
import pycocotools.mask as mask_util
from wildlife_tools.data import ImageDataset
from sklearn.metrics import average_precision_score
import numpy as np
import timm
from transformers import AutoModel
import torch
import numpy as np
from wildlife_tools.inference import TopkClassifier, KnnClassifier
from wildlife_tools.features import DeepFeatures
import torchvision.transforms as T
from PIL import Image
import kaggle
import pandas as pd
from wildlife_tools.data import ImageDataset, FeatureDataset, FeatureDatabase
from gcn_reid.segmentation import decode_rle_mask
from gcn_reid.newt_dataset import upload_to_kaggle
from pathlib import Path
from gcn_reid.newt_dataset import download_kaggle_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import cv2

# %%
dataset_name = 'mshahoyi/gcn-id-2024'
dataset_path = Path('data/gcn-id-2024')
download_kaggle_dataset(dataset_name, dataset_path)

# %%
#| output: false
mega = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0)
miewid = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True)

# %% [markdown]
# ## Run both models on all test sets and save the results
# Artifacts are a dataframe like the newt dataframe but that contains two new columns representing the mega and miewid embeddings.

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mega_extractor = DeepFeatures(mega, device=device, batch_size=32, num_workers=4)
miewid_extractor = DeepFeatures(miewid, device=device, batch_size=32, num_workers=4)

# %%
df = pd.read_csv(dataset_path / 'metadata.csv')
mask = df.is_hard_test_query.notna() | df.is_least_similar_test_query.notna() | df.is_random_test_query.notna()
df = df[mask & ~df.is_video].reset_index(drop=True).rename(columns={"file_name": "image_name", "file_path": "path"})
df

# %%
mega_transform = T.Compose([T.Resize(384),
                            T.CenterCrop(384),
                            T.ToTensor(), 
                            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

miewid_transform = T.Compose([
    T.Resize(400),
    T.CenterCrop(400),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mega_transform_rotated = T.Compose([T.Resize(384),
                            T.CenterCrop(384),
                            T.RandomRotation([90, 90]),  # Add 90 degree clockwise rotation
                            T.ToTensor(), 
                            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

miewid_transform_rotated = T.Compose([
    T.Resize(400),
    T.CenterCrop(400),
    T.RandomRotation([90, 90]),  # Add 90 degree clockwise rotation
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# %%
#| export
def get_cropped_newt(path, rle):
    import cv2
    from gcn_reid.segmentation import decode_rle_mask
    from PIL import Image
    import numpy as np

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = decode_rle_mask(rle)
    if mask is None: return Image.fromarray(img)
    
    img_array = np.array(img)
    masked_img = img_array * mask[:, :, np.newaxis]
    return Image.fromarray(masked_img)

# %%
#| export
def get_cropping_image_dataset():
    import cv2
    from PIL import Image
    from wildlife_tools.data import ImageDataset

    class CroppingImageDataset(ImageDataset):
        """Dataset that crops an image using an RLE segmentation mask."""
        
        def __init__(self, *image_dataset_args, crop_out=True, rle_col="segmentation_mask_rle", **image_dataset_kwargs):
            super().__init__(*image_dataset_args, **image_dataset_kwargs)
            self.crop_out = crop_out
            self.rle_col = rle_col

        def get_image(self, path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not self.crop_out: return Image.fromarray(img)
            
            relative_path = path.replace(str(self.root), "")[1:] # remove leading /
            rle = self.metadata[self.metadata[self.col_path] == relative_path][self.rle_col].values
            if len(rle) == 0: return Image.fromarray(img)
            return get_cropped_newt(path, rle[0])

    return CroppingImageDataset

# %%
CroppingImageDataset = get_cropping_image_dataset()

mega_cropping_dataset = CroppingImageDataset(df, root=dataset_path, transform=mega_transform, crop_out=True)
miewid_cropping_dataset = CroppingImageDataset(df, root=dataset_path, transform=miewid_transform, crop_out=True)
mega_cropping_dataset_rotated = CroppingImageDataset(df, root=dataset_path, transform=mega_transform_rotated, crop_out=True)
miewid_cropping_dataset_rotated = CroppingImageDataset(df, root=dataset_path, transform=miewid_transform_rotated, crop_out=True)

mega_dataset = ImageDataset(df, root=dataset_path, transform=mega_transform)
miewid_dataset = ImageDataset(df, root=dataset_path, transform=miewid_transform)
mega_dataset_rotated = ImageDataset(df, root=dataset_path, transform=mega_transform_rotated)
miewid_dataset_rotated = ImageDataset(df, root=dataset_path, transform=miewid_transform_rotated)

# %%
num_images = 1
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    x, y = next(iter(mega_cropping_dataset_rotated))
    plt.imshow(x.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mega_images.png')

# %%
artifacts_path = Path('artifacts')
artifacts_path.mkdir(exist_ok=True)
artifacts_name = 'baseline_features.csv'

# %%
if not (artifacts_path/artifacts_name).exists():
    mega_results = mega_extractor(mega_dataset)
    miewid_results = miewid_extractor(miewid_dataset)
    mega_results_cropped = mega_extractor(mega_cropping_dataset)
    miewid_results_cropped = miewid_extractor(miewid_cropping_dataset)
    mega_results_cropped_rotated = mega_extractor(mega_cropping_dataset_rotated)
    miewid_results_cropped_rotated = miewid_extractor(miewid_cropping_dataset_rotated)
    mega_results_rotated = mega_extractor(mega_dataset_rotated)
    miewid_results_rotated = miewid_extractor(miewid_dataset_rotated)

    df['mega_features'] = [features.tolist() for features in mega_results.features]
    df['miewid_features'] = [features.tolist() for features in miewid_results.features]
    df['mega_features_cropped'] = [features.tolist() for features in mega_results_cropped.features]
    df['miewid_features_cropped'] = [features.tolist() for features in miewid_results_cropped.features]
    df['mega_features_cropped_rotated'] = [features.tolist() for features in mega_results_cropped_rotated.features]
    df['miewid_features_cropped_rotated'] = [features.tolist() for features in miewid_results_cropped_rotated.features]
    df['mega_features_rotated'] = [features.tolist() for features in mega_results_rotated.features]
    df['miewid_features_rotated'] = [features.tolist() for features in miewid_results_rotated.features]
    df.to_csv(artifacts_path/artifacts_name, index=False)
else: 
    df = pd.read_csv(artifacts_path/artifacts_name)
    df['mega_features'] = df['mega_features'].apply(eval)
    df['miewid_features'] = df['miewid_features'].apply(eval)
    df['mega_features_cropped'] = df['mega_features_cropped'].apply(eval)
    df['miewid_features_cropped'] = df['miewid_features_cropped'].apply(eval)
    df['mega_features_cropped_rotated'] = df['mega_features_cropped_rotated'].apply(eval)
    df['miewid_features_cropped_rotated'] = df['miewid_features_cropped_rotated'].apply(eval)
    df['mega_features_rotated'] = df['mega_features_rotated'].apply(eval)
    df['miewid_features_rotated'] = df['miewid_features_rotated'].apply(eval)

# %%
similarity_function = CosineSimilarity()

def get_top_k_accuracy(df, query_col, feature_col, k=1, database_feature_col=None):
    query_df = df[df[query_col] == True].reset_index(drop=True)
    database_df = df[df[query_col] != True].reset_index(drop=True)

    query_dataset = FeatureDataset(features=query_df[feature_col].tolist(), metadata=query_df)
    database_dataset = FeatureDataset(features=database_df[database_feature_col if database_feature_col else feature_col].tolist(), metadata=database_df)

    similarity = similarity_function(query_dataset, database_dataset)

    classifier = TopkClassifier(k=k, database_labels=database_dataset.labels_string, return_all=True)
    predictions, scores, idx = classifier(similarity)
    y = query_dataset.labels_string[:, np.newaxis]
    accuracy = np.mean(np.any(y == predictions, axis=1))

    query_identities = query_dataset.labels_string
    query_filenames = query_df['image_name'].values

    preds_df = pd.DataFrame(predictions, columns=pd.MultiIndex.from_product([['predicted_identity'], range(1, k+1)]))
    scores_df = pd.DataFrame(scores, columns=pd.MultiIndex.from_product([['score'], range(1, k+1)]))
    filenames_df = pd.DataFrame(database_df.iloc[idx.flatten()]['image_name'].values.reshape(idx.shape), columns=pd.MultiIndex.from_product([['predicted_image_name'], range(1, k+1)]))
    query_df = pd.DataFrame(np.column_stack([query_identities, query_filenames]), columns=pd.MultiIndex.from_product([['query'], ['identity', 'image_name']]))
    top_k_df = pd.concat([query_df, preds_df, scores_df, filenames_df], axis=1)

    return accuracy, top_k_df

accuracy, top_k_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features', k=5)
top_k_df

# %%
acc_mega_top_1_hard, acc_mega_top_1_hard_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features', k=1)
acc_mega_top_5_hard, acc_mega_top_5_hard_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features', k=5)
acc_miewid_top_1_hard, acc_miewid_top_1_hard_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features', k=1)
acc_miewid_top_5_hard, acc_miewid_top_5_hard_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features', k=5)

# %%
acc_mega_top_1_hard_cropped, acc_mega_top_1_hard_cropped_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features_cropped', k=1)
acc_mega_top_5_hard_cropped, acc_mega_top_5_hard_cropped_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features_cropped', k=5)
acc_miewid_top_1_hard_cropped, acc_miewid_top_1_hard_cropped_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features_cropped', k=1)
acc_miewid_top_5_hard_cropped, acc_miewid_top_5_hard_cropped_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features_cropped', k=5)

# %%
acc_mega_top_1_hard_rotated, acc_mega_top_1_hard_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features', k=1, database_feature_col='mega_features_rotated')
acc_mega_top_5_hard_rotated, acc_mega_top_5_hard_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features', k=5, database_feature_col='mega_features_rotated')
acc_miewid_top_1_hard_rotated, acc_miewid_top_1_hard_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features', k=1, database_feature_col='miewid_features_rotated')
acc_miewid_top_5_hard_rotated, acc_miewid_top_5_hard_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features', k=5, database_feature_col='miewid_features_rotated')

# %%
acc_mega_top_1_hard_cropped_rotated, acc_mega_top_1_hard_cropped_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features_cropped', k=1, database_feature_col='mega_features_cropped_rotated')
acc_mega_top_5_hard_cropped_rotated, acc_mega_top_5_hard_cropped_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'mega_features_cropped', k=5, database_feature_col='mega_features_cropped_rotated')
acc_miewid_top_1_hard_cropped_rotated, acc_miewid_top_1_hard_cropped_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features_cropped', k=1, database_feature_col='miewid_features_cropped_rotated')
acc_miewid_top_5_hard_cropped_rotated, acc_miewid_top_5_hard_cropped_rotated_df = get_top_k_accuracy(df, 'is_hard_test_query', 'miewid_features_cropped', k=5, database_feature_col='miewid_features_cropped_rotated')

# %%
acc_mega_top_1_random, acc_mega_top_1_random_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features', k=1)
acc_mega_top_5_random, acc_mega_top_5_random_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features', k=5)
acc_miewid_top_1_random, acc_miewid_top_1_random_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features', k=1)
acc_miewid_top_5_random, acc_miewid_top_5_random_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features', k=5)

# %%
acc_mega_top_1_random_cropped, acc_mega_top_1_random_cropped_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features_cropped', k=1)
acc_mega_top_5_random_cropped, acc_mega_top_5_random_cropped_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features_cropped', k=5)
acc_miewid_top_1_random_cropped, acc_miewid_top_1_random_cropped_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features_cropped', k=1)
acc_miewid_top_5_random_cropped, acc_miewid_top_5_random_cropped_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features_cropped', k=5)

# %%
acc_mega_top_1_random_rotated, acc_mega_top_1_random_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features', k=1, database_feature_col='mega_features_rotated')
acc_mega_top_5_random_rotated, acc_mega_top_5_random_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features', k=5, database_feature_col='mega_features_rotated')
acc_miewid_top_1_random_rotated, acc_miewid_top_1_random_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features', k=1, database_feature_col='miewid_features_rotated')
acc_miewid_top_5_random_rotated, acc_miewid_top_5_random_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features', k=5, database_feature_col='miewid_features_rotated')

# %%
acc_mega_top_1_random_cropped_rotated, acc_mega_top_1_random_cropped_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features_cropped', k=1, database_feature_col='mega_features_cropped_rotated')
acc_mega_top_5_random_cropped_rotated, acc_mega_top_5_random_cropped_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'mega_features_cropped', k=5, database_feature_col='mega_features_cropped_rotated')
acc_miewid_top_1_random_cropped_rotated, acc_miewid_top_1_random_cropped_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features_cropped', k=1, database_feature_col='miewid_features_cropped_rotated')
acc_miewid_top_5_random_cropped_rotated, acc_miewid_top_5_random_cropped_rotated_df = get_top_k_accuracy(df, 'is_random_test_query', 'miewid_features_cropped', k=5, database_feature_col='miewid_features_cropped_rotated')
 
# %%
acc_mega_top_1_least, acc_mega_top_1_least_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features', k=1)
acc_mega_top_5_least, acc_mega_top_5_least_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features', k=5)
acc_miewid_top_1_least, acc_miewid_top_1_least_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features', k=1)
acc_miewid_top_5_least, acc_miewid_top_5_least_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features', k=5)

# %%
acc_mega_top_1_least_cropped, acc_mega_top_1_least_cropped_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features_cropped', k=1)
acc_mega_top_5_least_cropped, acc_mega_top_5_least_cropped_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features_cropped', k=5)
acc_miewid_top_1_least_cropped, acc_miewid_top_1_least_cropped_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features_cropped', k=1)
acc_miewid_top_5_least_cropped, acc_miewid_top_5_least_cropped_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features_cropped', k=5)

# %%
acc_mega_top_1_least_rotated, acc_mega_top_1_least_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features', k=1, database_feature_col='mega_features_rotated')
acc_mega_top_5_least_rotated, acc_mega_top_5_least_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features', k=5, database_feature_col='mega_features_rotated')
acc_miewid_top_1_least_rotated, acc_miewid_top_1_least_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features', k=1, database_feature_col='miewid_features_rotated')
acc_miewid_top_5_least_rotated, acc_miewid_top_5_least_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features', k=5, database_feature_col='miewid_features_rotated')

# %%  
acc_mega_top_1_least_cropped_rotated, acc_mega_top_1_least_cropped_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features_cropped', k=1, database_feature_col='mega_features_cropped_rotated')
acc_mega_top_5_least_cropped_rotated, acc_mega_top_5_least_cropped_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'mega_features_cropped', k=5, database_feature_col='mega_features_cropped_rotated')
acc_miewid_top_1_least_cropped_rotated, acc_miewid_top_1_least_cropped_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features_cropped', k=1, database_feature_col='miewid_features_cropped_rotated')
acc_miewid_top_5_least_cropped_rotated, acc_miewid_top_5_least_cropped_rotated_df = get_top_k_accuracy(df, 'is_least_similar_test_query', 'miewid_features_cropped', k=5, database_feature_col='miewid_features_cropped_rotated')

# %%
# Splits
random = 'Random'
least_similar = 'Least Similar'
hard = 'Hard'

# Debiasing
original = 'Original'
bg_removed = 'Background Removed'
rotated = 'Rotated'
bg_removed_rotated = 'Background Removed + Rotated'

# Models
mega = 'MegaDescriptor'
miewid = 'MiewID'

# Metrics
top_1 = 'Top-1'
top_5 = 'Top-5'

# Create hierarchical index for rows
row_idx = pd.MultiIndex.from_product([
    [random, least_similar, hard],
    [original, bg_removed, rotated, bg_removed_rotated]
], names=['split', 'debiasing'])

# Create hierarchical columns
col_idx = pd.MultiIndex.from_product([
    [mega, miewid], 
    [top_1, top_5]
], names=['model', 'metric'])

# Create data matrix
data = [
    # Random split
    [acc_mega_top_1_random, acc_mega_top_5_random, acc_miewid_top_1_random, acc_miewid_top_5_random],
    [acc_mega_top_1_random_cropped, acc_mega_top_5_random_cropped, acc_miewid_top_1_random_cropped, acc_miewid_top_5_random_cropped],
    [acc_mega_top_1_random_rotated, acc_mega_top_5_random_rotated, acc_miewid_top_1_random_rotated, acc_miewid_top_5_random_rotated],
    [acc_mega_top_1_random_cropped_rotated, acc_mega_top_5_random_cropped_rotated, acc_miewid_top_1_random_cropped_rotated, acc_miewid_top_5_random_cropped_rotated],
    
    # Least similar split
    [acc_mega_top_1_least, acc_mega_top_5_least, acc_miewid_top_1_least, acc_miewid_top_5_least],
    [acc_mega_top_1_least_cropped, acc_mega_top_5_least_cropped, acc_miewid_top_1_least_cropped, acc_miewid_top_5_least_cropped],
    [acc_mega_top_1_least_rotated, acc_mega_top_5_least_rotated, acc_miewid_top_1_least_rotated, acc_miewid_top_5_least_rotated],
    [acc_mega_top_1_least_cropped_rotated, acc_mega_top_5_least_cropped_rotated, acc_miewid_top_1_least_cropped_rotated, acc_miewid_top_5_least_cropped_rotated],
    
    # Hard split
    [acc_mega_top_1_hard, acc_mega_top_5_hard, acc_miewid_top_1_hard, acc_miewid_top_5_hard],
    [acc_mega_top_1_hard_cropped, acc_mega_top_5_hard_cropped, acc_miewid_top_1_hard_cropped, acc_miewid_top_5_hard_cropped],
    [acc_mega_top_1_hard_rotated, acc_mega_top_5_hard_rotated, acc_miewid_top_1_hard_rotated, acc_miewid_top_5_hard_rotated],
    [acc_mega_top_1_hard_cropped_rotated, acc_mega_top_5_hard_cropped_rotated, acc_miewid_top_1_hard_cropped_rotated, acc_miewid_top_5_hard_cropped_rotated]
]

results_df = pd.DataFrame(data, index=row_idx, columns=col_idx)
results_df


# %%
# Save results as LaTeX table
latex_table = results_df.to_latex(
    float_format=lambda x: '{:.1f}\%'.format(x*100), # Convert to percentages with 1 decimal
    bold_rows=True,
    multicolumn=True,
    multicolumn_format='c',
    multirow=True
)

with open(artifacts_path/'baseline_results.tex', 'w') as f:
    f.write(latex_table)

# %%
all_results_df = pd.concat([
    acc_mega_top_5_hard_df.assign(split=hard, model=mega, debiasing=original),
    acc_mega_top_5_least_df.assign(split=least_similar, model=mega, debiasing=original), 
    acc_mega_top_5_random_df.assign(split=random, model=mega, debiasing=original),
    acc_miewid_top_5_hard_df.assign(split=hard, model=miewid, debiasing=original),
    acc_miewid_top_5_least_df.assign(split=least_similar, model=miewid, debiasing=original),
    acc_miewid_top_5_random_df.assign(split=random, model=miewid, debiasing=original),
    acc_mega_top_5_hard_cropped_df.assign(split=hard, model=mega, debiasing=bg_removed),
    acc_mega_top_5_least_cropped_df.assign(split=least_similar, model=mega, debiasing=bg_removed), 
    acc_mega_top_5_random_cropped_df.assign(split=random, model=mega, debiasing=bg_removed),
    acc_miewid_top_5_hard_cropped_df.assign(split=hard, model=miewid, debiasing=bg_removed),
    acc_miewid_top_5_least_cropped_df.assign(split=least_similar, model=miewid, debiasing=bg_removed), 
    acc_miewid_top_5_random_cropped_df.assign(split=random, model=miewid, debiasing=bg_removed),
    acc_mega_top_5_hard_rotated_df.assign(split=hard, model=mega, debiasing=rotated),
    acc_mega_top_5_least_rotated_df.assign(split=least_similar, model=mega, debiasing=rotated),
    acc_mega_top_5_random_rotated_df.assign(split=random, model=mega, debiasing=rotated),
    acc_miewid_top_5_hard_rotated_df.assign(split=hard, model=miewid, debiasing=rotated),
    acc_miewid_top_5_least_rotated_df.assign(split=least_similar, model=miewid, debiasing=rotated),
    acc_miewid_top_5_random_rotated_df.assign(split=random, model=miewid, debiasing=rotated),
    acc_mega_top_5_hard_cropped_rotated_df.assign(split=hard, model=mega, debiasing=bg_removed_rotated),
    acc_mega_top_5_least_cropped_rotated_df.assign(split=least_similar, model=mega, debiasing=bg_removed_rotated),
    acc_mega_top_5_random_cropped_rotated_df.assign(split=random, model=mega, debiasing=bg_removed_rotated),
    acc_miewid_top_5_hard_cropped_rotated_df.assign(split=hard, model=miewid, debiasing=bg_removed_rotated),
    acc_miewid_top_5_least_cropped_rotated_df.assign(split=least_similar, model=miewid, debiasing=bg_removed_rotated),
    acc_miewid_top_5_random_cropped_rotated_df.assign(split=random, model=miewid, debiasing=bg_removed_rotated),
])

all_results_df.to_csv(artifacts_path/'baseline_results.csv', index=False)
all_results_df

# %%
# Plot predictions from acc_mega_top_5_hard_df
def plot_predictions(query_id, query_image_name, pred_scores, pred_image_ids, pred_image_names, master_df, data_path, query_transforms, pred_transforms, denorm_mean, denorm_std, masked, model_name):
    plt.subplot(1, 6, 1)
    
    query_image_path = data_path/master_df.loc[(master_df['identity'] == int(query_id)) & (master_df['image_name'] == query_image_name)]['path'].values[0]
    
    if masked:
        rle = master_df.loc[(master_df['identity'] == int(query_id)) & (master_df['image_name'] == query_image_name)]['segmentation_mask_rle'].values[0]
        img = get_cropped_newt(query_image_path, rle)
    else:
        img = Image.open(query_image_path)

    # Un-normalize for visualization
    img_tensor = query_transforms(img)
    
    # Convert lists to tensors for proper broadcasting
    if isinstance(denorm_mean, list):
        denorm_mean = torch.tensor(denorm_mean).view(3, 1, 1)
    if isinstance(denorm_std, list):
        denorm_std = torch.tensor(denorm_std).view(3, 1, 1)
    
    img_tensor = img_tensor * denorm_std + denorm_mean
    
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.title(f'Query\nID: {query_id}', fontsize=8, bbox=dict(facecolor='lightblue', boxstyle='round', edgecolor='lightblue'))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.xlabel(f'{query_image_name}', fontsize=6, color='gray')
    plt.ylabel(f'{model_name}', fontsize=6, color='black')
    
    # Plot predictions
    for i in range(1, 6):
        plt.subplot(1, 6, i+1)
        pred_img_path = data_path/master_df.loc[(master_df['identity'] == int(pred_image_ids[i])) & (master_df['image_name'] == pred_image_names[i])]['path'].values[0]

        if masked:
            rle = master_df.loc[(master_df['identity'] == int(pred_image_ids[i])) & (master_df['image_name'] == pred_image_names[i])]['segmentation_mask_rle'].values[0]
            img = get_cropped_newt(pred_img_path, rle)
        else:
            img = Image.open(pred_img_path)
        
        # Un-normalize for visualization
        img_tensor = pred_transforms(img)
        
        # Convert lists to tensors for proper broadcasting
        if isinstance(denorm_mean, list):
            denorm_mean = torch.tensor(denorm_mean).view(3, 1, 1)
        if isinstance(denorm_std, list):
            denorm_std = torch.tensor(denorm_std).view(3, 1, 1)
        
        img_tensor = img_tensor * denorm_std + denorm_mean

        plt.imshow(img_tensor.permute(1, 2, 0))
        
        # Color code based on correctness
        color = 'lightgreen' if pred_image_ids[i] == query_id else 'lightcoral'
        plt.title(f'Score: {pred_scores[i]:.2f}\nID: {pred_image_ids[i]}', fontsize=8, bbox=dict(facecolor=color, boxstyle='round', edgecolor=color))
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.xlabel(f'{pred_image_names[i]}', fontsize=6, color='gray')
        # plt.axis('off')
    
    plt.tight_layout()

# %%
# Get image paths and plot first example
row = acc_mega_top_5_hard_df.iloc[0]
plt.figure(figsize=(7, 7))

plot_predictions(query_id=row['query', 'identity'], 
                query_image_name=row['query', 'image_name'], 
                pred_scores=row['score'],
                pred_image_ids=row['predicted_identity'], 
                pred_image_names=row['predicted_image_name'], 
                master_df=df,
                data_path=dataset_path,
                query_transforms=mega_transform,
                pred_transforms=mega_transform,
                denorm_mean=0.5,
                denorm_std=0.5,
                masked=True,
                model_name=mega)

results_preview_path = artifacts_path/'results_preview'
results_preview_path.mkdir(exist_ok=True)

# plt.savefig(results_preview_path/'prediction_example.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.close()

# %%
dict_of_pred_transforms = {
    f'{mega}/{original}': mega_transform,
    f'{miewid}/{original}': miewid_transform,
    f'{mega}/{bg_removed}': mega_transform,
    f'{miewid}/{bg_removed}': miewid_transform,
    f'{mega}/{rotated}': mega_transform_rotated,
    f'{miewid}/{rotated}': miewid_transform_rotated,
    f'{mega}/{bg_removed_rotated}': mega_transform_rotated,
    f'{miewid}/{bg_removed_rotated}': miewid_transform_rotated,
}

save_path = artifacts_path/'results_preview'
save_path.mkdir(exist_ok=True)

for i, row in tqdm(all_results_df[:5].iterrows(), total=len(all_results_df)):
    key = f'{row.model.iloc[0]}/{row.debiasing.iloc[0]}'
    pred_transforms = dict_of_pred_transforms[key]
    query_transforms = mega_transform if row.model.iloc[0] == mega else miewid_transform

    if row.model.iloc[0] == mega:
        denorm_mean = 0.5
        denorm_std = 0.5
    else:
        denorm_mean = [0.485, 0.456, 0.406] 
        denorm_std = [0.229, 0.224, 0.225]

    masked = row.debiasing.iloc[0] == bg_removed or row.debiasing.iloc[0] == bg_removed_rotated

    plt.figure(figsize=(7, 7))
    plot_predictions(query_id=row['query', 'identity'], 
                query_image_name=row['query', 'image_name'], 
                pred_scores=row['score'],
                pred_image_ids=row['predicted_identity'], 
                pred_image_names=row['predicted_image_name'], 
                master_df=df,
                data_path=dataset_path,
                query_transforms=query_transforms,
                pred_transforms=pred_transforms,
                denorm_mean=denorm_mean,
                denorm_std=denorm_std,
                masked=masked,
                model_name=row.model.iloc[0])

    file_name = f'query_id_{row["query", "identity"]}_{row["query", "image_name"]}_{row.model.iloc[0]}_{row.debiasing.iloc[0]}_{row.split.iloc[0]}.svg'
    plt.savefig(save_path/file_name, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# %%
all_results_df[all_results_df.model == miewid]