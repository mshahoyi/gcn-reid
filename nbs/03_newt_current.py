# %% [markdown]
# # Newt Current
# > This notebook shows the current state of the newt re-identification models on our dataset.

# %%
#| default_exp newt

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
from wildlife_tools.data import ImageDataset
from gcn_reid.segmentation import decode_rle_mask

# %% [markdown]
# # Create Dataset Class

#%%
#| export
def get_newt_dataset():
    from wildlife_datasets import datasets
    import os
    import kaggle
    import pandas as pd

    class BarhillNewtsDataset(datasets.WildlifeDataset):
        @classmethod
        def _download(cls, dataset_name, download_path):
            if not os.path.exists(download_path):
                os.makedirs(download_path, exist_ok=True)
                kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
                print(f"Dataset downloaded to {download_path}")
            else:
                print(f"Dataset already exists at {download_path}")

        def create_catalogue(self) -> pd.DataFrame:
            df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
            return df.rename(columns={"newt_id": "identity", "image_id": "image_name", "image_path": "path"})
    
    return BarhillNewtsDataset

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
            
            relative_path = path.replace(self.root, "")[1:] # remove leading /
            rle = self.metadata[self.metadata[self.col_path] == relative_path][self.rle_col].values
            if len(rle) == 0: return Image.fromarray(img)
            return get_cropped_newt(path, rle[0])

    return CroppingImageDataset

# %%
CroppingImageDataset = get_cropping_image_dataset()

# %%
dataset_path = "data/newt_dataset"
NewtDataset = get_newt_dataset()
NewtDataset._download(dataset_name="mshahoyi/barhill-newts-segmented", download_path=dataset_path)
dataset = NewtDataset(dataset_path)
dataset.df.head()

# %%
dataset.plot_grid()

# %%
analysis.display_statistics(dataset.df)

# %% [markdown]
# # Create Query and Database Sets

# %%
splitter = splits.ClosedSetSplit(0.9)
for idx_database, idx_query in splitter.split(dataset.df):
    df_database, df_query = dataset.df.loc[idx_database], dataset.df.loc[idx_query]
    splits.analyze_split(dataset.df, idx_database, idx_query)

# %% [markdown]
# # Test MegaDescriptor

# %%
transform = T.Compose([T.Resize([384, 384]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset_database = datasets.WildlifeDataset(df=df_database, root=dataset.root, transform=transform)
dataset_query = datasets.WildlifeDataset(df=df_query, root=dataset.root, transform=transform)

# %%
name = 'hf-hub:BVRA/MegaDescriptor-L-384'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True), 
                         device=device,
                         batch_size=32,
                         num_workers=4,
                         )
print("model loaded to device:", device)

# %%
query = extractor(CroppingImageDataset(dataset_query.df, root=dataset_query.root, transform=dataset_query.transform, crop_out=False))
database = extractor(CroppingImageDataset(dataset_database.df, root=dataset_database.root, transform=dataset_database.transform, crop_out=False))

# %%
similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)

# %%
top_5_classifier = TopkClassifier(k=5, database_labels=dataset_database.labels_string, return_all=True)
predictions_top_5, scores_top_5, _ = top_5_classifier(similarity)

#%%
accuracy_top_1 = np.mean(dataset_query.labels_string == predictions_top_5[:, 0])
accuracy_top_5 = np.mean(np.any(predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))

accuracy_top_1, accuracy_top_5

# %%
def calculate_map(query_labels, database_labels, similarity_matrix):
    """
    Calculate mean Average Precision (mAP) for retrieval task.
    
    Args:
        query_labels: Array of query labels
        database_labels: Array of database labels  
        similarity_matrix: Similarity scores between queries and database
    
    Returns:
        mAP: Mean Average Precision
    """
    aps = []
    
    for i, query_label in enumerate(query_labels):
        # Get similarity scores for this query
        scores = similarity_matrix[i]
        
        # Create binary relevance labels (1 if same identity, 0 otherwise)
        relevance = (database_labels == query_label).astype(int)
        
        # Calculate Average Precision for this query
        if np.sum(relevance) > 0:  # Only if there are relevant items
            ap = average_precision_score(relevance, scores)
            aps.append(ap)
    
    return np.mean(aps)

# Calculate mAP
map_score = calculate_map(dataset_query.labels_string, dataset_database.labels_string, similarity)
print(f"Mean Average Precision (mAP): {map_score:.4f}")

# %%
#| export
def plot_retrieval_results(dataset_query, dataset_database, similarity_matrix, crop_out=False, mode = "mistakes", num_results=4, num_queries=5, figsize=(15, 20)):
    """
    Plot retrieval results showing query images and their most similar matches.
    
    Args:
        dataset_query: Query dataset with images and labels
        dataset_database: Database dataset with images and labels
        similarity_matrix: Similarity scores between queries and database
        num_results: Number of top similar images to show per query
        num_queries: Number of query images to display
        figsize: Figure size for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(num_queries, num_results + 1, figsize=figsize)
    
    if mode == "mistakes":
        # We'll identify mistakes by checking if the top match has the same label as query
        # Get top matches for each query
        top_matches = np.argmax(similarity_matrix, axis=1)
        top_match_labels = dataset_database.labels_string[top_matches]
        query_indices = np.where(dataset_query.labels_string != top_match_labels)[0]
    elif mode == "all":
        query_indices = np.arange(len(dataset_query))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for row, query_idx in enumerate(query_indices[:num_queries]):
        query_label = dataset_query.labels_string[query_idx]
        
        get_image_path = lambda idx, ds: os.path.join(ds.root, ds.df.iloc[idx].path)
        if crop_out:
            query_image = get_cropped_newt(get_image_path(query_idx, dataset_query), dataset_query.df.iloc[query_idx].segmentation_mask_rle)
        else:
            query_image = dataset_query.get_image(query_idx)
        
        # Get top similar images for this query
        similarities = similarity_matrix[query_idx]
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # Plot query image
        axes[row, 0].imshow(query_image)
        axes[row, 0].set_title(f'Query: {query_label}', fontweight='bold')
        
        # Set blue border for query image and remove ticks
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        
        # Plot top similar images
        for col, db_idx in enumerate(top_indices):
            db_label = dataset_database.labels_string[db_idx]
            if crop_out:
                db_image = get_cropped_newt(get_image_path(db_idx, dataset_database), dataset_database.df.iloc[db_idx].segmentation_mask_rle)
            else:
                db_image = dataset_database.get_image(db_idx)    
            similarity_score = similarities[db_idx]
            
            axes[row, col + 1].imshow(db_image)
            axes[row, col + 1].set_title(f'{db_label}\nSim: {similarity_score:.3f}', fontsize=10)
            
            # Set border color based on whether it's a correct match
            border_color = 'green' if db_label == query_label else 'red'
            for spine in axes[row, col + 1].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle('Retrieval Results: Query (Blue) vs Retrieved Images (Green=Correct, Red=Incorrect)', 
                 fontsize=14, y=1.02)
    plt.show()

# %%
# Plot retrieval results
plot_retrieval_results(dataset_query, dataset_database, similarity, num_results=4, num_queries=5)

# %% [markdown]
# # Test MiewID

# %%
#| hide_output: true
miew_id_model = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True)

miew_id_extractor = DeepFeatures(miew_id_model, 
                         device=device,
                         batch_size=32,
                         num_workers=4,
                         )

# %%
miew_id_query = miew_id_extractor(ImageDataset(dataset_query.df, root=dataset_query.root, transform=dataset_query.transform))
miew_id_database = miew_id_extractor(ImageDataset(dataset_database.df, root=dataset_database.root, transform=dataset_database.transform))

# %%
miew_id_similarity = similarity_function(miew_id_query, miew_id_database)
miew_id_predictions_top_5, miew_id_scores_top_5, _ = top_5_classifier(miew_id_similarity)

#%%
miew_id_accuracy_top_1 = np.mean(dataset_query.labels_string == miew_id_predictions_top_5[:, 0])
miew_id_accuracy_top_5 = np.mean(np.any(miew_id_predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))

miew_id_accuracy_top_1, miew_id_accuracy_top_5

# %%
miew_id_map_score = calculate_map(dataset_query.labels_string, dataset_database.labels_string, miew_id_similarity)
miew_id_map_score

# %%
plot_retrieval_results(dataset_query, dataset_database, miew_id_similarity, mode="mistakes", num_results=4, num_queries=5)

# %% [markdown]
# # Test on Cropped Newts

# %%

# %%
cropped_mega_query = extractor(CroppingImageDataset(dataset_query.df, root=dataset_query.root, transform=dataset_query.transform))
cropped_mega_database = extractor(CroppingImageDataset(dataset_database.df, root=dataset_database.root, transform=dataset_database.transform))

# %%
cropped_mega_similarity = similarity_function(cropped_mega_query, cropped_mega_database)
cropped_mega_predictions_top_5, cropped_mega_scores_top_5, _ = top_5_classifier(cropped_mega_similarity)

#%%
cropped_mega_accuracy_top_1 = np.mean(dataset_query.labels_string == cropped_mega_predictions_top_5[:, 0])
cropped_mega_accuracy_top_5 = np.mean(np.any(cropped_mega_predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))

cropped_mega_accuracy_top_1, cropped_mega_accuracy_top_5

# %% 
cropped_mega_map_score = calculate_map(dataset_query.labels_string, dataset_database.labels_string, cropped_mega_similarity)
cropped_mega_map_score

# %%
plot_retrieval_results(dataset_query, dataset_database, cropped_mega_similarity, crop_out=True, mode="mistakes", num_results=4, num_queries=5)

# %% [markdown]
# # Test Cropped out newts on MiewID  

# %%
miew_id_cropped_query = miew_id_extractor(CroppingImageDataset(dataset_query.df, root=dataset_query.root, transform=dataset_query.transform))
miew_id_cropped_database = miew_id_extractor(CroppingImageDataset(dataset_database.df, root=dataset_database.root, transform=dataset_database.transform))

# %%
miew_id_cropped_similarity = similarity_function(miew_id_cropped_query, miew_id_cropped_database)
miew_id_cropped_predictions_top_5, miew_id_cropped_scores_top_5, _ = top_5_classifier(miew_id_cropped_similarity)

# %%
miew_id_cropped_accuracy_top_1 = np.mean(dataset_query.labels_string == miew_id_cropped_predictions_top_5[:, 0])
miew_id_cropped_accuracy_top_5 = np.mean(np.any(miew_id_cropped_predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))

miew_id_cropped_accuracy_top_1, miew_id_cropped_accuracy_top_5

# %%
miew_id_cropped_map_score = calculate_map(dataset_query.labels_string, dataset_database.labels_string, miew_id_cropped_similarity)
miew_id_cropped_map_score

# %%
plot_retrieval_results(dataset_query, dataset_database, miew_id_cropped_similarity, crop_out=True, mode="mistakes", num_results=4, num_queries=5)

# %% [markdown]
# # Create a dataframe of the results

# %%
results = pd.DataFrame({
    "model": ["MegaDescriptor-L-384", "MiewID", "MegaDescriptor-L-384 (cropped)", "MiewID (cropped)"],
    "accuracy_top_5": [accuracy_top_5, miew_id_accuracy_top_5, cropped_mega_accuracy_top_5, miew_id_cropped_accuracy_top_5],
    "accuracy_top_1": [accuracy_top_1, miew_id_accuracy_top_1, cropped_mega_accuracy_top_1, miew_id_cropped_accuracy_top_1],
    "map_score": [map_score, miew_id_map_score, cropped_mega_map_score, miew_id_cropped_map_score]
})

results

# %%
results.set_index("model").transpose().plot.bar(figsize=(10, 5), rot=0)

# %%
import nbdev
nbdev.export.nbdev_export()