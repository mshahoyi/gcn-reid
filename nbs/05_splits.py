# %% [markdown]
# # Splits
# > This notebook deals with creating splits for the newt dataset.

# %%
#| default_exp splits

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
from pathlib import Path
from gcn_reid.newt_dataset import download_kaggle_dataset

# %%
dataset_name = 'mshahoyi/barhill-newts-all'
dataset_path = Path('data/gcns-processed')
download_kaggle_dataset(dataset_name, dataset_path)

# %% [markdown]
# ## Download and ready both models

# %%
mega = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0)
miewid = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True)

# %% [markdown]
# ## Run both models on all images and save the results
# Artifacts are a dataframe like the newt dataframe but that contains two new columns representing the mega and miewid embeddings.

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mega_extractor = DeepFeatures(mega, device=device, batch_size=32, num_workers=4)
miewid_extractor = DeepFeatures(miewid, device=device, batch_size=32, num_workers=4)

# %%
df = pd.read_csv(dataset_path / 'metadata.csv')
df = df[~df.is_video].reset_index(drop=True)
df = df[~df.is_video].rename(columns={"image_name": "image_name", "file_path": "path"})
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

mega_dataset = ImageDataset(df, root=dataset_path, transform=mega_transform)
miewid_dataset = ImageDataset(df, root=dataset_path, transform=miewid_transform)

# %%
num_images = 1
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    x, y= next(iter(mega_dataset))
    plt.imshow(x.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mega_images.png')

# %%
artifacts_path = Path('artifacts')
artifacts_path.mkdir(exist_ok=True)
artifacts_name = 'metadata_with_features.csv'

# %%
if not (artifacts_path/artifacts_name).exists():
    mega_results = mega_extractor(mega_dataset)
    miewid_results = miewid_extractor(miewid_dataset)

# %%
df['mega_features'] = [features.tolist() for features in mega_results.features]
df['miewid_features'] = [features.tolist() for features in miewid_results.features]

# %%
df.head()

# %%
df.to_csv(artifacts_path/artifacts_name, index=False)

# %%

# %% [markdown]
# ## Get all cosine similarities and save the highest correct match and the highest incorrect match scores and indices
# We will have 8 new columns: mega_highest_correct_score, mega_highest_correct_idx, mega_highest_incorrect_score, mega_highest_incorrect_idx, miewid_highest_correct_score, miewid_highest_correct_idx, miewid_highest_incorrect_score, miewid_highest_incorrect_idx

# %%
similarity_function = CosineSimilarity()
mega_similarities = similarity_function(mega_results, mega_results)
miewid_similarities = similarity_function(miewid_results, miewid_results)

# %%
other_indices = np.arange(len(df))

def get_highest_correct_and_incorrect_matches(df, similarities, i, row):
    other_indices = np.arange(len(df))

    # Get current newt ID
    current_newt_id = row['identity']
    
    # Get similarities for this image
    sims = similarities[i]

    # Get masks for correct and incorrect matches
    correct_mask = df['identity'] == current_newt_id
    incorrect_mask = df['identity'] != current_newt_id

    # Remove self from correct matches
    correct_mask[i] = False

    # Get highest correct and incorrect similarities for mega
    correct_sims = sims[correct_mask]
    incorrect_sims = sims[incorrect_mask]

    highest_correct_idx = other_indices[correct_mask][np.argmax(correct_sims)]
    highest_correct_score = np.max(correct_sims)

    highest_incorrect_idx = other_indices[incorrect_mask][np.argmax(incorrect_sims)]
    highest_incorrect_score = np.max(incorrect_sims)

    return highest_correct_idx, highest_correct_score, highest_incorrect_idx, highest_incorrect_score

# %%
# Test the get_highest_correct_and_incorrect_matches function
def test_get_highest_correct_and_incorrect_matches():
    # Create a small test dataset
    test_df = pd.DataFrame({
        'identity': ['A', 'A', 'A', 'B', 'B', 'C'],
    })
    
    # Create a test similarity matrix
    # Each row represents similarities to all other images
    test_similarities = np.array([
        [1.0, 0.8, 0.7, 0.9, 0.3, 0.2],  # Image 0 similarities
        [0.8, 1.0, 0.9, 0.4, 0.5, 0.3],  # Image 1 similarities 
        [0.7, 0.9, 1.0, 0.3, 0.4, 0.6],  # Image 2 similarities
        [0.9, 0.4, 0.3, 1.0, 0.8, 0.4],  # Image 3 similarities
        [0.3, 0.5, 0.4, 0.8, 1.0, 0.5],  # Image 4 similarities
        [0.2, 0.3, 0.6, 0.4, 0.5, 1.0],  # Image 5 similarities
    ])
    
    # Test cases
    test_cases = [
        {
            'idx': 0,  # Testing first image (identity A)
            'expected': {
                'correct_idx': 1,  # Should match with image 1 (identity A)
                'correct_score': 0.8,
                'incorrect_idx': 3,  # Should match with image 3 (identity B) 
                'incorrect_score': 0.9
            }
        },
        {
            'idx': 3,  # Testing fourth image (identity B)
            'expected': {
                'correct_idx': 4,  # Should match with image 4 (identity B)
                'correct_score': 0.8,
                'incorrect_idx': 0,  # Should match with image 0 (identity A)
                'incorrect_score': 0.9
            }
        }
    ]
    
    for test in test_cases:
        idx = test['idx']
        expected = test['expected']
        
        correct_idx, correct_score, incorrect_idx, incorrect_score = get_highest_correct_and_incorrect_matches(
            test_df, test_similarities, idx, test_df.iloc[idx]
        )
        
        # Assert the results match expected values
        assert correct_idx == expected['correct_idx'], f"Test failed for idx {idx}: Expected correct_idx {expected['correct_idx']}, got {correct_idx}"
        assert np.isclose(correct_score, expected['correct_score']), f"Test failed for idx {idx}: Expected correct_score {expected['correct_score']}, got {correct_score}"
        assert incorrect_idx == expected['incorrect_idx'], f"Test failed for idx {idx}: Expected incorrect_idx {expected['incorrect_idx']}, got {incorrect_idx}"
        assert np.isclose(incorrect_score, expected['incorrect_score']), f"Test failed for idx {idx}: Expected incorrect_score {expected['incorrect_score']}, got {incorrect_score}"
    
    print("All tests passed!")

# Run the tests
test_get_highest_correct_and_incorrect_matches()


# %%
for i, row in df.iterrows():
    # Get current newt ID
    mega_highest_correct_idx, mega_highest_correct_score, mega_highest_incorrect_idx, mega_highest_incorrect_score = get_highest_correct_and_incorrect_matches(df, mega_similarities, i, row)
    miewid_highest_correct_idx, miewid_highest_correct_score, miewid_highest_incorrect_idx, miewid_highest_incorrect_score = get_highest_correct_and_incorrect_matches(df, miewid_similarities, i, row)
    
    # Assign values to dataframe
    df.at[i, 'mega_highest_correct_score'] = mega_highest_correct_score
    df.at[i, 'mega_highest_correct_idx'] = mega_highest_correct_idx
    df.at[i, 'mega_highest_incorrect_score'] = mega_highest_incorrect_score
    df.at[i, 'mega_highest_incorrect_idx'] = mega_highest_incorrect_idx
    
    df.at[i, 'miewid_highest_correct_score'] = miewid_highest_correct_score
    df.at[i, 'miewid_highest_correct_idx'] = miewid_highest_correct_idx
    df.at[i, 'miewid_highest_incorrect_score'] = miewid_highest_incorrect_score
    df.at[i, 'miewid_highest_incorrect_idx'] = miewid_highest_incorrect_idx

# %%
df.to_csv(artifacts_path/artifacts_name, index=False)
df

# %% [markdown]
# ## Calculate the rightness score for each image and model.

# %%
df['mega_rightness_score'] = df['mega_highest_correct_score'] - df['mega_highest_incorrect_score']
df['miewid_rightness_score'] = df['miewid_highest_correct_score'] - df['miewid_highest_incorrect_score']
df['rightness_score'] = df['mega_rightness_score'] + df['miewid_rightness_score']

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
df.mega_rightness_score.hist(bins=50)
plt.title('Mega Rightness Score')
plt.subplot(1, 3, 2)
df.miewid_rightness_score.hist(bins=50)
plt.title('Miewid Rightness Score')
plt.subplot(1, 3, 3)
df.rightness_score.hist(bins=50)
plt.title('Rightness Score')
plt.savefig(artifacts_path/'rightness_scores.png')

# %%


# %% [markdown]
# ## Sort images by rightness score in an ascending order

# %%
df = df.sort_values(by=['miewid_rightness_score'], ascending=True)

# %% [markdown]
# ## Mark query and database images
# Starting with the least right images, mark the query and database images. Skip images that are already marked (this means they are the database for another image).

# %% [markdown]
# ## Create splits
# The first 50 queries, along with all the other images of that newt go to the test set and automatically become the database. The marked false images also need to go to the test set. Do this until the test set is the right proportion.

# %% [markdown]
# ## Create splits based on the rightness score

# %% [markdown]
# ## Create splits based on the rightness score