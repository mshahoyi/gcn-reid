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
from tqdm import tqdm

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
    df['mega_features'] = [features.tolist() for features in mega_results.features]
    df['miewid_features'] = [features.tolist() for features in miewid_results.features]
    df.to_csv(artifacts_path/artifacts_name, index=False)
else: df = pd.read_csv(artifacts_path/artifacts_name)

# %% [markdown]
# ## Get all cosine similarities and save the highest correct match and the highest incorrect match scores and indices
# We will have 8 new columns: mega_highest_correct_score, mega_highest_correct_idx, mega_highest_incorrect_score, mega_highest_incorrect_idx, miewid_highest_correct_score, miewid_highest_correct_idx, miewid_highest_incorrect_score, miewid_highest_incorrect_idx
# Convert string representation of features back to arrays

# %%
df['mega_features'] = df['mega_features'].apply(eval)
df['miewid_features'] = df['miewid_features'].apply(eval)

# %%
mega_features = np.array(df['mega_features'].tolist())
miewid_features = np.array(df['miewid_features'].tolist())

# Calculate cosine similarities manually
def cosine_similarity(a, b):
    # Normalize the vectors
    a_norm = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    # Calculate similarity matrix
    return np.dot(a_norm, b_norm.T)

mega_similarities = cosine_similarity(mega_features, mega_features)
miewid_similarities = cosine_similarity(miewid_features, miewid_features)

mega_similarities.shape, miewid_similarities.shape

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


# %% [markdown]
# ## Calculate the rightness score for each image and model.

# %%
df['mega_rightness_score'] = df['mega_highest_correct_score'] - df['mega_highest_incorrect_score']
df['miewid_rightness_score'] = df['miewid_highest_correct_score'] - df['miewid_highest_incorrect_score']
df['rightness_score'] = df['mega_rightness_score'] + df['miewid_rightness_score']

# %%
df.to_csv(artifacts_path/artifacts_name, index=False)
df

# %%
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# df.mega_rightness_score.hist(bins=50)
# plt.title('Mega Rightness Score')
# plt.subplot(1, 3, 2)
# df.miewid_rightness_score.hist(bins=50)
# plt.title('Miewid Rightness Score')
# plt.subplot(1, 3, 3)
# df.rightness_score.hist(bins=50)
# plt.title('Rightness Score')
# plt.savefig(artifacts_path/'rightness_scores.png')

# %%
# Plot the 5 least correct images with their matches
num_images = 100

sorted_df = df.sort_values(by=['mega_highest_correct_score'], ascending=False).reset_index(drop=True)
for i, row in tqdm(sorted_df[:num_images:10].iterrows(), total=num_images):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    # Plot query image
    query_path = dataset_path / row['path']
    axes[0].imshow(plt.imread(query_path))
    axes[0].set_title(f'Query\nID: {row["identity"]} - {row.file_name}')
    axes[0].axis('off')

    # Define matches to plot
    matches = [
        {'type': 'Mega Correct', 'score_col': 'mega_highest_correct_score', 'idx_col': 'mega_highest_correct_idx', 'ax_idx': 1},
        {'type': 'Mega Incorrect', 'score_col': 'mega_highest_incorrect_score', 'idx_col': 'mega_highest_incorrect_idx', 'ax_idx': 2},
        {'type': 'Miewid Correct', 'score_col': 'miewid_highest_correct_score', 'idx_col': 'miewid_highest_correct_idx', 'ax_idx': 3},
        {'type': 'Miewid Incorrect', 'score_col': 'miewid_highest_incorrect_score', 'idx_col': 'miewid_highest_incorrect_idx', 'ax_idx': 4},
    ]

    # Plot each match
    for match in matches:
        match_row = df.iloc[int(row[match['idx_col']])]
        match_path = dataset_path / match_row['path']
        ax = axes[match['ax_idx']]
        ax.imshow(plt.imread(match_path))
        ax.set_title(f'{match["type"]}\nScore: {row[match["score_col"]]:.3f}\n{match_row.identity}-{match_row.file_name}')
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(artifacts_path/f'correct_matches_mega_idx_{i}.png')
    plt.close(fig)


# %% [markdown]
# ## Sort images by rightness score in an ascending order

# %% [markdown]
# ## Mark query and database images
# Starting with the least right images, mark the query and database images. Skip images that are already marked (this means they are the database for another image).

# %%
def mark_query_and_database(df):
    df['is_query'] = pd.NA

    for i, row in df.sort_values(by=['rightness_score'], ascending=True).iterrows():
        mega_incorrect_idx = row['mega_highest_incorrect_idx']
        miewid_incorrect_idx = row['miewid_highest_incorrect_idx']
        if pd.notna(df.at[i, 'is_query']) or (pd.notna(df.at[mega_incorrect_idx, 'is_query']) or pd.notna(df.at[miewid_incorrect_idx, 'is_query'])) or (df[df['identity'] == row['identity']]['is_query'] == True).any():
            continue

        # All the other images of the same newt become the database
        df.loc[df['identity'] == row['identity'], 'is_query'] = False

        # Mark the image itself as query
        df.at[i, 'is_query'] = True

        # Mark the incorrect matches as database
        df.at[int(mega_incorrect_idx), 'is_query'] = False
        df.at[int(miewid_incorrect_idx), 'is_query'] = False

    return df

# %%
def test_mark_query_and_database():
    data = {
        'identity':                   ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
        'rightness_score':            [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
        'mega_highest_incorrect_idx': [2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0],
        'miewid_highest_incorrect_idx':[3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0]
    }
    test_df = pd.DataFrame(data)
    
    processed_df = mark_query_and_database(test_df.copy())
    
    expected_is_query_list = [True, False, False, False, True, False, False, False]
    expected_is_query = pd.Series(expected_is_query_list, name='is_query')
    
    unassigned_mask = processed_df['is_query'].isna()
    assert not unassigned_mask.any(), f"Found unassigned queries: \n{processed_df[unassigned_mask]}"

    pd.testing.assert_series_equal(
        processed_df['is_query'].astype(bool), 
        expected_is_query,
        check_names=False,
        check_dtype=False
    )
    print("test_mark_query_and_database passed!")

test_mark_query_and_database()

# %%
df = mark_query_and_database(df)
df.is_query.value_counts()

# %% [markdown]
# ## Create splits
# The first 50 queries, along with all the other images of that newt go to the test set and automatically become the database. The marked false images also need to go to the test set. Do this until the test set is the right proportion.

# %% [markdown]
# ## Create splits based on the rightness score

# %% [markdown]
# ## Create splits based on the rightness score