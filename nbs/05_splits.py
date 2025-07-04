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
from gcn_reid.newt_dataset import upload_to_kaggle
from pathlib import Path
from gcn_reid.newt_dataset import download_kaggle_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# %%
dataset_name = 'mshahoyi/newts-segmented-new'
dataset_path = Path('data/newts-segmented-new')
download_kaggle_dataset(dataset_name, dataset_path)

# %% [markdown]
# ## Download and ready both models

# %%

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# %% [markdown]
# ## Run both models on all images and save the results
# Artifacts are a dataframe like the newt dataframe but that contains two new columns representing the mega and miewid embeddings.

# %%
df_original = pd.read_csv(dataset_path / 'metadata.csv')
df = df_original.copy()
df = df[~df.is_video].reset_index(drop=True)
df

# %%
artifacts_path = Path('artifacts')
artifacts_path.mkdir(exist_ok=True)
artifacts_name = 'metadata_with_features.csv'

# %%
if not (artifacts_path/artifacts_name).exists():
    batch_size = 64
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        images = [Image.open(dataset_path / row['file_path']) for _, row in batch.iterrows()]
        inputs = dinov2_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            last_hidden_states = outputs.last_hidden_state[:, 0, :] # select the CLS token embedding
        features = pd.Series(last_hidden_states.cpu().tolist(), index=batch.index)
        df.loc[batch.index, 'dinov2_features'] = features
    df.to_csv(artifacts_path/artifacts_name, index=False)
else: 
    df = pd.read_csv(artifacts_path/artifacts_name)
    df['dinov2_features'] = df['dinov2_features'].apply(eval)

# %% [markdown]
# ## Get all cosine similarities and save the highest correct match and the highest incorrect match scores and indices
# We will have 8 new columns: mega_highest_correct_score, mega_highest_correct_idx, mega_highest_incorrect_score, mega_highest_incorrect_idx, miewid_highest_correct_score, miewid_highest_correct_idx, miewid_highest_incorrect_score, miewid_highest_incorrect_idx
# Convert string representation of features back to arrays

# %%
dinov2_features = np.array(df['dinov2_features'].tolist())

# Calculate cosine similarities manually
def cosine_similarity(a, b):
    # Normalize the vectors
    a_norm = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    # Calculate similarity matrix
    return np.dot(a_norm, b_norm.T)

dinov2_similarities = cosine_similarity(dinov2_features, dinov2_features)

dinov2_similarities.shape

# %%
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

    # Get highest correct and incorrect similarities
    correct_sims = sims[correct_mask]
    incorrect_sims = sims[incorrect_mask]

    if correct_sims.size > 0:
        highest_correct_idx = other_indices[correct_mask][np.argmax(correct_sims)]
        highest_correct_score = np.max(correct_sims)
    else:
        highest_correct_idx = np.nan
        highest_correct_score = np.nan

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
for i, (k, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
    # Get current newt ID
    dinov2_highest_correct_idx, dinov2_highest_correct_score, dinov2_highest_incorrect_idx, dinov2_highest_incorrect_score = get_highest_correct_and_incorrect_matches(df, dinov2_similarities, i, row)
    
    df.at[k, 'highest_correct_score'] = dinov2_highest_correct_score
    df.at[k, 'highest_correct_idx'] = dinov2_highest_correct_idx
    df.at[k, 'highest_incorrect_score'] = dinov2_highest_incorrect_score
    df.at[k, 'highest_incorrect_idx'] = dinov2_highest_incorrect_idx


# %% [markdown]
# ## Calculate the rightness score for each image and model.

# %%
df['rightness_score'] = df['highest_correct_score'] - df['highest_incorrect_score']

# %%
df.highest_correct_score.hist(bins=50)

# %%
# Plot the 5 least correct images with their matches
num_images = 1

sorted_df = df.sort_values(by=['rightness_score'], ascending=True).reset_index(drop=True)
for i, row in tqdm(sorted_df[:num_images].iterrows(), total=num_images):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot query image
    query_path = dataset_path / row['file_path']
    print(query_path)
    axes[0].imshow(plt.imread(query_path))
    axes[0].set_title(f'Query\nID: {row["identity"]} - {row.file_name}\n{row.creation_date}')
    axes[0].axis('off')

    # Define matches to plot
    matches = [
        {'type': 'DINOv2 Correct', 'score_col': 'highest_correct_score', 'idx_col': 'highest_correct_idx', 'ax_idx': 1},
        {'type': 'DINOv2 Incorrect', 'score_col': 'highest_incorrect_score', 'idx_col': 'highest_incorrect_idx', 'ax_idx': 2},
    ]

    # Plot each match
    for match in matches:
        match_row = df.iloc[int(row[match['idx_col']])]
        match_path = dataset_path / match_row['file_path']
        ax = axes[match['ax_idx']]
        ax.imshow(plt.imread(match_path))
        ax.set_title(f'{match["type"]}\nScore: {row[match["score_col"]]:.3f}\n{match_row.identity}/{match_row.file_name}\n{row.creation_date}', fontsize=10)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(artifacts_path/f'least_correct_matches_rightness_score_{i}.png')
    plt.close(fig)


# %% [markdown]
# ## Sort images by rightness score in an ascending order

# %% [markdown]
# ## Mark query and database images
# Starting with the least right images, mark the query and database images. Skip images that are already marked (this means they are the database for another image).

# %%
n_ind_test = 30
n_ind_val = 30

df_original['is_hard_test_query'] = pd.NA
df_original['is_hard_val_query'] = pd.NA

sorted_df = df.loc[df.groupby('identity')['rightness_score'].idxmin()].sort_values(by=['rightness_score'], ascending=True).head(n_ind_test + n_ind_val)
sorted_df.identity.nunique()

# %%
for count, (i, row) in enumerate(sorted_df.iterrows()):
    col = 'is_hard_test_query' if count < n_ind_test else 'is_hard_val_query'

    # Make other images of the same newt a database
    df_original.loc[df_original['identity'] == row['identity'], col] = False

    # Make the newt itself a query
    df_original.loc[(df_original['identity'] == row['identity']) & (df_original.file_name == row['file_name']), col] = True


# %%
df_original.is_hard_test_query.value_counts()

# %%
df_original.is_hard_val_query.value_counts()

# %% [markdown]
# # Create least similar split

# %%
df_original['is_least_similar_test_query'] = pd.NA
df_original['is_least_similar_val_query'] = pd.NA

least_similar_df = df.loc[df.groupby('identity')['highest_correct_score'].idxmin()].sort_values(by=['highest_correct_score'], ascending=True).head(n_ind_test + n_ind_val)
least_similar_df.identity.nunique()

# %%
for count, (i, row) in enumerate(least_similar_df.iterrows()):
    col = 'is_least_similar_test_query' if count < n_ind_test else 'is_least_similar_val_query'

    # Make other images of the same newt a database
    df_original.loc[df_original['identity'] == row['identity'], col] = False

    # Make the newt itself a query
    df_original.loc[(df_original['identity'] == row['identity']) & (df_original.file_name == row['file_name']), col] = True


# %%
df_original.is_least_similar_test_query.value_counts()

# %%
df_original.is_least_similar_val_query.value_counts()

# %% [markdown]
# # Create random split

# %%
df_original['is_random_test_query'] = pd.NA
df_original['is_random_val_query'] = pd.NA

# Set random seed for reproducibility
rng = np.random.default_rng(seed=42)

random_df = df.loc[df.groupby('identity').apply(lambda x: x.sample(n=1, random_state=rng).index[0], include_groups=False)].head(n_ind_test + n_ind_val)
random_df.identity.nunique()

# %%
for count, (i, row) in enumerate(random_df.iterrows()):
    col = 'is_random_test_query' if count < n_ind_test else 'is_random_val_query'

    # Make other images of the same newt a database
    df_original.loc[df_original['identity'] == row['identity'], col] = False

    # Make the newt itself a query
    df_original.loc[(df_original['identity'] == row['identity']) & (df_original.file_name == row['file_name']), col] = True



# %%
df_original.is_random_test_query.value_counts()

# %%
df_original.is_random_val_query.value_counts()

# %% [markdown]
# # Save the splits

# %%
df_original

# %%
df_original.to_csv(dataset_path/'metadata.csv', index=False)

# %% [markdown]
# # Create Kaggle dataset

# %%
upload_to_kaggle(
    user_id='mshahoyi',
    title='GCN-ID 2024',
    id='gcn-id-2024',
    licenses=[{"name": "CC0-1.0"}],
    keywords=['gcn-id', '2024'],
    dataset_dir=dataset_path
)

# %%
import nbdev; nbdev.nbdev_export()
# %%
