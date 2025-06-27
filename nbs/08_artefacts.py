# %% [markdown]
# # Artefacts
# > This notebook is used to create the artefacts for the GCN-ID 2024 dataset paper.

# %%
#| default_exp artefacts

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
from IPython import display


# %%
dataset_name = 'mshahoyi/gcn-id-2024'
dataset_path = Path('data/gcn-id-2024')
download_kaggle_dataset(dataset_name, dataset_path)

# %%
metadata = pd.read_csv(dataset_path / 'metadata.csv')
metadata

# %%
artifacts_path = Path('artifacts')

# %%
dino_features_df = pd.read_csv(artifacts_path / 'metadata_with_features.csv')
dino_features_df['dinov2_features'] = dino_features_df['dinov2_features'].apply(eval)
dino_features_df

# %%
deep_features_df = pd.read_csv(artifacts_path/'baseline_features.csv')

deep_features_df['mega_features'] = deep_features_df['mega_features'].apply(eval)
deep_features_df['miewid_features'] = deep_features_df['miewid_features'].apply(eval)
deep_features_df['mega_features_cropped'] = deep_features_df['mega_features_cropped'].apply(eval)
deep_features_df['miewid_features_cropped'] = deep_features_df['miewid_features_cropped'].apply(eval)
deep_features_df['mega_features_cropped_rotated'] = deep_features_df['mega_features_cropped_rotated'].apply(eval)
deep_features_df['miewid_features_cropped_rotated'] = deep_features_df['miewid_features_cropped_rotated'].apply(eval)
deep_features_df['mega_features_rotated'] = deep_features_df['mega_features_rotated'].apply(eval)
deep_features_df['miewid_features_rotated'] = deep_features_df['miewid_features_rotated'].apply(eval)
deep_features_df

# %% [markdown]
# # Output least similar images

# %%

# %%
dinov2_features = np.array(dino_features_df['dinov2_features'].tolist())

# %%
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
# Here I will create a dataframe of all similarities for each image
dino_features_df['id_and_image_name'] = dino_features_df['identity'].astype(str) + '_' + dino_features_df['file_name']
dino_features_df['id_and_image_name']

# %%
dinov2_similarities_df = pd.DataFrame(dinov2_similarities, index=dino_features_df['id_and_image_name'], columns=dino_features_df['id_and_image_name'])
plt.imshow(dinov2_similarities_df.to_numpy())
plt.title('Dinov2 Similarities for all images')
plt.colorbar()
plt.show()

# %%
dinov2_similarities_df.to_csv(artifacts_path/'dinov2_similarities.csv')
dinov2_similarities_df

# %%

def plot_identity_similarities(identity, dino_features_df, dinov2_similarities, dataset_path):
    """Plot all images for a given identity with their similarity scores.
    
    Args:
        identity: The identity ID to plot
        dino_features_df: DataFrame containing image features and metadata
        dinov2_similarities: Matrix of similarity scores between all images
        dataset_path: Path to the dataset containing the images
    """
    identity_df = dino_features_df[dino_features_df['identity'] == identity]

    n_images = len(identity_df)
    n_cols = n_images
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(3*n_cols, 5*n_rows))

    for idx, (_, row) in enumerate(identity_df.iterrows()):
        # Get similarities for this image with other images of same identity
        image_idx = dino_features_df[dino_features_df['file_name'] == row['file_name']].index[0]
        similarities = dinov2_similarities[image_idx]
        
        # Get max similarity with other images of same identity (excluding self)
        same_identity_mask = (dino_features_df['identity'] == identity) & (dino_features_df['file_name'] != row['file_name'])
        if same_identity_mask.any():
            max_similarity = np.max(similarities[same_identity_mask])
        else:
            max_similarity = 0.0

        # Plot image
        plt.subplot(n_rows, n_cols, idx + 1)
        img = plt.imread(dataset_path/row['file_path'])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        
        # Get name of most similar image
        if same_identity_mask.any():
            most_similar_idx = np.argmax(similarities[same_identity_mask])
            most_similar_name = dino_features_df.loc[same_identity_mask, 'file_name'].iloc[most_similar_idx]
            plt.title(f'Max similarity: {max_similarity:.3f}\nwith {most_similar_name}')
        else:
            plt.title(f'Max similarity: {max_similarity:.3f}\nNo other images')
        plt.xlabel(row['file_name'])

    plt.suptitle(f'Images for identity {identity}', fontsize=16)
    plt.tight_layout()

# %%
identity = dino_features_df['identity'].iloc[0]
plot_identity_similarities(identity, dino_features_df, dinov2_similarities, dataset_path)

# %%
dinov2_similarities_df.loc[dinov2_similarities_df.index.str.startswith('1_'), dinov2_similarities_df.columns.str.startswith('1_')]

# %%
n_images = 4

image_counts = dino_features_df.groupby('identity').file_name.count()
identities_with_n_images = image_counts[image_counts == n_images].index.tolist()
intra_identity_similarities_path = artifacts_path / 'intra_identity_similarities'
intra_identity_similarities_path.mkdir(parents=True, exist_ok=True)

for identity in identities_with_n_images:
    plot_identity_similarities(identity, dino_features_df, dinov2_similarities, dataset_path)
    plt.savefig(intra_identity_similarities_path / f'identity_{identity}_similarities.png')
    plt.close()
    df = dinov2_similarities_df.loc[dinov2_similarities_df.index.str.startswith(f'{identity}_'), dinov2_similarities_df.columns.str.startswith(f'{identity}_')]
    df.to_csv(intra_identity_similarities_path / f'identity_{identity}_similarities.csv')
    
    # Preview the similarity matrix for this identity
    display.display(df.style.background_gradient(cmap='RdYlBu', vmin=-1, vmax=1)
              .format("{:.3f}")
              .set_caption(f"Similarity matrix for identity {identity}"))

# %%

# %%

# %%

# %%