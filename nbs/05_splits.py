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
df = df[~df.is_video]
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
mega_results = mega_extractor(mega_dataset)
miewid_results = miewid_extractor(miewid_dataset)

# %%
df['mega_features'] = [features.tolist() for features in mega_results.features]
df['miewid_features'] = [features.tolist() for features in miewid_results.features]

# %%
df.head()

# %%
artifacts_path = Path('artifacts')
artifacts_path.mkdir(exist_ok=True)
df.to_csv(artifacts_path/'metadata_with_features.csv', index=False)

# %%

# %% [markdown]
# ## Get all cosine similarities and save the highest correct match and the highest incorrect match scores and indices

# %% [markdown]
# ## Calculate the rightness score for each image and model.

# %% [markdown]
# ## Sort images by rightness score in an ascending order

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