# %% [markdown]
# # Newt Current
# > This notebook shows the current state of the newt re-identification models on our dataset.

# %%
#| default_exp newt

# %%
#| export
import os
import sys
import torch


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from PIL import Image

# %% [markdown]
# # Create Dataset Class

#%%
#| export
import pandas as pd
from wildlife_datasets import datasets

class BarhillNewtsDataset(datasets.WildlifeDataset):
    @classmethod
    def _download(cls, dataset_name, download_path):
        if not os.path.exists(download_path):
            os.makedirs(download_path, exist_ok=True)
            import kaggle
            kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            print(f"Dataset downloaded to {download_path}")
        else:
            print(f"Dataset already exists at {download_path}")

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        return df.rename(columns={"newt_id": "identity", "image_id": "image_name", "image_path": "path"})

# %%
dataset_path = "data/newt_dataset"
BarhillNewtsDataset._download(dataset_name="mshahoyi/barhill-newts-segmented", download_path=dataset_path)
dataset = BarhillNewtsDataset(dataset_path)
dataset.df.head()

# %%
dataset.plot_grid()

# %%
from wildlife_datasets import analysis

analysis.display_statistics(dataset.df)

# %%
from wildlife_datasets import metrics

y_pred = ['GCN63-P6-S2']*len(dataset.df)
y_true = dataset.df['identity']

metrics.accuracy(y_pred, y_true)

# %% [markdown]
# # Create Query and Database Sets

# %%
from wildlife_datasets import datasets, splits

splitter = splits.ClosedSetSplit(0.9)
for idx_database, idx_query in splitter.split(dataset.df):
    df_database, df_query = dataset.df.loc[idx_database], dataset.df.loc[idx_query]
    splits.analyze_split(dataset.df, idx_database, idx_query)

# %% [markdown]
# # Test MegaDescriptor

# %%
from wildlife_tools.data import ImageDataset
import torchvision.transforms as T

transform = T.Compose([T.Resize([384, 384]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset_database = datasets.WildlifeDataset(df=df_database, root=dataset.root, transform=transform)
dataset_query = datasets.WildlifeDataset(df=df_query, root=dataset.root, transform=transform)

# %%
import timm
import torch
from wildlife_tools.features import DeepFeatures

name = 'hf-hub:BVRA/MegaDescriptor-L-384'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True), 
                         device=device,
                         batch_size=32,
                         num_workers=4,
                         )
print("model loaded to device:", device)

# %%
query = extractor(ImageDataset(dataset_query.df, root=dataset_query.root, transform=dataset_query.transform))
database = extractor(ImageDataset(dataset_database.df, root=dataset_database.root, transform=dataset_database.transform))

# %%
from wildlife_tools.similarity import CosineSimilarity

similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)

# %%
import numpy as np
from wildlife_tools.inference import TopkClassifier, KnnClassifier

top_5_classifier = TopkClassifier(k=5, database_labels=dataset_database.labels_string, return_all=True)
knn_classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string, return_scores=True)

predictions_top_5, scores_top_5, _ = top_5_classifier(similarity)
predictions_knn, scores_knn = knn_classifier(similarity)

#%%
accuracy_top_1 = np.mean(dataset_query.labels_string == predictions_top_5[:, 0])
accuracy_top_5 = np.mean(np.any(predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))
accuracy_knn = np.mean(dataset_query.labels_string == predictions_knn)

accuracy_top_1, accuracy_knn, accuracy_top_5

# %%
# Calculate mean Average Precision (mAP)
from sklearn.metrics import average_precision_score
import numpy as np

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
dataset_query.get_image(10)
# %%
import matplotlib.pyplot as plt
import numpy as np
#| export
def plot_retrieval_results(dataset_query, dataset_database, similarity_matrix, mode = "mistakes", num_results=4, num_queries=5, figsize=(15, 20)):
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
from transformers import AutoModel

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
miew_id_predictions_knn, miew_id_scores_knn = knn_classifier(miew_id_similarity)

#%%
miew_id_accuracy_top_1 = np.mean(dataset_query.labels_string == miew_id_predictions_top_5[:, 0])
miew_id_accuracy_top_5 = np.mean(np.any(miew_id_predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))
miew_id_accuracy_knn = np.mean(dataset_query.labels_string == miew_id_predictions_knn)

miew_id_accuracy_top_1, miew_id_accuracy_knn, miew_id_accuracy_top_5

# %%
miew_id_map_score = calculate_map(dataset_query.labels_string, dataset_database.labels_string, miew_id_similarity)
miew_id_map_score

# %%
plot_retrieval_results(dataset_query, dataset_database, miew_id_similarity, mode="mistakes", num_results=4, num_queries=5)

# %% [markdown]
# # Test on Cropped Newts

# %%
path = 'original_images/GCN63-P6-S2/IMG_2725.JPEG'
dataset.df[dataset.df.path == path].segmentation_mask_rle
# %%
import cv2
from gcn_reid.segmentation import decode_rle_mask

class CroppingImageDataset(ImageDataset):
    """Dataset that crops an image using an RLE segmentation mask."""
    
    def __init__(self, crop=True, **image_dataset_kwargs):
        super().__init__(**image_dataset_kwargs)
        self.crop = crop

    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        relative_path = path.replace(self.root, "")
        rle = self.df[self.df[self.col_path] == relative_path].segmentation_mask_rle.values[0]
        decoded_mask = decode_rle_mask(rle, img.shape[:2])
        return img
    
    def __call__(self, image, rle):
        """
        Args:
            image: PIL Image or tensor
            rle: RLE encoded segmentation mask
        
        Returns:
            Cropped image containing only the segmented region
        """
        from gcn_reid.seg import decode_rle_mask
        import numpy as np
        from PIL import Image
        
        # Convert image to numpy if it's a PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            is_pil = True
        else:
            img_array = image
            is_pil = False
        
        # Decode the RLE mask
        mask = decode_rle_mask(rle, img_array.shape[:2])
        
        # Find bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # If mask is empty, return original image
            return image
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        h, w = img_array.shape[:2]
        rmin = max(0, rmin - self.padding)
        rmax = min(h, rmax + self.padding + 1)
        cmin = max(0, cmin - self.padding)
        cmax = min(w, cmax + self.padding + 1)
        
        # Crop the image
        cropped = img_array[rmin:rmax, cmin:cmax]
        
        # Convert back to PIL if input was PIL
        if is_pil:
            return Image.fromarray(cropped)
        else:
            return cropped


# %%
import nbdev
nbdev.export.nbdev_export()