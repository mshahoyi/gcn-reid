# %% [markdown]
# # BSL Experiment
#> Finetuning MegaDescriptor with Background Supporession Loss (BSL)

# %%
#| default_exp bsl_exp

# %%
#| eval: false
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import timm
from pathlib import Path
import kaggle
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from wildlife_datasets import loader, datasets, splits
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.train import ArcFaceLoss, set_seed
from tqdm import tqdm
import random
from gcn_reid.segmentation import decode_rle_mask, visualize_segmentation, visualize_segmentation_from_metadata
from gcn_reid.attribution import my_occlusion_sensitivity
import itertools

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %%
# Download and verify dataset
def download_newt_dataset():
    dataset_name = "mshahoyi/barhill-newts-segmented"
    download_path = "data/newt_dataset"
    
    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset downloaded to {download_path}")
    else:
        print(f"Dataset already exists at {download_path}")
    
    return download_path

dataset_path = download_newt_dataset()

# Verify dataset structure
print(f"\nDataset path: {dataset_path}")
print("Dataset contents:")
for item in os.listdir(dataset_path):
    print(f"  {item}")

# %%
# Load and examine metadata
metadata_path = os.path.join(dataset_path, "metadata.csv")
df = pd.read_csv(metadata_path)

print(f"Dataset contains {len(df)} images")
print(f"Number of unique newts: {df['newt_id'].nunique()}")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nNewt ID distribution:")
print(df['newt_id'].value_counts().head(10))

# Test RLE decoding with a sample
sample_row = df.iloc[0]
print(f"Testing RLE decoding with sample:")
print(f"Image path: {sample_row['image_path']}")
print(f"Newt ID: {sample_row['newt_id']}")

# Load sample image to get dimensions
sample_img_path = Path(dataset_path) / sample_row['image_path']
sample_img = Image.open(sample_img_path)
print(f"Image size: {sample_img.size}")

# Decode mask
h, w = sample_img.size[1], sample_img.size[0]
mask = decode_rle_mask(sample_row['segmentation_mask_rle'])
print(f"Mask shape: {mask.shape}")
print(f"Mask unique values: {np.unique(mask)}")
print(f"Mask coverage: {mask.sum() / mask.size:.2%}")

# Visualize sample
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sample_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

axes[2].imshow(sample_img)
axes[2].imshow(mask, alpha=0.5, cmap='Reds')
axes[2].set_title('Image + Mask Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %%
# Create and test dataset splits
def create_newt_splits(df, train_ratio=0.8, random_state=42):
    """Create disjoint splits ensuring each newt appears in only one split"""
    unique_newts = df['newt_id'].unique()
    
    train_newts, test_newts = train_test_split(
        unique_newts, 
        train_size=train_ratio, 
        random_state=random_state,
        stratify=None
    )
    
    df_train = df[df['newt_id'].isin(train_newts)].copy()
    df_test = df[df['newt_id'].isin(test_newts)].copy()
    
    print(f"Train split: {len(df_train)} images from {len(train_newts)} newts")
    print(f"Test split: {len(df_test)} images from {len(test_newts)} newts")
    
    return df_train, df_test

df_train, df_test = create_newt_splits(df)

# Verify no overlap between train and test
train_newts = set(df_train['newt_id'].unique())
test_newts = set(df_test['newt_id'].unique())
overlap = train_newts.intersection(test_newts)
print(f"Overlap between train and test newts: {len(overlap)} (should be 0)")

print("\nTrain newt distribution (top 10):")
print(df_train['newt_id'].value_counts().head(10))

print("\nTest newt distribution (top 10):")
print(df_test['newt_id'].value_counts().head(10))

# %%
# Create and test custom dataset class
class NewtDataset(Dataset):
    def __init__(self, dataframe, root_path, transform=None, return_mask=True):
        self.df = dataframe.reset_index(drop=True)
        self.root_path = Path(root_path)
        self.transform = transform
        self.return_mask = return_mask
        self.labels_string = self.df['newt_id'].astype(str).tolist()
        
        # Create label mapping
        unique_labels = sorted(self.df['newt_id'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.df['newt_id']]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.root_path / row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Decode segmentation mask
        mask = None
        if self.return_mask and 'segmentation_mask_rle' in row:
            h, w = image.size[1], image.size[0]
            try:
                decoded_mask = decode_rle_mask(row['segmentation_mask_rle'])
                if decoded_mask is not None:
                    mask = Image.fromarray(decoded_mask * 255).convert('L')
                else:
                    # Create a default mask (all foreground) when RLE decoding fails
                    mask = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255).convert('L')
            except Exception as e:
                # Create a default mask if there's any error in decoding
                h, w = image.size[1], image.size[0]
                mask = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255).convert('L')
                print(f"Warning: Error decoding mask for image {idx}: {e}, using default full mask")
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                # Apply same transform to both image and mask
                seed = np.random.randint(2147483647)
                
                random.seed(seed)
                torch.manual_seed(seed)
                image = self.transform(image)
                
                random.seed(seed)
                torch.manual_seed(seed)
                mask = T.ToTensor()(mask)
                mask = T.Resize(image.shape[-2:])(mask)
            else:
                image = self.transform(image)
        
        if mask is not None:
            return image, label, mask.squeeze(0)
        else:
            return image, label

# Test dataset creation
transform_test = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset_small = NewtDataset(df_train.head(10), dataset_path, transform=transform_test, return_mask=True)

print(f"Test dataset size: {len(test_dataset_small)}")
print(f"Number of classes in test: {len(test_dataset_small.label_to_idx)}")
print(f"Label mapping: {test_dataset_small.label_to_idx}")

# Test loading a sample
sample_data = test_dataset_small[0]
print(f"Sample data shapes:")
print(f"  Image: {sample_data[0].shape}")
print(f"  Label: {sample_data[1]}")
print(f"  Mask: {sample_data[2].shape}")

# %%
# Create full datasets with sanity checks
transform_train = T.Compose([
    T.Resize([224, 224]),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=180),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = NewtDataset(df_train, dataset_path, transform=transform_train, return_mask=True)
test_dataset = NewtDataset(df_test, dataset_path, transform=transform_test, return_mask=True)

print(f"Training dataset: {len(train_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print(f"Number of classes: {len(train_dataset.label_to_idx)}")

# Verify datasets
train_sample = train_dataset[0]
test_sample = test_dataset[0]

print(f"\nTrain sample shapes: image={train_sample[0].shape}, label={train_sample[1]}, mask={train_sample[2].shape}")
print(f"Test sample shapes: image={test_sample[0].shape}, label={test_sample[1]}, mask={test_sample[2].shape}")

# Check label consistency
print(f"Train labels range: {min(train_dataset.labels)} to {max(train_dataset.labels)}")
print(f"Test labels range: {min(test_dataset.labels)} to {max(test_dataset.labels)}")

# %%
# Test model loading and feature extraction
def test_megadescriptor_loading():
    print("Testing MegaDescriptor loading...")
    
    # Test loading the model
    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        features = backbone(dummy_input)
        print(f"Model loaded successfully!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output features shape: {features.shape}")
        print(f"Feature dimension: {features.shape[1]}")
    
    return backbone, features.shape[1]

backbone, embedding_size = test_megadescriptor_loading()
backbone

# %%
# Create and test ArcFace loss
def test_arcface_loss():
    print("Testing ArcFace loss...")
    
    num_classes = len(train_dataset.label_to_idx)
    
    # Create ArcFace loss
    arcface_loss = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_size,
        margin=0.5,
        scale=64
    )
    
    print(f"ArcFace loss created for {num_classes} classes, embedding size {embedding_size}")
    
    # Test forward pass
    with torch.no_grad():
        dummy_embeddings = torch.randn(4, embedding_size)
        dummy_labels = torch.randint(0, num_classes, (4,))
        
        loss = arcface_loss(dummy_embeddings, dummy_labels)
        print(f"Test loss: {loss.item():.4f}")
        print("ArcFace loss working correctly!")
        
    return arcface_loss

arcface_loss = test_arcface_loss()

# %%
# Define Background Suppression ArcFace Loss
class BackgroundSuppressionArcFaceLoss(nn.Module):
    """
    Custom loss that combines ArcFace loss with background suppression
    Uses segmentation masks to focus learning on the newt regions
    """
    
    def __init__(self, num_classes, embedding_size, margin=0.5, scale=64, alpha=1.0, beta=0.5):
        super().__init__()
        self.arcface_loss = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale
        )
        self.alpha = alpha  # Weight for ArcFace loss
        self.beta = beta    # Weight for background suppression loss
        
    def forward(self, embeddings, labels, masks, patch_features=None):
        """
        Args:
            embeddings: Output embeddings from the backbone [B, embedding_size]
            labels: Ground truth labels [B]
            masks: Binary segmentation masks (1 for newt, 0 for background) [B, H, W]
            patch_features: Intermediate feature maps for background suppression [B, C, Hf, Wf]
        """
        # ArcFace loss on embeddings
        arcface_loss = self.arcface_loss(embeddings, labels)
        
        # Background suppression loss
        background_penalty = torch.tensor(0.0, device=embeddings.device)
        
        if patch_features is not None and masks is not None:
            B, C, Hf, Wf = patch_features.shape

            print(f"Patch features shape: {patch_features.shape}")
            
            # Resize masks to match feature map size
            masks_resized = F.interpolate(
                masks.unsqueeze(1).float(), 
                size=(Hf, Wf), 
                mode='nearest'
            ).squeeze(1)
            
            # Background mask (1 for background, 0 for foreground)
            background_mask = 1.0 - masks_resized
            
            # Compute L2 norm of patch features
            patch_norm = patch_features.pow(2).sum(1).sqrt()  # [B, Hf, Wf]
            
            # Background suppression: penalize high activations in background regions
            background_penalty = (patch_norm * background_mask).mean()
        
        total_loss = self.alpha * arcface_loss + self.beta * background_penalty
        
        return total_loss, arcface_loss, background_penalty

# Test BSL Loss
def test_bsl_loss():
    print("Testing Background Suppression ArcFace Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_dataset.label_to_idx)
    
    bsl_loss = BackgroundSuppressionArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_size,
        margin=0.5,
        scale=64,
        alpha=1.0,
        beta=0.5
    ).to(device)
    
    # Test with dummy data
    with torch.no_grad():
        # Get real data samples from training dataset
        sample_batch = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=True)))
        images, labels, masks = sample_batch
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        
        # Get embeddings and patch features from model
        model.eval()
        embeddings = model(images)
        patch_features = model.patch_features
        
        total_loss, arcface_loss, bg_loss = bsl_loss(
            embeddings, labels, masks, patch_features
        )
        
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"ArcFace loss: {arcface_loss.item():.4f}")
        print(f"Background loss: {bg_loss.item():.4f}")
        
    return bsl_loss

bsl_loss = test_bsl_loss()

# %%
model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', pretrained=True, num_classes=0)
model

# %%
# Create model with feature extraction hooks
class MegaDescriptorWithBSL(nn.Module):
    def __init__(self, num_classes, model_name='hf-hub:BVRA/MegaDescriptor-T-224'):
        super().__init__()
        
        # Load pretrained MegaDescriptor backbone
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.embedding_size = features.shape[1]
        
        # Store intermediate features for BSL
        self.patch_features = None
        
        # Register hook to capture intermediate features
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture intermediate feature maps"""
        def hook_fn(module, input, output):
            # Swin Transformer outputs features in [B, H, W, C] format
            if len(output.shape) == 4:
                B, H, W, C = output.shape
                # Convert to [B, C, H, W] format for compatibility
                self.patch_features = output.permute(0, 3, 1, 2)
            elif len(output.shape) == 3:
                # Some layers might output [B, N, C], try to reshape
                B, N, C = output.shape
                H = W = int(np.sqrt(N))
                if H * W == N:
                    self.patch_features = output.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Hook into one of the later Swin Transformer stages
        # Stage 2 has 384 channels and good spatial resolution
        # Stage 3 has 768 channels (final) but lower spatial resolution
        
        # Try to hook into stage 2 (layers.2) - 384 channels
        if hasattr(self.backbone, 'layers') and len(self.backbone.layers) > 2:
            target_stage = self.backbone.layers[2]  # Stage 2
            print(f"Hooking to Swin stage 2 with {384} channels")
            target_stage.register_forward_hook(hook_fn)
            return
        
        # Fallback: try to hook into any layer with 'layers' in the name
        hooked = False
        for name, module in self.backbone.named_modules():
            if 'layers.2' in name and not hooked:  # Prefer stage 2
                print(f"Hooking to layer: {name}")
                module.register_forward_hook(hook_fn)
                hooked = True
                break
            elif 'layers.1' in name and not hooked:  # Fallback to stage 1
                print(f"Hooking to layer: {name}")
                module.register_forward_hook(hook_fn)
                hooked = True
                break
        
        if not hooked:
            print("Warning: Could not find suitable Swin Transformer layer to hook")
        
    def forward(self, x):
        # Reset patch features
        self.patch_features = None
        
        # Forward through backbone
        embeddings = self.backbone(x)
        
        return embeddings
    
    def get_patch_features(self):
        """Get the stored patch features for background suppression"""
        return self.patch_features

# Test model creation
def test_model_creation():
    print("Testing model creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_dataset.label_to_idx)
    
    model = MegaDescriptorWithBSL(num_classes).to(device)
    
    print(f"Model created with {model.embedding_size} embedding size")
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        embeddings = model(dummy_input)
        patch_features = model.get_patch_features()
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        if patch_features is not None:
            print(f"Patch features shape: {patch_features.shape}")
        else:
            print("Warning: No patch features captured")
    
    return model

model = test_model_creation()

# %%

# %%
# Test data loading with actual data
def test_data_loading():
    print("Testing data loading...")
    
    # Create small data loaders for testing
    small_train_dataset = NewtDataset(df_train.head(20), dataset_path, transform=transform_train, return_mask=True)
    train_loader = DataLoader(small_train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Test loading one batch
    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            images, labels, masks = batch
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels: {labels}")
            print(f"  Masks shape: {masks.shape}")
            print(f"  Mask value ranges: {masks.min().item():.3f} to {masks.max().item():.3f}")
            
            # Visualize one sample from batch
            img = images[0]
            mask = masks[0]
            
            # Denormalize image for visualization
            img_denorm = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_denorm.permute(1, 2, 0))
            axes[0].set_title(f'Image (Label: {labels[0].item()})')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            break
    
    return train_loader

test_loader = test_data_loading()

# %%
# Test full training setup
def test_training_setup():
    print("Testing full training setup...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    num_classes = len(train_dataset.label_to_idx)
    model = MegaDescriptorWithBSL(num_classes).to(device)
    
    # Loss function
    bsl_loss = BackgroundSuppressionArcFaceLoss(
        num_classes=num_classes,
        embedding_size=model.embedding_size,
        margin=0.5,
        scale=64,
        alpha=1.0,
        beta=0.5
    ).to(device)
    
    # Optimizer
    params = itertools.chain(model.parameters(), bsl_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    
    # Test one training step
    model.train()
    bsl_loss.train()
    
    # Get a small batch
    small_dataset = NewtDataset(df_train.head(8), dataset_path, transform=transform_train, return_mask=True)
    loader = DataLoader(small_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for batch in loader:
        images, labels, masks = batch
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        patch_features = model.get_patch_features()
        
        print(f"Embeddings shape: {embeddings.shape}")
        if patch_features is not None:
            print(f"Patch features shape: {patch_features.shape}")
        
        # Compute loss
        loss, arcface_loss, bg_loss = bsl_loss(embeddings, labels, masks, patch_features)
        
        print(f"Losses - Total: {loss.item():.4f}, ArcFace: {arcface_loss.item():.4f}, BG: {bg_loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print("Training step completed successfully!")
        break
    
    return model, bsl_loss, optimizer

model, bsl_loss, optimizer = test_training_setup()

# %%
# Now we can proceed with the actual training
print("Setup complete! Ready for full training...")
print(f"Total training samples: {len(train_dataset)}")
print(f"Total test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.label_to_idx)}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# %%
# Test and define training epoch function with sanity checks
def train_epoch(model, train_loader, bsl_loss, optimizer, device, epoch):
    model.train()
    bsl_loss.train()
    
    total_loss = 0
    total_arcface_loss = 0
    total_bg_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:  # With masks
            images, labels, masks = batch
            masks = masks.to(device)
        else:  # Without masks
            images, labels = batch
            masks = torch.ones(images.shape[0], images.shape[2], images.shape[3]).to(device)
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        patch_features = model.get_patch_features()
        
        # Compute BSL loss with ArcFace
        loss, arcface_loss, bg_loss = bsl_loss(embeddings, labels, masks, patch_features)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_arcface_loss += arcface_loss.item()
        total_bg_loss += bg_loss.item()
        
        # For accuracy calculation, get predictions from ArcFace weights
        with torch.no_grad():
            # Access the classifier weights from the pytorch_metric_learning ArcFace loss
            W = bsl_loss.arcface_loss.loss.W  # The classifier weights
            # Normalize embeddings and weights for cosine similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            W_norm = F.normalize(W, p=2, dim=0)
            # Compute logits as cosine similarity * scale
            logits = F.linear(embeddings_norm, W_norm.T) * bsl_loss.arcface_loss.loss.scale
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Arc': f'{arcface_loss.item():.4f}',
            'BG': f'{bg_loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_arcface_loss = total_arcface_loss / len(train_loader)
    avg_bg_loss = total_bg_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_arcface_loss, avg_bg_loss, accuracy

# Test the training epoch function with a tiny dataset
print("Testing training epoch function...")

# Create a tiny test dataset
tiny_dataset = NewtDataset(df_train.head(16), dataset_path, transform=transform_train, return_mask=True)
tiny_loader = DataLoader(tiny_dataset, batch_size=4, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test training epoch
test_loss, test_arcface, test_bg, test_acc = train_epoch(
    model, tiny_loader, bsl_loss, optimizer, device, epoch=0
)

print(f"✅ Training epoch test passed!")
print(f"   Loss: {test_loss:.4f} (ArcFace: {test_arcface:.4f}, BG: {test_bg:.4f})")
print(f"   Accuracy: {test_acc:.2f}%")

# %%
# Test and define evaluation function
def evaluate(model, test_loader, bsl_loss, device):
    model.eval()
    bsl_loss.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if len(batch) == 3:  # With masks
                images, labels, _ = batch
            else:  # Without masks
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)
            
            # Get embeddings
            embeddings = model(images)
            
            # Get predictions from ArcFace weights
            W = bsl_loss.arcface_loss.loss.W  # The classifier weights
            # Normalize embeddings and weights for cosine similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            W_norm = F.normalize(W, p=2, dim=0)
            # Compute logits as cosine similarity * scale
            logits = F.linear(embeddings_norm, W_norm.T) * bsl_loss.arcface_loss.loss.scale
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# Test evaluation function
print("Testing evaluation function...")

tiny_test_dataset = NewtDataset(df_test.head(16), dataset_path, transform=transform_test, return_mask=True)
tiny_test_loader = DataLoader(tiny_test_dataset, batch_size=4, shuffle=False, num_workers=0)

eval_acc = evaluate(model, tiny_test_loader, bsl_loss, device)
print(f"✅ Evaluation test passed!")
print(f"   Test accuracy: {eval_acc:.2f}%")

# %%
# Test occlusion sensitivity function with detailed checks
def run_occlusion_sensitivity_test(model, bsl_loss, dataset, device, epoch, save_dir):
    """Run occlusion sensitivity on pairs of different newts to test similarity"""
    print(f"Starting occlusion sensitivity test for epoch {epoch}")
    
    model.eval()
    bsl_loss.eval()
    
    # Create save directory
    epoch_dir = Path(save_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {epoch_dir}")
    
    # Find pairs of different newts
    newt_indices_by_id = {}
    for idx in range(len(dataset)):
        newt_id = dataset.labels_string[idx]
        if newt_id not in newt_indices_by_id:
            newt_indices_by_id[newt_id] = []
        newt_indices_by_id[newt_id].append(idx)
    
    # Select 2 pairs of different newts
    newt_ids = list(newt_indices_by_id.keys())
    if len(newt_ids) < 2:
        print("Not enough different newts for similarity testing")
        return
    
    # Create similarity model function
    def similarity_model(image1, image2):
        """
        Compute cosine similarity between two images using the trained model
        
        Args:
            image1: First image tensor [1, C, H, W]
            image2: Second image tensor [1, C, H, W] 
            
        Returns:
            Cosine similarity score as tensor
        """
        with torch.no_grad():
            # Get embeddings for both images
            emb1 = model(image1)
            emb2 = model(image2)
            
            # Compute cosine similarity
            emb1_norm = F.normalize(emb1, p=2, dim=1)
            emb2_norm = F.normalize(emb2, p=2, dim=1)
            similarity = F.cosine_similarity(emb1_norm, emb2_norm, dim=1)
            
            return similarity
    
    # Test 2 pairs
    for pair_idx in range(2):
        try:
            # Select two different newts
            newt_id1, newt_id2 = random.sample(newt_ids, 2)
            
            # Get one image from each newt
            idx1 = random.choice(newt_indices_by_id[newt_id1])
            idx2 = random.choice(newt_indices_by_id[newt_id2])
            
            print(f"  Processing pair {pair_idx+1}: Newt {newt_id1} (idx {idx1}) vs Newt {newt_id2} (idx {idx2})")
            
            # Get the images
            if len(dataset[idx1]) == 3:
                image1, label1, _ = dataset[idx1]
            else:
                image1, label1 = dataset[idx1]
                
            if len(dataset[idx2]) == 3:
                image2, label2, _ = dataset[idx2]
            else:
                image2, label2 = dataset[idx2]
            
            print(f"    Image1: {image1.shape}, Label1: {label1}")
            print(f"    Image2: {image2.shape}, Label2: {label2}")
            
            # Convert images to tensors for similarity computation (add batch dimension)
            image1_tensor = image1.unsqueeze(0).to(device)  # [1, C, H, W]
            image2_tensor = image2.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # Test baseline similarity
            baseline_similarity = similarity_model(image1_tensor, image2_tensor).item()
            print(f"    Baseline similarity: {baseline_similarity:.4f}")
            
            # Run occlusion sensitivity using your function with tensors
            print(f"    Running occlusion sensitivity...")
            occlusion_map = my_occlusion_sensitivity(
                similarity_model, 
                image1_tensor, 
                image2_tensor, 
                patch_size=16, 
                stride=8, 
                occlusion_value=0.5, 
                device=device
            )
            print(f"    Occlusion map shape: {occlusion_map.shape}, range [{occlusion_map.min():.3f}, {occlusion_map.max():.3f}]")
            
            # Convert images to numpy for visualization only
            img1_np = image1.cpu().numpy().transpose(1, 2, 0)
            img1_np = (img1_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img1_np = np.clip(img1_np, 0, 1)
            
            img2_np = image2.cpu().numpy().transpose(1, 2, 0)
            img2_np = (img2_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img2_np = np.clip(img2_np, 0, 1)
            
            # Save visualization
            save_path = epoch_dir / f"similarity_pair_{pair_idx+1}_newt_{newt_id1}_vs_{newt_id2}.png"
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original images
            axes[0,0].imshow(img1_np)
            axes[0,0].set_title(f'Image 1: Newt {newt_id1}\n(Index {idx1})')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(img2_np)
            axes[0,1].set_title(f'Image 2: Newt {newt_id2}\n(Index {idx2})')
            axes[0,1].axis('off')
            
            # Occlusion sensitivity overlay
            axes[1,0].imshow(img1_np)
            axes[1,0].imshow(occlusion_map, cmap='hot', alpha=0.6)
            axes[1,0].set_title(f'Occlusion Sensitivity on Image 1\n(Similarity: {baseline_similarity:.3f})')
            axes[1,0].axis('off')
            
            # Pure occlusion map
            im = axes[1,1].imshow(occlusion_map, cmap='hot')
            axes[1,1].set_title('Occlusion Sensitivity Map')
            axes[1,1].axis('off')
            plt.colorbar(im, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Saved similarity occlusion test to {save_path}")
            
        except Exception as e:
            print(f"    ❌ Error in similarity occlusion test for pair {pair_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed occlusion sensitivity test for epoch {epoch}")
    model.train()
    bsl_loss.train()

# Test occlusion sensitivity function
print("Testing occlusion sensitivity function...")

test_occlusion_dir = Path("data/test_occlusion")
test_occlusion_dir.mkdir(parents=True, exist_ok=True)

run_occlusion_sensitivity_test(
    model, bsl_loss, tiny_test_dataset, device, epoch=-1, save_dir=test_occlusion_dir
)

print("✅ Occlusion sensitivity test completed!")

# %%
# Set up full training configuration with verification
def setup_full_training():
    print("Setting up full training configuration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    print("✅ Seed set for reproducibility")
    
    # Model
    num_classes = len(train_dataset.label_to_idx)
    model = MegaDescriptorWithBSL(num_classes).to(device)
    print(f"✅ Model created with {model.embedding_size} embedding size for {num_classes} classes")
    
    # Loss function with ArcFace + Background suppression
    bsl_loss = BackgroundSuppressionArcFaceLoss(
        num_classes=num_classes,
        embedding_size=model.embedding_size,
        margin=0.5,
        scale=64,
        alpha=1.0,    # ArcFace weight
        beta=0.5      # Background suppression weight
    ).to(device)
    print(f"✅ BSL loss created (alpha={1.0}, beta={0.5})")
    
    # Optimizer for both model and ArcFace parameters
    params = itertools.chain(model.parameters(), bsl_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    print(f"✅ Optimizer created with lr=1e-4")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    print("✅ Cosine annealing scheduler created")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✅ Data loaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Create directories for saving
    os.makedirs("data", exist_ok=True)
    occlusion_dir = Path("data/occlusion_maps")
    occlusion_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories created: data/, {occlusion_dir}")
    
    # Test one forward pass with real data
    print("Testing forward pass with real data...")
    with torch.no_grad():
        for batch in train_loader:
            images, labels, masks = batch
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            embeddings = model(images)
            patch_features = model.get_patch_features()
            loss, arcface_loss, bg_loss = bsl_loss(embeddings, labels, masks, patch_features)
            
            print(f"✅ Forward pass successful:")
            print(f"   Batch shape: {images.shape}")
            print(f"   Embeddings: {embeddings.shape}")
            print(f"   Patch features: {patch_features.shape if patch_features is not None else None}")
            print(f"   Losses: total={loss.item():.4f}, arcface={arcface_loss.item():.4f}, bg={bg_loss.item():.4f}")
            break
    
    return model, bsl_loss, optimizer, scheduler, train_loader, test_loader, device, occlusion_dir

# Setup training with verification
model, bsl_loss, optimizer, scheduler, train_loader, test_loader, device, occlusion_dir = setup_full_training()

# %%
# Test one complete epoch to verify everything works
print("=" * 60)
print("TESTING ONE COMPLETE EPOCH")
print("=" * 60)

# Initialize best accuracy for testing
best_acc = 0

# Create a subset for testing
test_train_dataset = NewtDataset(df_train.head(64), dataset_path, transform=transform_train, return_mask=True)
test_train_loader = DataLoader(test_train_dataset, batch_size=8, shuffle=True, num_workers=0)

test_test_dataset = NewtDataset(df_test.head(32), dataset_path, transform=transform_test, return_mask=True)
test_test_loader = DataLoader(test_test_dataset, batch_size=8, shuffle=False, num_workers=0)

print("Running test training epoch...")
train_loss, arcface_loss, bg_loss, train_acc = train_epoch(
    model, test_train_loader, bsl_loss, optimizer, device, epoch=0
)

print("Running test evaluation...")
test_acc = evaluate(model, test_test_loader, bsl_loss, device)

print("Testing scheduler step...")
old_lr = optimizer.param_groups[0]['lr']
scheduler.step()
new_lr = optimizer.param_groups[0]['lr']

print(f"✅ Complete epoch test passed!")
print(f"   Train Loss: {train_loss:.4f} (ArcFace: {arcface_loss:.4f}, BG: {bg_loss:.4f})")
print(f"   Train Acc:  {train_acc:.2f}%")
print(f"   Test Acc:   {test_acc:.2f}% {'🌟 BEST!' if test_acc > best_acc else ''}")
print(f"   Best Acc:   {best_acc:.2f}%")
print(f"   LR:         {old_lr:.2e} → {new_lr:.2e}")

# %%
# Test training visualization with actual results
def test_training_visualization_with_actual_results(train_loss, arcface_loss, bg_loss, train_acc, test_acc, lr):
    print("Testing training visualization with actual results...")
    
    # Use actual training results
    epochs = [0]  # Just one epoch for testing
    train_losses = [train_loss]
    arcface_losses = [arcface_loss]
    bg_losses = [bg_loss]
    train_accs = [train_acc]
    test_accs = [test_acc]
    lrs = [lr]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0,0].plot(epochs, train_losses, label='Total Loss', marker='o', markersize=8)
    axes[0,0].plot(epochs, arcface_losses, label='ArcFace Loss', marker='s', markersize=8)
    axes[0,0].plot(epochs, bg_losses, label='Background Loss', marker='^', markersize=8)
    axes[0,0].set_title('Training Losses (Actual Results)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_xlim(-0.1, 0.1)
    
    # Accuracy curves
    axes[0,1].plot(epochs, train_accs, label='Train Acc', marker='o', markersize=8)
    axes[0,1].plot(epochs, test_accs, label='Test Acc', marker='s', markersize=8)
    axes[0,1].set_title('Accuracy (Actual Results)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_xlim(-0.1, 0.1)
    
    # Loss breakdown
    axes[1,0].plot(epochs, arcface_losses, label='ArcFace', marker='o', markersize=8)
    axes[1,0].plot(epochs, bg_losses, label='Background Suppression', marker='s', markersize=8)
    axes[1,0].set_title('Loss Components (Actual Results)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_xlim(-0.1, 0.1)
    
    # Learning rate
    axes[1,1].plot(epochs, lrs, marker='o', markersize=8)
    axes[1,1].set_title('Learning Rate (Actual)')
    axes[1,1].grid(True)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_xlim(-0.1, 0.1)
    
    # Add actual values as text - fix the max() calls
    max_loss = max(train_loss, arcface_loss, bg_loss)  # Compare actual values, not lists
    max_acc = max(train_acc, test_acc)  # Compare actual values, not lists
    
    axes[0,0].text(0, max_loss, f'Total: {train_loss:.4f}\nArcFace: {arcface_loss:.4f}\nBG: {bg_loss:.4f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[0,1].text(0, max_acc, f'Train: {train_acc:.2f}%\nTest: {test_acc:.2f}%', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/actual_training_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Training visualization with actual results completed!")
    print(f"   Results saved to: data/actual_training_test_results.png")
    print(f"   Train Loss: {train_loss:.4f} (ArcFace: {arcface_loss:.4f}, BG: {bg_loss:.4f})")
    print(f"   Accuracies: Train {train_acc:.2f}%, Test {test_acc:.2f}%")
    print(f"   Learning Rate: {lr:.2e}")

# Test with actual results from the training epoch
print("\n" + "="*60)
print("TESTING VISUALIZATION WITH ACTUAL RESULTS")
print("="*60)

test_training_visualization_with_actual_results(
    train_loss=train_loss,
    arcface_loss=arcface_loss, 
    bg_loss=bg_loss,
    train_acc=train_acc,
    test_acc=test_acc,
    lr=new_lr  # Use the learning rate after scheduler step
)

# %%
# Main training function with comprehensive testing
def train_newt_reid_with_bsl():
    print("🚀 STARTING COMPREHENSIVE TRAINING")
    print("=" * 70)
    
    num_epochs = 5
    best_acc = 0
    train_history = []
    
    # Initial sanity check
    print("Performing initial sanity checks...")
    with torch.no_grad():
        for batch in train_loader:
            images, labels, masks = batch
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            embeddings = model(images)
            patch_features = model.get_patch_features()
            loss, arcface_loss, bg_loss = bsl_loss(embeddings, labels, masks, patch_features)
            print(f"✅ Initial forward pass: Loss={loss.item():.4f}")
            break
    
    print("Starting training loop...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*20} EPOCH {epoch+1}/{num_epochs} {'='*20}")
        
        # Training
        print("Training...")
        train_loss, arcface_loss, bg_loss, train_acc = train_epoch(
            model, train_loader, bsl_loss, optimizer, device, epoch
        )
        
        # Evaluation
        print("Evaluating...")
        test_acc = evaluate(model, test_loader, bsl_loss, device)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save training history
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'arcface_loss': arcface_loss,
            'bg_loss': bg_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'lr': current_lr
        }
        train_history.append(epoch_data)
        
        # Occlusion sensitivity testing every 5 epochs
        if epoch % 5 == 0:
            print(f"Running occlusion sensitivity test...")
            run_occlusion_sensitivity_test(model, bsl_loss, test_dataset, device, epoch, occlusion_dir)
        
        # Save best model
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_state_dict': bsl_loss.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'embedding_size': model.embedding_size,
                'num_classes': len(train_dataset.label_to_idx),
                'train_history': train_history
            }
            torch.save(checkpoint, 'data/best_newt_reid_bsl_model.pth')
            print(f"💾 NEW BEST MODEL SAVED! Accuracy: {best_acc:.2f}%")
        
        # Print epoch summary
        print(f"\n📊 EPOCH {epoch} SUMMARY:")
        print(f"   Train Loss: {train_loss:.4f} (ArcFace: {arcface_loss:.4f}, BG: {bg_loss:.4f})")
        print(f"   Train Acc:  {train_acc:.2f}%")
        print(f"   Test Acc:   {test_acc:.2f}% {'🌟 BEST!' if is_best else ''}")
        print(f"   Best Acc:   {best_acc:.2f}%")
        print(f"   LR:         {old_lr:.2e} → {current_lr:.2e}")
        
        # Visualization every 5 epochs
        if epoch > 0 and epoch % 5 == 0:
            print("Creating training visualizations...")
            
            epochs = [h['epoch'] for h in train_history]
            train_losses = [h['train_loss'] for h in train_history]
            arcface_losses = [h['arcface_loss'] for h in train_history]
            bg_losses = [h['bg_loss'] for h in train_history]
            train_accs = [h['train_acc'] for h in train_history]
            test_accs = [h['test_acc'] for h in train_history]
            lrs = [h['lr'] for h in train_history]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss curves
            axes[0,0].plot(epochs, train_losses, label='Total Loss', marker='o')
            axes[0,0].plot(epochs, arcface_losses, label='ArcFace Loss', marker='s')
            axes[0,0].plot(epochs, bg_losses, label='Background Loss', marker='^')
            axes[0,0].set_title('Training Losses')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Accuracy curves
            axes[0,1].plot(epochs, train_accs, label='Train Acc', marker='o')
            axes[0,1].plot(epochs, test_accs, label='Test Acc', marker='s')
            axes[0,1].set_title('Accuracy')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Loss breakdown
            axes[1,0].plot(epochs, arcface_losses, label='ArcFace', marker='o')
            axes[1,0].plot(epochs, bg_losses, label='Background Suppression', marker='s')
            axes[1,0].set_title('Loss Components')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Learning rate
            axes[1,1].plot(epochs, lrs, marker='o')
            axes[1,1].set_title('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'data/training_progress_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📈 Best test accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved to: data/best_newt_reid_bsl_model.pth")
    print(f"🔍 Occlusion maps saved to: {occlusion_dir}")
    
    return model, bsl_loss, train_history

print("✅ All tests passed! Ready to start main training...")

# %%
# Final pre-training verification
print("FINAL PRE-TRAINING VERIFICATION")
print("=" * 50)

# Check GPU memory if available
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Free: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")

# Check dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {len(train_dataset.label_to_idx)}")
print(f"Batches per epoch: {len(train_loader)}")

# Final forward pass test
print("Final forward pass test...")
with torch.no_grad():
    test_batch = next(iter(train_loader))
    images, labels, masks = test_batch
    images, labels, masks = images.to(device), labels.to(device), masks.to(device)
    
    embeddings = model(images)
    patch_features = model.get_patch_features()
    loss, arcface_loss, bg_loss = bsl_loss(embeddings, labels, masks, patch_features)
    
    print(f"✅ Final test successful!")
    print(f"   Batch size: {images.shape[0]}")
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Memory usage: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB" if torch.cuda.is_available() else "CPU mode")

print("\n🚀 READY TO START TRAINING!")

# %%
# Start the actual training!
trained_model, trained_bsl_loss, history = train_newt_reid_with_bsl()

# %%
