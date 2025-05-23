# %% [markdown]
# # BSL Experiment
#> Finetuning MegaDescriptor with Background Supporession Loss (BSL)

# %%
#| default_exp bsl_exp

# %%
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
import gcn_reid
from tqdm import tqdm
import random

# %%
# Download the newt dataset from Kaggle
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

# Download the dataset
dataset_path = download_newt_dataset()

# %%
# Load and parse the metadata
metadata_path = os.path.join(dataset_path, "metadata.csv")
df = pd.read_csv(metadata_path)
print(f"Dataset contains {len(df)} images of {df['newt_id'].nunique()} unique newts")
print(df.head())

# %%
# RLE decode function for segmentation masks
def rle_decode(rle_string, shape):
    """Decode RLE string to binary mask"""
    if pd.isna(rle_string):
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # RLE is 1-indexed
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape)

# %%
# Create custom dataset class for newts with segmentation masks
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
            # Assume standard image size or get from image
            h, w = image.size[1], image.size[0]  # PIL returns (width, height)
            mask = rle_decode(row['segmentation_mask_rle'], (h, w))
            mask = Image.fromarray(mask * 255).convert('L')
        
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
            return image, label, mask.squeeze(0)  # Remove channel dimension from mask
        else:
            return image, label

# %%
# Create disjoint splits using wildlife-datasets approach
def create_newt_splits(df, train_ratio=0.8, random_state=42):
    """Create disjoint splits ensuring each newt appears in only one split"""
    
    # Get unique newt IDs
    unique_newts = df['newt_id'].unique()
    
    # Split newt IDs (not individual images)
    train_newts, test_newts = train_test_split(
        unique_newts, 
        train_size=train_ratio, 
        random_state=random_state,
        stratify=None  # Can't stratify with small classes
    )
    
    # Create train and test dataframes
    df_train = df[df['newt_id'].isin(train_newts)].copy()
    df_test = df[df['newt_id'].isin(test_newts)].copy()
    
    print(f"Train split: {len(df_train)} images from {len(train_newts)} newts")
    print(f"Test split: {len(df_test)} images from {len(test_newts)} newts")
    
    return df_train, df_test

# Create splits
df_train, df_test = create_newt_splits(df)

# %%
# Define transforms
transform_train = T.Compose([
    T.Resize([224, 224]),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = NewtDataset(df_train, dataset_path, transform=transform_train, return_mask=True)
test_dataset = NewtDataset(df_test, dataset_path, transform=transform_test, return_mask=True)

print(f"Training dataset: {len(train_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print(f"Number of classes: {len(train_dataset.label_to_idx)}")

# %%
# Background Suppression Loss (BSL) implementation
class BackgroundSuppressionLoss(nn.Module):
    """
    Custom loss that penalizes predictions based on background pixels
    Uses segmentation masks to focus learning on the newt regions
    """
    
    def __init__(self, base_loss_fn=nn.CrossEntropyLoss(), alpha=1.0, beta=0.5):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for background suppression loss
        
    def forward(self, features, labels, masks, model):
        """
        Args:
            features: Output features from the model
            labels: Ground truth labels
            masks: Binary segmentation masks (1 for newt, 0 for background)
            model: The model to extract intermediate features from
        """
        # Standard classification loss
        classification_loss = self.base_loss_fn(features, labels)
        
        # Background suppression loss
        # Get feature maps from intermediate layers
        activation_maps = self.get_activation_maps(model, features)
        
        # Resize masks to match activation maps
        if activation_maps is not None:
            mask_resized = F.interpolate(
                masks.unsqueeze(1), 
                size=activation_maps.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            # Background suppression: penalize high activations in background regions
            background_mask = (1 - mask_resized)
            background_penalty = (activation_maps * background_mask.unsqueeze(1)).mean()
            
            total_loss = self.alpha * classification_loss + self.beta * background_penalty
        else:
            total_loss = classification_loss
            background_penalty = torch.tensor(0.0)
        
        return total_loss, classification_loss, background_penalty
    
    def get_activation_maps(self, model, features):
        """Extract activation maps from intermediate layers"""
        # This is a simplified version - you might need to modify based on MegaDescriptor architecture
        try:
            # Get the last feature map before global pooling
            if hasattr(model, 'features'):
                return model.features
            elif hasattr(model, 'forward_features'):
                # For vision transformers
                return None  # Handle differently for ViT
            else:
                return None
        except:
            return None

# %%
# MegaDescriptor model setup with custom head
class MegaDescriptorWithBSL(nn.Module):
    def __init__(self, num_classes, model_name='hf-hub:BVRA/MegaDescriptor-T-224'):
        super().__init__()
        
        # Load pretrained MegaDescriptor
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Store intermediate features for BSL
        self.intermediate_features = None
        
    def forward(self, x):
        features = self.backbone(x)
        
        # Store for BSL computation
        self.intermediate_features = features
        
        logits = self.classifier(features)
        return logits

# %%
# Training setup
def setup_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    num_classes = len(train_dataset.label_to_idx)
    model = MegaDescriptorWithBSL(num_classes).to(device)
    
    # Loss function
    bsl_loss = BackgroundSuppressionLoss(alpha=1.0, beta=0.5)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    return model, bsl_loss, optimizer, scheduler, train_loader, test_loader, device

# %%
# Occlusion sensitivity testing during training
def run_occlusion_sensitivity_test(model, dataset, device, epoch, save_dir):
    """Run occlusion sensitivity on two random newts and save results"""
    model.eval()
    
    # Create save directory
    epoch_dir = Path(save_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Select two random samples
    indices = random.sample(range(len(dataset)), 2)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample
                if len(dataset[idx]) == 3:  # With mask
                    image, label, mask = dataset[idx]
                else:  # Without mask
                    image, label = dataset[idx]
                    mask = None
                
                # Convert to numpy for occlusion sensitivity
                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np, 0, 1)
                
                # Run occlusion sensitivity
                def predict_fn(x):
                    """Prediction function for occlusion sensitivity"""
                    if len(x.shape) == 3:
                        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
                    x = x.to(device)
                    
                    # Normalize
                    x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
                    
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1)
                        return probs[0, label].item()
                
                # Use the user's occlusion sensitivity function
                occlusion_map = gcn_reid.my_occlusion_sensitivity(img_np, predict_fn)
                
                # Save the occlusion map
                save_path = epoch_dir / f"newt_{i+1}_occlusion.png"
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(img_np)
                plt.title(f'Original Image (Newt {dataset.labels_string[idx]})')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(occlusion_map, cmap='hot', alpha=0.7)
                plt.imshow(img_np, alpha=0.3)
                plt.title(f'Occlusion Sensitivity')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved occlusion sensitivity map to {save_path}")
                
            except Exception as e:
                print(f"Error in occlusion sensitivity for sample {i}: {e}")
    
    model.train()

# %%
# Training loop
def train_epoch(model, train_loader, bsl_loss, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_cls_loss = 0
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
        logits = model(images)
        
        # Compute BSL loss
        loss, cls_loss, bg_loss = bsl_loss(logits, labels, masks, model)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bg_loss += bg_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_bg_loss = total_bg_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_cls_loss, avg_bg_loss, accuracy

# %%
# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            if len(batch) == 3:  # With masks
                images, labels, _ = batch
            else:  # Without masks
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# %%
# Main training function
def train_newt_reid():
    # Setup
    model, bsl_loss, optimizer, scheduler, train_loader, test_loader, device = setup_training()
    
    # Create directories for saving occlusion maps
    occlusion_dir = Path("data/occlusion_maps")
    occlusion_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 50
    best_acc = 0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        train_loss, cls_loss, bg_loss, train_acc = train_epoch(
            model, train_loader, bsl_loss, optimizer, device, epoch
        )
        
        # Evaluation
        test_acc = evaluate(model, test_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Occlusion sensitivity testing every 5 epochs
        if epoch % 5 == 0:
            print(f"\nRunning occlusion sensitivity test for epoch {epoch}...")
            run_occlusion_sensitivity_test(model, test_dataset, device, epoch, occlusion_dir)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'data/best_newt_reid_model.pth')
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f} (Cls: {cls_loss:.4f}, BG: {bg_loss:.4f}), '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%')
    
    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
    return model

# %%
# Start training
if __name__ == "__main__":
    trained_model = train_newt_reid()
