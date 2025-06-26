#%% [markdown]
# Newt Re-Identification with ArcFace Loss
# This notebook demonstrates fine-tuning a pre-trained model for newt re-identification using ArcFace loss

#%% [markdown]
# ## 1. Setup and Data Loading
# First, we set up the data directory and load the dataset

#%%
import pathlib
DATA_DIR = pathlib.Path('/kaggle/working/barhill')
print(f"Data directory: {DATA_DIR}")

#%% [markdown]
# ## 2. Load and Preprocess CSV Data

#%%
import pandas as pd

df = pd.read_csv(DATA_DIR/'gallery_and_probes.csv')
print("Original dataframe head:")
df.head()

#%% [markdown]
# ## 3. Update Image Paths

#%%
# Create a new column with the path in gcns_cropped instead of GCNs
df['cropped_image_path'] = df.image_path.str.replace('GCNs', 'gcns-cropped')
df['cropped_image_path'] = str(DATA_DIR.parent) + "/" + df['cropped_image_path']
df['full_image_path'] = str(DATA_DIR.parent) + "/" + df['image_path']
print("Updated dataframe with full paths:")
df.tail()

#%% [markdown]
# ## 4. Prepare DataFrame for Wildlife Dataset Format

#%%
# Convert column names to match the expected format
import os
df = df.rename(columns={'image_name': 'image_id', 'cropped_image_path': 'path', 'newt_id': 'identity'})

# Filter out entries where the image file doesn't exist
valid_paths = []
for idx, row in df.iterrows():
    if os.path.exists(row['path']):
        valid_paths.append(idx)

# Keep only rows with valid paths
df = df.loc[valid_paths].reset_index(drop=True)

# Remove duplicate image_ids, keeping only the first occurrence
df = df.drop_duplicates(subset=['image_id'], keep='first')

# Reset index after removing duplicates
df = df.reset_index(drop=True)

print("DataFrame after cleaning:")
df.info()
print(f"Number of unique identities: {df['identity'].nunique()}")

#%% [markdown]
# ## 5. Create a Wildlife Dataset

#%%
import pandas as pd
from wildlife_datasets import datasets

class NewtDataset(datasets.WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        self.finalize_catalogue(df)
        return df

#%%
newt = NewtDataset('/kaggle/working/barhill')
print(f"Total images in dataset: {len(newt)}")
print(f"Number of unique newts: {len(newt.df['identity'].unique())}")

#%% [markdown]
# ## 6. Import Required Libraries for Fine-tuning

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import timm
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed
from wildlife_tools.data import ImageDataset
from torch.utils.data import random_split
import itertools

print(f"PyTorch version: {torch.__version__}")
# Device will be determined and printed in Cell 10

#%% [markdown]
# ## 7. Define Data Transforms

#%%
# Define transforms for training
train_transform = T.Compose([
    T.Resize([224, 224]),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Define transforms for validation (no augmentation)
val_transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

print("Training transforms applied:")
print(" - Resize to 224x224")
print(" - Random horizontal flip")
print(" - Random rotation (±10°)")
print(" - Color jitter (brightness, contrast, saturation)")
print(" - Convert to tensor")
print(" - Normalize with ImageNet stats")

print("\nValidation transforms applied:")
print(" - Resize to 224x224")
print(" - Convert to tensor")
print(" - Normalize with ImageNet stats")

#%% [markdown]
# ## 8. Create Dataset with Transforms

#%%
# Create dataset with the training transforms
dataset = ImageDataset(newt.df, newt.root, transform=train_transform)
print(f"Full dataset size: {len(dataset)} images")

#%% [markdown]
# ## 9. Split Dataset into Train and Validation Sets

#%%
# Use DisjointSetSplit to create train and validation sets with disjoint identities
from wildlife_datasets.splits import DisjointSetSplit

# Create a disjoint split with 30% of identities in validation set
splitter = DisjointSetSplit(0.3)
for idx_train, idx_val in splitter.split(newt.df):
    train_df, val_df = newt.df.loc[idx_train], newt.df.loc[idx_val]

# Create datasets from the split dataframes with appropriate transforms
train_dataset = ImageDataset(train_df, newt.root, transform=train_transform)
val_dataset = ImageDataset(val_df, newt.root, transform=val_transform)

print(f"Training set: {len(train_dataset)} images, {train_df['identity'].nunique()} unique identities")
print(f"Validation set: {len(val_dataset)} images, {val_df['identity'].nunique()} unique identities")

#%% [markdown]
# ## 9.1 Visualize Dataset Samples

#%%
import matplotlib.pyplot as plt
import random
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import pandas as pd

# Custom padding transform to maintain aspect ratio
def pad_to_square(img):
    width, height = img.size
    if width == height:
        return img
    
    size = max(width, height)
    new_img = Image.new('RGB', (size, size), color=(0, 0, 0))
    paste_x = (size - width) // 2
    paste_y = (size - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

# Function to display multiple samples for each identity
def display_identity_samples(df, root_path, identities=3, samples_per_identity=3):
    # Get a list of unique identities
    unique_identities = df['identity'].unique()
    
    # Select random identities if there are more than requested
    if len(unique_identities) > identities:
        selected_identities = random.sample(list(unique_identities), identities)
    else:
        selected_identities = unique_identities
    
    # Create figure with subplots
    fig, axes = plt.subplots(identities, samples_per_identity, figsize=(4*samples_per_identity, 3*identities))
    
    # If only one identity is selected, make sure axes is 2D
    if identities == 1:
        axes = axes.reshape(1, -1)
    
    # For each selected identity
    for i, identity in enumerate(selected_identities):
        # Get all images for this identity
        identity_df = df[df['identity'] == identity]
        
        # Select random samples if there are more than requested
        if len(identity_df) > samples_per_identity:
            sample_indices = random.sample(range(len(identity_df)), samples_per_identity)
            samples = identity_df.iloc[sample_indices]
        else:
            samples = identity_df
            # If not enough samples, repeat some
            while len(samples) < samples_per_identity:
                samples = pd.concat([samples, samples.iloc[0:1]])
        
        # Display each sample
        for j, (_, row) in enumerate(samples.iterrows()):
            # Load image
            img_path = os.path.join(root_path, row['path'])
            img = Image.open(img_path).convert('RGB')
            
            # Create a transform that preserves aspect ratio with padding
            transform = T.Compose([
                T.Resize(340),  # Resize the smaller dimension to 340
                pad_to_square,  # Custom function to pad to square
                T.Resize(224),  # Final resize to 224x224
                T.ToTensor()    # Convert to tensor (no normalization for visualization)
            ])
            
            # Apply transform
            img_tensor = transform(img)
            
            # Convert tensor to numpy for display
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            # Display the image
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(f"ID: {identity}")
            axes[i, j].axis('off')
            # Set background color to black for better visualization of padding
            axes[i, j].set_facecolor('black')
    
    plt.tight_layout()
    plt.suptitle("Sample Images by Identity", y=1.02)
    plt.show()

# Display samples from training set
print(f"\nTraining set: {len(train_dataset)} images, {train_df['identity'].nunique()} unique identities")
display_identity_samples(train_df, newt.root, identities=3, samples_per_identity=3)

# Display samples from validation set
print(f"\nValidation set: {len(val_dataset)} images, {val_df['identity'].nunique()} unique identities")
display_identity_samples(val_df, newt.root, identities=3, samples_per_identity=3)

# Print identity distribution
print("\nIdentity distribution in training set:")
train_identity_counts = train_df['identity'].value_counts().head(10)
print(train_identity_counts)

print("\nIdentity distribution in validation set:")
val_identity_counts = val_df['identity'].value_counts().head(10)
print(val_identity_counts)

# Check for identity overlap
train_identities = set(train_df['identity'].unique())
val_identities = set(val_df['identity'].unique())
common_identities = train_identities.intersection(val_identities)

print(f"\nNumber of identities in training set: {len(train_identities)}")
print(f"Number of identities in validation set: {len(val_identities)}")
print(f"Number of identities in both sets: {len(common_identities)}")

# If using DisjointSetSplit, common_identities should be empty
if len(common_identities) == 0:
    print("✅ Validation successful: Training and validation sets have disjoint identities")
else:
    print("⚠️ Warning: There are overlapping identities between training and validation sets")
    print("Common identities:", list(common_identities)[:5], "..." if len(common_identities) > 5 else "")

#%% [markdown]
# ## 10. Load Pre-trained Model

#%%
# Load MegaDescriptor model from HuggingFace hub
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'hf-hub:BVRA/MegaDescriptor-T-224' # Ensure this matches your input size (e.g., 224x224)
device = torch.device('cuda:0')

model = (
    timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224',
                      pretrained=True, num_classes=0)        # reset head here
          .to(device, non_blocking=True)
)


def find_off_device(mod, target):
    wrong = []
    for n,p in list(mod.named_parameters()) + list(mod.named_buffers()):
        if p.device != target:
            wrong.append((n, p.device))
    return wrong


def ensure_model_on_device(model, device):
    """
    Ensures all model parameters are on the specified device.
    
    Args:
        model: The PyTorch model to check
        device: The target device where all parameters should be
        
    Returns:
        model: The model with all parameters moved to the target device
    """
    # Check for parameters on wrong device
    wrong = find_off_device(model, device)
    if wrong:
        print(f"Found {len(wrong)} parameters on wrong device. First few: {wrong[:10]}")
    
    # Get the device of the model
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
    
    # Check if all model parameters are on the same device
    all_on_same_device = all(p.device == device for p in model.parameters())
    print(f"All model parameters on target device: {all_on_same_device}")
    
    # If there are parameters on wrong device, move them
    if not all_on_same_device:
        print(f"Moving all model parameters to {device}...")
        model = model.to(device)
        # Check again after moving
        all_on_same_device = all(p.device == device for p in model.parameters())
        print(f"All model parameters on target device after fix: {all_on_same_device}")

#%%
ensure_model_on_device(model, device)

#%%
# Get the embedding size from the model's configuration
embedding_size = model.num_features
print(f"Model: {model_name}")
print(f"Embedding size: {embedding_size}")

# Replace the classifier with identity (we'll use ArcFace instead)
model.reset_classifier(num_classes=0)
print(f"Classifier replaced with identity function")

# Determine target device
if torch.cuda.is_available():
    target_device_str = 'cuda:0' # Default to cuda:0
    print(f"CUDA is available. Attempting to use device: {target_device_str}")
    try:
        device = torch.device(target_device_str)
        _ = torch.tensor([1]).to(device) # Test device accessibility
        print(f"Successfully initialized device: {device}")
    except RuntimeError as e:
        print(f"Error initializing device {target_device_str}: {e}. Falling back to CPU.")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print(f"CUDA not available. Using device: {device}")

print(f"Final device for model and training: {device}")

model = model.to(device) # Move entire model to the determined device

# --- CRUCIAL DEBUG CHECK & FIX ---
print(f"--- DEBUG (Cell 10) ---")
print(f"Device of model's first parameter after model.to(device): {next(model.parameters()).device}")
if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
    print(f"Device of model.patch_embed.proj.weight after model.to(device): {model.patch_embed.proj.weight.device}")
    if model.patch_embed.proj.weight.device.type == 'cpu' and device.type == 'cuda':
        print(f"WARNING: model.patch_embed.proj.weight is still on CPU despite model.to({device})!")
        print(f"Attempting explicit move of model.patch_embed.proj to {device}...")
        model.patch_embed.proj = model.patch_embed.proj.to(device)
        print(f"Device of model.patch_embed.proj.weight after EXPLICIT move: {model.patch_embed.proj.weight.device}")
else:
    print("Could not access model.patch_embed.proj.weight for debugging in Cell 10.")
print(f"--- END DEBUG (Cell 10) ---")

print(f"Model setup complete on device: {device}")

#%% [markdown]
# ## 11. Define ArcFace Loss and Optimizer

#%%
# Define number of identities
num_identities = len(newt.df['identity'].unique())

# Define ArcFace loss (will be moved to device in Cell 15)
criterion = ArcFaceLoss(num_classes=num_identities, embedding_size=embedding_size)
print(f"ArcFace loss initialized with {num_identities} classes and embedding size {embedding_size}")

#%% [markdown]
# ## 12. Define Learning Rate Scheduler

#%%
# Define learning rate scheduler
epochs = 20 # Moved epochs definition here as it's used by scheduler

#%% [markdown]
# ## 13. Set Training Parameters

#%%
# The 'device' is ALREADY DEFINED and USED in Cell 10.
# This cell now primarily sets other training params.

# Define training parameters
batch_size = 8
num_workers = 4

# Set seed for reproducibility
set_seed(42)

print(f"Device being used for training (from Cell 10): {device}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Number of workers: {num_workers}")
print("Random seed set to 42 for reproducibility")

#%%
ensure_model_on_device(model, device)

#%% [markdown]
# ## 14. Define Validation Callback with Re-identification Metrics

#%%
# Define a validation callback function with re-identification metrics
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier

def validation_callback(trainer): # Ensure it takes trainer if called by track_loss_callback
    model = trainer.model
    # device = trainer.device # Use trainer's device for consistency within callback
    # The global `device` from Cell 10 should be the same as trainer.device if passed correctly.
    # For safety, let's use trainer.device within the callback.
    current_device = trainer.device
    model.eval() # model is trainer.model, which is moved to trainer.device by the trainer
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(current_device), targets.to(current_device)
            embeddings = model(inputs)
            output = criterion(embeddings, targets) # criterion should also be on current_device
            
            if isinstance(output, tuple):
                loss, logits = output
            else:
                loss = output
                logits = None
            
            val_loss += loss.item()
            
            if logits is not None:
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    print(f'Epoch {trainer.epoch}, Validation Loss: {avg_val_loss:.4f}', end='')
    if total > 0:
        val_accuracy = 100.0 * correct / total
        print(f', Classification Accuracy: {val_accuracy:.2f}%')
    else:
        print('')
    
    # Re-ID Metrics part
    # Ensure extractor uses the correct device, model already on trainer.device
    extractor = DeepFeatures(model, device=current_device, batch_size=batch_size, num_workers=num_workers)
    # ... rest of re-id logic ...
    # (ensure test_transform, query_dataset, gallery_dataset are correctly defined)
    # Create a gallery and query split from validation set
    val_identities_unique = val_df['identity'].unique() # Use a different name to avoid conflict
    query_indices = []
    gallery_indices = []
    
    for identity_val in val_identities_unique: # Use different loop var name
        identity_indices = val_df[val_df['identity'] == identity_val].index.tolist()
        if len(identity_indices) > 1:
            query_indices.append(identity_indices[0])
            gallery_indices.extend(identity_indices[1:])
    
    if not query_indices or not gallery_indices:
        print("Not enough validation data for re-identification metrics this epoch.")
        return
    
    query_df_val = val_df.loc[query_indices].reset_index(drop=True) # Use different df name
    gallery_df_val = val_df.loc[gallery_indices].reset_index(drop=True) # Use different df name
    
    test_transform_val = T.Compose([ # Use different transform name if needed
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    query_dataset_val = ImageDataset(query_df_val, newt.root, transform=test_transform_val)
    gallery_dataset_val = ImageDataset(gallery_df_val, newt.root, transform=test_transform_val)
    
    if not query_dataset_val or not gallery_dataset_val:
        print("Query or Gallery dataset for Re-ID is empty.")
        return

    query_features = extractor(query_dataset_val)
    gallery_features = extractor(gallery_dataset_val)
    
    similarity_function = CosineSimilarity()
    similarity_matrix = similarity_function(query_features, gallery_features) # Store the matrix directly
    
    k_values = [1, 3, 5]
    top_k_accuracies = []
    
    for k_val in k_values: # Use different loop var name
        correct_count = 0
        total_count = len(query_df_val)
        
        for i_idx, query_id_val in enumerate(query_df_val['identity'].values): # Use different loop var name
            scores = similarity_matrix[i_idx]
            top_k_indices = np.argsort(scores)[-k_val:]
            top_k_ids = gallery_df_val['identity'].values[top_k_indices]
            
            if query_id_val in top_k_ids:
                correct_count += 1
        
        top_k_acc = correct_count / total_count if total_count > 0 else 0.0
        top_k_accuracies.append(top_k_acc)
    
    print(f"Re-ID Metrics - Top-1: {top_k_accuracies[0]:.4f}, Top-3: {top_k_accuracies[1]:.4f}, Top-5: {top_k_accuracies[2]:.4f}")


print("Validation callback defined with re-identification metrics") #This line was outside the callback
# print("Will report validation loss, accuracy, and re-ID metrics after each epoch") #This line was outside callback
# Imports for validation callback should be at the top of the script or cell
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier # Already imported but good practice

#%% [markdown]
# ## 15. Create and Configure Trainer

#%%
ensure_model_on_device(model, device)

#%%
# The 'device' object from Cell 10 is used here.
# The 'model' object from Cell 10 (already moved to 'device' and potentially fixed) is used here.

# Move criterion to the defined device
criterion = criterion.to(device) 
if list(criterion.parameters()): 
    print(f"Criterion weight device after .to(device) in Cell 15: {next(criterion.parameters()).device}")
else:
    print(f"Criterion is on device: {device} (or has no parameters to move)")

# --- DEBUG (Cell 15 - Before Optimizer/Trainer) ---
print(f"--- DEBUG (Cell 15 - Before Optimizer/Trainer) ---")
print(f"Device of model's first parameter: {next(model.parameters()).device}")
if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
    print(f"Device of model.patch_embed.proj.weight: {model.patch_embed.proj.weight.device}")
    if model.patch_embed.proj.weight.device.type == 'cpu' and device.type == 'cuda':
         print(f"WARNING: model.patch_embed.proj.weight is on CPU before trainer init!")
else:
    print("Could not access model.patch_embed.proj.weight for debugging in Cell 15.")
print(f"--- END DEBUG (Cell 15 - Before Optimizer/Trainer) ---")

# Define optimizer with parameters from model and ArcFace
params = itertools.chain(model.parameters(), criterion.parameters()) 
optimizer = optim.Adam(params, lr=0.0001)
print("Optimizer: Adam with learning rate 0.0001")

# Define learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
print("Scheduler: CosineAnnealingLR")
print(f" - Learning rate will follow cosine decay from 0.0001 to 1e-6 over {epochs} epochs")

# Create a list to store training losses
train_losses = []

# Define a custom callback to track losses
def track_loss_callback(trainer, epoch_data):
    train_losses.append(epoch_data["train_loss_epoch_avg"])
    validation_callback(trainer)  # Pass the trainer object
    
    # Ensure model is fully on the correct device after validation
    if hasattr(trainer.model, 'patch_embed') and hasattr(trainer.model.patch_embed, 'proj'):
        if trainer.model.patch_embed.proj.weight.device != trainer.device:
            print(f"Moving model.patch_embed.proj back to {trainer.device}...")
            trainer.model.patch_embed.proj = trainer.model.patch_embed.proj.to(trainer.device)
    
    # Check for any remaining off-device parameters and move them
    wrong = find_off_device(trainer.model, trainer.device)
    if wrong:
        print(f"Found {len(wrong)} parameters on wrong device after validation. Moving them...")
        trainer.model.to(trainer.device)

# Update trainer with the combined callback
trainer = BasicTrainer(
    dataset=train_dataset,
    model=model, # This model instance should be fully on `device`           
    objective=criterion,    
    optimizer=optimizer,
    epochs=epochs,
    scheduler=scheduler,
    device=device, # Pass the canonical `device` to the trainer
    batch_size=batch_size,
    num_workers=num_workers,
    epoch_callback=track_loss_callback
)

print("Trainer configured with:")
print(f" - {len(train_dataset)} training images")
print(f" - {epochs} epochs")
print(" - ArcFace loss")
print(" - Cosine annealing learning rate scheduler")
print(" - Loss tracking and validation callbacks")

# --- DEBUG (Cell 15 - After Trainer Init) ---
print(f"--- DEBUG (Cell 15 - After Trainer Init) ---")
if hasattr(trainer.model, 'patch_embed') and hasattr(trainer.model.patch_embed, 'proj'):
    print(f"Device of trainer.model.patch_embed.proj.weight: {trainer.model.patch_embed.proj.weight.device}")
else:
    print("Could not access trainer.model.patch_embed.proj.weight for debugging.")
print(f"Trainer's own device attribute (trainer.device): {trainer.device}")
print(f"--- END DEBUG (Cell 15 - After Trainer Init) ---")

#%% [markdown]
# ## 16. Train the Model
# This cell will take some time to run as it trains the model

#%%
ensure_model_on_device(model, device)

#%%
# Train the model
validation_callback(trainer)
ensure_model_on_device(model, device)
trainer.train()

#%% [markdown]
# ## 17. Generate Training Loss Graph

#%%
import matplotlib.pyplot as plt

# Use the manually tracked losses
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, 'o-', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

print(f"Final training loss: {train_losses[-1]:.4f}")

#%% [markdown]
# ## 18. Evaluate Model with Cosine Similarity for Re-identification

#%%
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create feature extractor using our fine-tuned model
extractor = DeepFeatures(model, device=device, batch_size=batch_size, num_workers=num_workers)

# Define transform for feature extraction
test_transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Extract features for validation set
val_features_dataset = ImageDataset(val_df, newt.root, transform=test_transform)
val_features = extractor(val_features_dataset)

# Create a gallery and query split from validation set
# Use first image of each identity as query, rest as gallery
identities = val_df['identity'].unique()
query_indices = []
gallery_indices = []

for identity in identities:
    identity_indices = val_df[val_df['identity'] == identity].index.tolist()
    if len(identity_indices) > 0:
        query_indices.append(identity_indices[0])
        gallery_indices.extend(identity_indices[1:])

# Create query and gallery dataframes
query_df = val_df.loc[query_indices].reset_index(drop=True)
gallery_df = val_df.loc[gallery_indices].reset_index(drop=True)

# Extract features for query and gallery
query_dataset = ImageDataset(query_df, newt.root, transform=test_transform)
gallery_dataset = ImageDataset(gallery_df, newt.root, transform=test_transform)

query_features = extractor(query_dataset)
gallery_features = extractor(gallery_dataset)

# Calculate cosine similarity between query and gallery features
similarity_function = CosineSimilarity()
similarity = similarity_function(query_features, gallery_features)
#%%
# Use KNN classifier for re-identification
classifier = KnnClassifier(k=1, database_labels=gallery_df['identity'].values)
predictions = classifier(similarity)

# Calculate accuracy
accuracy = np.mean(query_df['identity'].values == predictions)
print(f"Re-identification accuracy using cosine similarity: {accuracy:.4f}")

# Calculate additional metrics (precision, recall, F1 for each class)
y_true = query_df['identity'].values
y_pred = predictions

# Calculate metrics
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot top-k accuracy
k_values = [1, 3, 5]
top_k_accuracies = []

for k in k_values:
    classifier = KnnClassifier(k=k, database_labels=gallery_df['identity'].values)
    predictions = classifier(similarity)
    top_k_acc = np.mean([true_id in pred_ids for true_id, pred_ids in zip(query_df['identity'].values, predictions)])
    top_k_accuracies.append(top_k_acc)
    print(f"Top-{k} accuracy: {top_k_acc:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(k_values, top_k_accuracies, color='skyblue')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Top-k Re-identification Accuracy')
plt.xticks(k_values)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('top_k_accuracy.png')
plt.show()

#%% [markdown]
# ## 19. Save the Fine-tuned Model

#%%
# Save the fine-tuned model
torch.save(model.state_dict(), 'finetuned_newt_model.pth')
print("Model saved to 'finetuned_newt_model.pth'")

#%% [markdown]
# ## 20. Extract Features Using the Fine-tuned Model

#%%
# Extract features using the fine-tuned model
from wildlife_tools.features import DeepFeatures

# Define transform for feature extraction
test_transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Create dataset for feature extraction
test_dataset = ImageDataset(newt.df, newt.root, transform=test_transform)

# Create feature extractor
extractor = DeepFeatures(model, device=device, batch_size=batch_size, num_workers=num_workers)

print("Feature extractor created")
print("Extracting features for all images...")

#%% [markdown]
# ## 20. Next Steps
# Now you can use these features for:
# - Similarity calculation
# - Nearest neighbor search
# - Clustering
# - Visualization with t-SNE or UMAP
