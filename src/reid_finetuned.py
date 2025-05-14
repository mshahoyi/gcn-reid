# %% [markdown]
# # Newt Re-Identification with Fine-tuned Model
# This notebook evaluates a fine-tuned model for newt re-identification.

# %% [markdown]
# ## 1. Setup and Imports

# %%
# Define constants
IMAGE_PATH_COL = 'cropped_image_path'

# %%
# Set up data paths
from pathlib import Path

# Fix the base directory path
base_dir = Path('/kaggle/working/barhill')  # Changed from '/kaggle/working/barhill_output'
data_dir = base_dir
print(f"Data directory: {data_dir}")

# %%
# Import libraries
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import ramda as R

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Load and Prepare Dataset

# %%
# Load the dataset
df = pd.read_csv(data_dir/"gallery_and_probes.csv")
print(f"Loaded dataset with {len(df)} images")
df.head()

# %%
# Fix the path issue - ensure we don't duplicate the base path
df['cropped_image_path'] = df.image_path.str.replace('GCNs', 'gcns-cropped')

# Check if paths already contain the base directory
if not df['cropped_image_path'].iloc[0].startswith('/kaggle'):
    # If not, add the base directory
    df['cropped_image_path'] = str(base_dir.parent) + "/" + df['cropped_image_path']
    df['image_path'] = str(base_dir.parent) + "/" + df['image_path']
else:
    print("Paths already contain base directory, not prepending again")

print("Path columns added to dataframe")
print(f"Sample cropped path: {df['cropped_image_path'].iloc[0]}")
df.tail()

# %%
# Check if paths exist
valid_cropped_paths = df['cropped_image_path'].apply(lambda x: os.path.exists(x))
valid_full_paths = df['image_path'].apply(lambda x: os.path.exists(x))

print(f"Cropped images exist: {valid_cropped_paths.sum()} out of {len(df)}")
print(f"Full images exist: {valid_full_paths.sum()} out of {len(df)}")

if valid_cropped_paths.sum() == 0:
    print("WARNING: No cropped image paths exist. Let's check a few paths:")
    for i in range(min(5, len(df))):
        print(f"Path {i+1}: {df['cropped_image_path'].iloc[i]} - Exists: {os.path.exists(df['cropped_image_path'].iloc[i])}")

# %%
# Count probes and gallery images
probe_count = df[df.is_probe == 1].shape[0]
gallery_count = df[df.is_probe == 0].shape[0]
print(f"Dataset contains {probe_count} probe images and {gallery_count} gallery images")
print(f"Number of unique newt IDs: {df.newt_id.nunique()}")

# %% [markdown]
# ## 3. Load Fine-tuned Model

# %%
# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Load model architecture and fine-tuned weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)

try:
    model.load_state_dict(torch.load('finetuned_newt_model.pth', map_location=device))
    model.eval()
    model.to(device)
    print(f"✅ Successfully loaded fine-tuned model from finetuned_newt_model.pth")
    print(f"Model is on device: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# %% [markdown]
# ## 4. Feature Extraction

# %%
# Define feature extraction function
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor).cpu().numpy()
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

# %%
# Extract features for gallery images
gallery_features = {}
for index, row in tqdm(df[df.is_probe == 0].iterrows(), desc="Gallery features"):
    if os.path.exists(row[IMAGE_PATH_COL]):
        features = extract_features(row[IMAGE_PATH_COL])
        if features is not None:
            gallery_features[row.image_name] = (row.newt_id, features)

print(f"Extracted features for {len(gallery_features)} gallery images")
if len(gallery_features) == 0:
    print("WARNING: No gallery features extracted. Check image paths.")

# %%
# Extract features for probe images
probe_features = {}
for index, row in tqdm(df[df.is_probe == 1].iterrows(), desc="Probe features"):
    if os.path.exists(row[IMAGE_PATH_COL]):
        features = extract_features(row[IMAGE_PATH_COL])
        if features is not None:
            probe_features[row.image_name] = (row.newt_id, features)

print(f"Extracted features for {len(probe_features)} probe images")
if len(probe_features) == 0:
    print("WARNING: No probe features extracted. Check image paths.")

# %% [markdown]
# ## 5. Re-Identification

# %%
# Compute similarities and rank matches
def re_identify(probe_features, gallery_features):
    results = {}
    for probe_name, (probe_id, probe_feat) in tqdm(probe_features.items(), desc="Re-identifying"):
        similarities = {}
        for gallery_name, (gallery_id, gallery_feat) in gallery_features.items():
            sim = cosine_similarity(probe_feat.reshape(1, -1), gallery_feat.reshape(1, -1))[0][0]
            similarities[gallery_name] = (gallery_id, sim)
        
        # Sort by similarity (descending)
        ranked = sorted(similarities.items(), key=lambda x: x[1][1], reverse=True)
        results[probe_name] = (probe_id, ranked)
    return results

# %%
# Run re-identification
if len(probe_features) > 0 and len(gallery_features) > 0:
    results = re_identify(probe_features, gallery_features)
    print(f"Completed re-identification for {len(results)} probe images")
else:
    print("Cannot perform re-identification: no features extracted")
    results = {}

# %% [markdown]
# ## 6. Evaluation Metrics

# %%
# Calculate evaluation metrics
def evaluate_results(results):
    if not results:
        print("No results to evaluate")
        return {"Top-1 Accuracy": 0.0, "Top-5 Accuracy": 0.0, "mAP": 0.0}
        
    top1_correct = 0
    top5_correct = 0
    ap_scores = []
    
    for probe_name, (true_id, ranked) in results.items():
        # Top-1 and Top-5 accuracy
        top_ids = [gallery_id for _, (gallery_id, _) in ranked[:5]]
        if ranked and true_id == ranked[0][1][0]:
            top1_correct += 1
        if true_id in top_ids:
            top5_correct += 1
        
        # Average Precision (AP)
        relevant = [1 if gallery_id == true_id else 0 for _, (gallery_id, _) in ranked]
        if sum(relevant) > 0:
            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.sum(precision_at_k * relevant) / sum(relevant)
            ap_scores.append(ap)
    
    result_count = len(results)
    if result_count > 0:
        top1_acc = top1_correct / result_count
        top5_acc = top5_correct / result_count
        mAP = np.mean(ap_scores) if ap_scores else 0.0
    else:
        top1_acc = top5_acc = mAP = 0.0
    
    return {"Top-1 Accuracy": top1_acc, "Top-5 Accuracy": top5_acc, "mAP": mAP}

# %%
# Evaluate and display metrics
metrics = evaluate_results(results)
print("Re-identification Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# %%
# Plot metrics as a bar chart
if any(metrics.values()):
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylim(0, 1)
    plt.title('Re-identification Performance Metrics')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.show()
else:
    print("No metrics to plot")

# %% [markdown]
# ## 7. Visualize Results

# %%
# Function to plot re-identification results
def plot_results(slice_start, slice_end, folder="gcns-cropped"):
    if not results:
        print("No results to visualize")
        return
        
    result_list = list(results.items())
    
    for i in range(slice_start, slice_end):
        if i >= len(result_list):
            print(f"Index {i} out of range. Only {len(result_list)} results available.")
            continue
            
        probe_file_name = result_list[i][0]
        probe_newt_id = result_list[i][1][0]
        top_5_file_info = result_list[i][1][1][:5]
        
        # Load probe image
        probe_path = os.path.join(data_dir, folder, probe_newt_id, probe_file_name)
        if not os.path.exists(probe_path):
            print(f"Probe image not found: {probe_path}")
            # Try alternate path
            probe_path = os.path.join(folder, probe_newt_id, probe_file_name)
            if not os.path.exists(probe_path):
                print(f"Alternate probe path also not found: {probe_path}")
                continue
            
        prb_img = [(probe_file_name, "Source", Image.open(probe_path), probe_newt_id)]
        
        # Load gallery images
        gallery_imgs = []
        for gallery_name, (gallery_id, score) in top_5_file_info:
            gallery_path = os.path.join(data_dir, folder, gallery_id, gallery_name)
            if not os.path.exists(gallery_path):
                # Try alternate path
                gallery_path = os.path.join(folder, gallery_id, gallery_name)
            
            if os.path.exists(gallery_path):
                gallery_imgs.append((gallery_name, score, Image.open(gallery_path), gallery_id))
            else:
                print(f"Gallery image not found: {gallery_path}")
                
        images = prb_img + gallery_imgs
        
        plt.figure(figsize=(25, 10))
        for j, (file_name, score, img, newt_id) in enumerate(images):
            plt.subplot(1, len(images), j + 1)
            plt.imshow(img)
            if score == "Source":
                plt.xlabel(f"{file_name} - {newt_id}", fontsize=12)
            else:
                color = 'green' if newt_id == probe_newt_id else 'red'
                plt.xlabel(f"{file_name} - {newt_id} - {score:.2f}", 
                          color='white', 
                          fontsize=12,
                          bbox=dict(facecolor=color, edgecolor='none', pad=2.0))
            plt.yticks([])
            plt.xticks([])
        
        plt.suptitle(f"Probe: {probe_file_name} (ID: {probe_newt_id})", fontsize=16)
        plt.tight_layout()
        plt.show()

# %%
# Show first 3 results (if available)
if results:
    plot_results(0, min(3, len(results)))
else:
    print("No results to visualize")

# %% [markdown]
# ## 8. Interactive Visualization with Error Prioritization

# %%
# Create interactive visualization that prioritizes incorrect matches
from ipywidgets import widgets
from IPython.display import display, clear_output

# Sort results to prioritize incorrect matches
def sort_results_by_errors(results_dict):
    if not results_dict:
        return []
    
    # Create a list of (probe_name, probe_id, ranked_matches, error_type) tuples
    # error_type: 0 = correct top-1, 1 = incorrect top-1, 2 = incorrect top-1 with high confidence
    sorted_items = []
    
    for probe_name, (probe_id, ranked_matches) in results_dict.items():
        if not ranked_matches:
            continue
            
        top_match_id = ranked_matches[0][1][0]
        top_match_score = ranked_matches[0][1][1]
        
        if probe_id == top_match_id:
            # Correct top-1 match
            error_type = 0
        else:
            # Incorrect top-1 match
            # Higher score = more confident wrong match = higher priority
            error_type = 1 + top_match_score
            
        sorted_items.append((probe_name, probe_id, ranked_matches, error_type))
    
    # Sort by error_type (descending) to prioritize incorrect matches with high confidence
    sorted_items.sort(key=lambda x: x[3], reverse=True)
    return sorted_items

# Sort the results
sorted_results = sort_results_by_errors(results)
current_index = 0
max_index = len(sorted_results) - 1 if sorted_results else -1

# Print summary of sorting
if sorted_results:
    incorrect_count = sum(1 for item in sorted_results if item[3] > 0)
    print(f"Sorted {len(sorted_results)} results with {incorrect_count} incorrect top-1 matches")
    print("Showing incorrect matches first, sorted by confidence score")

def show_plot(index, use_cropped=False):
    """
    Display the probe image and top matches.
    
    Args:
        index: Index of the result to display
        use_cropped: If True, use images from 'gcns-cropped' folder, otherwise use 'GCNs'
    """
    folder = "gcns-cropped" if use_cropped else "GCNs"
    if not sorted_results or max_index < 0:
        print("No results to visualize")
        return
        
    if index < len(sorted_results):
        probe_name, probe_id, ranked_matches, error_type = sorted_results[index]
        top_matches = ranked_matches[:5]  # Show top 5 matches
        
        # Load and display images
        plt.figure(figsize=(25, 10))
        
        # Try different possible paths for the probe image, prioritizing the selected folder type
        probe_paths = []
        if use_cropped:
            probe_paths = [
                os.path.join(data_dir, "gcns-cropped", probe_id, probe_name),
                os.path.join("gcns-cropped", probe_id, probe_name),
                os.path.join(data_dir, "GCNs", probe_id, probe_name),
                os.path.join("GCNs", probe_id, probe_name)
            ]
        else:
            probe_paths = [
                os.path.join(data_dir, "GCNs", probe_id, probe_name),
                os.path.join("GCNs", probe_id, probe_name),
                os.path.join(data_dir, "gcns-cropped", probe_id, probe_name),
                os.path.join("gcns-cropped", probe_id, probe_name)
            ]
        
        probe_img = None
        probe_path_used = None
        for path in probe_paths:
            if os.path.exists(path):
                probe_img = Image.open(path)
                probe_path_used = path
                break
                
        # Display probe image
        plt.subplot(1, 6, 1)  # 1 row, 6 columns (probe + 5 matches)
        if probe_img:
            plt.imshow(probe_img)
            plt.xlabel(f"{probe_name} - {probe_id}", fontsize=12)
        else:
            plt.text(0.5, 0.5, "Image not found", ha='center', va='center')
            plt.xlabel(f"{probe_name} - {probe_id} (missing)", fontsize=12)
        plt.yticks([])
        plt.xticks([])
        
        # Display top matches
        for i, (gallery_name, (gallery_id, score)) in enumerate(top_matches):
            plt.subplot(1, 6, i + 2)
            
            # Prioritize the selected folder type for gallery images
            gallery_paths = []
            if use_cropped:
                gallery_paths = [
                    os.path.join(data_dir, "gcns-cropped", gallery_id, gallery_name),
                    os.path.join("gcns-cropped", gallery_id, gallery_name),
                    os.path.join(data_dir, "GCNs", gallery_id, gallery_name),
                    os.path.join("GCNs", gallery_id, gallery_name)
                ]
            else:
                gallery_paths = [
                    os.path.join(data_dir, "GCNs", gallery_id, gallery_name),
                    os.path.join("GCNs", gallery_id, gallery_name),
                    os.path.join(data_dir, "gcns-cropped", gallery_id, gallery_name),
                    os.path.join("gcns-cropped", gallery_id, gallery_name)
                ]
            
            gallery_img = None
            for path in gallery_paths:
                if path and os.path.exists(path):
                    gallery_img = Image.open(path)
                    break
                    
            if gallery_img:
                plt.imshow(gallery_img)
            else:
                plt.text(0.5, 0.5, "Image not found", ha='center', va='center')
                
            color = 'green' if gallery_id == probe_id else 'red'
            plt.xlabel(f"{gallery_name} - {gallery_id} - {score:.2f}", 
                      color='white', 
                      fontsize=12,
                      bbox=dict(facecolor=color, edgecolor='none', pad=2.0))
            plt.yticks([])
            plt.xticks([])
        
        # Add error status to title
        if error_type > 0:
            error_status = "❌ INCORRECT TOP MATCH"
        else:
            error_status = "✅ CORRECT TOP MATCH"
            
        plt.suptitle(f"Probe {index+1}/{max_index+1}: {probe_name} (ID: {probe_id}) - {error_status}", 
                    fontsize=16, 
                    color='red' if error_type > 0 else 'green')
        plt.tight_layout()
        plt.show()
    
def on_prev_clicked(b):
    global current_index
    current_index = max(0, current_index - 1)
    with output:
        clear_output(wait=True)
        show_plot(current_index, use_cropped=True)
        print(f"Image {current_index + 1} of {max_index + 1}")
    
def on_next_clicked(b):
    global current_index
    current_index = min(max_index, current_index + 1)
    with output:
        clear_output(wait=True)
        show_plot(current_index, use_cropped=True)
        print(f"Image {current_index + 1} of {max_index + 1}")

# Only create interactive widgets if we have results
if sorted_results:
    # Create buttons
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")

    # Set button callbacks
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)

    # Create output widget for displaying plots
    output = widgets.Output()

    # Display buttons and output widget
    button_box = widgets.HBox([prev_button, next_button])
    display(button_box)
    display(output)

    # Show initial plot
    with output:
        show_plot(current_index, use_cropped=True)
        print(f"Image {current_index + 1} of {max_index + 1}")
        if sorted_results[current_index][3] > 0:
            print("⚠️ This is an incorrect match")
else:
    print("No results available for interactive visualization")

# %% [markdown]
# ## 9. Save Results

# %%
# Save re-identification results to CSV
if results:
    results_df = []
    for probe_name, (probe_id, ranked) in results.items():
        for rank, (gallery_name, (gallery_id, similarity)) in enumerate(ranked[:10]):  # Save top 10 matches
            results_df.append({
                'probe_name': probe_name,
                'probe_id': probe_id,
                'rank': rank + 1,
                'gallery_name': gallery_name,
                'gallery_id': gallery_id,
                'similarity': similarity,
                'correct_match': probe_id == gallery_id
            })

    results_df = pd.DataFrame(results_df)
    results_df.to_csv('reid_results.csv', index=False)
    print(f"Results saved to reid_results.csv")
    results_df.head(10)
else:
    print("No results to save")

gcns_to_merge = [
    ("GCN10-P7-S8", "GCN11-P7-S8", 
     "GCN13-P7-S8"),
    ("GCN53-P2-S4", "GCN52-P2-S4"),
    
]

images_to_delete = [
    ("GCN54-P2-S2", "IMG_2665.JPEG"),
]