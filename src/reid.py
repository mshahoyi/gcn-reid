# %%
IMAGE_PATH_COL = 'image_path'

# %%
import os
os.system('pip -q install ramda')

#%%
from pathlib import Path

base_dir = Path('/kaggle/working/barhill_output')
data_dir = base_dir

#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ramda as R


# %%
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import timm
from pathlib import Path
import pandas as pd

# %%
df = pd.read_csv(data_dir/"gallery_and_probes.csv")
df.head()

# %%
# Create a new column with the path in gcns_cropped instead of GCNs
# df['cropped_image_path'] = df.image_path.str.replace('GCNs', 'gcns-cropped')
# df['cropped_image_path'] = str(base_dir) + "/" + df['cropped_image_path']
# df['image_path'] = str(base_dir) + "/" + df['image_path']
# df.tail()

# %%
# Preprocessing for MegaDescriptor
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load MegaDescriptor model
model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);

# %%
# Function to extract features
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor).cpu().numpy()
    return features.flatten()

# %%
# Extract features for gallery and probe
gallery_features = {}
probe_features = {}

for index, row in tqdm(df[df.is_probe == 0].iterrows()):
    if Path(row[IMAGE_PATH_COL]).exists():
        gallery_features[row.image_name] = (row.newt_id, extract_features(row[IMAGE_PATH_COL]))

for index, row in tqdm(df[df.is_probe == 1].iterrows()):
    if Path(row[IMAGE_PATH_COL]).exists():
        probe_features[row.image_name] = (row.newt_id, extract_features(row[IMAGE_PATH_COL]))

print(f"Extracted features for {len(gallery_features)} gallery and {len(probe_features)} probe images.")

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarities and rank matches
def re_identify(probe_features, gallery_features):
    results = {}
    for probe_name, (probe_id, probe_feat) in tqdm(probe_features.items()):
        similarities = {}
        for gallery_name, (gallery_id, gallery_feat) in gallery_features.items():
            sim = cosine_similarity(probe_feat.reshape(1, -1), gallery_feat.reshape(1, -1))[0][0]
            similarities[gallery_name] = (gallery_id, sim)
        
        # Sort by similarity (descending)
        ranked = sorted(similarities.items(), key=lambda x: x[1][1], reverse=True)
        results[probe_name] = (probe_id, ranked)
    return results

# Run re-identification
results = re_identify(probe_features, gallery_features)
# %%
import matplotlib.pyplot as plt
import random
import os

result_list = list(results.items())

def plot_results(slice: slice, folder="gcns-cropped"):
    rows = slice.stop - slice.start

    for i in range(rows):
        image = i
        number_of_images = 5
        
        probe_file_name = result_list[image][0]
        probe_newt_id = result_list[image][1][0]
        top_5_file_info = result_list[image][1][1][:number_of_images]
        prb_img = [(probe_file_name, "Source", Image.open(os.path.join(data_dir, folder, probe_newt_id, probe_file_name)), probe_newt_id)] 
        gallery_imgs = R.map(lambda x: (x[0], x[1][1], Image.open(os.path.join(data_dir, folder, x[1][0], x[0])), x[1][0]))(top_5_file_info)
        images = prb_img + gallery_imgs
        
        plt.figure(figsize=(25, 10))
        for i, (file_name, score, img, newt_id) in enumerate(images):
            plt.subplot(1, number_of_images + 1, i + 1)
            plt.imshow(img)
            if score == "Source":
                plt.xlabel(f"{file_name} - {newt_id}")
            else:
                plt.xlabel(f"{file_name} - {newt_id} - {score:.2f}", color='white', bbox=dict(facecolor='green' if newt_id == probe_newt_id else 'red', edgecolor='none', pad=2.0))
            plt.yticks([])
            plt.xticks([])
        plt.show()

# %%
def evaluate_results(results):
    top1_correct = 0
    top5_correct = 0
    ap_scores = []
    
    for probe_name, (true_id, ranked) in results.items():
        # Top-1 and Top-5 accuracy
        top_ids = [gallery_id for _, (gallery_id, _) in ranked[:5]]
        if true_id == ranked[0][1][0]:
            top1_correct += 1
        if true_id in top_ids:
            top5_correct += 1
        
        # Average Precision (AP)
        relevant = [1 if gallery_id == true_id else 0 for _, (gallery_id, _) in ranked]
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = np.sum(precision_at_k * relevant) / sum(relevant) if sum(relevant) > 0 else 0
        ap_scores.append(ap)
    
    top1_acc = top1_correct / len(results)
    top5_acc = top5_correct / len(results)
    mAP = np.mean(ap_scores)
    
    return {"Top-1 Accuracy": top1_acc, "Top-5 Accuracy": top5_acc, "mAP": mAP}

# Evaluate
metrics = evaluate_results(results)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
#%%
from ipywidgets import widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Create interactive widgets
current_index = 0
max_index = len(results) - 1

def show_plot(index):
    # Convert the index to a list to properly slice the results dictionary
    probe_names = list(results.keys())
    if index < len(probe_names):
        probe_name = probe_names[index]
        # Fix: The plot_results function expects a dictionary, not a slice
        # Also, make sure the plot_results function handles dictionaries correctly
        result_dict = {probe_name: results[probe_name]}
        
        # Display the result directly instead of using plot_results
        probe_file_name = probe_name
        probe_newt_id, ranked_matches = results[probe_name]
        top_matches = ranked_matches[:5]  # Show top 5 matches
        
        # Load and display images
        plt.figure(figsize=(25, 10))
        
        # Display probe image
        plt.subplot(1, 6, 1)  # 1 row, 6 columns (probe + 5 matches)
        probe_img = Image.open(os.path.join(data_dir, "GCNs", probe_newt_id, probe_file_name))
        plt.imshow(probe_img)
        plt.xlabel(f"{probe_file_name} - {probe_newt_id}", fontsize=12)
        plt.yticks([])
        plt.xticks([])
        
        # Display top matches
        for i, (gallery_name, (gallery_id, score)) in enumerate(top_matches):
            plt.subplot(1, 6, i + 2)
            gallery_img = Image.open(os.path.join(data_dir, "GCNs", gallery_id, gallery_name))
            plt.imshow(gallery_img)
            plt.xlabel(f"{gallery_name} - {gallery_id} - {score:.2f}", 
                      color='white', 
                      fontsize=12,  # Increased font size
                      bbox=dict(facecolor='green' if gallery_id == probe_newt_id else 'red', 
                               edgecolor='none', pad=2.0))
            plt.yticks([])
            plt.xticks([])
        
        plt.tight_layout()
        plt.show()
    
def on_prev_clicked(b):
    global current_index
    current_index = max(0, current_index - 1)
    with output:
        clear_output(wait=True)
        show_plot(current_index)
        # Display current position
        print(f"Image {current_index + 1} of {max_index + 1}")
    
def on_next_clicked(b):
    global current_index
    current_index = min(max_index, current_index + 1)
    with output:
        clear_output(wait=True)
        show_plot(current_index)
        # Display current position
        print(f"Image {current_index + 1} of {max_index + 1}")

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
    show_plot(current_index)
    print(f"Image 1 of {max_index + 1}")
# %%

gcns_to_merge = [
    ("GCN10-P7-S8", "GCN11-P7-S8"),
    
]