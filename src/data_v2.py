# %%
gcns_to_merge = [
    ("GCN10-P7-S8", "GCN11-P7-S8", "GCN13-P7-S8"),
    ("GCN53-P2-S4", "GCN52-P2-S4"),
]

images_to_delete = [
    ("GCN54-P2-S2", "IMG_2665.JPEG"),
]

# %%
import os
os.system('kaggle datasets download --unzip mshahoyi/bar-hill-surveys -p /kaggle/working')
#%%
os.system('mv /kaggle/working/"Bar Hill Surveys 2024" /kaggle/working/barhill')

# %%
os.system('pip install -q ramda')
#%%
os.system('ls /kaggle/working/barhill')

# %%
import ramda as R
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd
import json

data_root = Path('/kaggle/working/barhill')

# %%
# Delete specified images
print("Deleting specified images...")
for gcn_id, image_name in images_to_delete:
    # Find the image by searching through the directory structure
    found = False
    for root, _, files in os.walk(data_root):
        # Check if this folder contains the GCN ID
        if gcn_id.lower() in root.lower():
            # Look for the image in this folder
            for file in files:
                if file.lower() == image_name.lower():
                    image_path = os.path.join(root, file)
                    os.remove(image_path)
                    print(f"Deleted {image_path}")
                    found = True
                    break
        if found:
            break
    
    if not found:
        print(f"Warning: Image {image_name} for {gcn_id} not found")

# %% 
# Merge GCN folders as specified
print("Merging GCN folders...")
for merge_group in gcns_to_merge:
    target_gcn_id = merge_group[0]
    target_folder = None
    
    # Find the target folder
    for root, dirs, _ in os.walk(data_root):
        for dir_name in dirs:
            if target_gcn_id.lower() in os.path.join(root, dir_name).lower():
                target_folder = os.path.join(root, dir_name)
                break
        if target_folder:
            break
    
    if not target_folder:
        print(f"Warning: Target folder for {target_gcn_id} not found")
        continue
    
    # Process each source folder to merge into target
    for source_gcn_id in merge_group[1:]:
        source_folders = []
        
        # Find all folders containing the source GCN ID
        for root, dirs, _ in os.walk(data_root):
            for dir_name in dirs:
                if source_gcn_id.lower() in os.path.join(root, dir_name).lower():
                    source_folders.append(os.path.join(root, dir_name))
        
        if not source_folders:
            print(f"Warning: No folders found for {source_gcn_id}")
            continue
        
        # Process each found source folder
        for source_folder in source_folders:
            # Copy all files from source to target
            for file_name in os.listdir(source_folder):
                source_file = os.path.join(source_folder, file_name)
                target_file = os.path.join(target_folder, file_name)
                
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {source_file} to {target_folder}")
            
            # Remove the source folder after merging
            shutil.rmtree(source_folder)
            print(f"Removed {source_folder} after merging")

# %% 
# REGENERATE the walk list AFTER merges and deletions
print("Regenerating file lists after merges and deletions...")
walk = list(os.walk(data_root))

# %% Only get the GCN folders
gcns = R.filter(lambda x: 'gcn' in os.path.basename(x[0]).lower(), walk)
gcns[:2]

# %% Get the image files
image_extensions = (".jpg", ".jpeg", ".png")

data = [(os.path.basename(root), R.compose(
    R.map(lambda f: os.path.join(root, f)), 
    R.filter(lambda f: f.lower().endswith(image_extensions))
)(files)) for root, _, files in gcns]

dict(data[:2])    

# %%
base_dir = Path("/kaggle/working/barhill_output_v2")
output_dir = base_dir/"GCNs"
os.system(f'rm -rf {base_dir}')
Path(base_dir).mkdir(exist_ok=True)
Path(output_dir).mkdir(exist_ok=True)
os.system(f'ls {base_dir}')

# %%
# All GCN folders to create
label_folders = R.map(lambda x: os.path.join(output_dir, x[0]))(data) # x[0] is newt id
label_folders[:5]

# %%
# File source to destination mapping
src_to_dest_map = R.chain(lambda x: R.map(lambda y: (y, os.path.join(output_dir, x[0], os.path.basename(y))), x[1]))(data) # x[1] is filenames
src_to_dest_map[:5]

# %%
# Do the copying
for f in label_folders: Path(f).mkdir(exist_ok=True)

# Verify all source files exist before attempting to copy
missing_files = []
for src, _ in src_to_dest_map:
    if not os.path.exists(src):
        missing_files.append(src)

if missing_files:
    print(f"Warning: {len(missing_files)} source files are missing:")
    for f in missing_files[:10]:  # Show first 10 missing files
        print(f"  - {f}")
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")
        
    # Filter out missing files from src_to_dest_map
    src_to_dest_map = [(src, dst) for src, dst in src_to_dest_map if os.path.exists(src)]
    print(f"Filtered src_to_dest_map to {len(src_to_dest_map)} valid files")

# Copy the files
for src, dst in tqdm(src_to_dest_map): 
    shutil.copy(src, dst)

# %%
list(os.walk(output_dir))[:3]

# %%
# Split data into gallery and probe
gallery_and_probe_data = []

random.seed(42)
for newt_id in os.listdir(output_dir):
    newt_folder = os.path.join(output_dir, newt_id)
    images = [f for f in os.listdir(newt_folder)]
    random.shuffle(images)
    
    # Split: 70% gallery, 30% probe (adjust as needed)
    split_idx = int(0.7 * len(images))
    gallery_images = images[:split_idx]
    probe_images = images[split_idx:]
    
    # Copy to respective folders
    for img in gallery_images:
        gallery_and_probe_data.append({
            'image_path': os.path.join(newt_folder, img),
            'image_name': img,
            'newt_id': newt_id,
            'is_probe': 0
        })
        
    for img in probe_images:
        gallery_and_probe_data.append({
            'image_path': os.path.join(newt_folder, img),
            'image_name': img,
            'newt_id': newt_id,
            'is_probe': 1
        })
gallery_and_probe_data[:5]

# %%
df = pd.DataFrame(gallery_and_probe_data)
df.head()

# %%
df.is_probe.mean()

# %%
df.to_csv(base_dir/"gallery_and_probes.csv")

# %%
os.system(f'ls {base_dir}')

# %%
# Update Kaggle dataset metadata
import json

os.system(f'kaggle datasets init -p {base_dir}')
metadata_path = os.path.join(base_dir, "dataset-metadata.json")

# Load existing metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Update title and id
metadata["title"] = "Barhill Processed V2"
metadata["id"] = "mshahoyi/barhill-processed-v2"

# Add description if not present
if "description" not in metadata:
    metadata["description"] = """# Bar Hill Newt Surveys Processed Dataset V2

This dataset contains processed images of great crested newts (GCNs) from Bar Hill surveys.

## Processing
- Merged duplicate GCN identities (e.g., merged GCN11-P7-S8 and GCN13-P7-S8 into GCN10-P7-S8)
- Removed problematic images
- Split into gallery (70%) and probe (30%) sets for re-identification tasks

## Contents
- `GCNs/`: Folders organized by newt ID containing newt images
- `gallery_and_probes.csv`: Information about each image, including paths and labels
"""

# Save updated metadata
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Updated metadata file at {metadata_path}")
os.system(f'cat {metadata_path}')

# %%
# Push the dataset to Kaggle with updated metadata
print("Pushing processed dataset to Kaggle...")
os.system(f'kaggle datasets create -p {base_dir} -r zip')
