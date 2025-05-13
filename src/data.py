# %%
import os
os.system('kaggle datasets download -d bar-hill-surveys')

# %%
os.system('ls /kaggle/input/bar-hill-surveys')

# %%
import ramda as R
import os
import shutil
from pathlib import Path

data_root = Path("/kaggle/input/bar-hill-surveys")/'Bar Hill Surveys 2024'
walk = list(os.walk(data_root))
walk[:4]

# %%
gcns = R.filter(lambda x: 'gcn' in os.path.basename(x[0]).lower(), walk) #
gcns[:2]

# %%
image_extensions = (".jpg", ".jpeg", ".png")

data = [(os.path.basename(root), R.compose(
    R.map(lambda f: os.path.join(root, f)), 
    R.filter(lambda f: f.lower().endswith(image_extensions))
)(files)) for root, _, files in gcns]

dict(data[:2])    

# %%
base_dir = Path("barhill")
output_dir = base_dir/"GCNs"
!rm -rf {base_dir}
Path(base_dir).mkdir(exist_ok=True)
Path(output_dir).mkdir(exist_ok=True)
!ls

# %%
# All GCN folders to create
label_folders = R.map(lambda x: os.path.join(output_dir, x[0]))(data) # x[0] is newt id
label_folders[:5]

# %%
# File source to destination mapping
src_to_dest_map = R.chain(lambda x: R.map(lambda y: (y, os.path.join(output_dir, x[0], os.path.basename(y))), x[1]))(data) # x[1] is filenames
src_to_dest_map[:5]

# %%
from tqdm import tqdm

# Do the copying
for f in label_folders: Path(f).mkdir(exist_ok=True)
for src, dst in tqdm(src_to_dest_map): shutil.copy(src, dst)

# %%
list(os.walk(output_dir))[:3]

# %%
import random

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
ls {base_dir}

# %%



