# %% [markdown]
# # Newt Dataset Preparation
# > This notebook is used to prepare the newt dataset for training and evaluation.

# %%
#| default_exp newt_dataset

# %%
#| eval: false
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from pathlib import Path
from wildlife_datasets import datasets, analysis, splits
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
# Set pandas display options to show all columns and wide output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#%%
#| export
def download_kaggle_dataset(dataset_name, download_path):
    import os
    import kaggle

    if not os.path.exists(download_path):
        os.makedirs(download_path, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset downloaded to {download_path}")
    else:
        print(f"Dataset already exists at {download_path}")

    return download_path

# %%
download_kaggle_dataset("mshahoyi/bar-hill-surveys", "./data/barhill-unprocessed")

# %%
data_root = Path("./data/barhill-unprocessed")/'Bar Hill Surveys 2024'
walk = list(os.walk(data_root))
walk[:4]

# %%
gcns = [x for x in walk if 'gcn' in os.path.basename(x[0]).lower()]
gcns[:2]

# %%
data = [(os.path.basename(root), list(map(lambda f: os.path.join(root, f), files))) for root, _, files in gcns]

dict(data[:2])

# %%
metadata = pd.DataFrame(data).explode(1)
metadata.columns = ["identity", "file_path"]
metadata.shape

# %%
metadata.head()

# %%
metadata.info()

# %%
metadata['file_name'] = metadata['file_path'].apply(lambda x: os.path.basename(x))
metadata['is_video'] = metadata['file_name'].apply(lambda x: 'mov' in x.lower())
metadata.head()

# %% [markdown]
# # Merging and deleting images

# %%
gcns_to_merge = [
    # ("GCN10-P7-S8", "GCN11-P7-S8", "GCN13-P7-S8"), These are not the same newts. There is probably image level contamination.
    ("GCN53-P2-S4", "GCN52-P2-S4"),
]

files_to_delete = [
    ("GCN54-P2-S2", "IMG_2665.JPEG"),
]

# %%
print(f"Number of identities before merging: {metadata.identity.nunique()}")
for merge_group in gcns_to_merge:
    metadata.loc[metadata.identity.isin(merge_group[1:]), 'identity'] = merge_group[0]
print(f"Number of identities after merging: {metadata.identity.nunique()}")

# %%
print(f"Number of files before deleting: {metadata.file_name.nunique()}")
for gcn_id, file_name in files_to_delete:
    metadata = metadata[~((metadata.identity == gcn_id) & (metadata.file_name == file_name))]
print(f"Number of files after deleting: {metadata.file_name.nunique()}")

# %%
metadata.head()

# %%
output_dir = Path("./data/gcns-processed")
shutil.rmtree(output_dir, ignore_errors=True)
Path(output_dir).mkdir(exist_ok=True)

# %%
class UnprocessedNewtsDataset(datasets.WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        return metadata[~metadata.is_video].rename(columns={"file_name": "image_name", "file_path": "path"})

# %%
dataset = UnprocessedNewtsDataset('.')
dataset.plot_grid()
plt.savefig(output_dir/'distribution.png')

# %% [markdown]
# # Split the dataset

# %%
analysis.display_statistics(dataset.df)


# %%
metadata_new = metadata.copy().reset_index(drop=True)

for i, row in tqdm(metadata_new.iterrows()):
    new_path = Path('newts')/row.identity/row.file_name
    Path(output_dir/new_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(row.file_path, output_dir/new_path)
    metadata_new.loc[i, 'file_path'] = new_path

# %%
metadata_new.to_csv(output_dir/'metadata.csv', index=False)
metadata_new

# %% [markdown]
# # Upload to Kaggle

# %%
#| export

def upload_to_kaggle(user_id, title, id, licenses, keywords, dataset_dir):
    import json
    import subprocess
    import os
    from pathlib import Path

    dataset_dir = Path(dataset_dir)
    original_cwd = os.getcwd()

    try:
        # The 'id' in metadata must be in the format 'user_id/dataset_id'
        full_dataset_id = f"{user_id}/{id}"

        # Create dataset metadata for Kaggle
        kaggle_metadata = {
            "title": title,
            "id": full_dataset_id,
            "licenses": licenses,
            "keywords": keywords,
        }
        
        # Write Kaggle dataset metadata
        with open(dataset_dir / "dataset-metadata.json", "w") as f:
            json.dump(kaggle_metadata, f, indent=2)
        
        print("Checking if dataset exists...")
        
        # Check if dataset exists
        check_result = subprocess.run(
            ["kaggle", "datasets", "list", "--user", user_id, "--search", id],
            capture_output=True, text=True
        )
        
        dataset_exists = id in check_result.stdout

        os.chdir(dataset_dir)
        
        print(f"Current directory: {os.getcwd()}")
        print(f"Files to upload: {list(Path('.').iterdir())}")
        
        command = []
        if dataset_exists:
            print("Dataset exists, updating...")
            command = [
                "kaggle", "datasets", "version",
                "-p", ".",
                "-m", "Updated with new dataset",
                "--dir-mode", "zip"
            ]
        else:
            print("Dataset does not exist, creating new...")
            command = [
                "kaggle", "datasets", "create",
                "-p", ".",
                "--dir-mode", "zip"
            ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("✅ Dataset operation successful!")
        print("Output:", result.stdout)

        print(f"\nDataset available at: https://www.kaggle.com/datasets/{full_dataset_id}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during Kaggle CLI command.")
        print("--- STDERR ---")
        print(e.stderr)
        print("--- STDOUT ---")
        print(e.stdout)
        raise e
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        raise e
    finally:
        os.chdir(original_cwd)

# %%
upload_to_kaggle(user_id="mshahoyi",
                title="Barhill Great Crested Newts", 
                id="barhill-newts-all", 
                licenses=[{"name": "CC0-1.0"}], 
                keywords=["biology", "computer-vision", "animals", "great crested newts"], 
                dataset_dir="./data/gcns-processed")
# %%
import nbdev; nbdev.nbdev_export()
# %%
