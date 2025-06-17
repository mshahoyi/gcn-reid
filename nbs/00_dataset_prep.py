# %% [markdown]
# # Newt Dataset Preparation
# > This notebook is used to prepare the newt dataset for training and evaluation.

# %%
#| default_exp newt_dataset

# %%
#| eval: false
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ramda as R
import os
import shutil
from pathlib import Path
from wildlife_datasets import datasets, analysis, splits
from tqdm import tqdm
import matplotlib.pyplot as plt

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
gcns = R.filter(lambda x: 'gcn' in os.path.basename(x[0]).lower(), walk) #
gcns[:2]

# %%
data = [(os.path.basename(root), R.map(lambda f: os.path.join(root, f), files)) for root, _, files in gcns]

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

# %%
class UnprocessedNewtsDataset(datasets.WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        return metadata[~metadata.is_video].rename(columns={"file_name": "image_name", "file_path": "path"})

# %%
dataset = UnprocessedNewtsDataset('.')
dataset.plot_grid()

# %% [markdown]
# # Split the dataset

# %%
analysis.display_statistics(dataset.df)

# %%
def create_train_test_split(df, split_ratio=0.5):
    disjoint_splitter = splits.DisjointSetSplit(split_ratio)
    for idx_train, idx_test in disjoint_splitter.split(df):
        df_train, df_test = df.loc[idx_train], df.loc[idx_test]
        splits.analyze_split(df, idx_train, idx_test)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

df_train, df_test = create_train_test_split(dataset.df, split_ratio=0.5)
df_test, df_val = create_train_test_split(df_test, split_ratio=0.5)

print(f"Train: {len(df_train)}, Test: {len(df_test)}, Validation: {len(df_val)}")

# %%
df_train.head()

# %%
output_dir = Path("./data/gcns-processed")
train_dir = output_dir/"train"
val_dir = output_dir/"val"
test_dir = output_dir/"test"

Path(output_dir).mkdir(exist_ok=True)
Path(train_dir).mkdir(exist_ok=True)
Path(val_dir).mkdir(exist_ok=True)
Path(test_dir).mkdir(exist_ok=True)

# %%
train_ids = df_train.identity.unique()
val_ids = df_val.identity.unique()
test_ids = df_test.identity.unique()

# %%
metadata_new = metadata.copy()

for i, row in tqdm(metadata_new.iterrows()):
    if row.identity in train_ids:
        split = 'train'
    elif row.identity in val_ids:
        split = 'val'
    elif row.identity in test_ids:
        split = 'test'
    else:
        raise ValueError(f"Unknown identity: {row.identity}")
    
    new_path = Path(split)/row.identity/row.file_name
    Path(output_dir/new_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(row.file_path, output_dir/new_path)

    metadata_new.loc[i, 'split'] = split
    metadata_new.loc[i, 'file_path'] = new_path

metadata_new.head()

# %%
metadata_new.split.value_counts()

# %%
metadata_new.to_csv("./data/gcns-processed/metadata.csv", index=False)

# %%
print("Number of identities in each split:")
metadata_new.groupby('split').identity.nunique()

# %%
print("Number of files in each split (including videos):")
metadata_new.groupby('split').file_name.nunique()

# %%
print("Number of images and videos:")
metadata_new.is_video.value_counts()

# %%
analysis.display_statistics(metadata_new[metadata_new.split == 'train'])
plt.title('Train set statistics')
plt.savefig('./data/gcns-processed/train_statistics.png')
plt.show()
plt.close()

# %%
analysis.display_statistics(metadata_new[metadata_new.split == 'val'])
plt.title('Validation set statistics')
plt.savefig('./data/gcns-processed/val_statistics.png')
plt.show()
plt.close()

# %%
analysis.display_statistics(metadata_new[metadata_new.split == 'test'])
plt.title('Test set statistics')
plt.savefig('./data/gcns-processed/test_statistics.png')
plt.show()
plt.close()

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
