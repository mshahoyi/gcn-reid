# %% [markdown]
# # Finetuning
# > This notebook finetunes MegaDescriptor and MiewID models on our newt dataset to improve their performance.

# %%
#| default_exp finetuning

# %%
#| eval: false
import os
import numpy as np
import sys
import torch
from gcn_reid.newt import get_newt_dataset, get_cropping_image_dataset
from wildlife_datasets import splits
import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from wildlife_tools.train import set_seed
from torchvision import transforms as T
from wildlife_tools.features import DeepFeatures
from wildlife_tools.inference import TopkClassifier
from wildlife_tools.similarity import CosineSimilarity
from transformers import AutoModel

# %% [markdown]
# # Download Newt Dataset
# %%
#| eval: false
NewtDataset = get_newt_dataset()
dataset_path = "data/newt_dataset"
NewtDataset._download(dataset_name="mshahoyi/barhill-newts-segmented", download_path=dataset_path)
dataset = NewtDataset(dataset_path)
dataset.df.head()

# %% [markdown]
# # Create Data Splits

# %% [markdown]
# ## Create Train/Test Split

# %%
def create_train_test_split(df, split_ratio=0.5):
    disjoint_splitter = splits.DisjointSetSplit(split_ratio)
    for idx_train, idx_test in disjoint_splitter.split(df):
        df_train, df_test = df.loc[idx_train], df.loc[idx_test]
        splits.analyze_split(df, idx_train, idx_test)
    return df_train, df_test

df_train, df_test = create_train_test_split(dataset.df, split_ratio=0.5)
df_test, df_val = create_train_test_split(df_test, split_ratio=0.5)

print(f"Train: {len(df_train)}, Test: {len(df_test)}, Validation: {len(df_val)}")

# %% [markdown]
# ## Closed Set Split (for database and query sets)

# %%
def create_database_query_split(df, split_ratio=0.9):
    splitter = splits.ClosedSetSplit(split_ratio)
    for idx_database, idx_query in splitter.split(df):
        df_database, df_query = df.loc[idx_database], df.loc[idx_query]
        splits.analyze_split(df, idx_database, idx_query)
    return df_database, df_query

df_test_database, df_test_query = create_database_query_split(df_test, split_ratio=0.9)
print(f"Test Database: {len(df_test_database)}, Test Query: {len(df_test_query)}\n\n\n")

df_val_database, df_val_query = create_database_query_split(df_val, split_ratio=0.9)
print(f"Validation Database: {len(df_val_database)}, Validation Query: {len(df_val_query)}\n\n\n")

df_train_database, df_train_query = create_database_query_split(df_train, split_ratio=0.9)
print(f"Train Database: {len(df_train_database)}, Train Query: {len(df_train_query)}\n\n\n")

# %% [markdown]
# # Train MegaDescriptor

# %%
# Download MegaDescriptor-T backbone from HuggingFace Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True).to(device)

# %%
CroppingImageDataset = get_cropping_image_dataset()
transform = T.Compose([
    T.Resize([224, 224]), 
    T.ToTensor(), 
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

train_dataset = CroppingImageDataset(df_train, root=dataset_path, transform=transform, crop_out=True)

# %%
def evaluate(model, df_query, df_database, data_root, transform, crop_out, batch_size, num_workers, device):
    # Calculate retrieval results
    extractor = DeepFeatures(model, 
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers)
    
    print("Extracting features for query set")
    dataset_query = CroppingImageDataset(df_query, root=data_root, transform=transform, crop_out=crop_out)
    query = extractor(dataset_query)

    print("Extracting features for database set")
    dataset_database = CroppingImageDataset(df_database, root=data_root, transform=transform, crop_out=crop_out)
    database = extractor(dataset_database)

    similarity_function = CosineSimilarity()
    similarity = similarity_function(query, database)
    top_5_classifier = TopkClassifier(k=5, database_labels=dataset_database.labels_string, return_all=True)
    
    predictions_top_5, scores_top_5, _ = top_5_classifier(similarity)
    accuracy_top_1 = np.mean(dataset_query.labels_string == predictions_top_5[:, 0])
    accuracy_top_5 = np.mean(np.any(predictions_top_5 == dataset_query.labels_string[:, np.newaxis], axis=1))

    return dict(accuracy_top_1=accuracy_top_1, accuracy_top_5=accuracy_top_5)

# %%
val_transform = T.Compose([
    T.Resize([224, 224]), 
    T.ToTensor(), 
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

evaluate(model=backbone, 
         df_query=df_val_query, 
         df_database=df_val_database, 
         data_root=dataset_path, 
         transform=val_transform, 
         crop_out=True, 
         batch_size=32, num_workers=4, device=device)

# %%
class EvaluationCb:
    def __init__(self, df_query, df_database, root, transform, crop_out):
        self.df_query = df_query
        self.df_database = df_database
        self.root = root
        self.transform = transform
        self.crop_out = crop_out
        self.history = []
        
    def __call__(self, trainer, epoch_data):
        eval_results = evaluate(model=trainer.model, 
                           df_query=self.df_query, 
                           df_database=self.df_database, 
                           data_root=self.root, 
                           transform=self.transform, 
                           crop_out=self.crop_out, 
                           batch_size=trainer.batch_size, 
                           num_workers=trainer.num_workers, 
                           device=trainer.device)
        
        epoch_data.update(eval_results)
        self.history.append(epoch_data)
        print(f"Accuracy top 1: {eval_results['accuracy_top_1']}, Accuracy top 5: {eval_results['accuracy_top_5']}\n\n")
        trainer.model = trainer.model.to(trainer.device)

# %%
epoch_callback = EvaluationCb(df_val_query, 
                                df_val_database, 
                                dataset_path, 
                                val_transform, 
                                crop_out=True)

# %%
# Arcface loss - needs backbone output size and number of classes.
objective = ArcFaceLoss(
    num_classes=train_dataset.num_classes,
    embedding_size=768,
    margin=0.5,
    scale=64    
    )

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

set_seed(0)
trainer = BasicTrainer(
    dataset=train_dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=5,
    device=device,
    num_workers=4,
    epoch_callback=epoch_callback
    )

trainer.train()

# %%
epoch_callback.history

# %% [markdown]
# ## MegaDescriptor on Test Set

# %%
mega_descriptor_results = evaluate(model=backbone, 
         df_query=df_test_query, 
         df_database=df_test_database, 
         data_root=dataset_path, 
         transform=val_transform, 
         crop_out=True, 
         batch_size=32, 
         num_workers=4, 
         device=device)

print("MegaDescriptor Results:", mega_descriptor_results)

# %% [markdown]
# # Train MiewID

# %%
#| hide_output: true
miew_id_model = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True).to(device)

# %%
miew_id_results = evaluate(model=miew_id_model, 
         df_query=df_val_query, 
         df_database=df_val_database, 
         data_root=dataset_path, 
         transform=val_transform, 
         crop_out=True, 
         batch_size=32, num_workers=4, device=device)

print("MiewID Results before finetuning:", miew_id_results)

# %%
miew_id_epoch_callback = EvaluationCb(df_val_query, 
                                df_val_database, 
                                dataset_path, 
                                val_transform, 
                                crop_out=True)

# %%


# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

set_seed(0)
trainer = BasicTrainer(
    dataset=train_dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=5,
    device=device,
    num_workers=4,
    epoch_callback=epoch_callback
    )

trainer.train()

# %%
epoch_callback.history

# %%
import nbdev; nbdev.nbdev_export()