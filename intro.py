# %%
from wildlife_datasets import analysis, datasets, loader

# %%
d = loader.load_dataset(datasets.MacaqueFaces, 'data', 'dataframes')

# %%
from wildlife_datasets import splits
splits

# %%
splitter = splits.ClosedSetSplit(0.8)
idx_train, idx_test = splitter.split(d.df)[0]

    # %%
df_train = d.df.loc[idx_train]
df_train
# %%
df_test = d.df.loc[idx_test]
df_test
# %%
y_pred = ['Dan']*len(df_test)

# %%
from wildlife_datasets import metrics

y_true = df_test['identity']
metrics.accuracy(y_true, y_pred)
# %%
from wildlife_datasets import loader, datasets
from wildlife_tools.data import ImageDataset
import torchvision.transforms as T

metadata = loader.load_dataset(datasets.MacaqueFaces, 'data', 'dataframes')
transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset = ImageDataset(metadata.df, metadata.root, transform=transform)
len(dataset)

# %%
dataset_database = ImageDataset(metadata.df.iloc[100:,:], metadata.root, transform=transform)
dataset_query = ImageDataset(metadata.df.iloc[:100,:], metadata.root, transform=transform)

# %%
import timm
from wildlife_tools.features import DeepFeatures
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#%%
name = 'hf-hub:BVRA/MegaDescriptor-T-224'
model = timm.create_model(name, num_classes=0, pretrained=True)
model.to(device)
extractor = DeepFeatures(model)

#%%
query = extractor(dataset_query)
# %%
query.features.shape, len(dataset_query)
#%%
database = extractor(dataset_database)
# %%
from wildlife_tools.similarity import CosineSimilarity

similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)
similarity.shape
# %%
import numpy as np
from wildlife_tools.inference import KnnClassifier

classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
predictions = classifier(similarity)
predictions[:2], dataset_query.labels_string[:2], dataset_query.labels[:2]
#%%
accuracy = np.mean(dataset_query.labels_string == predictions)
accuracy
# %%
