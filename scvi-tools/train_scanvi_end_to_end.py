print(f"{'='*10}Importing Libraries{'='*10}")
import numpy as np
import pandas as pd 
from tqdm import tqdm
import scanpy as sc
import anndata as ad
import scvi
import os
from sklearn.metrics import classification_report

# Paths of train and test files
TRAIN_PATH = 'hed train data.h5ad'
TEST_PATH = 'hed test data.h5ad'

# Name you want to save scVI with
SCVI_MODEL_NAME = 'scvi_hed_end_to_end'
# Name you want to save scANVI with
SCANVI_MODEL_NAME = 'scanvi_hed_end_to_end'
# Observation to classify (cell_type, cell_state etc etc)
TARGET_CLASS = 'cell_type'

# Whether a saved scVI model already exists or not
scvi_trained = False

# Number of epochs to train for
SCVI_EPOCHS = 100
SCANVI_EPOCHS = 25


print(f"{'='*10}Reading data{'='*10}")
adata = ad.read_h5ad(TRAIN_PATH)


print(f"{'='*10}Filtering data{'='*10}")
sc.pp.filter_cells(adata, min_genes=1)


print(f"{'='*10}Setting up the data{'='*10}")
unique_cell_types = list(adata.obs[TARGET_CLASS].unique())
cell_type_masks = []

for cell_type in unique_cell_types:
    bool_arr = adata.obs[TARGET_CLASS] == cell_type
    cell_type_masks.append(bool_arr)

cell_type_masks = np.array(cell_type_masks)

seed_labels = np.array(adata.shape[0] * ["Unknown"])

for i in range(len(unique_cell_types)):
    seed_labels[cell_type_masks[i]] = unique_cell_types[i]

adata.obs["seed_labels"] = seed_labels


if not scvi_trained: 
    # In case scVI is not yet trained
    print(f"{'='*10}Training scVI{'='*10}")
    scvi.model.SCVI.setup_anndata(adata, batch_key=None, labels_key="seed_labels")
    scvi_model = scvi.model.SCVI(adata) #, n_latent=N_LATENT) #, n_layers=N_LAYERS)
    scvi_model.train(SCVI_EPOCHS)

    print(f"{'='*10}Saving scVI model{'='*10}")
    scvi_model.save(SCVI_MODEL_NAME, overwrite=True)

else:
    # In case scVI model is saved but scANVI is yet to be trained
    print(f"{'='*10}Loading scVI{'='*10}")
    scvi_model = scvi.model.SCVI.load(SCVI_MODEL_NAME, adata=adata)


print(f"{'='*10}Training scANVI{'='*10}")
scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, "Unknown")
scanvi_model.train(SCANVI_EPOCHS)


print(f"{'='*10}Saving scANVI model{'='*10}")
scanvi_model.save(SCANVI_MODEL_NAME, overwrite=True)


print(f"{'='*10}Training Completed{'='*10}")


print(f"{'='*10}Reading test data{'='*10}")
test_adata = ad.read_h5ad(TEST_PATH)


print(f"{'='*10}Getting Latent Representation of Data{'='*10}")
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "C_scANVI"
adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
adata.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(adata)
sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata)
print(f"{'='*10}Saving Latent Representation of Data{'='*10}")
sc.pl.umap(adata, color=[TARGET_CLASS, SCANVI_PREDICTIONS_KEY], save=f'{SCANVI_MODEL_NAME}.png')


print(f"{'='*10}Setting up the test data{'='*10}")
test_adata.layers['counts'] = test_adata.X.copy()
unique_cell_types = test_adata.obs[TARGET_CLASS].unique()
test_cell_type_masks = []

for cell_type in unique_cell_types:
    bool_arr = test_adata.obs[TARGET_CLASS] == cell_type
    test_cell_type_masks.append(bool_arr)

test_cell_type_masks = np.array(test_cell_type_masks)

test_seed_labels = np.array(test_adata.shape[0] * ["Unknown"])

for i in range(len(unique_cell_types)):
    test_seed_labels[test_cell_type_masks[i]] = unique_cell_types[i]

test_adata.obs["seed_labels"] = test_seed_labels


print(f"{'='*10}Getting Predictions{'='*10}")
predictions = scanvi_model.predict(test_adata)
print(classification_report(test_adata.obs['seed_labels'], predictions))


results = classification_report(test_adata.obs['seed_labels'], predictions, output_dict=True)
df = pd.DataFrame(results).transpose()
df.to_csv(f'results/{SCANVI_MODEL_NAME}.csv')
print(f"{'='*10}THE END{'='*10}")
