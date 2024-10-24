# %%
import celltypist
from celltypist import models
import pandas as pd
import anndata
import scanpy as sc
import time
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

target = 'cell_type'
dataset_name = 'pbmc_120k_3500_sgd'
train_data_path = f'/home/biocusp/data/{dataset_name}/{target}/train.h5ad'
test_data_path = f'/home/biocusp/data/{dataset_name}/{target}/test.h5ad'
store_results_path = f'/home/anand/single_cell_classification/{dataset_name}/{target}'

if not os.path.exists(store_results_path):
    os.makedirs(store_results_path, exist_ok=True)

# Data Loading
a_data_train = sc.read_h5ad(train_data_path)
a_data_test = sc.read_h5ad(test_data_path)
print('Train and test data loaded...')
print('Train data shape : ', a_data_train.shape)
print('Test data shape : ', a_data_test.shape)


if 'Cell_Type' in a_data_train.obs.columns:
    a_data_train.obs.rename(columns={'Cell_Type': 'cell_type'}, inplace=True)
    a_data_test.obs.rename(columns={'Cell_Type': 'cell_type'}, inplace=True)

if 'Cell_State' in a_data_train.obs.columns:
    a_data_train.obs.rename(columns={'Cell_State': 'cell_state'}, inplace=True)
    a_data_test.obs.rename(columns={'Cell_State': 'cell_state'}, inplace=True)

# %%
"""
# SGD Regression & Feature selection based approach - two round pass
"""

# %%
# The `cell_Type` in `adata_2000.obs` will be used as cell Type labels for training.
print('Running CellTypist SGD + Feature Selection & again SGD Regression model...')
start_time = time.time()
model = celltypist.train(a_data_train, labels = target, n_jobs = 1, use_SGD=True, feature_selection = True, check_expression=False)
model_running_time = time.time() - start_time
print(f'Model training time : {model_running_time/(60)} mins')

# %%
# Save the model.
model.write(f'{store_results_path}/sgd_TRP_model.pkl')

# %%
print('Starting predictions on test data...')
start_time = time.time()
predictions = celltypist.annotate(a_data_test, model = model, majority_voting = True)
predictions_end_time = time.time() - start_time
print(f'Model Inference time : {predictions_end_time/(60)} mins')

# %%
predictions

# %%
a_data_pred = predictions.to_adata()

# %%
a_data_pred

# %%
sc.tl.umap(a_data_pred)

# %%
sc.pl.umap(a_data_pred, color = [target, 'predicted_labels', 'majority_voting'], legend_loc = 'on data')

# %%
predictions.predicted_labels

# %%
a_data_test.obs[target].values

# %%
test_targets = a_data_test.obs[target].values
test_preds = predictions.predicted_labels['predicted_labels'].values

# %%
confusion_matrix(test_targets, test_preds)

# %%
print(classification_report(test_targets, test_preds))

# %%
print(pd.DataFrame(classification_report(test_targets, test_preds, output_dict=True)).transpose())

# %%

if not os.path.exists(f'{store_results_path}/results'):
    os.makedirs(f'{store_results_path}/results', exist_ok=True)

pd.DataFrame(classification_report(test_targets, test_preds, output_dict=True)).transpose().to_csv(f'{store_results_path}/results/sgd_TRP_{target}_top_3500_feat_stats.csv')

