import scBalance as sb
import scanpy as sc
import scBalance.scbalance_IO as ss

import numpy as np
import pandas as pd
import random
import tqdm
from warnings import simplefilter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



target = 'cell_type'
train_data_path = f'path/to/train.h5ad'
test_data_path = f'path/to/test.h5ad'
store_results_path = f'path/to/results_directory'
scale = 'False'


# Scanpy function modified.
def Scanpy_Obj_IO(test_obj=None, ref_obj=None, label_obj=None, scale = False):
    '''

    :param test_obj: Input Anndata object for cell type annotation
    :param ref_obj: Input reference cell atlas, please note that an csv file required
    :param label_obj: (IF APPLICABLE) If the reference dataset is splited as expression matrix.csv and label.csv, input the label path
    :return: test_matrix (pd.dataframe) for annotation,
    ref_matrix (pd.dataframe) contain reference expression matrix,
    label (pd.dataframe) contain the label list.
    :scale: Whether to scale the reference dataset
    '''

    ref_adata = ref_obj
    # normallize and log reference dataset
    if label_obj is None:
        label = ref_adata[:, ref_adata.n_vars - 1].to_df()
        ref_adata = ref_adata[:, 0:ref_adata.n_vars - 1]

    else:
        label = label_obj

    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)
    simplefilter(action='ignore', category=FutureWarning)
    # gene = ref_adata.var_names & test_obj.var_names
    gene = list(set(ref_adata.var_names).intersection(test_obj.var_names))

    if 'log1p' in test_obj.uns:
        # test whether the dataset is in log format
        ref = ref_adata[:,gene]
        test_matrix = test_obj.to_df()[gene]
    else:
        sc.pp.log1p(test_obj)
        ref = ref_adata[:,gene]
        test_matrix = test_obj.to_df()[gene]

    if scale:
        sc.pp.scale(ref, max_value=10)
        ref_matrix = ref.to_df()
    else:
        ref_matrix = ref.to_df()

    return test_matrix, ref_matrix, label



# ScBalance function modified
import torch
import os
import pickle
import numpy as np
import torch.nn as nn
import time
import torch.utils.data as Data
import torch.nn.functional as F
from scBalance.scBalance_classifier import classifier
from scBalance.weightsampling import Weighted_Sampling

def scBalance(test = None, 
            reference = None, 
            label = None, 
            processing_unit = 'cpu',
            weighted_sampling = True, 
            save_model = None,
            save_path =  None,
            load_model = None,
            load_dict = None):

    '''

    :param test: Input your test dataset to annotate
    :param reference: Input your reference dataset
    :param label: Input label set
    :param processing_unit: Choose which processing unit you would like to use,
    if you choose 'gpu', than the software will try to access CUDA,
    if you choose 'cpu', the software will use CPU as default.
    :param save_model: Save trained network structure for reuse
    :param load_model: load model for reuse
    :param weighted_sampling: whether to use the internal sampling method. Default is "True"
    :return:
    '''
    if load_model:

        X_test = test.values
        X_test = torch.from_numpy(X_test)

        if load_dict:
            dict_read = open(load_dict, 'rb')
            dict2 = pickle.load(dict_read)
            status_dict = dict2

        print("Loading model")
        model = torch.load(load_model)
        model.eval()

        with torch.no_grad():
            #X_test = X_test.to(device)
            output = model(X_test)

        temp = F.softmax(output, dim=1)
        pred = torch.argmax(temp, dim=1).cpu()

        result = list(pred.numpy())
        results = []
        for i in result:
            results.append(status_dict[i])
        print("Prediction finished")
    else:
        # label.columns = ['Label']
        # label.Label.value_counts()
        label = pd.DataFrame(label.values, columns=['Label'])
        status_dict = label['Label'].unique().tolist()
        int_label = label['Label'].apply(lambda x: status_dict.index(x))
        label.insert(1,'transformed',int_label)
        label['transformed'] = label['transformed'].astype('int')

        X_train = reference.values
        X_test = test.values
        y_train = label['transformed'].values

        dtype = torch.float
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)

        if weighted_sampling == True:
            sampler = Weighted_Sampling(y_train)

            #construct pytorch object
            train_data = Data.TensorDataset(X_train, y_train)
            train_loader = Data.DataLoader(dataset=train_data,
                                    batch_size=128,
                                    sampler = sampler,
                                    num_workers=1)
        else:
            train_data = Data.TensorDataset(X_train, y_train)
            train_loader = Data.DataLoader(dataset=train_data,
                                                batch_size=128)

        input_size = X_train.shape[1]
        num_class = len(status_dict)
        num_epochs = 20

        model = classifier(input_size, num_class)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        total_step = len(train_loader)

        print("--------Start annotating----------")
        start = time.perf_counter()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if processing_unit == 'cpu':
            device = torch.device('cpu')
        elif processing_unit == 'gpu':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                print("No GPUs are available on your server.")

        print("Computational unit be used is:", device)

        model.to(device)

        model.train()
        for epoch in range(num_epochs):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                train_loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            a = "=" * (epoch + 1)
            b = "." * (num_epochs - epoch - 1)
            c = ((epoch + 1) / num_epochs) * 100
            dur = time.perf_counter() - start
            print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end='')
            time.sleep(0.1)

        print("\n--------Annotation Finished----------")

        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            output = model(X_test)

        temp = F.softmax(output, dim=1)
        pred = torch.argmax(temp, dim=1).cpu()

        result = list(pred.numpy())
        results = []
        for i in result:
            results.append(status_dict[i])

        if save_model == True:
            if save_path:
                torch.save(model, save_path+"/scbalance.pkl")
                dict_path = save_path+'/dict_file.pkl'
                dict_save = open(dict_path, 'wb')
                pickle.dump(status_dict, dict_save)
                dict_save.close()

            else:
                save_path = os.path.abspath(os.path.dirname(os.getcwd()))+'/scbalance.pkl'
                torch.save(model, save_path)
                dict_path = os.path.abspath(os.path.dirname(os.getcwd()))+"/dict_file.pkl"
                dict_save = open(dict_path, 'wb')
                pickle.dump(status_dict, dict_save)
                dict_save.close()

            print("Model is saved at:"+save_path+"\n")
            print("Dict file is saved at:"+dict_path)

    return results



if not os.path.exists(f'{store_results_path}/scbalance_scale_{scale}_results'):
    os.makedirs(f'{store_results_path}/scbalance_scale_{scale}_results', exist_ok=True)

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


# Train and test labels
train_label = a_data_train.obs[target]
test_label = a_data_test.obs[target]

# Running scBalance
print('Running scBalance annotation tool...')
test, reference, ref_label = Scanpy_Obj_IO(test_obj=a_data_test.copy(), ref_obj=a_data_train.copy(), label_obj=train_label.copy(), scale = scale)
pred_label = scBalance(test, reference, ref_label, save_model=True, save_path=f'{store_results_path}/scbalance_scale_{scale}_results',
                      processing_unit='gpu')

print('Test accuracy score :', accuracy_score(test_label, pred_label))

print('\nConfusion matrix : \n', confusion_matrix(test_label, pred_label))

print('\nClassification report : \n', print(classification_report(test_label, pred_label)))

if not os.path.exists(f'{store_results_path}/scbalance_scale_{scale}_results/results'):
    os.makedirs(f'{store_results_path}/scbalance_scale_{scale}_results/results', exist_ok=True)

pd.DataFrame(classification_report(test_label, pred_label, output_dict=True)).transpose().to_csv(f'{store_results_path}/scbalance_scale_{scale}_results/results/classification_report.csv')
