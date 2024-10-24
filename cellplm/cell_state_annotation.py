import warnings
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import anndata as ad
import torch
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig


DATASET = 'MS' # 'hPancreas'
PRETRAIN_VERSION = '20230926_85M'
DEVICE = 'cuda:3'

set_seed(42)
# Top 3500 features
# data_train = ad.read_h5ad("/home/anand/scFCSB_tool/apr18_fs_fc_1/Cell_State/train_top_3500_features.h5ad")
# data_test = ad.read_h5ad("/home/anand/scFCSB_tool/apr18_fs_fc_1/Cell_State/test_top_3500_features.h5ad")

# Top 3500 features from NN
data_train = ad.read_h5ad("/home/biocusp/data/pbmc_120k_3500_sgd/cell_state/train.h5ad")
data_val = ad.read_h5ad("/home/biocusp/data/pbmc_120k_3500_sgd/cell_state/val.h5ad")
data_test = ad.read_h5ad("/home/biocusp/data/pbmc_120k_3500_sgd/cell_state/test.h5ad")

# print(data_train.shape, data_test.shape)

# Top 3500 features
# train_num = data_train.shape[0]
# data = ad.concat([data_train, data_test])
# data.X = csr_matrix(data.X)

# print(data.obs)

# data.obs['split'] = 'test'
# tr = np.random.permutation(train_num) #torch.randperm(train_num).numpy()
# data.obs['split'][tr[:int(train_num*0.85)]] = 'train'
# data.obs['split'][tr[int(train_num*0.85):train_num]] = 'valid'


# Top 3500 features NN
data_train.obs['split'] = 'train'
data_val.obs['split'] = 'valid'
data_test.obs['split'] = 'test'
data = ad.concat([data_train, data_val, data_test])
data.X = csr_matrix(data.X)

data.obs['cellstate'] = data.obs['Cell_State']

print(data.obs['split'].value_counts())


pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()

model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = data.obs['cellstate'].nunique()

pipeline_config.update({"epochs": 200})
pipeline_config.update({"max_eval_batch_size": 200000})

print(pipeline_config, model_config)

pipeline = CellTypeAnnotationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='../ckpt')

pipeline.fit(data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            label_fields = ['cellstate'],
            # device='cuda'
            ) # Specify a column in .obs that contains cell type labels


# torch.save(pipeline.model.state_dict(), "cellplm_celltype_200.pt")

pipeline.predict(
                data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
            )


print(pipeline.score(data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', # Specify a column in .obs to specify train and valid split, optional
                target_split = 'test', # Specify a target split to predict, optional
                label_fields = ['cellstate'])  # Specify a column in .obs that contains cell type labels
    )