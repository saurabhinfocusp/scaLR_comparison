This script contains a Python script that leverages the scANVI model to annotate and analyze single-cell RNA-seq data using a semi-supervised variational autoencoder. > This is a script for running `scvi-tools` followed by scANVI for cell type and state classification.


**scVI Homepage:** [Link](https://scvi-tools.org/)


### Pre-requisites
**Note:** To run scVI on GPU, install [jaxlib](https://jax.readthedocs.io/en/latest/installation.html) and [PyTorch](https://pytorch.org/get-started/locally/) for GPU before installing scVI. For CPU, directly install the below mentioned libraries.


Install required libraries:
```bash
pip install scvi-tools scanpy
```

### Variables
* Paths of train and test files (.h5ad format)
* Names of scVI and scANVI models
* Target name you wish to classify (cell_type, cell_state etc.)
* Whether a trained scVI model exists or not
* Number of epochs to train scVI and scANVI for


### Running the script
With time stats

```bash
nohup /usr/bin/time --verbose python train_scanvi_end_to_end.py > nohup_logs.out 2>&1 &
```
