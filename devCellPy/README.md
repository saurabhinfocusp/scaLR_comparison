
Please follow enviornment setup guide from their [github](https://github.com/devCellPy-Team/devCellPy)

We had made a change in their config file to use standard params for training and skip finetuning as it was taking too much time.
- file path: `/path/to/conda/envs/devcellpy_env/lib/python3.11/site-packages/devCellPy/config.py`
- change: `std_params = True # edited to True, default is False`

Also the run was giving error on line 20 of `/path/to/conda/envs/devcellpy_env/lib/python3.11/site-packages/devCellPy/helpers.py`  - `norm_express = pd.DataFrame(adata.X.toarray(), columns = adata.var.index, index = adata.obs.index)`.
- So we need to change it to `norm_express = pd.DataFrame(adata.X, columns = adata.var.index, index = adata.obs.index)` i.e. removed `toarray()`.`


Example command to run the tool
- training : 
    - devCellPy --runMode trainAll --trainNormExpr '/path/to/train.h5ad' --labelInfo /path/to/dcp_labels_cell_type.csv --trainMetadata /path/to/dcp_pbmc_metadata_cell_type.csv --rejectionCutoff 0.5


- predicting -
    - devCellPy --runMode predictOne --predNormExpr "/path/to/test.h5ad" --layerObjectPaths "/path/to/Root_object.pkl" --rejectionCutoff 0.5


Examples of labelInfo and trainMetadata are stored here in same directory, but you need to follow instructions from their github/documentation for making those.