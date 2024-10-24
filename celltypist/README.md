There are 3 variants in celltypist tool.
1. Logistic classifier - script -> logistic_classifier.py
2. SGD classifier - script -> sgd_classifier.py
3. Feature selection followed by SGD classifier - script -> sgd_classifier_FS_sgd_again.py


Create a virtual enviornment following the instructions on celltypist tool [github](https://github.com/Teichlab/celltypist)


You will need to edit below variables in the scripts while running, example below...

1. target = 'cell_type' # define this as per your target name.
2. train_data_path = f'path/to/train.h5ad'
3. test_data_path = f'path/to/test.h5ad'
4. store_results_path = f'path/to/results_directory'


Run the file using below command to get nohup output as well with the memory and time profiling details...

- nohup /usr/bin/time --verbose python -u sgd_classifier.py > sgd_cs_nohup.txt 2>&1 &