There are 2 scripts made for scBalance run
- single run -> scbalance_single_run.py
- bootstrap run -> scbalance_bootstrap_run.py - this was made as scBalance results are not deterministic - so averaging the results for 10 runs now. You can modify the #runs in scripts.

Please follow enviornment setup from their [github](https://github.com/yuqcheng/scBalance).

You will need to edit below variables in the scripts while running, example below...

1. target = 'cell_type' # define this as per your target name.
2. train_data_path = f'path/to/train.h5ad'
3. test_data_path = f'path/to/test.h5ad'
4. store_results_path = f'path/to/results_directory'
5. scale = True|False -# whether to scale the data or not.


Example command to run the scripts
- nohup /usr/bin/time --verbose python scbalance_bootstrap_run.py > scBalance_cs_False_bootstrap_nohup.txt 2>&1 &