# FSD-CLCD
FSD-CLCD: Functional Semantic Distillation Graph Learning for Cross-Language Code Clone Detection

The _saved_model_ folder is used to save the trained model. Currently, there is a trained model with a pruning ratio of 0.4.

The _saved_var_ folder is used to store preprocessed dataset variables, thereby reducing the time spent repeatedly processing the dataset. Due to the size limit of GitHub, we have only stored test data here.

The log folder is used to save the logs generated during the runtime process.

The code for data preprocessing is currently not publicly available.

To get the result in paper, run main.py


environment
pytorch 2.1.0
pyg 2.5.0
