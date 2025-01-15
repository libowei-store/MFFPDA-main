# Requirement

* python == 3.11
* pytorch == 2.4.1+cu118
* Numpy == 1.23.5
* scikit-learn == 1.3.2
* scipy == 1.14.1
* pandas == 1.5.3
* matplotlib == 3.10.0
* tqdm == 4.67.1
* fitlog == 0.9.15

# Files

## 1. data

It contains a probiotic-disease association dataset, four calculated disease similarities, and a probiotic similarity matrix.

## 2. source
It includes several methods for extracting probiotics and disease features, such as jaccard similarity, cosine similarity, etc.

## 3. network.py

 This function contains the network framework of our entire model and is based on pytorch.

## 4. utils.py

This function contains the necessary processing subroutines.

## 5. main.py

The main function of the model includes some hyperparameter settings of the model, data reading methods, etc. If you want to modify the hyperparameters, you can modify them directly in this file.

# Train and test 


mode: Set the mode to train or test, then you can train the model or test the model

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training.

data_bath: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

weight_path: Define the path to save the model.

All files of Data and Code should be stored in the same folder to run the model.


run
```
main.py
```



