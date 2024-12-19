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


## 3. network.py

 This function contains the network framework of our entire model and is based on pytorch 1.10.

## 4. utils.py

This function contains the necessary processing subroutines.

# Train and test 

python main.py --mode mode --epochs number  --batch_size number  --rawpath path --weight_path path

mode: Set the mode to train or test, then you can train the model or test the model

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training.

data_bath: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

weight_path: Define the path to save the model.

All files of Data and Code should be stored in the same folder to run the model.



If you want to run the model
run
```
main.py
```



