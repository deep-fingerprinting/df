Deep Fingerprinting
:warning: experimental - PLEASE BE CAREFUL. Intended for reasearch purposes only.

The source code and dataset are used to demonstrate the DF model, and reproduce the results of the ACM CCS2018 paper:

## ACM Reference Formant
```
Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018.
Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning. 
In 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS ’18), 
October 15–19, 2018, Toronto, ON, Canada. ACM, New York, NY, USA, 16 pages. 
https://doi.org/10.1145/3243734.3243768
```
You can find our paper on [here](https://dl.acm.org/citation.cfm?id=3243768)

# Closed-World Evaluation
## Dataset
We publish the datasets of web traffic traces produced for the closed-world evaluations on non-defended, WTF-PAD and Walkie-Talkie datasets. However, due to the limitation on the size of uploaded files set by Github, we upload our dataset to google drive repository instead. 

The dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1kxjqlaWWJbMW56E3kbXlttqcKJFkeCy6?usp=sharing)

## Dataset Structure
- We serialized the dataset to the pickle file.
- The researcher can simply use the cPickle python's library to load the dataset

## Descriptions of Dataset
- In each sub-folder, it contains 6 different files 
```   
X_<type of data>_<type of evaluation>.pkl : Packet's directions sequence
y_<type of data>_<type of evaluation>.pkl : Corresponding website's classes sequece

<type of data> --Three different data sets used for training, validation, and testing 

<type_of_evaluation> --Three different evaluations: 
                       NoDef: Tor's trafic traces without defense
                       WTFPAD: Tor's trafic traces with WTF-PAD defense
                       WalkieTalkie: WTFPAD: Tor's trafic traces with Walkie-Talkie defense
```

## Dataset Format
In all datasets, we use the same data structure as following:
```
X_<type of data>_<type of evaluation>.pkl --Array of network traffic sequences.
    The dimension of X's dataset is [n x 5000] in which 
    n -- Total number of network traffic sequences instances
    5000 -- Fixed length of each network traffic sequence instance.
y_<type of data>_<type of evaluation>.pkl --Array of the websites' labels
    The dimension of y's dataset is [n] in which
    n --Total number of network traffic sequences instances

E.g.

X_<type of data>_<type of evaluation>.pkl = [[+1,-1, ..., -1], ... ,[+1, +1, ..., -1]]
y_<type of data>_<type of evaluation>.pkl = [45, ... , 12]
In this case:
   - the 1st packet sequence [+1,-1, ..., -1] belongs to website number 45
   - the last packet sequence [+1, +1, ..., -1] belongs to website number 12
```
Before training and evaluating DF model please place the downloaded datasets into ../df/dataset/ClosedWorld/
## Reproduce Results
First of all, you will need to download the corresponding dataset and place it in: 
  ../dataset/ClosedWorld/NoDef/ directory for non-defended   
  ../dataset/ClosedWorld/WTFPAD/ directory for WTF-PAD
  ../dataset/ClosedWorld/WalkieTalkie/ directory for Walkie-Talkie
  
## Attack Accuracy
#### Non-Defended Evaluation
```
python src/ClosedWorld_DF_NoDef.py

Training and evaluating DF model for closed-world scenario on non-defended dataset
Number of Epoch:  30
Loading and preparing data for training, and evaluating the model
Loading non-defended dataset for closed-world scenario
Data dimensions:
X: Training data's shape :  (76000, 5000)
y: Training data's shape :  (76000,)
X: Validation data's shape :  (9500, 5000)
y: Validation data's shape :  (9500,)
X: Testing data's shape :  (9500, 5000)
y: Testing data's shape :  (9500,)
(76000, 'train samples')
(9500, 'validation samples')
(9500, 'test samples')
Building and training DF model
Model compiled
Train on 76000 samples, validate on 9500 samples
Epoch 1/30
 - 78s - loss: 1.9622 - acc: 0.4976 - val_loss: 0.6449 - val_acc: 0.8456
Epoch 2/30
 - 75s - loss: 0.6925 - acc: 0.8249 - val_loss: 0.3391 - val_acc: 0.9153
Epoch 3/30
 - 75s - loss: 0.4304 - acc: 0.8952 - val_loss: 0.2284 - val_acc: 0.9433

...

Epoch 29/30
 - 75s - loss: 0.0416 - acc: 0.9892 - val_loss: 0.0757 - val_acc: 0.9837
Epoch 30/30
 - 75s - loss: 0.0408 - acc: 0.9892 - val_loss: 0.0790 - val_acc: 0.9817
('Testing accuracy:', '0.9827368421052631')
```
#### WTF-PAD Evaluation
```
python src/ClosedWorld_DF_WTFPAD.py

Training and evaluating DF model for closed-world scenario on WTF-PAD dataset
Number of Epoch:  40
Loading and preparing data for training, and evaluating the model
Loading WTF-PAD dataset for closed-world scenario
Data dimensions:
X: Training data's shape :  (76000, 5000)
y: Training data's shape :  (76000,)
X: Validation data's shape :  (9500, 5000)
y: Validation data's shape :  (9500,)
X: Testing data's shape :  (9500, 5000)
y: Testing data's shape :  (9500,)
(76000, 'train samples')
(9500, 'validation samples')
(9500, 'test samples')
Building and training DF model
Model compiled
Train on 76000 samples, validate on 9500 samples
Epoch 1/40
 - 78s - loss: 3.1377 - acc: 0.2031 - val_loss: 2.2293 - val_acc: 0.3932
Epoch 2/40
 - 75s - loss: 1.9602 - acc: 0.4607 - val_loss: 1.3017 - val_acc: 0.6458
Epoch 3/40
 - 75s - loss: 1.4927 - acc: 0.5839 - val_loss: 0.9280 - val_acc: 0.7469

...

Epoch 39/40
 - 75s - loss: 0.2608 - acc: 0.9246 - val_loss: 0.3522 - val_acc: 0.9111
Epoch 40/40
 - 75s - loss: 0.2573 - acc: 0.9250 - val_loss: 0.3709 - val_acc: 0.9069
('Testing accuracy:', '0.906947368471246')
```
#### Walkie-Talkie Evaluation (also include top-2 prediction)
```
python src/ClosedWorld_DF_WalkieTalkie.py

Training and evaluating DF model for closed-world scenario on Walkie-Talkie dataset
Number of Epoch:  30
Loading and preparing data for training, and evaluating the model
Loading Walkie-Talkie dataset for closed-world scenario
Data dimensions:
X: Training data's shape :  (80000, 5000)
y: Training data's shape :  (80000,)
X: Validation data's shape :  (5000, 5000)
y: Validation data's shape :  (5000,)
X: Testing data's shape :  (5000, 5000)
y: Testing data's shape :  (5000,)
(80000, 'train samples')
(5000, 'validation samples')
(5000, 'test samples')
Building and training DF model
Model compiled
Train on 80000 samples, validate on 5000 samples
Epoch 1/30
 - 80s - loss: 2.5954 - acc: 0.2657 - val_loss: 1.7162 - val_acc: 0.3588
Epoch 2/30
 - 77s - loss: 1.4514 - acc: 0.4143 - val_loss: 1.0211 - val_acc: 0.4578
Epoch 3/30
 - 77s - loss: 1.1592 - acc: 0.4457 - val_loss: 0.8824 - val_acc: 0.4780

...

Epoch 29/30
 - 77s - loss: 0.7291 - acc: 0.4932 - val_loss: 0.7300 - val_acc: 0.4964
Epoch 30/30
 - 77s - loss: 0.7286 - acc: 0.4947 - val_loss: 0.7340 - val_acc: 0.4965
('Testing accuracy:', '0.497')
Start evaluating Top-2 Accuracy
Top-2 Accuracy: 0.992000 
```
# Open-World Evaluation
## Dataset
- The dataset format and description are the same as closed-world dataset
- The dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1EDgNIM98PlayyUplKbaviHlDpVhVa1yS?usp=sharing)

## Reproduce Results
First of all, you will need to download the corresponding dataset and place it in: 
  ../dataset/OpenWorld/NoDef/ directory for non-defended   
  ../dataset/OpenWorld/WTFPAD/ directory for WTF-PAD
  ../dataset/OpenWorld/WalkieTalkie/ directory for Walkie-Talkie

## Source Code's Description
The source codes contain two part:
### Training the WF classifier
- The model includes both monitored and unmonitored websites used to train the DF model with respect to each evaluation.
- To train the model
```
python src/OpenWorld_DF_<type of evaluation>_Training.py
```
- The output of this part is the trained DF model that will be used in the next part.
- After finishing training the model, the trained DF model will be automatically saved at 
```
../saved_trained_models/OpenWorld_<type of evaluation>.h5 
```
### Evaluating the performance of the attacks
- The performance of the attack in open-world scenario is evaluated by running  
```
python src/OpenWorld_DF_<type of evaluation>_Evaluation.py
```
- The output of the evaluation will be automatically saved in the 
```
../results/OpenWorld_<type of evaluation>.csv
```
- In each row of the csv file consists of the related performance metrics with respect to different thresholds including
```
True Positive (TP) False Positive (FP) True Negative (TN) False Negative)
True Positive Rate (TPR) False Positive Rate (FPR)
Precision and Recall
```
- The researcher can use this performance metric to plot ROC or Precision&Recall curves.
- Note that our research work mainly focus on the use of precision and recall as the main performance metric.

## Questions and comments
Please, address any questions or comments to the authors of the paper. The main developers of this code are:
* Payap Sirinam (payap.sirinam@mail.rit.edu)
* Mohsen Imani (imani.moh@gmail.com)
* Marc Juarez (marc.juarez@kuleuven.be)


