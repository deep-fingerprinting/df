# Deep Fingerprinting
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

## Dataset
We published the datasets of web traffic traces produced for the closed-world evaluations on non-defended and WTF-PAD datasets. However, due to the limitation on the size of uploaded files on Github, we uploaded our dataset to google drive repository instead. 

They can be downloaded in links below.

The datasets for the open world are also available upon request to the authors.

### Dataset used for non-defended dataset in the closed-world scenario
https://drive.google.com/open?id=1sFjUqo3r4E0KZsTWF0f-3zfLM5FfRot9

The dataset including 6 files:
- `X_train_NoDef.pkl` : Packet's directions sequence (used as the training data)
- `y_train_NoDef.pkl` : Corresponding website's classes sequece (used as the training data)
- `X_valid_NoDef.pkl` : Packet's directions sequence (used as the validation data)
- `y_valid_NoDef.pkl` : Corresponding website's classes sequece (used as the validation data)
- `X_test_NoDef.pkl` : Packet's directions sequence (used as the testing data)
- `y_test_NoDef.pkl` : Corresponding website's classes sequece (used as the testing data)

Before training and evaluating DF model please place these files into `../df/dataset/NoDef/`

### Dataset used for defended dataset(WTF-PAD) in the closed-world scenario
https://drive.google.com/open?id=187JjQ-Dz4g4zMBkOE4yo_rZcDkkT5zth

The dataset including 6 files: 
- `X_train_WTFPAD.pkl` : Packet's directions sequence (used as the training data)
- `y_train_WTFPAD.pkl` : Corresponding website's classes sequece (used as the training data)
- `X_valid_WTFPAD.pkl` : Packet's directions sequence (used as the validation data)
- `y_valid_WTFPAD.pkl` : Corresponding website's classes sequece (used as the validation data)
- `X_test_WTFPAD.pkl` : Packet's directions sequence (used as the testing data)
- `y_test_WTFPAD.pkl` : Corresponding website's classes sequece (used as the testing data)

Before training and evaluating DF model please place these files into `../df/dataset/WTFPAD/`

## Reproduce results
First of all, you will need to download the corresponding datase and place it in the 
1. `../dataset/nodef/` directory for non-defended closed world 
2. `../dataset/WTFPAD/` directory for WTF-PAD closed world 
  
### Attack accuracy
#### For non-defended closed world
```
python src/DF_NoDef.py

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

#### For WTF-PAD closed world
```
python src/DF_WTFPAD.py

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

## Questions and comments
Please, address any questions or comments to the authors of the paper. The main developers of this code are:
* Payap Sirinam (payap.sirinam@mail.rit.edu)
* Mohsen Imani (imani.moh@gmail.com)
* Marc Juarez (marc.juarez@kuleuven.be)


