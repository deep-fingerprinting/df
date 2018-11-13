# Dataset Structure
- We serialized the dataset to the pickle file.
- The researcher can simply use the cPickle python's library to load the dataset

## Dataset formant
In all datasets, we all use the same data structure as following:

- X_****.pkl file contains the array of network traffic sequences.
  - The dimension of X's dataset is [n x 5000] in which 
    n --> total number of network traffic sequences instances
    5000 --> refers to the fixed length of each network traffic sequence instance.
- y_***.pkl file contains the array of the websites' labels
  - The dimension of y's dataset is [n] in which
    n --> total number of network traffic sequences instances

E.g.   
```
X_train.pkl = [[+1,-1, ..., -1], ... ,[+1, +1, ..., -1]]
y_train.pkl = [45, ... , 12]
In this case:
   - the 1st packet sequence [+1,-1, ..., -1] belongs to website number 45
   - the last packet sequence [+1, +1, ..., -1] belongs to website number 12
```

