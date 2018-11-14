from keras import backend as K
from utility import LoadDataWalkieTalkieOW_Training
from keras.utils import np_utils
from keras.optimizers import Adamax
from Model_WalkieTalkie import DFNet
import numpy as np
import time
import random
import os

random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EXP_Type = 'OpenWorld_WalkieTalkie'
print "Experimental Type: ", EXP_Type
# network and training
NB_EPOCH = 30
print "Number of Epoch: ", NB_EPOCH
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

NB_CLASSES = 101 # number of outputs: 100 Monitored websites + 1 Unmonitored websites
INPUT_SHAPE = (LENGTH,1)

X_train, y_train, X_valid, y_valid = LoadDataWalkieTalkieOW_Training()
K.set_image_dim_ordering("tf") # tf is tensorflow
# consider them as float and normalize
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')

# we need a [Length x 1] x n shape as input to the DFNet (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]

print(X_train.shape[0], 'training samples')
print(X_valid.shape[0], 'validation samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)


print ("Preparing Data for training")
# initialize the optimizer and model
print time.sleep(2)
model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print ("Model compiled")

# Start training
history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
		validation_data=(X_valid, y_valid))

# Save model
print "Saving Model"
savedpath ='../saved_trained_models/%s.h5'%str(EXP_Type)
model.save(savedpath)
print "Saving Model Done!", savedpath

