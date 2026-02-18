# Keras is a powerful and easy-to-use free open source Python library
# for developing and evaluating deep learning models.
# It wraps the efficient numerical computation libraries Theano and TensorFlow
# and allows you to define and train neural network models in just a few lines of code.
####################################################################
# Install the required packages
# conda install -c anaconda keras
# conda install -c anaconda tensorflow
# conda install -c anaconda scipy
# pip install -U numpy
####################################################################
#The steps in this tutorial are as follows:

# 1. Load Data.
# 2. Define Keras Model.
# 3. Compile Keras Model.
# 4. Fit Keras Model.
# 5. Evaluate Keras Model.
# 6. Make Predictions.
# 7. Save models for the future

############################################################
# 0. Quality Control of the data
############################################################
# Before loading the data, a critical step should be processed: Quality Control.
# The QC process is required to provide routine and consistent checks of the design and plan details.
# The details can be extended in another note or series.
############################################################
# 1. Load Data.
############################################################
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
import dill
t0 = time.time()

# load the dataset

# a model to map rows of input variables (X) to an output variable (y),
# which we often summarize as y = f(X).

# Input Variables (X):
# Gene expression profiles, with rows as cancer samples and columns as genes

X = pd.read_csv('PRAD_TCGA_RNA_seq.txt',delimiter=',')

# Output Variables (y):
# Class variable (0 or 1)
y = pd.read_csv('PRAD_TCGA_Types.txt',delimiter=',')


#############################################################
# 2. Define Keras Model
#############################################################
# Models in Keras are defined as a sequence of layers.
# This is a a Sequential model and add layers one at a time
# until we are happy with our network architecture.
# How do we know the number of layers and their types?
# There are heuristics that we can use and often the best network structure
# is found through a process of trial and error experimentation

# This practice is a fully-connected network structure with three layers.
# The model expects rows of data with 577 variables, ie genes (the input_dim=577 argument)
# The first hidden layer has 1000 nodes and uses the relu activation function.
# The second hidden layer has 200 nodes and uses the relu activation function.
# The third hidden layer has 100 nodes and uses the relu activation function.
# The output layer has one node and uses the sigmoid activation function.

model = Sequential()
model.add(Dense(1000, input_dim=577, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#############################################################
# 3. Compile Keras Model
#############################################################
# Compiling the model uses the efficient numerical libraries under the covers
# (the so-called backend) such as Theano or TensorFlow.
# The backend automatically chooses the best way to represent the network for training and
# making predictions to run on your hardware, such as CPU or GPU or even distributed.
# We must specify
# 1) the loss function to use to evaluate a set of weights,
# 2) the optimizer is used to search through different weights for the network
# 3) and any optional metrics we would like to collect and report during training.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#############################################################
# 4. Fit Keras Model
#############################################################
# We have defined our model and compiled it ready for efficient computation.
# Training occurs over epochs and each epoch is split into batches.
# Epoch: One pass through all of the rows in the training dataset.
# Batch: One or more samples considered by the model within an epoch before weights are updated.
# These configurations can be chosen experimentally by trial and error.
# We want to train the model enough so that it learns a good mapping of rows of input data
# to the output classification. The model will always have some error,
# but the amount of error will level out after some point for a given model configuration.
# This is called model convergence.
model.fit(X, y, epochs=1000, batch_size=200, verbose=0)
#############################################################
#5. Evaluate Keras Model
#############################################################
...
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
#############################################################
# 6. Make Predictions
#############################################################
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases

for i in range(5):
	print('%s => %d (expected %d)' % (X.index[i], predictions[i], y.Types[i]))

#############################################################
# 7. Save models
#############################################################
t1 = time.time()
total = t1-t0
print('The practice takes %s seconds' % total)
dill.dump_session('./PRAD_Classifier_from_TCGA_RNA_seq.pkl')
#to restore session:
#dill.load_session('./PRAD_Classifier_from_TCGA_06212021.pkl')
