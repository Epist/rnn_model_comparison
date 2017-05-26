######################################################
#Omnidirectional learning with concatenated models
######################################################

#To be used as a baseline for comparison with the shared omnidirectional model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.io as sio
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
import tensorflow as tf

import copy

params = {}
params["batch_size"] = 32
params["timesteps"] = 64
params["num_lstm_layers"] = 2
params["num_channels"] = 64
params["val_split"] = .8 #The percentage to asigne to the training set
params["num_epochs"] = 5
params["val_gap"] = .5 #In seconds
params["cue_length_in_secs"] = 4


class model_part(object):
	def __init__(self, inputs, outputs, params):
		self.inputs = inputs
		self.dim = params["num_channels"]
		self.input_mask = self.create_mask(inputs, self.dim)

		#if outputs == None: #If the outputs should be defined based on the inputs
		#	self.outputs = [1 if x==0 else 0 for x in inputs]#not the input vector
		#else:
		#	self.outputs = outputs
		self.outputs = outputs
		self.output_mask = self.create_mask(outputs, self.dim)
		self.model = self.build_model(params) #Build model based on this I/O configuration

	def create_mask(self, vals, dim):
		mask = np.zeros([dim])
		for i, val in enumerate(vals):
			mask[val] = 1
		return mask

	def build_model(self, params, weights = None):
		#Build a Keras or Tensorflow model corresponding to the paramaters
		batch_size_ = params["batch_size"]
		timesteps_ = params["timesteps"]
		num_lstm_layers = params["num_lstm_layers"]
		full_data_dim = params["num_channels"]
	    model = Sequential()
	    model.add(LSTM(32, return_sequences=True, stateful=True,
	                   batch_input_shape=(batch_size_, timesteps_, full_data_dim)))
	    for i in range(num_lstm_layers-1):
	        model.add(LSTM(32, return_sequences=True, stateful=True))
	    #model.add(LSTM(32, stateful=True))
	    model.add(Dense(full_data_dim, activation='linear'))

	    model.compile(loss='mean_squared_error', #categorical_crossentropy
	                  optimizer='rmsprop') #,
	                  #metrics=[target_MSE])
	    if weights==None:
	        return model
	    else:
	        #Change the weights of the model to the weights specified in the weights variable
	        model.set_weights(weights)#Transfer the weights from the old model to the new model
	        return model

num_channels = params["num_channels"]

#Iterate through all models where one output is predicted from the rest of the data variables
all_models = []
for i in range(num_channels):
	#Construct a model predicting that channel from the others
	inputs = [x for x in range(num_channels) if x != i]
	outputs = [i]
	all_models.append(model_part(inputs, outputs, params))


def train_model(model_part, data):
	#Train and validate a model on the data using the i/o configuration specified by the model_part object
	pass

for i, model_part in enumerate(all_models):
	train_model(model_part, data)



#Test model

#To be implemented later. Not important right now...