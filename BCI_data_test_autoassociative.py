#########################################
#Autoassociative model implementation
#########################################

#Need to:
#Implement validation and testing
#Implement training with multiple subjects and a contextual vector/embedding for their subject IDs (the embedding might better facilitate testing with an unknown subject?)

#Implement an additional error metric for just the targets

#Don't use categorical cross entropy...

#Potentially scale the targets...

#Might want hybrid loss functions and metrics and therefore might want to implement this in raw tensorflow, or at least dig deeper into Keras...

#Add an additional sequence to the input that says whether or not the timepoint should be considered in the accuracy metric (is it less that 1 sec after the onset of the stimulus?)
#Write a custom loss function that removes this auxilliary sequence and then performs the desired loss function
#Write a custom metric that gives the accuracy metric only for the trials that should be considered according to this auxilliary sequence

import numpy as np
import pandas as pd
import scipy.io as sio
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
import tensorflow as tf

def target_acc(y_true, y_pred, targetCols = (59, 62)):
    tars = y_true[:, :, targetCols[0]:targetCols[1]]
    preds = y_pred[:, :, targetCols[0]:targetCols[1]]

    return metrics.categorical_accuracy(tars, preds)
    #print(y_true.get_shape().as_list())
    #return tf.shape(y_true)[2]

print("Loading data")
raw_data = sio.loadmat('/home/larry/Data/BCI_Competition/IV/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1a_1000Hz.mat')


subjectA_recordings = np.array(raw_data['cnt'])
subjectA_cues_raw = raw_data['mrk'][0,0]
sample_rate = raw_data['nfo'][0,0][0][0][0]
cue_length_in_secs = 4
cue_steps = cue_length_in_secs * sample_rate
target_value = 1000 #For changing the importance of the targets vs. the sequence in prediction

#np.shape(subjectA_cues)
#print(raw_data.keys())
cues = raw_data["mrk"]
subjectA_cue_times = subjectA_cues_raw[0][0]
subjectA_cue_values = subjectA_cues_raw[1][0]
numCues = len(subjectA_cue_times)

print("Building targets")
recording_len = np.shape(subjectA_recordings)[0]
targets = np.zeros([recording_len, 3])
cue_steps_remaining = 0
cue_index = 0 #{0:0, -1:1, 1:2}
cue_number = 0
for t in range(recording_len):
    if cue_number < numCues:
        if subjectA_cue_times[cue_number] == t:
            #If new queue is reached
            cue_type = subjectA_cue_values[cue_number]
            if cue_type == -1:
                cue_index = 1
            elif cue_type== 1:
                cue_index = 2
            else:
                raise(exception("unrecognized cue type"))
            cue_steps_remaining = cue_steps-1
            targets[t, cue_index] = target_value
            cue_number += 1
        elif cue_steps_remaining > 0:
            #If last queue is still active
            cue_steps_remaining -= 1
            targets[t, cue_index] = target_value
        else:
            targets[t, 0] = target_value #Append a value representing no active cue
    else:
            targets[t, 0] = target_value #Append a value representing no active cue


#Autoassociative model

#Stateful model

data_dim = np.shape(subjectA_recordings)[1]
timesteps = 64
num_classes = 3
batch_size = 32
num_epochs = 5
num_lstm_layers = 2

full_data_dim = data_dim + num_classes


# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, full_data_dim)))
for i in range(num_lstm_layers-1):
    model.add(LSTM(32, return_sequences=True, stateful=True))
#model.add(LSTM(32, stateful=True))
model.add(Dense(full_data_dim, activation='softmax'))

model.compile(loss='mean_squared_error', #categorical_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy', target_acc])


all_data_vars = np.concatenate((subjectA_recordings, targets), axis=1)

shifted_data_vars = np.concatenate((np.zeros([1,np.shape(all_data_vars)[1]]),
                                    all_data_vars[:(recording_len-1), :]) , axis=0)


truncation_length = (recording_len-(recording_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))

reshaped_inputs = np.reshape(all_data_vars[0:truncation_length,:], [-1,timesteps, full_data_dim])
reshaped_targets = np.reshape(shifted_data_vars[0:truncation_length,:], [-1,timesteps, full_data_dim])

x_train = reshaped_inputs
y_train = reshaped_targets


print("Training model")
model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs)#, epochs=num_epochs)

#Need to implement prediction/validation
