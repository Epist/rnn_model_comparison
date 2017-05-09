#########################################
#Sequence to sequence model implementation
#########################################

#Need to:
#Implement validation and testing
#Implement training with multiple subjects and a contextual vector/embedding for their subject IDs (the embedding might better facilitate testing with an unknown subject?)

import numpy as np
import pandas as pd

import scipy.io as sio

print("Loading data")
raw_data = sio.loadmat('/home/larry/Data/BCI_Competition/IV/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1b_1000Hz.mat')


subjectA_recordings = np.array(raw_data['cnt'])
subjectA_cues_raw = raw_data['mrk'][0,0]
sample_rate = raw_data['nfo'][0,0][0][0][0]
cue_length_in_secs = 4
cue_steps = cue_length_in_secs * sample_rate

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
            targets[t, cue_index] = 1
            cue_number += 1
        elif cue_steps_remaining > 0:
            #If last queue is still active
            cue_steps_remaining -= 1
            targets[t, cue_index] = 1
        else:
            targets[t, 0] = 1 #Append a value representing no active cue
    else:
            targets[t, 0] = 1 #Append a value representing no active cue

#Stateful model


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = np.shape(subjectA_recordings)[1]
timesteps = 64
num_classes = 3
batch_size = 32
num_epochs = 10
num_lstm_layers = 2


# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
print("Building model")
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
for i in range(num_lstm_layers-1):
    model.add(LSTM(32, return_sequences=True, stateful=True))
#model.add(LSTM(32, stateful=True))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
#x_train = np.random.random((batch_size * 10, timesteps, data_dim))
#y_train = np.random.random((batch_size * 10, num_classes))
truncation_length = (recording_len-(recording_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))

reshaped_inputs = np.reshape(subjectA_recordings[0:truncation_length,:], [-1,timesteps,data_dim])
reshaped_targets = np.reshape(targets[0:truncation_length,:], [-1,timesteps,num_classes])

x_train = reshaped_inputs
y_train = reshaped_targets


#model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs)#, epochs=num_epochs)

#Need to implement prediction/validation


eval_folder_data = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1b_1000Hz.mat"
eval_folder_labels = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/true_labels_1/BCICIV_eval_ds1b_1000Hz_true_y.mat"
raw_eval_data = sio.loadmat(eval_folder_data)
subject_eval_recordings = np.array(raw_eval_data['cnt'])
raw_label_data = sio.loadmat(eval_folder_labels)
un_nan_labels = np.array([x if np.isfinite(x) else 0 for x in raw_label_data["true_y"]]) #This is not the true test, as the NAN vals shouldnot be counted...


val_recording_len = np.shape(subject_eval_recordings)[0]


print("Building val targets")
val_targets = np.zeros([val_recording_len, 3])
cue_steps_remaining = 0
cue_index = 0 #{0:0, -1:1, 1:2}
cue_number = 0
for t in range(val_recording_len):

            #If new queue is reached
        cue_type = un_nan_labels[t]
        if cue_type == -1:
            cue_index = 1
        elif cue_type== 1:
            cue_index = 2
	elif cue_type == 0:
	    cue_index = 0
        else:
                raise(exception("unrecognized cue type"))
        val_targets[t, cue_index] = 1



val_truncation_length = (val_recording_len-(val_recording_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))


reshaped_inputs_val = np.reshape(subject_eval_recordings[0:val_truncation_length,:], [-1,timesteps,data_dim])
reshaped_targets_val = np.reshape(val_targets[0:val_truncation_length,:], [-1,timesteps,num_classes])

x_val = reshaped_inputs_val
y_val = reshaped_targets_val


print("Training model")

model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs, validation_data=(x_val, y_val))#, epochs=num_epochs)
