#########################################
#Sequence to sequence model implementation
#########################################

#Need to:
#Implement validation and testing
#Implement training with multiple subjects and a contextual vector/embedding for their subject IDs (the embedding might better facilitate testing with an unknown subject?)

#Add early stopping

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

import scipy.io as sio
from sklearn.metrics import mean_squared_error


val_split = .8 #The percentage to asigne to the training set
val_gap = .5 #In seconds
timesteps = 64
num_classes = 1
batch_size = 32
num_epochs = 1
num_lstm_layers = 2

cue_length_in_secs = 4


print("Loading data")
raw_data = sio.loadmat('/home/larry/Data/BCI_Competition/IV/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1b_1000Hz.mat')


subject_recordings = np.array(raw_data['cnt'])
subject_cues_raw = raw_data['mrk'][0,0]
sample_rate = raw_data['nfo'][0,0][0][0][0]
cue_steps = cue_length_in_secs * sample_rate
val_gap_samples = int(val_gap * sample_rate)

#np.shape(subject_cues)
#print(raw_data.keys())
cues = raw_data["mrk"]
subject_cue_times = subject_cues_raw[0][0]
subject_cue_values = subject_cues_raw[1][0]
numCues = len(subject_cue_times)


print("Building targets")
recording_len = np.shape(subject_recordings)[0]
targets = np.zeros([recording_len, 1])
#cue_steps_remaining = 0
#cue_index = 0 #{0:0, -1:1, 1:2}
#cue_number = 0
for i, cue_index in enumerate(subject_cue_times):
    targets[cue_index:cue_index+cue_steps] = subject_cue_values[i]



data_dim = np.shape(subject_recordings)[1]

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
print("Building model")
def build_model(batch_size_, timesteps_, weights = None):

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size_, timesteps_, data_dim)))
    for i in range(num_lstm_layers-1):
        model.add(LSTM(32, return_sequences=True, stateful=True))
    #model.add(LSTM(32, stateful=True))
    model.add(Dense(num_classes, activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['mean_squared_error'])

    if weights==None:
        return model
    else:
        #Change the weights of the model to the weights specified in the weights variable
        model.set_weights(weights)#Transfer the weights from the old model to the new model
        return model

model = build_model(batch_size, timesteps) #Define and build the model for training

#Split the test data for validation

split_point = int(recording_len * val_split)

train_input_seq = subject_recordings[0:split_point]
val_input_seq = subject_recordings[split_point+val_gap_samples:]
train_target_seq = targets[0:split_point]
val_target_seq = targets[split_point+val_gap_samples:]

train_raw_len = len(train_input_seq)
truncation_length_train = (train_raw_len-(train_raw_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))
val_raw_len = len(val_input_seq)
truncation_length_val = (val_raw_len-(val_raw_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))


reshaped_inputs_train = np.reshape(train_input_seq[0:truncation_length_train,:], [-1,timesteps,data_dim])
reshaped_targets_train = np.reshape(train_target_seq[0:truncation_length_train,:], [-1,timesteps,num_classes])
reshaped_inputs_val = np.reshape(val_input_seq[0:truncation_length_val,:], [-1,timesteps,data_dim])
reshaped_targets_val = np.reshape(val_target_seq[0:truncation_length_val,:], [-1,timesteps,num_classes])

x_train = reshaped_inputs_train
y_train = reshaped_targets_train

x_val = reshaped_inputs_val
y_val = reshaped_targets_val

#model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs)#, epochs=num_epochs)

#Need to implement prediction/validation


eval_folder_data = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1b_1000Hz.mat"
eval_folder_labels = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/true_labels_1/BCICIV_eval_ds1b_1000Hz_true_y.mat"
raw_eval_data = sio.loadmat(eval_folder_data)
subject_eval_recordings = np.array(raw_eval_data['cnt'])
raw_label_data = sio.loadmat(eval_folder_labels)
un_nan_labels = np.array([x if np.isfinite(x) else 0 for x in raw_label_data["true_y"]]) #This is not the true test, as the NAN vals shouldnot be counted...


test_recording_len = np.shape(subject_eval_recordings)[0]
test_targets = np.reshape(un_nan_labels, [-1,1])



test_truncation_length = (test_recording_len-(test_recording_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))


#reshaped_inputs_test = np.reshape(subject_eval_recordings[0:test_truncation_length,:], [-1,timesteps,data_dim])
#reshaped_targets_test = np.reshape(test_targets[0:test_truncation_length,:], [-1,timesteps,num_classes])

#x_test = reshaped_inputs_test
#y_test = reshaped_targets_test

x_test = subject_eval_recordings
y_test = test_targets

print("Training model")

model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs, validation_data=(x_val, y_val))#, epochs=num_epochs)

#Need to figure out how to only include some of the score for the summary validation score


#Now run the test on the  test set as per the competition specs
#Could predict on batches for this one though...
#Batch testing not presently implemented


#To determine when to count the validation metric. (Designed to conform with the contest parameters)
val_toggle = np.array([1 if np.isfinite(x) else 0 for x in raw_label_data["true_y"]]) #This line gets rid of the NANs (which presumably should not be counted)

last = 0
for i, val in enumerate(test_targets[:,0]):
    if last == 0 and val != 0:
        val_toggle[i:i+sample_rate] = 0
    last = val


def target_MSE(y_true, y_pred, val=False):
    tars = y_true[:, :, 0]
    preds = y_pred[:, :, 0]
    #This only works for a single example at a time
    #print(shape(tars))
    #print(shape(preds))
    #tars = y_true[:, :, data_dim]
    #preds = y_pred[:, :, num_classes]
    return mean_squared_error(tars, preds)

def rebuild_model_for_test(model):
    #Take the model, and build the same model with the same weights but with different batch size and numsteps parameters
    
    test_model = build_model(1, 1, weights = model.get_weights())#Transfer the weights from the old model to the new model)
    return test_model

print("Testing model")
test_model = rebuild_model_for_test(model)
#prediciton_inputs = [0] #Initial prediciton inputs. This is equivalent to a zero in the target sequence
accuracy_sum = 0
num_val_measures = 0
for i in range(test_recording_len):
    #current_recording_input = x_test[i,:]
    #val_input_vec = np.concatenate((current_recording_input, prediciton_inputs), axis=0)
    test_input_vec = x_test[i,:]
    test_tar_vec = y_test[i,:]

    test_input_mat = np.reshape(test_input_vec, [1, 1, data_dim])
    test_target_mat = np.reshape(test_tar_vec, [1, 1, num_classes]) 

    predictions = test_model.predict_on_batch(test_input_mat)

    #Now compute error between predictions and test_target_mat
    #print("targets ", type(test_target_mat))
    #print("prediciton", type(predictions))
    if val_toggle[i] == 1:
        num_val_measures += 1
        cur_target_MSE = target_MSE(test_target_mat, predictions, val=True)
        accuracy_sum += cur_target_MSE

    #Might want to include additional metrics in here...

    #Now extract the new prediciton_inputs from predictions
    #prediciton_inputs = predictions[0,0,59:60]

    if i%100000==0:
        if num_val_measures > 0:
            acc_so_far = accuracy_sum/num_val_measures
        else:
            acc_so_far = "N/A"
        print("     Target MSE: ",  acc_so_far, "  ;  {0:.3f}".format((i/test_recording_len)*100), " percent complete")
print("Target MSE for test is ", acc_so_far)
