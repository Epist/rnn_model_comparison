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


#Change the predictions to be continuous values between 0 and 1
#Change the loss function to be MSE
#Add an additional metric that reflects the MSE (keep the accuracy metric)


#SHOULD GET RID OF THE SOFTMAX OUTPUT LAYER...

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


val_split = .8 #The percentage to asigne to the training set
val_gap = .5 #In seconds
timesteps = 64
num_classes = 1
batch_size = 32
num_epochs = 5
num_lstm_layers = 2

cue_length_in_secs = 4

    #print(y_true.get_shape().as_list())
    #return tf.shape(y_true)[2]

print("Loading data")
raw_data = sio.loadmat('/home/larry/Data/BCI_Competition/IV/BCICIV_1calib_1000Hz_mat/BCICIV_calib_ds1b_1000Hz.mat')


subject_recordings = np.array(raw_data['cnt'])
subject_cues_raw = raw_data['mrk'][0,0]
sample_rate = raw_data['nfo'][0,0][0][0][0]
cue_steps = cue_length_in_secs * sample_rate
val_gap_samples = int(val_gap * sample_rate)

#target_value = 1000 #For changing the importance of the targets vs. the sequence in prediction

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


#Autoassociative model

#Stateful model

data_dim = np.shape(subject_recordings)[1]


full_data_dim = data_dim + num_classes


def target_MSE(y_true, y_pred, val=False):
    tars = y_true[:, :, data_dim]
    preds = y_pred[:, :, data_dim]
    if not val:
        return metrics.mse(tars, preds)
    else:
        #This only works for a single example at a time
        #print(tars)
        #print(preds)
        tars = tars[0,0]
        preds = preds[0,0]
        #return mean_squared_error(tars, preds)
        return np.mean(np.square(tars-preds))


# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
def build_model(batch_size_, timesteps_, weights = None):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size_, timesteps_, full_data_dim)))
    for i in range(num_lstm_layers-1):
        model.add(LSTM(32, return_sequences=True, stateful=True))
    #model.add(LSTM(32, stateful=True))
    model.add(Dense(full_data_dim, activation='linear'))

    model.compile(loss='mean_squared_error', #categorical_crossentropy
                  optimizer='rmsprop',
                  metrics=[target_MSE])
    if weights==None:
        return model
    else:
        #Change the weights of the model to the weights specified in the weights variable
        model.set_weights(weights)#Transfer the weights from the old model to the new model
        return model

model = build_model(batch_size, timesteps) #Define and build the model for training

split_point = int(recording_len * val_split)


all_data_vars = np.concatenate((subject_recordings, targets), axis=1)

all_train_seq = all_data_vars[0:split_point]
all_valid_seq = all_data_vars[split_point+val_gap_samples:]

train_raw_len = np.shape(all_train_seq)[0]
valid_raw_len = np.shape(all_valid_seq)[0]
print(train_raw_len, " training data points")
print(valid_raw_len, " validation data points")


shifted_data_vars_train = np.concatenate((np.zeros([1,np.shape(all_train_seq)[1]]),
                                    all_train_seq[:(train_raw_len-1), :]) , axis=0)

shifted_data_vars_valid = np.concatenate((np.zeros([1,np.shape(all_valid_seq)[1]]),
                                    all_valid_seq[:(valid_raw_len-1), :]) , axis=0)

truncation_length_train = (train_raw_len-(train_raw_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))
#truncation_length_valid = (valid_raw_len-(valid_raw_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))

reshaped_inputs_train = np.reshape(all_train_seq[0:truncation_length_train,:], [-1,timesteps, full_data_dim])
reshaped_targets_train = np.reshape(shifted_data_vars_train[0:truncation_length_train,:], [-1,timesteps, full_data_dim])
#reshaped_inputs_valid = np.reshape(all_valid_seq[0:truncation_length_valid,:], [-1,timesteps, full_data_dim])
#reshaped_targets_valid = np.reshape(shifted_data_vars_valid[0:truncation_length_valid,:], [-1,timesteps, full_data_dim])

x_train = reshaped_inputs_train
y_train = reshaped_targets_train

#x_val = reshaped_inputs_valid
#y_val = reshaped_targets_valid
#x_val = all_valid_seq
x_val = subject_recordings[split_point+val_gap_samples:] #Don't include the target as input. This will be inferred...
y_val = shifted_data_vars_valid

#Need to build val toggle sequence for validation data


#model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = num_epochs)#, epochs=num_epochs)

#Need to implement prediction/validation


eval_folder_data = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/BCICIV_1eval_1000Hz_mat/BCICIV_eval_ds1b_1000Hz.mat"
eval_folder_labels = "/home/larry/Data/BCI_Competition/IV/BCICIV_1_eval/true_labels_1/BCICIV_eval_ds1b_1000Hz_true_y.mat"
raw_eval_data = sio.loadmat(eval_folder_data)
subject_eval_recordings = np.array(raw_eval_data['cnt'])
raw_label_data = sio.loadmat(eval_folder_labels)
un_nan_labels = np.array([x if np.isfinite(x) else 0 for x in raw_label_data["true_y"]]) #This is not the true test, as the NAN vals shouldnot be counted...


test_recording_len = np.shape(subject_eval_recordings)[0]
print(test_recording_len, " test data points")

test_targets = np.reshape(un_nan_labels, [-1,1])

#To determine when to count the validation metric. (Designed to conform with the contest parameters)
val_toggle_test = np.array([1 if np.isfinite(x) else 0 for x in raw_label_data["true_y"]]) #This line gets rid of the NANs (which presumably should not be counted)

last = 0
for i, val in enumerate(test_targets[:,0]):
    if last == 0 and val != 0:
        val_toggle_test[i:i+sample_rate] = 0
    last = val



###################################
#All of this needs to be written s.t. the validation inputs contain the previous predictions from the model and not the ground truth. 
#This means that the validation/testing cannot be done in batch.
#The current setup is cheating (making use of privilidged information) and is therefore not acceptable...

#To do this, I will need to feed the val data in one by one. This means that the model must be validated with timesteps and num_batches both equal to 1.
#I need to figure out how to do this (will probably need to rebuild the model...)
###################################

all_data_vars_tar = np.concatenate((subject_eval_recordings, test_targets), axis=1)

shifted_data_vars_tar = np.concatenate((np.zeros([1,np.shape(all_data_vars_tar)[1]]),
                                    all_data_vars_tar[:(test_recording_len-1), :]) , axis=0)

x_test = subject_eval_recordings
y_test = shifted_data_vars_tar

#print(np.shape(all_data_vars_tar))
#print(np.shape(shifted_data_vars_tar))
#val_truncation_length = (test_recording_len-(test_recording_len%(timesteps*batch_size))) #Might need to make it  (recording_len-(recording_len%(timesteps*batch_size)))

#reshaped_inputs_val = np.reshape(all_data_vars_tar[0:val_truncation_length,:], [-1,timesteps, full_data_dim])
#reshaped_targets_val = np.reshape(shifted_data_vars_tar[0:val_truncation_length,:], [-1,timesteps, full_data_dim])


#x_val = reshaped_inputs_val
#y_val = reshaped_targets_val
#print(np.shape(subject_eval_recordings))




print("Training model")
def rebuild_model_for_validation(model):
    #Take the model, and build the same model with the same weights but with different batch size and numsteps parameters
    
    val_model = build_model(1, 1, weights = model.get_weights())#Transfer the weights from the old model to the new model)
    return val_model

for e in range(num_epochs):
    print("Training for epoch: ", e+1)
    model.fit(x_train, y_train, batch_size=batch_size, shuffle=False, nb_epoch = 1)#, epochs=num_epochs)

    #validation_data=(x_val, y_val)
    val_model = rebuild_model_for_validation(model)

    print("Validating model")
    prediciton_inputs = [0] #Initial prediciton inputs. This is equivalent to a zero in the target sequence
    accuracy_sum = 0
    num_val_measures = 0
    for i in range(len(x_val)):
        current_recording_input = x_val[i,:]
        #print(current_recording_input)
        #print(prediciton_inputs)
        val_input_vec = np.concatenate((current_recording_input, prediciton_inputs), axis=0)
        val_tar_vec = y_val[i,:]

        val_input_mat = np.reshape(val_input_vec, [1, 1, full_data_dim])
        val_target_mat = np.reshape(val_tar_vec, [1, 1, full_data_dim]) 

        predictions = val_model.predict_on_batch(val_input_mat)

        #Now compute error between predictions and val_target_mat
        """
        if val_toggle_val[i] == 1:
            num_val_measures += 1
            cur_target_MSE = target_MSE(val_target_mat, predictions, val=True)
            accuracy_sum += cur_target_MSE
        """
        num_val_measures += 1
        cur_target_MSE = target_MSE(val_target_mat, predictions, val=True)
        accuracy_sum += cur_target_MSE

        #Might want to include additional metrics in here...

        #Now extract the new prediciton_inputs from predictions
        prediciton_inputs = predictions[0,0,59:60]

        if i%10000==0:
            if num_val_measures > 0:
                acc_so_far = accuracy_sum/num_val_measures
            else:
                acc_so_far = "N/A"
            print("     Target MSE: ",  acc_so_far, "  ;  {0:.2f}".format((i/len(x_val))*100), " percent complete")
    print("Target MSE for val epoch ", e+1, " is ", acc_so_far)




#TEST

print("Testing model")
test_model = rebuild_model_for_validation(model)
prediciton_inputs = [0] #Initial prediciton inputs. This is equivalent to a zero in the target sequence
accuracy_sum = 0
num_val_measures = 0
for i in range(test_recording_len):
    current_recording_input = x_test[i,:]
    test_input_vec = np.concatenate((current_recording_input, prediciton_inputs), axis=0)
    test_tar_vec = y_test[i,:]

    test_input_mat = np.reshape(test_input_vec, [1, 1, full_data_dim])
    test_target_mat = np.reshape(test_tar_vec, [1, 1, full_data_dim]) 

    predictions = test_model.predict_on_batch(test_input_mat)

    #Now compute error between predictions and test_target_mat
    #print("targets ", type(test_target_mat))
    #print("prediciton", type(predictions))
    if val_toggle_test[i] == 1:
        num_val_measures += 1
        cur_target_MSE = target_MSE(test_target_mat, predictions, val=True)
        accuracy_sum += cur_target_MSE

    #Might want to include additional metrics in here...

    #Now extract the new prediciton_inputs from predictions
    prediciton_inputs = predictions[0,0,59:60]

    if i%100000==0:
        if num_val_measures > 0:
            acc_so_far = accuracy_sum/num_val_measures
        else:
            acc_so_far = "N/A"
        print("     Target MSE: ",  acc_so_far, "  ;  {0:.2f}".format((i/test_recording_len)*100), " percent complete")
print("Target MSE for test is ", acc_so_far)