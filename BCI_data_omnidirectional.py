##############################################
#Omnidirectional learning for autoimputation
#(Or Self-supervised learning for autoimputation)
#RNN/LSTM version
##############################################

#Need to set up a test paradigm and data feeder (randomly choose a sequence to be output instead of input )
#Need to set up the model to be able to handle this

#Choose a target electrode to ablate
#Set this as the target (possibly along with the actual target)
#Set the rest of the electrodes as inputs
#Build the model with this configuration (transferring the weights from any previously trained models)
#Train the model to predict the missing electrode trace from the other electrode traces
#Repeat for other random electrodes

#When do I stop? How do I prevent overwriting of previosuly trained relations?

#Test using seperate data predicting each of the electrodes from the others

#Could try this using conditional GANs
#Could also try this using a more explicitly omnidirectional architechture

#Will do this using a dropconnect strategy designed to facilitate learning of the proper joint distribution whereby each input variable contirubtes 
#information when present but its existence does not penalize the model when it is not. (Model will also receive an additional vector capturing whether or not the variable is receiving information)

#Use the Keras functional API to handle the conditional i/o dropconnect

inputs = []
outputs = []
io_control = [] #A binary vector with a 1 if a variable should be treated as an input and a zero if it should be treated as an output