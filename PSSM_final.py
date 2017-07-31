'''
Problem: Denoising of position-specific scoring matrices (PSSMs)

A protein consists of a linear sequence of amino acid residues. 
Briefly, a PSSM of a protein contains the statistics of mutations of amino acid residues of the protein encountered in nature. 
These statistics may be "corrupted" (noisy), The task was to teach a deep network to denoise them.

Data Format:

Input is a CSV file. The first four columns are identifiers (protein id, sequence length, residue name, and residue sequence id), 
the next 20 columns are "corrupted" PSSM values, and the last 20 columns are PSSM values computed from multiple sequence 
alignments obtained with a high-quality method.

Approach:

Each protein (identified by its protein id) is a sample (in the machine learning sense) over which 1D convolutions were done 
along the sequence length dimension. The corrupted values are the inputs to the network, 
and the high-quality values are the output targets (ground truth). 
There are 20 input channels and 20 output channels (corresponding to the PSSM entries).


Instructions to run the code :

1) Install following libraries first in your machine before running the code:
    - numpy
    - theano
    - lasagne
    - matplotlib
    - scikit-learn

All of them can be installed using "pip install <insert_library_name>"

2) All the data that is needed to run this program should have been placed in "data" directory


'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
# read the data in numpy as a array.
data = np.genfromtxt("pssm_denoising_dataset.csv", delimiter=",") # `data` holds our training data

# we get the input and output data as 2d arrays.
'''
############## This section deals with rearranging data that we need to train the function ##############
'''
row_values = data[ : , 1]
size_list = [0]

current_index = 0
array_size = len(row_values)


while current_index < array_size:
    size = int(row_values[current_index])
    current_index += size
    size_list.append(current_index)
print(len(size_list))

'''
############## END OF SECTION : that deals with rearranging data that we need to train the function ##############
'''

'''
############## Network Architecture Part ##############
'''

# symbolic input and output variables
# our input is 3D
input_var = T.tensor3('inputs')
target_var = T.tensor3('targets')

# start building our network here in lasagne:
# shape is batch size, number of channels, size of the input
# in our case the batch size  = 1, number of channels = 20

# Input dim --> (batch_size, num_input_channels, input_length)
input0 = lasagne.layers.InputLayer(shape=(1, 20, None), input_var=input_var)
# Weight dim --> (num_filters, num_input_channels, filter_length)

'''
# Simplest Archeitecture 
conv1 = lasagne.layers.Conv1DLayer(input0, num_filters=32, filter_size=5,pad='valid', 
                                   W=lasagne.init.Uniform(), nonlinearity=None  ) #nonlinearity=lasagne.nonlinearities.sigmoid)
network = lasagne.layers.Conv1DLayer(conv1, num_filters=20, filter_size=5,pad='full', 
                                     W=conv1.W.dimshuffle((1,0,2)), nonlinearity=None  )
'''


# Building Network
conv1 = lasagne.layers.Conv1DLayer(input0, num_filters=32, filter_size=5,pad='valid', 
                                   W=lasagne.init.Uniform(), nonlinearity=None  ) #nonlinearity=lasagne.nonlinearities.sigmoid)
conv2 = lasagne.layers.Conv1DLayer(conv1, num_filters=32, filter_size=5,pad='valid',
                                   W=lasagne.init.Uniform(), nonlinearity=None  ) #nonlinearity=lasagne.nonlinearities.sigmoid)
conv3 = lasagne.layers.Conv1DLayer(conv2, num_filters=32, filter_size=5,pad='valid',
                                   W=lasagne.init.Uniform(), nonlinearity=None  ) #nonlinearity=lasagne.nonlinearities.sigmoid)
deconv1 = lasagne.layers.Conv1DLayer(conv3, num_filters=32, filter_size=5,pad='full',  
                                     W=conv3.W.dimshuffle((1,0,2)), nonlinearity=None ) #nonlinearity=lasagne.nonlinearities.sigmoid)
deconv2 = lasagne.layers.Conv1DLayer(deconv1, num_filters=32, filter_size=5,pad='full', 
                                     W=conv2.W.dimshuffle((1,0,2)), nonlinearity=None ) #nonlinearity=lasagne.nonlinearities.sigmoid)
# Last layer always have to be linear 
network = lasagne.layers.Conv1DLayer(deconv2, num_filters=20, filter_size=5,pad='full', 
                                     W=conv1.W.dimshuffle((1,0,2)), nonlinearity=None  )

#Prediction and Loss Calculation
predictions = lasagne.layers.get_output(network)
trainable_params = lasagne.layers.get_all_params(network, trainable=True)
loss = lasagne.objectives.squared_error(predictions, target_var).mean()


adam_updates = lasagne.updates.adam(loss,trainable_params, learning_rate=0.0005)
train_fn = theano.function( [ input_var, target_var ], loss, updates=adam_updates )

'''
############## END OF Network Architecture Part ##############
'''

'''
############## Training Part of the code ##############
'''
num_epochs = 50
epochs = range(num_epochs)

train_loss_epochs = []

for epoch in epochs:
    train_err = 0
    train_batches = 0
# Real learning with all the data
    for i in range(len(size_list)-1):
        # (batch_size, num_input_channels, input_length)
        # Input & Output are 2d arrays of { [1] x [Sequence length] x [20 corrupted values] }
        X = data[ size_list[i] : size_list[i+1] , 4:24 ]
        Y = data[ size_list[i] : size_list[i+1] , 24:44 ]

        input = np.reshape(X, (1,20,size_list[(i+1)]-size_list[i])  )
        output = np.reshape(Y, (1,20,size_list[(i+1)]-size_list[i])  )
        train_err = train_err+ train_fn(input, output)
        train_batches += 1

    train_loss = (train_err / train_batches)

    train_loss_epochs.append(train_loss)
    print("  training loss:\t\t{:.6f}".format(train_loss))

'''
############## END OF Training ##############
'''

#Plotting the loss graph
plt.plot(epochs, train_loss_epochs)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('graphs/pssm_train_loss_graph_1.png')



