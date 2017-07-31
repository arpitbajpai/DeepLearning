'''
Problem : PU Learning for Virtual Screening

Goal : Predict the biological activity of small molecules from the set of chemical descriptors.
Build a classifier to identify the positive class samples from the test samples.

Data :
• Small molecules samples labeled and unlabeled
• Each sample consists of 1194 chemical descriptors such as Number of aromatic rings, Topological polar surface area etc.

Approach: 
Treat small molecules which are biologically active as a labeled samples and remaining samples as unlabeled samples.


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
import warnings
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

########################## Utility Functions(get_mini_batch)##########################

# this function returns random batches from the given data with given batch size.
def get_minibatch( inputs, batchsize):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_index in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_index:start_index + batchsize]
        yield inputs[excerpt]

# partition data in 80% (traning data) and 20% (testing data)
def get_train_and_test_data(inputs):

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    # 80% 20% for training and test data
    train_data_size = np.int ( math.ceil(len(indices) * 0.8) )
    train_data = inputs[ : train_data_size ]
    test_data = inputs[ train_data_size : ]

    return train_data, test_data

########################## Utility Functions End ##########################

# read the data in numpy as a array.
data_positive = np.genfromtxt("data/drug_bank_selected.csv", delimiter=",") # `data` holds our training data
data_negative_1 = np.genfromtxt("data/reaction_built.csv", delimiter=",") # `data` holds our training data
data_negative_2 = np.genfromtxt("data/rotbond_built.csv", delimiter=",") # `data` holds our training data

# associate the data with their targets
unlabeled_data = np.vstack((data_negative_1,data_negative_2))
unlabeled_data = np.hstack( ( unlabeled_data, np.zeros((len(unlabeled_data),1), dtype=np.int)) )
labeled_data = np.hstack( ( data_positive, np.ones((len(data_positive),1), dtype=np.int)) )

print "Number of labeled data samples are ", len(labeled_data)
print "Number of unlabeled data samples are ", len(unlabeled_data)

data_combined = np.vstack((labeled_data, unlabeled_data))

# this is shuffling the data : mixing unlabeled and labeled data
indices = np.arange(len(data_combined))
np.random.shuffle(indices)

# this is the data we need to split for training and testing.
data_combined = data_combined[indices]


'''
############## Network Architecture Part ##############
'''

# symbolic input and output variables
# our input is 3D
input_var = T.tensor4('inputs')
target_var = T.dmatrix('targets')

# start building our network here in lasagne:
# shape is batch size, number of channels, size of the input
# in our case the batch size  = 1, number of channels = 20
batch_size = 500

network = lasagne.layers.InputLayer(shape=(batch_size, 1, 1, 1115), input_var=input_var)
network = lasagne.layers.DenseLayer(network, num_units=1115, W=lasagne.init.GlorotUniform())
network = lasagne.layers.DenseLayer(network, num_units=50, W=lasagne.init.GlorotUniform())
network = lasagne.layers.DenseLayer(network, num_units=25, W=lasagne.init.GlorotUniform())
network = lasagne.layers.DenseLayer(network, num_units=1, W=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.sigmoid)

########################## Train Function ##############################

predictions = lasagne.layers.get_output(network)

trainable_params = lasagne.layers.get_all_params(network, trainable=True)

# because we are training in mini-batches, it will return the array of values and hence we use mean
loss_var = lasagne.objectives.binary_crossentropy(predictions, target_var).mean()

adam_updates = lasagne.updates.adam(loss_var, trainable_params, learning_rate=0.008)

train_fn = theano.function( [ input_var, target_var ], [ predictions, loss_var ], updates=adam_updates )

#######################################################################

########################## Test Function ##############################

test_predictions = lasagne.layers.get_output(network, deterministic=True )
test_loss_var = lasagne.objectives.binary_crossentropy(test_predictions, target_var).mean()

binary_predictions = test_predictions > 0.5

test_acc = T.eq(binary_predictions, target_var)

# testing function returns predictions and testing loss in the same order
test_fn = theano.function( [ input_var, target_var ], [ binary_predictions, test_loss_var ])

#######################################################################


num_epochs = 10
train_loss_epochs = []
test_loss_epochs = []
epochs = range(num_epochs)

training_data, testing_data = get_train_and_test_data(data_combined)

testing_targets = testing_data[: , -1:]
num_zeros_testing_data = np.count_nonzero(testing_targets==0)
num_ones_testing_data = np.count_nonzero(testing_targets)

print "number of negative samples in testing data are ", num_zeros_testing_data

print "And number of positive samples in testing data are ", num_ones_testing_data

for i in epochs:
    # these calculations are per epoch.
    train_loss = 0;
    test_loss = 0;
    test_acc = 0;

    num_train_batches = 0;
    num_test_batches = 0;

    # training the samples
    for inputs in get_minibatch(training_data, batch_size):

        targets = inputs[:, -1:]  # last column contains the targets
        inputs = inputs[: , : -1]  # last column should be stripped in input
        temp_batch_size = min( batch_size , len(inputs) )
        inputs = np.reshape(inputs, (temp_batch_size, 1, 1, 1115))

        train_loss += train_fn(inputs, targets)[1]

        num_train_batches += 1

    train_loss = (train_loss / num_train_batches)

    train_loss_epochs.append(train_loss)
    #Uncomment this code for validation loss analysis
    # test_loss_epochs.append(test_loss)

    print(" ==================== ==================== ==================== ")
    print "Results for Epoch ", i
    print("  training loss:\t\t{:.6f}".format(train_loss))
    #Uncomment this code for validation loss analysis
    # print("  validation loss:\t\t{:.6f}".format(test_loss))
    print(" ==================== ==================== ==================== ")

# testing the samples (one at a time)
ground_truth_arr = []
predictions_arr = []
for inputs in get_minibatch(testing_data, 1):
    
    targets = inputs[:, -1:]  # last column contains the targets
    inputs = inputs[:, : -1]  # last column should be stripped in input
    # temp_batch_size = min(batch_size, len(inputs)) #change number of batches
    inputs = np.reshape(inputs, (1, 1, 1, 1115))
    test_prediction, loss = test_fn(inputs, targets)
    ground_truth_arr.append((targets[0])[0]);

    if test_prediction:
        predictions_arr.append(1);
    else:
        predictions_arr.append(0);
    
    #Uncomment this code for validation loss analysis
    #test_loss += err
    #num_test_batches += 1

    #Uncomment this code for validation loss analysis
    # test_loss = (test_loss / num_train_batches)

#Generating Confusion Matrix
print(confusion_matrix(ground_truth_arr, predictions_arr))


#Uncomment this code for loss graph generation

#plt.plot(epochs, train_loss_epochs)
# plt.axis([min(epochs),max(epochs),min(train_loss_epochs),max(train_loss_epochs)])
#plt.savefig('train_loss_new_arch.png')
#
# plt.plot(epochs, test_loss_epochs)
# plt.axis([min(epochs),max(epochs),min(test_loss_epochs),max(test_loss_epochs)])
# plt.savefig('test_loss_graph.png')