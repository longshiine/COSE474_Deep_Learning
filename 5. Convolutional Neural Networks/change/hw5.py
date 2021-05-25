import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle

# put nn_layers and mnist_loader modules to your working directory (or set appropriate paths)
import nn_layers as nnl

class nn_mnist_classifier:
    def __init__(self, mmt_friction=0.9, lr=1e-2):
        ## initialize each layer of the entire classifier

        # convolutional layer
        # input image size: 28 x 28
        # filter size: 3 x 3
        # input channel size: 1
        # output channel size (number of filters): 28

        self.conv_layer_1 = nnl.nn_convolutional_layer(Wx_size=3, Wy_size=3, input_size=28,
                                                       in_ch_size=1, out_ch_size=28)

        # activation layer
        self.act_1 = nnl.nn_activation_layer()

        # activaition map output: map size 26 x 26, 28 channels

        # maxpool
        self.maxpool_layer_1 = nnl.nn_max_pooling_layer(stride=2, pool_size=2)

        # after max pool, map size 13 x 13, 28 channels

        # fully connected layer 1
        # input: 13 x 13 with 28 channels => total length of 28*13*13
        # output 128
        self.fc1 = nnl.nn_fc_layer(input_size=28*13**2, output_size=128)
        self.act_2 = nnl.nn_activation_layer()

        # fully connected layer 1
        # input 128
        # output 10
        self.fc2 = nnl.nn_fc_layer(input_size=128, output_size=10)

        # softmax
        self.sm1 = nnl.nn_softmax_layer()

        # cross entropy
        self.xent = nnl.nn_cross_entropy_layer()

        # initialize momentum parameter
        # momentum parameter v in the momentum update equation

        # momentum v for convolutional layer, one for W and for b
        self.v_w_cv1 = 0
        self.v_b_cv1 = 0

        # momentum v for fully connected layer 1, one for W and for b
        self.v_w_fc1 = 0
        self.v_b_fc1 = 0

        # momentum v for fully connected layer 2, one for W and for b
        self.v_w_fc2 = 0
        self.v_b_fc2 = 0

        # learning rate
        self.lr = lr
        # friction parameter (alpha) for momentum update
        self.mmt_friction = mmt_friction

    # forward method
    # parameters:
    #   x: input MNIST images in batch
    #   y: ground truth/labels of the batch
    #   backprop_req: set this to True if backprop method is called next
    #                 set this to False if only forward pass (inference) needed
    def forward(self, x, y, backprop_req=True):
        ########################
        # Q1. Complete forward method
        ########################
        # cv1_f, ac1_f, mp1_f, fc1_f, ac2_f, fc2_f, sm1_f, cn_f
        # are outputs from each layer of the CNN
        # cv1_f is the output from the convolutional layer
        # ac1_f is the output from 1st activation layer
        # mp1_f is the output from maxpooling layer
        # ... and so on

        cv1_f = self.conv_layer_1.forward(x)

        # similarly, fill in ... part in the below
        ac1_f = ...
        mp1_f = ...

        fc1_f = ...
        ac2_f = ...

        fc2_f = ...

        sm1_f = ...

        cn_f = ...

        # cn_f should be the loss of the current input batch

        ########################
        # Q1 ends here
        ########################

        scores = sm1_f
        loss = cn_f

        # store intermediate variables, later to be used for backprop
        # store required only when backprop_req is True
        if backprop_req:
            self.fwd_cache = (x, y, cv1_f, ac1_f, mp1_f, fc1_f, ac2_f, fc2_f, sm1_f, cn_f)

        # forward will return, scores (sm1_f), and loss (cn_f)
        return scores, loss

    # backprop method
    def backprop(self):
        # note that the backprop will use the saved structures,
        (x, y, cv1_f, ac1_f, mp1_f, fc1_f, ac2_f, fc2_f, sm1_f, cn_f) = self.fwd_cache

        ########################
        # Q2. Complete backprop method
        ########################
        cn_b = self.xent.backprop(sm1_f, y)

        # similarly, fill in ... part in the below
        sm1_b = ...

        fc2_b, dldw_fc2, dldb_fc2 = ...
        ac2_b = ...

        fc1_b, dldw_fc1, dldb_fc1 = ...
        mp1_b = ...

        ac1_b = ...
        cv1_b, dldw_cv1, dldb_cv1 = ...

        ########################
        # Q2 ends here
        ########################

        # cache upstream gradients for weight updates!
        self.bkp_cache = (dldw_fc2, dldb_fc2, dldw_fc1, dldb_fc1, dldw_cv1, dldb_cv1)

    def update_weights(self):
        # restore upstream gradients
        (dldw_fc2, dldb_fc2, dldw_fc1, dldb_fc1, dldw_cv1, dldb_cv1) = self.bkp_cache

        friction = self.mmt_friction
        lr = self.lr

        ########################
        # Q3. Complete momentum update
        ########################


        # fill in ... part in the below
        # perform momentum update on each v variable for W and b at convolutional and FC layers
        self.v_w_fc2 = ...
        self.v_b_fc2 = ...

        self.v_w_fc1 = ...
        self.v_b_fc1 = ...

        self.v_w_cv1 = ...
        self.v_b_cv1 = ...

        # using v, perform weight update for each layer
        self.fc2.update_weights(...)
        self.fc1.update_weights(...)
        self.conv_layer_1.update_weights(...)
        ########################
        # Q3 ends here
        ########################


########################
## classification: dataset preparation
########################

# load MNIST data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# insert channel dimension of 1
X_train=np.expand_dims(X_train,axis=1)
X_test=np.expand_dims(X_test,axis=1)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# divide train into training and validation
# set the dataset size
# 50000 training and 10000 validation samples
n_train_sample = 50000
n_val_sample = len(y_train) - n_train_sample

# data preprocessing
# normalize pixel values to (0,1)
X_train = X_train.astype('float64') / 255.0
X_test = X_test.astype('float64') / 255.0

########################
# Q. Set learning rate, batch size and total number of epochs for training
# There are no definitive answers, experiement with several hyperparameters
########################
lr = ...
n_epoch = ...
batch_size = ...

# set friction (alpha) for momentum
friction = 0.9

# define classifier
classifier = nn_mnist_classifier(mmt_friction=friction, lr=lr)

# number of steps per epoch
numsteps = int(n_train_sample / batch_size)

# split data into training and validation dataset
X_s = np.split(X_train, [n_val_sample, ])
X_val = X_s[0]
X_trn = X_s[1]

y_s = np.split(y_train, [n_val_sample, ])
y_val = y_s[0]
y_trn = y_s[1]

do_validation = True

########################
# training
########################

for i in range(n_epoch):

    # randomly shuffle training data
    shuffled_index_train = np.arange(y_trn.shape[0])
    np.random.shuffle(shuffled_index_train)

    # shuffled for randomized mini-batch
    X_trn = X_trn[shuffled_index_train]
    y_trn = y_trn[shuffled_index_train]

    print('epoch number:', i)

    # for tracking training accuracy
    trn_accy = 0

    for j in range(numsteps):

        # take mini-batch from training set
        x = X_trn[j * batch_size:(j + 1) * batch_size, ]
        y = y_trn[j * batch_size:(j + 1) * batch_size, ]

        # perform forward, backprop and weight update
        scores, loss = classifier.forward(x, y)
        classifier.backprop()
        classifier.update_weights()

        # for tracking training accuracy
        estim = np.ravel(np.argmax(scores, axis=1))
        trn_accy += np.sum((estim == y).astype('uint8')) / batch_size

        # every 50 loops, check loss
        if (j + 1) % 50 == 0:
            print('loop count', j + 1)
            print('loss', loss)

            # every 200 loops, print training accuracy
            if (j + 1) % 200 == 0:
                print('training accuracy:', trn_accy / 2, '%')
                trn_accy = 0

                # evaluate the validation accuarcy
                if do_validation:
                    # pick 100 random samples from validation set
                    val_idx = np.random.randint(low=0, high=y_val.shape[0], size=(100,))

                    x = X_val[val_idx]
                    y = y_val[val_idx]

                    # take random batch of batch_size
                    # forward pass!
                    scores, loss = classifier.forward(x, y, backprop_req=False)
                    estim = np.ravel(np.argmax(scores, axis=1))

                    # compare softmax vs y
                    val_accy = np.sum((estim == y).astype('uint8'))
                    print('validation accuracy:', val_accy, '%')

########################
# testing
########################
# test_batch: accuracy is measured in this batch size
# test_iter: total number of batch iterations to complete testing over test data
# tot_accy: total accuracy

test_batch = 100
test_iter = int(y_test.shape[0] / test_batch)
tot_accy = 0

for j in range(test_iter):
    x = X_test[j * test_batch:(j + 1) * test_batch, ]
    y = y_test[j * test_batch:(j + 1) * test_batch, ]

    # forward pass!
    scores, loss = classifier.forward(x, y, backprop_req=False)
    estim = np.ravel(np.argmax(scores, axis=1))
    accy = np.sum((estim == y).astype('uint8')) / test_batch
    tot_accy += accy
    print('batch accuracy:', accy)

# print out final accuracy
print('total accuray', tot_accy / test_iter)

# set this to True if we want to plot sample predictions
plot_sample_prediction = True

# test plot randomly picked 10 MNIST numbers
if plot_sample_prediction:
    num_plot = 10
    sample_index = np.random.randint(0, X_test.shape[0], (num_plot,))
    plt.figure(figsize=(12, 4))

    for i in range(num_plot):
        idx = sample_index[i]
        img = np.squeeze(X_test[idx])
        ax = plt.subplot(1, num_plot, i + 1)
        plt.imshow(img, cmap=plt.get_cmap('gray'))

        x = X_test[idx:idx + 1]
        y = y_test[idx:idx + 1]

        # get prediction from our classifier
        score, _ = classifier.forward(x, y, backprop_req=False)
        pred = np.ravel(np.argmax(score, axis=1))

        # if our prediction is correct, the title will be in black color
        # otherwise, for incorrect predictions, the title will be in red
        if y_test[idx] == pred:
            title_color = 'k'
        else:
            title_color = 'r'

        ax.set_title('GT:' + str(y_test[idx]) + '\n Pred:' + str(int(pred)), color=title_color)

    plt.tight_layout()
    plt.show()
