import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    s = W@x+b

    s_exp = np.exp(s) # (num_class, n)
    s_prob = s_exp / np.sum(s_exp, axis=0) #(num_class, n)
    correct_prob_log = -1 * np.log(s_prob.T[np.arange(n), y]) #(n, )
    loss = np.sum(correct_prob_log) / n
    return loss
    pass

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:] # (1, feat_dim)
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim)) # (nun_class, feat_dim)
    x = np.reshape(x.T, (-1, n)) # (feat_dim, n)
    s = W@x + b
    
    s_yi = s.T[np.arange(n),y] # (n,)
    diff = s - s_yi + 1
    diff_overzero_bool = diff > 0
    diff_overzero_bool[y, np.arange(n)] = False
    loss = np.sum(diff * diff_overzero_bool) / n
    return loss
    pass

########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    # implement your function here
    #return accuracy
    X_test_Squared = np.sum(X_test ** 2, axis=1).reshape(n_test_sample,1)
    X_train_Squared = np.sum(X_train ** 2, axis=1)
    inner_prod = X_test @ X_train.T
    L2_dists = X_test_Squared + X_train_Squared - 2 * inner_prod # L2 ds between X_test, X_train (100,400)
    
    neighbors_label = y_train[np.argsort(L2_dists)] # (100, 400)
    k_neighbors_label = neighbors_label[:, :k] # (100, k)
    predict_label = stats.mode(k_neighbors_label, axis=1)[0].reshape((100)) # (100,)
    accuracy = (y_test - predict_label == 0).sum() / y_test.shape[0]
    return accuracy
    pass


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 3

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.5

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'svm'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
