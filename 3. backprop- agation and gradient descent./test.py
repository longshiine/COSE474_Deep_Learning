import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class nn_linear_layer:
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size)) 
        self.b = np.random.normal(0,std,(output_size,1))
    
    # x = (N, D)
    # W = (H, D)
    # b = (H, 1)
    # score = (H, N) --> 벡터의 관점

    ######
    ## Q1
    def forward(self,x):
        score = x @ self.W.T + self.b.T # (N, H)
        return score # (N, H)
    
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        # dLdy = (N,H)
        H = dLdy.shape[1]
        dLdb = np.sum(dLdy, axis=0).reshape(1,H) # (1,H)
        dLdW = dLdy.T @ x # (H, D)
        dLdx = dLdy @ self.W # (N, D)
        return dLdW,dLdb,dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        sigmoid_x = 1 / (1 + np.exp(-x))
        return dLdy * sigmoid_x * (1 - sigmoid_x) # (N, H)

class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        N, O = x.shape
        scores_exp = np.exp(x) # (N, O), (20, 2)
        scores_sum = np.sum(scores_exp, axis=1) # (N, ), (20, )
        softmax = scores_exp / scores_sum.reshape(N, 1) # (N, O), (20,2)
        return softmax
    
    ######
    ## Q6
    def backprop(self,x,dLdy):
        #dLdy = (N, O)
        N = x.shape[0]
        scores_exp = np.exp(x) #(N, 2)
        scores_sum = np.sum(scores_exp, axis=1) #(N, )
        sum_loss = - 1 / (scores_sum ** 2)
        dLdx = sum_loss.reshape((N,1)) * scores_exp * dLdy 
        dLdx += (1 / scores_sum).reshape((N,1)) * dLdy # (N, 2)
        return dLdx

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    def forward(self,x,y):
        N = x.shape[0]
        b_i = y.reshape(N) #(N, )
        before_log = x[np.arange(N),b_i] # (N, )
        log = np.log(before_log)
        cross_entropy_loss = -1 * np.sum(log)
        cross_entropy_loss /= N
        return cross_entropy_loss
        
    ######
    ## Q8
    def backprop(self,x,y):
        N = x.shape[0]
        dLdx = np.zeros_like(x)
        b_i = y.reshape(N)
        before_log = x[np.arange(N), b_i] # (N, )
        loss_before_log = -1 / before_log # (N, )
        dLdx[np.arange(N),b_i] = loss_before_log / N
        return dLdx # (N,2)

num_d=5

# number of test runs
num_test=5

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr=0.01
num_gd_step=60000


# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=True

# set this True if want to plot loss over gradient descent iteration
show_loss=True

for j in range(num_test):
    
    # create training data
    m_d1=(0,0)
    m_d2=(1,1)
    m_d3=(0,1)
    m_d4=(1,0)

    sig=0.05
    s_d1=sig**2*np.eye(2)

    d1=np.random.multivariate_normal(m_d1,s_d1,num_d)
    d2=np.random.multivariate_normal(m_d2,s_d1,num_d)
    d3=np.random.multivariate_normal(m_d3,s_d1,num_d)
    d4=np.random.multivariate_normal(m_d4,s_d1,num_d)

    # training data, and has dimension (4*num_d,2,1)
    x_train_d = np.vstack((d1,d2,d3,d4))
    # training data lables, and has dimension (4*num_d,1)
    y_train_d = np.vstack((np.zeros((2*num_d,1),dtype='uint8'),np.ones((2*num_d,1),dtype='uint8')))
    
    # plotting training data if needed
    if (show_train_data) & (j==0):
        plt.grid()
        plt.scatter(x_train_d[range(2*num_d),0], x_train_d[range(2*num_d),1], color='b', marker='o')
        plt.scatter(x_train_d[range(2*num_d,4*num_d),0], x_train_d[range(2*num_d,4*num_d),1], color='r', marker='x')
        plt.show()
                        
    # create layers

    # hidden layer
    # linear layer
    layer1= nn_linear_layer(input_size=2,output_size=4,)
    # activation layer
    act=nn_activation_layer()
                            
                            
    # output layer
    # linear
    layer2= nn_linear_layer(input_size=4,output_size=2,)
    # softmax
    smax=nn_softmax_layer()
    # cross entropy
    cent=nn_cross_entropy_layer()


    # variable for plotting loss
    loss_out=np.zeros((num_gd_step))

    for i in range(num_gd_step):
        
        # fetch data
        x_train = x_train_d
        y_train = y_train_d
            
        # create one-hot vectors from the ground truth labels
        y_onehot = np.zeros((batch_size,num_class))
        y_onehot[range(batch_size),y_train.reshape(batch_size,)]=1

        ################
        # forward pass
        
        # hidden layer
        # linear
        l1_out=layer1.forward(x_train)
        # activation
        a1_out=act.forward(l1_out)
    
        # output layer
        # linear
        l2_out=layer2.forward(a1_out)
        # softmax
        smax_out=smax.forward(l2_out)
        # cross entropy loss
        loss_out[i]=cent.forward(smax_out,y_train)
         
        ################
        # perform backprop
        # output layer
        
        # cross entropy
        b_cent_out=cent.backprop(smax_out,y_train)
        # softmax
        b_nce_smax_out=smax.backprop(l2_out,b_cent_out)
        # linear
        b_dLdW_2,b_dLdb_2,b_dLdx_2=layer2.backprop(x=a1_out,dLdy=b_nce_smax_out)
    
        # backprop, hidden layer
        # activation
        b_act_out=act.backprop(x=l1_out,dLdy=b_dLdx_2)
        # linear
        b_dLdW_1,b_dLdb_1,b_dLdx_1=layer1.backprop(x=x_train,dLdy=b_act_out)
    
        ################
        # update weights: perform gradient descent
        layer2.update_weights(dLdW=-b_dLdW_2*lr,dLdb=-b_dLdb_2.T*lr)
        layer1.update_weights(dLdW=-b_dLdW_1*lr,dLdb=-b_dLdb_1.T*lr)

        if (i+1) % 2000 ==0:
             print('gradient descent iteration:',i+1)

    # set show_loss to True to plot the loss over gradient descent iterations
    if (show_loss) & (j==0):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step),loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()
    
    
    ################
    # training done
    # now testing

    predicted=np.ones((4,))

    # predicting label for (1,1)
    l1_out=layer1.forward([[1,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)',smax_out,'predicted label:',int(predicted[0]))

    # predicting label for (0,0)
    l1_out=layer1.forward([[0,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)',smax_out,'predicted label:',int(predicted[1]))

    # predicting label for (1,0)
    l1_out=layer1.forward([[1,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)',smax_out,'predicted label:',int(predicted[2]))

    # predicting label for (0,1)
    l1_out=layer1.forward([[0,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)',smax_out,'predicted label:',int(predicted[3]))

    print('total predicted labels:',predicted.astype('uint8'))
        
    accuracy += (predicted[0]==0)&(predicted[1]==0)&(predicted[2]==1)&(predicted[3]==1)

    if (j+1)%10==0:
        print('test iteration:',j+1)

print('accuracy:',accuracy/num_test*100,'%')
