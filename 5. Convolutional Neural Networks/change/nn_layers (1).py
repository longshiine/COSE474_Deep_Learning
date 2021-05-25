import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return out

    def backprop(self, x, dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        return out

    def backprop(self,x,dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        
        return out
    
    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return out

    def backprop(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return dLdx
