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
        N, C, H, W = x.shape
        F, C, HH, WW = self.W.shape

        stride = 1
        pad = 0
        filter_size = C*HH*WW # 내적의 용이성을 위해 미리 계산해두는 값

        H_out = int(1 + (H - HH) / stride) # H'
        W_out = int(1 + (W - WW) / stride) # W'

        out = np.zeros((N, F, H_out, W_out)) # 최종적으로 구할 out 초기화 (N, F, H', W')

        for height in  range(H_out):
            for width in  range(W_out):
                x_crop = x[np.arange(N),  :, height*stride:height*stride+HH, width*stride:width*stride+WW]  # x에서 (N,C,HH,WW)크기만큼을 crop
                x_crop_stretch = x_crop.reshape(N, filter_size)  # (N, filter_size)
                w_stretch = self.W.reshape(F, filter_size)  # (F, filter_size)
                spatial_out = np.dot(x_crop_stretch, w_stretch.T) + self.b.reshape((1,F)) # (N,F)
                out[np.arange(N),  :, height, width] = spatial_out
        return out

    def backprop(self, x, dLdy):
        N, C, H, W = x.shape
        F, C, HH, WW = self.W.shape
        N, F, H_out, W_out = dLdy.shape

        stride = 1
        pad = 0

        filter_size = C*HH*WW # scalar, for stretch

        dLdx= np.zeros((N,C,H,W))
        dLdW = np.zeros((F,C,HH,WW))
        dLdb = np.zeros((1,F,1,1))

        for height in range(H_out):
            for width in range(W_out):
                dspatial_out = dLdy[np.arange(N), :, height, width] # (N, F)
                dLdb += np.sum(dspatial_out, axis=0).reshape(1,F,1,1) # (1, F, 1, 1)

                x_tmp = x[np.arange(N), :, height*stride:height*stride+HH, width*stride:width*stride+WW] # (N, C, HH, WW)
                dw_stretch = np.dot(dspatial_out.T, x_tmp.reshape(N, filter_size)) # (N,F).T @ (N,filter_size) = (F, filter_size)
                dLdW += dw_stretch.reshape(F,C,HH,WW)

                w_stretch = self.W.reshape(F, filter_size) # (F, filter_size)
                dx_tmp_stretch = np.dot(dspatial_out, w_stretch) # (N,F) @ (F,filter_size) = (N, filter_size)
                dx_tmp = dx_tmp_stretch.reshape(N,C,HH,WW)
                dLdx[np.arange(N), :, height*stride:height*stride+HH, width*stride:width*stride+WW] += dx_tmp
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
        N, C, H, W = x.shape

        H_out = int(1 + (H - self.pool_size) / self.stride) # H'
        W_out = int(1 + (W - self.pool_size) / self.stride) # W'
        
        out = np.zeros((N, C, H_out, W_out)) # 최종적으로 구할 out 초기화 (N, F, H', W')

        for height in  range(H_out):
            for width in  range(W_out):
                x_crop = x[np.arange(N),  :, height*self.stride:height*self.stride+self.pool_size, width*self.stride:width*self.stride+self.pool_size]  # x에서 (N,C,pool_size,pool_size)크기만큼을 crop
                max_value = np.max(x_crop, axis=(2,3)) # N,C
                out[np.arange(N), :, height, width] = max_value
        return out

    def backprop(self, x, dLdy):
        N, C, H, W = x.shape
        N, C, H_out, W_out = dLdy.shape

        pool_size = self.pool_size ** 2
        dLdx = np.zeros((N,C,H,W))

        for height in range(H_out):
            for width in range(W_out):
                dmax_value = dLdy[np.arange(N), :, height, width].reshape((N,C,1)) # N,C,1
                
                x_crop = x[np.arange(N), :, height*self.stride: height*self.stride+self.pool_size, width*self.stride: width*self.stride+self.pool_size] # (N,C,pool_height,pool_width)
                x_crop_stretch = x_crop.reshape((N,C,pool_size)) # N,C,pool_size
                max_value = np.max(x_crop_stretch, axis=2).reshape((N,C,1)) # N,C,1
                
                dx_crop_stretch = np.where(x_crop_stretch==max_value, dmax_value , 0) # N,C,pool_size
                dx_crop = dx_crop_stretch.reshape((N,C,self.pool_size,self.pool_size))
                dLdx[np.arange(N), :, height*self.stride: height*self.stride+self.pool_size, width*self.stride: width*self.stride+self.pool_size] += dx_crop
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
        N = x.shape[0]
        D = self.W.shape[1]
        H = self.b.shape[0]

        out = x.reshape((N, D)) @ self.W.T + self.b # (N, H)
        return out

    def backprop(self,x,dLdy):
        # dLdy = (N,H)
        N = x.shape[0]
        D = self.W.shape[1]
        H = dLdy.shape[1]
        dLdb = np.sum(dLdy, axis=0) # (H,)
        dLdW = dLdy.T @ x.reshape((N, D)) # (H, D)
        dLdx = (dLdy @ self.W).reshape(x.shape) # (N, D)
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
        out = np.where(x>0, x, 0)
        return out
    
    def backprop(self, x, dLdy):
        dLdx = np.where(x>0, dLdy, 0)
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        N, O = x.shape
        scores_exp = np.exp(x) # (N, O), (20, 2)
        scores_sum = np.sum(scores_exp, axis=1) # (N, ), (20, )
        softmax = scores_exp / scores_sum.reshape(N, 1) # (N, O), (20,2)
        return softmax

    def backprop(self, x, dLdy):
        N = x.shape[0]
        scores_exp = np.exp(x) #(N, 2)
        scores_sum = np.sum(scores_exp, axis=1) #(N, )
        sum_loss = - 1 / (scores_sum ** 2)
        dLdx = scores_exp * (sum_loss * dLdy.sum(axis=1)).reshape((N,1)) #(sum쪽은 전부 내려가야함!!)
        dLdx += (1 / scores_sum).reshape((N,1)) * dLdy # (N, 2)
        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        N = x.shape[0]
        b_i = y.reshape(N) #(N, )
        before_log = x[np.arange(N),b_i] # (N, )
        log = np.log(before_log)
        cross_entropy_loss = -1 * np.sum(log)
        cross_entropy_loss /= N
        return cross_entropy_loss

    def backprop(self, x, y):
        N = x.shape[0]
        dLdx = np.zeros_like(x)
        b_i = y.reshape(N)
        before_log = x[np.arange(N), b_i] # (N, )
        loss_before_log = -1 / before_log # (N, )
        dLdx[np.arange(N),b_i] = loss_before_log / N
        return dLdx # (N,2)
