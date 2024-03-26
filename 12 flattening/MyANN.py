# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:22:57 2024

@author: MMH_user
"""
import random          # for picking images randomly 
import numpy as np
import glob as gl      # for ls like in linux
from PIL import Image  # for resizing images



class Read_Scale_Imds:
    
    def __init__(self, minibatch_size, ANN_size):
        
        self.xANN = ANN_size[0]
        self.yANN = ANN_size[1]
        
        self.minibatch_size = minibatch_size
        
        my_path   = r'C:\Users\MMH_user\Desktop\QBM\QBM\courses\Python\ANN\pet pics'
        form      = '.jpg'

        path_dogs = my_path + '\Dog\\' + '*' + form
        path_cats = my_path + '\Cat\\' + '*' + form

        Cats      = gl.glob(path_cats)
        Dogs      = gl.glob(path_dogs)
        
        self.Ld   = len(Dogs)
        self.Lc   = len(Cats)
        
        D         = np.zeros((self.Ld))
        C         = np.zeros((self.Lc)) + 1
        
        self.All_classes = np.array((np.hstack((D,C))))
        self.All_imds    = np.hstack((Dogs,Cats))
        
    
    def Read_Scale(self):
        
        xANN = self.xANN
        yANN = self.yANN
        
        minibatch_size = self.minibatch_size
        
        idx  = random.sample(range(self.Ld + self.Lc), minibatch_size)
        
        classes = self.All_classes[idx]
        imds    = self.All_imds[idx]
        
        
        ImdsMatrix = np.zeros((xANN, yANN, 3, minibatch_size))
        
        for i in range(minibatch_size):
            I                   = Image.open(imds[i])
            Ire                 = I.resize((xANN, yANN))
            Iar                 = np.array(Ire)
            
            if len(Iar.shape) != 3:
                I3D        = np.zeros((xANN, yANN, 3))
                I3D[:,:,0] = Iar
                I3D[:,:,1] = Iar
                I3D[:,:,2] = Iar
                Iar        = I3D
            
            ImdsMatrix[:,:,:,i] = Iar
            
        ImdsMatrix.astype(np.uint8)
        
        return(ImdsMatrix, classes)
            
###############################################################################
###############################################################################
class ConvLayer:
    
    def __init__(self, xKernShape = 3, yKernShape = 3, Kernnumber = 10):
        self.xKernShape = xKernShape
        self.yKernShape = yKernShape
        self.Kernnumber = Kernnumber
        
        #creating kernels
        self.weights = 10*np.random.rand(xKernShape, yKernShape, Kernnumber)
        self.biases  = np.zeros((1,Kernnumber))
        
    def forward(self, M, padding = 0, stride = 1):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        xK = self.xKernShape
        yK = self.yKernShape
        NK = self.Kernnumber
        
        b  = self.biases
        
        W  = np.nan_to_num(self.weights)
        
        #we are going to need for backprop
        self.padding = padding
        self.stride  = stride
        
        xOutput = int((xImgShape - xK + 2*padding)/stride + 1)
        yOutput = int((yImgShape - yK + 2*padding)/stride + 1)
        
        output  = np.zeros((xOutput, yOutput, numChan, NK, numImds))
        
        imagePadded = np.zeros((xImgShape + 2*padding, yImgShape + 2*padding,\
                                numChan, NK, numImds))
        
        for k in range(NK):
            imagePadded[int(padding):int(padding + xImgShape),\
                    int(padding):int(padding + yImgShape), :, k, :] = M
                
        for i in range(numImds):
            currentIm_pad = imagePadded[:,:,:,:,i]
            for c in range(numChan):
                for k in range(NK):
                    for y in range(yOutput):
                        for x in range(xOutput):
                    
                            #finding corners of current slice
                            y_start = y*stride
                            y_end   = y_start + yK
                            x_start = x*stride
                            x_end   = x_start + xK
                    
                            current_slice = currentIm_pad[x_start:x_end,\
                                                          y_start:y_end,c,k]
                            s            = np.multiply(current_slice, W[:,:,k])
                            
                            output[x, y, c, k, i] = np.sum(s) + b[0,k].astype(float)
                            
        output = output.sum(2)
                    
        self.output = np.nan_to_num(output)
        self.input  = M
        self.impad  = imagePadded
                   
                    
                   
###############################################################################
###############################################################################
class Max_Pool:
    
    def forward(self, M, stride = 1, KernShape = 2):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        self.inputs = M
        
        xK = KernShape
        yK = KernShape

        xOutput = int((xImgShape - xK)/stride + 1)
        yOutput = int((yImgShape - yK)/stride + 1)
        
        imagePadded = M
        
        output  = np.zeros((xOutput, yOutput, numChan, numImds))

        imagePadded_copy = imagePadded.copy() * 0
        
        for i in range(numImds):
            currentIm_pad = imagePadded[:,:,:,i]
            for c in range(numChan):
                for y in range(yOutput):
                    for x in range(xOutput):
                
                        #finding corners of current slice
                        y_start = y*stride
                        y_end   = y_start + yK
                        x_start = x*stride
                        x_end   = x_start + xK
                        
                        sx = slice(x_start, x_end)
                        sy = slice(y_start, y_end)
                        
                        current_slice      = currentIm_pad[sx,sy,c]
                        slice_max          = float(current_slice.max())
                        output[x, y, c, i] = slice_max
                        
                        #keeping track of those pixel which contribute to the max
                        #for backprop
                        
                        imagePadded_copy[sx, sy, c, i] += np.equal(\
                                    currentIm_pad[sx,sy,c], slice_max).astype(float)
                            
        mask = imagePadded_copy
        mask = np.matrix.round(mask/(mask + 1e-7))
                        
        self.xKernShape = xK
        self.yKernShape = yK
        self.output     = output
        self.mask       = mask
        self.stride     = stride
        
###############################################################################
###############################################################################
class Flat:
    
    def forward(self, M):
        
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        
        self.input = M
        
        #length of vector (is one image, i.e. one data point with L features)
        L = xImgShape * yImgShape * numChan
        
        output = np.zeros((numImds, L))
        
        for i in range(numImds):
            output[i,:] = M[:,:,:,i].reshape((1,L))
            
        self.output = output

###############################################################################
###############################################################################

class Layer_Dense:
    
    def __init__(self, n_inputs,n_neurons):
        
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases  = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dinputs  = np.dot(dvalues,self.weights.T)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        
        
###############################################################################
###############################################################################
class Sigmoid:
    
    def forward(self, M):
        sigm        = np.clip(1/(1 + np.exp(-M)), 1e-7, 1 - 1e-7)
        self.output = sigm
        self.inputs = M
        
###############################################################################
###############################################################################
class Tanh:
    
    def forward(self, M):
        tanh        = np.tanh(M)
        self.output = tanh
        self.inputs = M
        
###############################################################################
###############################################################################

class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        self.inputs  = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#!!!
        
        
###############################################################################
###############################################################################

class Activation_Softmax:
    
    def forward(self, inputs):
        
        exp_values    = np.exp(inputs - np.max(inputs))
        probabilities = exp_values/np.sum(exp_values, axis = 1, \
                                          keepdims = True)
        self.output   = probabilities
        
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            
            single_output = single_output.reshape(-1,1)
            
            jacobMatr = np.diagflat(single_output) - \
                        np.dot(single_output,single_output.T)
            
            self.dinputs[i] = np.dot(jacobMatr,single_dvalues)
        
###############################################################################
###############################################################################

class Loss:
    
    def calculate(self, output, y):
        #output is the output from the softmax layer
        #vector/matrix of the actual class from the training data
        
        sample_losses = self.forward(output,y)
        data_loss     = np.mean(sample_losses)
        
        return(data_loss)
        
###############################################################################
###############################################################################

class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, prob_pred, y_true):
        
        Nsamples = len(prob_pred)
        y_pred_clipped = np.clip(prob_pred, 1e-7, 1-1e-7)
        
        #checking if we have one hot or sparse
        if len(y_true.shape) == 1:#sparse [0,2,3,1,1,0]
            correct_confidences = y_pred_clipped[range(Nsamples),y_true]
            
        elif len(y_true.shape) == 2:# one hot [[0,1,0], [1,0,0]]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return(negative_log_likelihoods)
    
    
    def backward(self, dvalues, y_true):
        
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 1:
            Nlabels = len(dvalues[0])
            y_true  = np.eye(Nlabels)[y_true]
        
        self.dinputs = - y_true/dvalues/Nsamples
        
###############################################################################
###############################################################################

class CalcSoftmaxLossGrad:
    
    def __init__(self):
        
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        #inputs comes from the last dense layer (output) --> needed for 
        #softmax
        #y_true: vector/matrix of the true classes/labels
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return(self.loss.calculate(self.output, y_true))
    
    def backward(self, dvalues, y_true):
        #output from softmax layer
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        #calculating the gradient
        self.dinputs[range(Nsamples),y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
        
###############################################################################
###############################################################################

class Optimizer_SGD:
    
    def __init__(self, learning_rate = 0.1, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.decay                 = decay
        self.current_learning_rate = learning_rate
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate* \
                (1 / (1 + self.iterations * self.decay))
        
    def update_params(self, layer):
        
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums -\
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums -\
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else: 
            weight_updates = -self.current_learning_rate* layer.dweights
            bias_updates   = -self.current_learning_rate* layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1








