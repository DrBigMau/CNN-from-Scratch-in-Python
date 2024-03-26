# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:22:57 2024

@author: MMH_user
"""

import numpy as np

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
###############################################################################
#
###############################################################################
class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

###############################################################################
#
###############################################################################
class Activation_Softmax:
    
    def forward(self, inputs):
        exp_values    = np.exp(inputs - np.max(inputs))
        probabilities = exp_values/np.sum(exp_values, axis = 1, keepdims = True)
        
        self.output   = probabilities

###############################################################################
#
###############################################################################
class Loss:
    
    def calculate(self, output, y):
        #output: is comming from the softmax layer
        #y     : class vector/matrix
        
        sample_losses = self.forward(output,y)
        data_loss     = np.mean(sample_losses)
        
        return(data_loss)
###############################################################################
#
###############################################################################
class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, prob_pred, y_true):
        
        s              = prob_pred.shape
        Nsamples       = s[0]
        
        y_pred_clipped = np.clip(prob_pred, 1e-7, 1 - 1e-7)
        
        #be careful with (600,) vs (600,1)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(Nsamples),y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped* y_true, axis = 1)
            
        negative_log_likelihoods =  -np.log(correct_confidences)
        
        return(negative_log_likelihoods)

















