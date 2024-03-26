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









