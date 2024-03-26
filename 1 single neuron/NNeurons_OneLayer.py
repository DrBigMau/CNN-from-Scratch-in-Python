# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 02:47:11 2024

@author: MMH_user
"""

import numpy as np

def NNeurons_OneLayer(inputs, n_neurons):
    
    l = len(inputs)
    
    weights = np.random.rand(n_neurons,l)
    bias    = np.random.rand(n_neurons,1)
    
    out     = np.dot(weights,inputs) + bias
    
    return(out)