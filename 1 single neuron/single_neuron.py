# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 01:20:05 2024

@author: MMH_user
"""

import numpy as np

def single_neuron(inputs):
    
    l = len(inputs)
    
    weights = np.random.rand(1,l)
    bias    = np.random.rand(1,1)
    
    out     = np.dot(weights,inputs) + bias
    
    return(out)