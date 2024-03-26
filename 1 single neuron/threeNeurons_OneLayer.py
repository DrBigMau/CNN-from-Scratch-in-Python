# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 02:42:58 2024

@author: MMH_user
"""

import numpy as np

def threeNeurons_OneLayer(inputs):
    
    l = len(inputs)
    
    weights = np.random.rand(3,l)
    bias    = np.random.rand(3,1)
    
    out     = np.dot(weights,inputs) + bias
    
    return(out)