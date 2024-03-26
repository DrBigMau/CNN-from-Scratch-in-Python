# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:38:04 2024

@author: MMH_user
"""
import numpy as np

I = [[1,-2,-5,7], [4,-5,7,-8]]
I = np.array(I)
S = I.shape

n_neu1 = 5
n_neu2 = 6

from MyANN import *

dense1      = Layer_Dense(S[1], n_neu1)
dense2      = Layer_Dense(n_neu1, n_neu2)
activation1 = Activation_ReLU()

dense1.forward(I)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
out = dense2.output
out.shape








