# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:38:04 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

[x, y] = spiral_data(samples = 200, classes = 3)

#plotting the data
idx0 = np.argwhere(y == 0)
idx1 = np.argwhere(y == 1)
idx2 = np.argwhere(y == 2)

plt.scatter(x[idx0,0], x[idx0,1], c = 'black')
plt.scatter(x[idx1,0], x[idx1,1], c = 'blue')
plt.scatter(x[idx2,0], x[idx2,1], c = 'red')



I = [[1,-2,-5,7], [4,-5,7,-8]]
I = np.array(I)
S = I.shape

n_neu1 = 5
n_neu2 = 6

from MyANN import *

dense1      = Layer_Dense(S[1], n_neu1)
dense2      = Layer_Dense(n_neu1, n_neu2)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

dense1.forward(I)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
out    = activation2.output








