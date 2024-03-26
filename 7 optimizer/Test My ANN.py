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


S = x.shape

n_neu1 = 64
n_neu2 = 3

from MyANN import *

dense1        = Layer_Dense(S[1], n_neu1)
dense2        = Layer_Dense(n_neu1, n_neu2)
activation1   = Activation_ReLU()
loss_function = CalcSoftmaxLossGrad()
optimizer     = Optimizer_SGD()

N       = 10000
Monitor = np.zeros((N,2))

for epoch in range(N):
    
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_function.forward(dense2.output, y)
    
    predictions = np.argmax(loss_function.output, axis = 1)
    
    if len(y.shape) == 2:
            y = np.argmax(y,axis = 1)
            
    accuracy = np.mean(predictions == y)
    
    loss_function.backward(loss_function.output, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
    Monitor[epoch,0] = accuracy *100
    Monitor[epoch,1] = loss
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'accuracy: {accuracy: .3f} ' +
              f'loss: {loss: .3f}')
    

fig, ax = plt.subplots(2,1, sharex= True)
ax[0].plot(np.arange(N),Monitor[:,0]) 
ax[0].set_ylabel('accuracy [%]')
ax[1].plot(np.arange(N),Monitor[:,1]) 
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
plt.xscale('log',base=10)




