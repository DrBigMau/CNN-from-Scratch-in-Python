# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:05:02 2024

@author: MMH_user
"""
import matplotlib.pyplot as plt
from MyANN import *

read_and_scale = Read_Scale_Imds(5, [100,100])
MP1            = Min_Pool()

[M, C] = read_and_scale.Read_Scale()

Conv1  = ConvLayer(3,3,5)

Conv1.forward(M,0,1)
MP1.forward(Conv1.output, 2, 3)
MP1.backward(MP1.output)

I1 = MP1.inputs[:,:,1,4]
I2 = MP1.output[:,:,1,4]
I3 = MP1.mask[:,:,1,4]

D  = MP1.dinputs[:,:,1,4]

plt.imshow(I1) 
plt.show()
plt.imshow(I2)
plt.show()
plt.imshow(I3)
plt.show()
plt.imshow(D)
plt.show()