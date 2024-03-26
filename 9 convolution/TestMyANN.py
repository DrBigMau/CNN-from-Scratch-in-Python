# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:05:02 2024

@author: MMH_user
"""
from MyANN import *


read_and_scale = Read_Scale_Imds(50, [100,100])
[M, C] = read_and_scale.Read_Scale()

Conv1 = ConvLayer(3,3,10)
Conv2 = ConvLayer(5,5,8)

Conv1.forward(M,0,1)
Conv2.forward(Conv1.output,2,4)





