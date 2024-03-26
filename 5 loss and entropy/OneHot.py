# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 00:59:33 2024

@author: MMH_user
"""

import numpy as np

def OneHot(Seq, *Nclass):
    #one hot encoder: turns the numerical vector Seq into a one hot encoded
    #matrix
    if not Nclass:
        Nclass = int(np.max(Seq)) + 1
    else:
        for i in Nclass:
            Nclass = i
    
    Seq = np.array(Seq,dtype=(int))
    M   = np.zeros((len(Seq),Nclass))
    
    for i,j in enumerate(Seq):
        M[i,j] = 1
    return(M)