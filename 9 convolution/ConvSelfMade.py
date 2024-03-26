# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:14:37 2024

@author: MMH_user
"""

import matplotlib.pyplot as plt
import numpy as np

def ConvSelfMade(*Image, K, padding = 0, stride = 1):
    
    my_path = r'C:\Users\MMH_user\Desktop\QBM\QBM\courses\Python\ANN\pet pics'
    form    = '.jpg'
    
    if Image:
        for Image in Image:
            plt.imshow(Image)
            
    else:
        Image = plt.imread(my_path + '\Dog\\' + str(2) + form)
        plt.imshow(Image)
    
    #shape of input image
    [xImgShape , yImgShape, numChan] = Image.shape
    
    [xK, yK] = K.shape
    
    xOutput = int((xImgShape - xK + 2*padding)/stride + 1)
    yOutput = int((yImgShape - yK + 2*padding)/stride + 1)
    
    output  = np.zeros((xOutput, yOutput, numChan))
    
    imagePadded = np.zeros((xImgShape + 2*padding, yImgShape + 2*padding,\
                            numChan))
    imagePadded[int(padding):int(padding + xImgShape),\
                int(padding):int(padding + yImgShape),:] = Image
    
    
    for c in range(numChan):
        for y in range(yOutput):
            for x in range(xOutput):
                
                #finding corners of current slice
                y_start = y*stride
                y_end   = y_start + yK
                x_start = x*stride
                x_end   = x_start + xK
                
                current_slice = imagePadded[x_start:x_end, y_start:y_end, c]
                s             = np.multiply(current_slice, K)
                output[x,y,c] = np.sum(s)
        
    plt.imshow(output.sum(2), cmap = 'gray')
    plt.title('after convolution with padding'\
                 '=' + str(padding) + ' and  stride=' + str(stride))
    plt.show()
      
    return(output)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        