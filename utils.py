#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:38:13 2020

@author: samar
"""

import math
import numpy as np
from matplotlib import pyplot as plt 

def displaydata(X,**kwargs):

    if 'width' in kwargs.keys():
        width=kwargs['width']
    else:
        width=round(math.sqrt(np.shape(X)[1]))
    
    (m, n)=np.shape(X)
    
    #number of pixels in each image
    height=round(n/width)
    
    #number of images to display
    if 'disp_rows' in kwargs.keys():
        display_rows=kwargs['disp_rows']
    else:
        display_rows=round(math.sqrt(m))
        
    display_cols=round(m/display_rows)
    
    
    disp_arr=np.ones([display_rows*height,display_cols*width])
    
    curr_ex=0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex>=m:
                break
            
            disp_arr[i*height:(i+1)*height,j*width:(j+1)*width]=np.reshape(X[curr_ex,:],[height,width])
            curr_ex+=1
                
            if curr_ex>=m:
                break
    
    plt.pcolormesh(np.rot90(disp_arr),cmap='Greys')
    plt.show()

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def lrCostFunction(theta, X, y):
    """ compute cost function and gradient """
    
    m=len(X)
    n=len(theta)
    theta=np.reshape(theta,[n,1]) #reshaping into vector
    h=sigmoid(np.matmul(X,theta))
    J=-y*np.log(h)-(1-y)*np.log(1-h)
    J=np.sum(J)/m
    
    grad=np.matmul(np.transpose(X),h-y)
    grad=grad/m
    
    return (J,grad)
 
def lrCostFunctionReg(theta, X, y, L):
    """ compute regularized cost function and gradient """
    
    m = len(y) # number of training examples
    n=len(theta)
    theta=np.reshape(theta,[n,1])

    (J,grad)=lrCostFunction(theta,X,y)
    
    theta2=np.copy(theta)
    theta2[0]=0
    
    J+=L*np.linalg.norm(theta2)**2/(2*m)
    grad+=L*theta2/m
    
    return (J,grad)