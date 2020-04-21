#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:16:25 2020

@author: samar
"""

import numpy as np

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def CostFunction(theta, X, y):
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
 
def CostFunctionReg(theta, X, y, L):
    """ compute regularized cost function and gradient """
    
    m = len(y) # number of training examples
    n=len(theta)
    theta=np.reshape(theta,[n,1])

    (J,grad)=CostFunction(theta,X,y)
    
    theta2=np.copy(theta)
    theta2[0]=0
    
    J+=L*np.linalg.norm(theta2)**2/(2*m)
    grad+=L*theta2/m
    
    return (J,grad)