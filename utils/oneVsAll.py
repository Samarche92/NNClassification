#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:18:09 2020

@author: samar
"""
import numpy as np
#from scipy.optimize import minimize
from .fmincg import fmincg

def oneVsAll(X, y, num_labels, L, cost_function):
    (m,n)=np.shape(X)
    all_theta=np.zeros([num_labels,n+1])
    # Add ones to training set for the intercept term
    X=np.column_stack((np.ones([m,]),X)) 
    
    #Set Initial theta
    initial_theta = np.zeros([n + 1,])

    for c in range(num_labels):      
        #res=minimize(lambda t : cost_function(t,X,(y==c+1).astype(float),L)[0],
         #            np.ndarray.flatten(initial_theta),
          #           jac=lambda t : np.ndarray.flatten(cost_function(t,X,(y==c+1).astype(float),L)[1]),
           #          options={'disp':True,'maxiter':50})

        res=fmincg(lambda t : cost_function(t,X,(y==c+1).astype(float),L),initial_theta,
                   MaxIter=50)
         
        theta=res[0]
        #theta=res.x
        all_theta[c,:]=np.reshape(theta,[n+1])

    return all_theta

def predictOneVsAll(all_theta,X):
    m=len(X)
    p=np.zeros([m,1])
    X=np.column_stack((np.ones([m,]),X)) 
    reg=np.matmul(X,np.transpose(all_theta))

    p=np.argmax(reg,axis=1)
    p+=np.ones_like(p)
    
    return p