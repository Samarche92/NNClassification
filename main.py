#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:19:56 2020

@author: samar
"""

import numpy as np
from utils import LogReg as lr
from utils import utils
from utils import oneVsAll as one

#Load data (handwritten images)
# Each example is a 20x20 pixel image
X=np.loadtxt("data.txt",delimiter='\t') 
y=np.loadtxt("labels.txt",delimiter='\t')
m=len(X) #number of training examples
y=np.reshape(y,[m,1])

# Randomly select 100 data points to display
rand_indices = np.random.permutation(np.arange(m))
sel = X[rand_indices[0:100], :]

#display data
utils.displaydata(sel)

wait = input("Program paused. Press enter to continue \n")

print('Testing lrCostFunction() with regularization')

theta_t = np.reshape(np.array([-2, -1, 1, 2]),[4,1])
X_t=np.ones([5,4])
X_t[:,1:]=np.transpose(np.reshape(np.arange(1,16),[3,5])/10)
y_t = np.reshape(np.array([1,0,1,0,1]),[5,1])
lambda_t = 3
(J, grad) = lr.CostFunctionReg(theta_t, X_t, y_t, lambda_t)

print('Cost: {}\n'.format(J))
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

wait = input("Program paused. Press enter to continue \n")

print('Training One-vs-All Logistic Regression...\n')

L = 0.1
all_theta = one.oneVsAll(X, y,10,L,lr.CostFunctionReg)

wait = input("Program paused. Press enter to continue \n")

pred=one.predictOneVsAll(all_theta,X)

print('Training Set Accuracy: {}\n'.format(np.mean((pred == y).astype(float))*100))