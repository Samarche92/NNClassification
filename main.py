#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:19:56 2020

@author: samar
"""

import numpy as np
from utils import *

#Load data (handwritten images)
# Each example is a 20x20 pixel image
X=np.loadtxt("data.txt",delimiter='\t') 
y=np.loadtxt("labels.txt",delimiter='\t')
m=len(X) #number of training examples

# Randomly select 100 data points to display
rand_indices = np.random.permutation(np.arange(m))
sel = X[rand_indices[0:100], :]

#display data
displaydata(sel)
