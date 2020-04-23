#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:25:52 2020

@author: samar
"""
import numpy as np
from .NeuralNetworkClass import NeuralNetwork

class FeedForward(NeuralNetwork):
    
    def __init__(self,input_nodes=0,num_layers=0,output_nodes=0):
        NeuralNetwork.__init__(self,input_nodes,num_layers)
        self.ouput=output_nodes
        self.hidden_nodes=[0]*(num_layers-2)
        self.weights=[]
        
    def AddHiddenNodes(self,A):
        """ Function used to set the number of nodes in each layer.
        A is an array of integers conatining the number of nodes in each hidden
        layer (should be of length num_layers-2)."""
        if len(A)==self.layers-2:
            for (i,a) in enumerate(A):
                self.hidden_nodes[i]=a
        else:
            raise ValueError("Dimension of input array should be {} but instead is {}"
                             .format(self.layers-2,len(A)))
            
    def InitializeWeights(self,W):