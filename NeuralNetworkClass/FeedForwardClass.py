#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:25:52 2020

@author: samar
"""
import numpy as np
from .NeuralNetworkClass import NeuralNetwork

class FeedForward(NeuralNetwork):
    """ Feed-forward neural networks used for classification. Inherits from 
    general Neural Network class"""
    
    def __init__(self,input_nodes=0,num_layers=0,output_nodes=0):
        """ Initialize the network with the number of input nodes, the number 
        of layers (=2+ number of hidden layers) and the number of output 
        nodes. """
        
        NeuralNetwork.__init__(self,input_nodes,num_layers)
        self.ouput=output_nodes
        self.hidden_nodes=[0]*(num_layers-2)
        self.weights=[]
        
    def AddHiddenNodes(self,A):
        """ Function used to set the number of nodes in each layer.
        A is a list of integers containing the number of nodes in each hidden
        layer (should be of length num_layers-2)."""
        
        if len(A)!=self.layers-2:
            raise ValueError("Dimension of input array should be {} but instead is {}"
                             .format(self.layers-2,len(A)))
            
        for (i,a) in enumerate(A):
            self.hidden_nodes[i]=a
            
    def InitializeWeights(self,W):
        """ Function used to initialize the weights for each layer.
        W is a list of matrices containing the initial weights for each 
        layer. W should be of length num_layers-2 and the size of each matrix
        is determined by the number of nodes in each layer."""
        
        if len(W)!=self.layers-2:
            raise ValueError("Dimension of input array should be {} but instead is {}"
                             .format(self.layers-2,len(W)))    
        self.weights=W #will check each matrix size size later