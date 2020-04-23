#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:53:56 2020

@author: samar
"""
import numpy as np

class NeuralNetwork:
    
    def __init__(self,input_nodes=0,num_layers=0):
        self.input=0
        self.layers=num_layers
        #self.weights=[]
        
        
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
        
class Maxnet(NeuralNetwork):
    
    def __init__(self,input_nodes=0,weight=-0.1):
        NeuralNetwork.__init__(self,input_nodes,1)
        self.weights=weight*np.ones([input_nodes,input_nodes])
        self.weights+=(1-weight)*np.eye(input_nodes)
        
class HemmingNet(NeuralNetwork):
    
    def __init__(self,input_nodes=0,output_nodes=0):
        NeuralNetwork.__init__(self,input_nodes,2)
        self.ouput=output_nodes
        self.weights=np.zeros([output_nodes,input_nodes+1])
        
class SimpleCompetitive(NeuralNetwork):
    
    def __init__(self,input_nodes=0,output_nodes=0):
        self.Hemming=HemmingNet(input_nodes,output_nodes)
        self.out=Maxnet(output_nodes)