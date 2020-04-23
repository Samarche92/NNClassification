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