#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:02:30 2020

@author: samar
"""

import unittest
from .FeedForwardClass import FeedForward

class FeedForwardTest(unittest.TestCase):
    
    """ Test case used to test functions of FeedForwardClass"""
    
    def test_AddHiddenNodes(self):
        """ Tests function Add Hidden Nodes """
        FF=FeedForward(3,4,3)
        A_long=[3,3,3]
        A_float=[2.1,2]
        A=[5,5]
        
        with self.assertRaises(ValueError):
            FF.AddHiddenNodes(A_long)
        
        with self.assertRaises(ValueError):
            FF.AddHiddenNodes(A_float)
            
        FF.AddHiddenNodes(A)
        self.assertEqual(FF.hidden_nodes,A)
        