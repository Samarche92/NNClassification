#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:28:02 2020

@author: samar
"""

import numpy as np
import math

def fmincg(f, X, **options):
    """ 
    Translation to python of Andrew Ng's fmincg function in matlab.
    
    Minimize a continuous differentialble multivariate function. Starting point
    is given by X (D by 1), and the function f must return a function value 
    and a vector of partial derivatives. The function returns when either its 
    length is up, or if no further progress can be made (ie, we are at a 
    minimum, or so close that due to numerical problems, we cannot get any 
    closer). The function returns the found solution X, a vector of function 
    values fX indicating the progress made and i the number of iterations 
    (line searches or function evaluations, depending on the sign of "length") used.

     Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13


     (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
 
    Permission is granted for anyone to copy, use, or modify these
    programs and accompanying documents for purposes of research or
    education, provided this copyright notice is retained, and note is
    made of any changes that have been made.
 
    These programs and documents are distributed without any warranty,
    express or implied.  As the programs were written for research
    purposes only, they have not been tested to the degree that would be
    advisable in any important application.  All use of these programs is
    entirely at the user's own risk."""
   
    if 'MaxIter' in options:
        length=options['MaxIter']
    else:
        length=100
        
    # a bunch of constants for line searches
    RHO = 0.01      # RHO and SIG are the constants in the Wolfe-Powell conditions
    SIG = 0.5       
    INT = 0.1    # don't reevaluate within INT of the limit of the current bracket
    EXT = 3.0       # extrapolate maximum EXT times the current bracket
    MAX = 20        # max function evaluations per line search
    RATIO = 100
    
    i = 0     # zero the run length counter
    ls_failed = 0   # no previous line search has failed
    fX = []
    
    # get function value and gradient
    (f1,df1)=f(X)
    i += int(length<0)       # count epochs?!
    s = -df1       # search direction is steepest
    d1=-np.dot(s,s) #this is the slope  
    red=1.0
    z1 = red/(1.0-d1)    # initial step is red/(|s|+1)
    
    while i<abs(length): #while not finished
        i+=int(length>0) #count interations
        #copy current values
        X0=X
        f0=f1
        df0=df1
        #begin line search
        X=X+z1*s
        (f2,df2)=f(X)
        i += int(length<0)       # count epochs?!
        d2= np.dot(df2,s)
        #initialize point 3 equal to point 1
        f3=f1
        d3=d1
        z3=-z1
        if length>0:
            M=MAX
        else:
            M=min(MAX,-length-i)
        #initialize quantities
        success=False
        limit=-1
        
        while 1:
            while ((f2>f1+z1*RHO*d1) or ((d2>-SIG*d1) and (M>0))):
                limit=z1 #tighten the bracket
                if f2>f1:
                    z2=z3-0.5*d3*z3*z3/(d3*z3+f2-f3) #quadratic fit
                else:
                    A=6.0*(f2-f3)/z3+3.0*(d2+d3) #cubic fit
                    B=3.0*(f3-f2)-z3*(d3+2.0*d2)
                    z2=(math.sqrt(B*B-A*d2*z3*z3)-B)/A
                
                if math.isnan(z2) or math.isinf(z2): #if numerical problem then bisect
                    z2=z3/2
                    
                z2=max([min([z2,INT*z3]),(1.0-INT)*z3]) #don't accept too close to limits
                #update step
                z1+=z2 
                X+=z2*s
                (f2,df2)=f(X)
                M-=1
                i += int(length<0)       # count epochs?!
                d2=np.dot(df2,s)
                z3-=z2 #z3 is now relative to the location of z2
            
            if f2>f1+z1*RHO*d1 or d2>-SIG*d1:
                break
            elif d2>SIG*d1:
                success=True
                break
            elif M==0:
                break
            
            A=6.0*(f2-f3)/z3+3.0*(d2+d3) #make cubic extrapolation
            B=3.0*(f3-f2)-z3*(d3+2.0*d2)
            z2=-d2*z2*z3/(B+math.sqrt(B*B-A*d2*z3*z3))
            if (not z2*z2>=0) or math.isnan(z2) or math.isinf(z2) or z2<0:
                #num problem or wrong sign
                if limit<-0.5: #if no upper limit
                    z2=z1*(EXT-1) #extrapolate max amount
                else:
                    z2=(limit-z1)/2 #bisect
            elif (limit>-0.5) and (z2+z1>limit): #extrapolation beyond max
                z2=(limit-z1)/2 #bisect
            elif (limit<0.5) and (z2+z1>z1*EXT): #extrapolaition beyond limit
                z2=z1*(EXT-1) #set to extrapolation limit
            elif z2<-z3*INT:
                z2=-z3*INT
            elif limit>-0.5 and z2<(limit-z1)*(1-INT): #too close to limit?
                z2=(limit-z1)*(1-INT)
                
            #set point 3 equal to point 2
            f3=f2
            d3=d2
            z3=-z2
            #update current estimates
            z1+=z2
            X+=z2*s
            (f2,df2)=f(X)
            M-=1
            i += int(length<0)       # count epochs?!
            d2=np.dot(df2,s)
            ## end of line search
    
        if success:
            f1=f2
            fX.append(f1)
            print('Iteration {} | Cost: {} \n'.format(i,f1))
            s=(np.dot(df2,df2)-np.dot(df1,df2))/(np.dot(df1,df1)*s)-df2 #Polack-Ribiere direction
            df1,df2=df2,df1 #swap derivatives
            d2=np.dot(df1,s)
            if d2>0: #new slope must be negative
                s=-df1
                d2=-np.dot(s,s) #otherwise use steepest direction
            z1*=min([RATIO,d1/(d2-np.finfo(float).tiny)]) #slope ratio but max RATIO
            d1=d2
            ls_failed=False
        else:
            #restore point from before failed line search
            X=X0
            f1=f0
            df1=df0
            if ls_failed or i>abs(length): 
                #line search failed twice in a row or we ran out of time
                break #so give up
            #swap derivatives
            df1,df2=df2,df1
            s=-df1 
            #try steepest
            d1=-np.dot(s,s)
            z1=1.0/(1-d1)
            ls_failed=True          
    
    return (X,fX,i)