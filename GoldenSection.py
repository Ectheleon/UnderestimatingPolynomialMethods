#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:56:55 2019

@author: Jonathan Grant-Peters
"""

import numpy as np

def goldenSection(bracket, xInit, fbracket, fInit, f, eps=1e-5, TrustUser=True):
    
    R = 0.61803399
    C = 1-R
    
    if not TrustUser:
        f0, f3 = fbracket
        
        if f0 < fInit or f3 < fInit:
            raise ValueError('Invalid input for Golden Section. The central point should'+
                 ' be the current minimum')
    
    
    x0, x3 = bracket if bracket[0] < bracket[1] else bracket[::-1]
    
    Accuracy = [x3 - x0]
    UBS = [x3]
    LBS = [x0]
    niter = 0

    if xInit - x0 > x3 - xInit:
        x2 = xInit
        x1 = xInit - C*(xInit - x0)       
    else:
        x1 = xInit
        x2 = xInit +C* (x3 - xInit)

    f1 = f(x1)
    f2 = f(x2)
    
    while Accuracy[-1] > eps:
        #print(x0, x1, x2, x3)
        
        if f2<f1:
            x0, x1, x2 = x1, x2, R*x2 + C*x3
            f0,f1, f2, = f1, f2, f(x2)
        else:
            x3, x2, x1 = x2, x1, R*x1 + C*x0
            f3, f2, f1 = f2, f1, f(x1)
        
        #print(f0,f1,f2,f3)
        
        Accuracy.append(x3 - x0)
        UBS.append(x3)
        LBS.append(x0)
        niter += 1
 
    return x1 if  f1<f2 else x2, niter, np.array(Accuracy), np.array(LBS), np.array(UBS)
