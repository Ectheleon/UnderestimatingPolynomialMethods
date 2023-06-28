#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:33:46 2019

@author: Jonathan GP
"""

import numpy as np

def U1(xn, gxn):
    return np.array([xn[0], xn[1], gxn, xn[2], xn[3]])

def U2(xn, gxn):
    return np.array([xn[1], xn[2], gxn, xn[3], xn[4]])

def U3(xn, gxn):
    return np.array([xn[0], xn[1], xn[2], gxn, xn[3]])

def U4(xn, gxn):
    return np.array([xn[1], gxn, xn[2], xn[3], xn[4]])

def g(x):
    return (x[3]*x[4]-x[0]*x[1])/(x[3]+x[4]-x[0]-x[1])

def b(x):
    return x[3]-x[1]

def r(x):
    return x[4] - x[1]

def l(x) :
    return x[3] - x[0]

def h(x):
    return x[2] - x[1]

def Utype(xn, gxn, fM, fgxn):
    if fgxn<fM:
        if gxn < xn[2]:
            return 1
        else:
            return 2
    else:
        if gxn < xn[2]:
            return 4
        else:
            return 3


def U(xn, gxn, fM, fgxn):
    if fgxn<fM:
        if gxn < xn[2]:
            return U1(xn, gxn), fgxn
        else:
            return U2(xn, gxn), fgxn
    else:
        if gxn < xn[2]:
            return U4(xn, gxn), fM
        else:
            return U3(xn, gxn), fM
        
def isBracket(x, fx):
    ordering = x[1]>x[0] and x[2]>x[1]
    minimum = fx[1]<fx[0] and fx[1] < fx[2]
    return ordering and minimum


def printMessage(i, x, xold=None, utype=None):
    if xold is None:
        print('{:6d} | {:.6e} | {:.6e} | {:.6e} | {:.6e}'.format(i,b(x),r(x), l(x), h(x)))
    else:
        print('{:6d} | {:.6e} | {:.6e} | {:.6e} | {:.6e} | {:.6e} '.format(i,b(x),r(x), l(x), h(x), b(x)/b(xold))+'U'+str(utype))

def dat(x, utype):
    return np.array([b(x), r(x), l(x), h(x), utype])

def UPML(xn, fM, f, tol, log=True):
    
    i=0
    printMessage(i, xn)
    
    if log:
        data = [dat(xn,-1)]
    
    while b(xn) > tol:
        gxn = g(xn)
        fgxn = f(gxn)

        #print(xn, gxn, fgxn, fM)
        xold = xn
        utype = Utype(xn, gxn, fM, fgxn)
        xn, fM = U(xn, gxn, fM, fgxn)
        i+=1
        printMessage(i,xn, xold, utype)
        data.append(dat(xn, utype))
        
    if log:
        return xn, fM, np.array(data)
    else:
        return xn, fM
        
        
        