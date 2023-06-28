#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:04:45 2019

@author: Jonathan GP
"""

import autograd.numpy as np
import importlib
import matplotlib.pyplot as plt
import sys
import scipy.optimize as so
sys.path.append('/home/peters/Documents/Projects/ResearchProject/MyCode/ContinuousOptimiser/python/')

sys.path.append('/home/user/Documents/Projects/ResearchProject/MyCode/ContinuousOptimiser/python/')
#import ExactLineSearch as els
import cvxopt as cvx
import qpsolvers
#import mosek
import autograd
import matplotlib.ticker as ticker


import LStstfn as lst
import GoldenSection 


import Optimize1D as opt
importlib.reload(opt)
import NSLS_merit_fctn as mt
importlib.reload(mt)

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tmp = lst.AS2()
f = tmp.func
# 2. Choose starting grids
xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR

# Record the true solution
ans = tmp.xtrue

# Identify the starting bracket
bracket = np.array([np.max(xL[0]), np.min(xR[0])])
fbracket = np.array([f(x) for x in bracket])

# 3. Set up the optimiser
alg0 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0 )
alg1 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.1)
alg2 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.5 )
alg3 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 1)
alg4 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 5 )
alg5 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 1e5 )


algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, delta=1e-6)


xx = np.arange(xL[-1], xR[-1], 0.01)
yy = np.array([f(x) for x in xx])
plt.plot(xx,yy)
plt.show()

#%%
#alg0.naive_solve(f)
alg1.naive_solve(f, maxiter=14)
#alg2.naive_solve(f)
#alg3.naive_solve(f)
#alg4.naive_solve(f)
#alg5.naive_solve(f)

#algbrent.solve(f)
#%%

xL1 = alg1.xL
xR1 = alg1.xR

pR1 = opt.underestimate_interp(xR1, f(xR1), 0.1)
pL1 = opt.underestimate_interp(xL1, f(xL1), 0.1)
guess1 = opt.min_max_poly(pL1, pR1, xL1[0], xR1[0])

#%%
alg1 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.1)
alg1.naive_solve(f, maxiter=15)

#%%
xL2 = alg1.xL
xR2 = alg1.xR

pR2 = opt.underestimate_interp(xR2, f(xR2), 0.1)
pL2 = opt.underestimate_interp(xL2, f(xL2), 0.1)
guess2 = opt.min_max_poly(pL2, pR2, xL2[0], xR2[0])

#%%

alg1 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.1)
alg1.naive_solve(f, maxiter=16)

#%%
xL3 = alg1.xL
xR3 = alg1.xR

pR3 = opt.underestimate_interp(xR3, f(xR3), 0.1)
pL3 = opt.underestimate_interp(xL3, f(xL3), 0.1)
guess3 = opt.min_max_poly(pL3, pR3, xL3[0], xR3[0])


#%%
print(guess1, guess2, guess3)


#%%


