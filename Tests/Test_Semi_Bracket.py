#!/usr/bin/env python3
# -*- coding: utf-8 -*-cocoa powder
"""
Created on Wed Mar 27 16:59:55 2019

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


import Optimize1D_saved as opt
importlib.reload(opt)
import NSLS_merit_fctn as mt
importlib.reload(mt)

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tmp = lst.SS1()
f = tmp.func

# 2. Choose starting grids
xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR
xR[1] = 9
fR[1] = f(xR[1])
xR[2] = 13
fR[2] = f(xR[2])

# Record the true solution
ans = tmp.xtrue

# Identify the starting bracket
bracket = np.array([np.max(xL[0]), np.min(xR[0])])
fbracket = np.array([f(x) for x in bracket])

# 3. Set up the optimiser
alg1 = opt.GenBrentSimple(xL, xR, fL, fR, xM, fM, Lipschitz = tmp.M, eps=1e-5, delta=1e-6)


algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, delta=1e-6)

#%%

alg1.solve(f)




#%%
#algbrent.solve(f)

_,_,gsacc,_,_ = GoldenSection.goldenSection(bracket, fbracket, xM, fM, f, eps=1e-5)

# 5. Plot convergence rate

plt.semilogy(alg1.acc, label='Generalised Brent 1')
#plt.semilogy(alg2.acc, label='Generalised Brent 2')

plt.semilogy(algbrent.acc, label='Brent')
plt.semilogy(gsacc, label='Golden Section')

plt.xlabel('Number of Function Evaluations')
plt.ylabel('Length of Bracket')
plt.legend()

# 6. Observe plot and note the bottleneck. For AN1, it occurs at iteration 9

#%%

alg = opt.GenBrent(xL, xR, fL, fR, xM, fM, limited_mem=False, Lipschitz = tmp.M, use_uncertainty = True, ptype = 4)

dat = alg.semibracket_solve(f)

#%%

acc1 = dat[:, 2] - dat[:,0]
acc2 = np.fabs(dat[:,3] - ans)
acc3 = np.fabs(dat[:,4] - ans)
acc4 = np.fabs(dat[:,5] - ans)

#%%

plt.semilogy(acc1, label='Bracket')
plt.semilogy(acc2, label='Conservative')
plt.semilogy(acc3, label='Left')
plt.semilogy(acc4, label='Right')
plt.legend()



#%%

plt.semilogy(acc1, label='Bracket')
plt.semilogy(acc2, label='Conservative')
plt.semilogy(acc3, label='Left')
plt.semilogy(acc4, label='Right')

plt.semilogy(alg1.acc, label='Generalised Brent 1')
plt.semilogy(alg2.acc, label='Generalised Brent 2')

plt.semilogy(algbrent.acc, label='Brent')
plt.semilogy(gsacc, label='Golden Section')

plt.xlabel('Number of Function Evaluations')
plt.ylabel('Length of Bracket')
plt.legend()



#%%



#%%


























