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
import GoldenSection as gs


import Optimize1D as opt
importlib.reload(opt)
import NSLS_merit_fctn as mt
importlib.reload(mt)

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tmp = lst.AN3()
f = tmp.func

# 2. Choose starting grids
xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR


M = tmp.M
# Record the true solution
ans = tmp.xtrue

i = 0
M = 10


#%%

tmp = lst.AN3()
f = tmp.func

# 2. Choose starting grids
xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR


M = tmp.M
# Record the true solution
ans = tmp.xtrue


print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xL0' , 'xM', 'xR0', 'xnew', 'bracket'))
print("-"*(20+14*5))


Niters = 20

bracket = np.zeros([Niters])
xMhist = np.zeros([Niters])
xnewhist = np.zeros([Niters])

for i in range(Niters):
   
   
   pL = opt.underestimate_interp(xL, fL, M)
   pR = opt.underestimate_interp(xR, fR, M)
   
   
   
   xnew = opt.min_max_poly(pL, pR, xL[0], xR[0])
   
   xMhist[i] = xM
   xnewhist[i] = xnew
   bracket[i] = xR[0] - xL[0]
   
   
   fnew = f(xnew)
   
   print("%10d |%14.10f |%14.10f |%14.10f |%14.10f |%14.10f" %(i,  xL[0], xM, xR[0], xnew, xR[0] - xL[0]))
   
   xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)


#%% Set up problem

# 1. Choose test function 
tmp = lst.AN3()
f = tmp.func

# 2. Choose starting grids
xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR


M = tmp.M
# Record the true solution
ans = tmp.xtrue

i = 0

#%% Iterate


M = 10


pL = opt.underestimate_interp(xL, fL, M)
pR = opt.underestimate_interp(xR, fR, M)



xnew = opt.min_max_poly(pL, pR, xL[0], xR[0])



fnew = f(xnew)

print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xL0' , 'xM', 'xR0', 'xnew', 'bracket'))
print("-"*(20+14*5))
print("%10d |%14.10f |%14.10f |%14.10f |%14.10f |%14.10f" %(i,  xL[0], xM, xR[0], xnew, xR[0] - xL[0]))





xmin  = xL[1] 
xmax = ans + np.fabs(ans - xL[1])
#xmax = xR[1]


ymax = max(f(xmax), f(xmin))
ymin = f(ans)

dist = ymax - ymin

ymax += 0.1*dist
ymin -= 0.1*dist


xx = np.arange(xmin, xmax, dist/500)

ytrue = np.array([f(x) for x in xx])
ypL = np.array([pL(x) for x in xx])
ypR = np.array([pR(x) for x in xx])




plt.plot(xx,ytrue)
plt.plot(xx,ypL, label='pL')

plt.plot(xx,ypR, label='pR')
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)

plt.vlines(xM, ymin, ymax, color='r', label=r'$x_{min}$')
plt.vlines(xnew, ymin, ymax, color='g', label=r'$x_{test}$')

plt.vlines(xL[0], ymin, ymax)
plt.vlines(xL[1], ymin, ymax)
plt.vlines(xR[0], ymin, ymax)
plt.vlines(xR[1], ymin, ymax)

plt.legend()

xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)

#figname = 'Stall_Type1'+str(i)
#plt.tight_layout()

#plt.savefig(figname+'.pdf')
#plt.savefig(figname+'.eps')
#plt.savefig(figname+'.png', dpi=600)


i+=1

#%%


#%%

a1 = opt.dd(2, [xR[0], xR[1]], [fR[0], fR[1]])
print(a1)
5
a2 = opt.dd(2, [xM, xR[1]], [fM, fR[1]])
print(a2)
print((a1-a2)/(xM - xR[0]))

a3 = opt.dd(3, [xR[0], xR[1], xM], [fR[0], fR[1], fM])
print(a3)


#%%



#%%


M = 6000

pR, pL = opt.underestimated_model(xL, xR, xM, fL, fR, fM, M)
xnew = opt.min_max_poly(pL, pR, xL[0], xR[0])


ypL = np.array([pL(x) for x in xx])
ypR = np.array([pR(x) for x in xx])


fnew = f(xnew)

print(xL)
print(xM)
print(xnew)
print(xR)
print(xR[0] - xL[0])

#%%

ymax = max(fR[1], fL[1])
ymin = f(ans)

dist = ymax - ymin

ymax += 0.1*dist
ymin -= 0.1*dist


xmin  = xL[1]
xmax = xR[1]



#%%
xx = np.arange(xmin, xmax, dist/500)

ytrue = np.array([f(x) for x in xx])
ypL = np.array([pL(x) for x in xx])
ypR = np.array([pR(x) for x in xx])




plt.plot(xx,ytrue)
plt.plot(xx,ypL, label='pL')

plt.plot(xx,ypR, label='pR')
#plt.ylim(ymin, ymax)
#plt.xlim(xmin, xmax)
plt.legend()

plt.vlines(xM, ymin, ymax, color='r')
plt.vlines(xnew, ymin, ymax, color='g')

plt.vlines(xL[0], ymin, ymax)
plt.vlines(xL[1], ymin, ymax)
plt.vlines(xR[0], ymin, ymax)
plt.vlines(xR[1], ymin, ymax)


xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)


#%%

xRerr = alg1.xR
xLerr = alg1.xL
xMerr = alg1.xmin

fRerr = f(xRerr)
fLerr = f(xLerr)
fMerr = f(xMerr)
M = 0.1
#%%

pL, pR = opt.underestimated_model(xLerr, xRerr, xMerr, fLerr, fRerr, fMerr, M)




xnew = opt.min_max_poly(pL, pR, xLerr[0], xRerr[0])




ymax = max(fRerr[1], fLerr[1])
ymin = f(ans)

dist = ymax - ymin

ymax += 0.1*dist
ymin -= 0.1*dist


xmin  = xLerr[1]
xmax = xRerr[1]

xx = np.arange(xmin, xmax, dist/500)

ytrue = np.array([f(x) for x in xx])
ypL = np.array([pL(x) for x in xx])
ypR = np.array([pR(x) for x in xx])
ypR2 = np.array([pRsaved(x) for x in xx])



plt.plot(xx,ytrue)
plt.plot(xx,ypL, label='pL')

plt.plot(xx,ypR, label='pR')
plt.plot(xx,ypR2, label='pR2')

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.legend()

plt.vlines(xM, ymin, ymax, color='r')
plt.vlines(xnew, ymin, ymax, color='g')

plt.vlines(xLerr[0], ymin, ymax)
plt.vlines(xLerr[1], ymin, ymax)
plt.vlines(xRerr[0], ymin, ymax)
plt.vlines(xRerr[1], ymin, ymax)