#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:43:12 2019

@author: Jonathan Grant-Peters
"""


import importlib
import matplotlib.pyplot as plt
import sys
import scipy

sys.path.append('/home/user/Documents/Projects/ResearchProject/SiemensCode/3dv1d/JonathansCode/ContinuousOptimiser/python/')

import LStstfn as lst
import GoldenSection 
import numpy as np

import Optimize1D as opt
importlib.reload(opt)
import UPM
importlib.reload(UPM)
import copy

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 4))
plt.rc('font', size=18)

def aveConvergence(acc):
    return np.exp((np.log(acc[-1])-np.log(acc[0]))/len(acc))

def DataPoint(tcname):
    tmp = getattr(lst, tcname)()

    xL, xM, xR, fL, fM, fR = tmp.setup_grid()
    f = tmp.func
    
    bracket = np.array([np.max(xL[0]), np.min(xR[0])])
    fbracket = np.array([f(x) for x in bracket])

    alg0 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0 )
    alg1 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0.1)
    alg2 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1 )
    alg3 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 10)
    alg4 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 100 )
    alg5 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1000 )
    algM = UPM.EUPM(xL, xR, fL, xR, xM, fM)
    
    algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, min_step=1e-6)
    _,_,gsacc,_,_ = GoldenSection.goldenSection(bracket, fbracket, xM, fM, f, eps=1e-5)

    alg0.solve(f)
    alg1.solve(f)
    alg2.solve(f)
    alg3.solve(f)
    alg4.solve(f)
    alg5.solve(f)
    
    algM.solve(f)
    algbrent.solve(f)

    return xL, xM, xR, fL, fM, fR, aveConvergence(alg0.acc), aveConvergence(alg1.acc), aveConvergence(alg2.acc), aveConvergence(alg3.acc), aveConvergence(alg4.acc), aveConvergence(alg5.acc), aveConvergence(algM.acc), aveConvergence(algbrent.acc), aveConvergence(gsacc)



#%%
N = 10
tags = ['SU', 'SM', 'NU']
ntests = np.sum(np.array([lst.ns[key] for key in lst.ns]))


table = np.empty([ntests, 9])
index = 0

for k in range(3):
    category = tags[k]
    for i in range(lst.ns[category]):
        dat = np.empty([ N, 9]) 
        tcname = category+str(i+1)
        print(tcname)
        for j in range(N):
            print(j)
            tmp = DataPoint(tcname)
            dat[j] = tmp[-9:]
            
        table[index] = np.mean(dat, axis=0)
        index+=1
            
        


#%%
    
N = 20
tcname = 'NU1'



for i in range(N):


#%% Set up problem

# 1. Choose test function 
tcname = 'SM7'

tmp = getattr(lst, tcname)()

f = tmp.func

#ans = tmp.xtrue
#start = np.random.uniform(tmp.llim,tmp.rlim)
#direction = np.sign(ans-start)

# 2. Choose starting grids
#xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR
xL, xM, xR, fL, fM, fR = tmp.setup_grid()

xL0 = copy.copy(xL)
xM0 = copy.copy(xM)
xR0 = copy.copy(xR)
fL0 = copy.copy(fL)
fM0 = copy.copy(fM)
fR0 = copy.copy(fR)

#%%
# Identify the starting bracket
bracket = np.array([np.max(xL[0]), np.min(xR[0])])
fbracket = np.array([f(x) for x in bracket])

# 3. Set up the optimiser
alg0 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0 )
alg1 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0.1)
alg2 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1 )
alg3 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 10)
alg4 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 100 )
alg5 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1000 )
algM = UPM.EUPM(xL, xR, fL, xR, xM, fM)

algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, min_step=1e-6)


xx = np.arange(xL[-1], xR[-1], 0.01)
yy = np.array([f(x) for x in xx])
plt.plot(xx,yy)
#plt.show()

figname = 'testcase'+tcname

plt.tight_layout()
plt.savefig(figname+'.pdf')
plt.savefig(figname+'.eps')
plt.savefig(figname+'.png', dpi=600)


#%%
alg0.solve(f)
alg1.solve(f)
alg2.solve(f)
alg3.solve(f)
alg4.solve(f)
alg5.solve(f)

algM.solve(f)
algbrent.solve(f)


#%%
_,_,gsacc,_,_ = GoldenSection.goldenSection(bracket, fbracket, xM, fM, f, eps=1e-5)

# 5. Plot convergence rate
plt.semilogy(alg0.acc[:30], label=r'$\alpha$ = '+ str(alg0.alpha))
plt.semilogy(alg1.acc[:30], label=r'$\alpha$ = '+ str(alg1.alpha))
plt.semilogy(alg2.acc[:30], label=r'$\alpha$ = '+ str(alg2.alpha))
plt.semilogy(alg3.acc[:30], label=r'$\alpha$ = '+ str(alg3.alpha))
plt.semilogy(alg4.acc[:30], label=r'$\alpha$ = '+ str(alg4.alpha))
plt.semilogy(alg5.acc[:30], label=r'$\alpha$ = '+ str(alg5.alpha))
plt.semilogy(algM.acc, label=r'$\alpha = \infty$')


plt.semilogy(algbrent.acc, label='Brent')
plt.semilogy(gsacc, label='Golden Section')

plt.xlabel('Number of Function Evaluations')
plt.ylabel('Length of Bracket')
plt.legend()

figname = 'testcase'+tcname+'_outcome'

plt.tight_layout()
plt.savefig(figname+'.pdf')
plt.savefig(figname+'.eps')
plt.savefig(figname+'.png', dpi=600)

plt.show()
