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
from scipy.stats import mstats

sys.path.append('/home/peters/Documents/Projects/ResearchProject/MyCode/ContinuousOptimiser/python/')

sys.path.append('/home/user/Documents/Projects/ResearchProject/SiemensCode/3dv1d/JonathansCode/ContinuousOptimiser/python/')
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
plt.rc('figure', figsize=(12, 4))
plt.rc('font', size=18)

#%%
# Trial GBrentLimit

# 1. Choose test function 
tmp = lst.AN1()
tcname = 'AS3'

f = tmp.func

ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

# 2. Choose starting grids
#xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR
xL, xM, xR, fL, fM, fR = lst.setup_grid(f, start, direction, 1e-5)




# Record the true solution

# Identify the starting bracket
bracket = np.array([np.max(xL[0]), np.min(xR[0])])
fbracket = np.array([f(x) for x in bracket])

# 3. Set up the optimiser
algM = opt.GBrentLimit(xL, xR, fL, xR, xM, fM)
algM.naive_solve(f)


#%%
_,_,gsacc,_,_ = GoldenSection.goldenSection(bracket, fbracket, xM, fM, f, eps=1e-5)

N = len(algM.acc)
breduction = np.array([algM.acc[i+1]/algM.acc[i] for i in range(N-1)])
gmean = scipy.stats.mstats.gmean(breduction)
print(gmean, xL[0] - xL[1], xR[1] - xR[0],xR[1] - xR[0]-xL[0] + xL[1] )

#%%
approx = np.array([gmean**i*algM.acc[0] for i in range(N)])
#plt.semilogy(bracket)
plt.semilogy(approx, label='average convergence')



# 5. Plot convergence rate
plt.semilogy(algM.acc, label=r'$M = \infty$')
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



#%% Set up problem

# 1. Choose test function 
tmp = lst.AN2()
tcname = 'AS2'

f = tmp.func

ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

# 2. Choose starting grids
#xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR
xL, xM, xR, fL, fM, fR = lst.setup_grid(f, start, direction, 1e-5)

# Record the true solution

# Identify the starting bracket
bracket = np.array([np.max(xL[0]), np.min(xR[0])])
fbracket = np.array([f(x) for x in bracket])

# 3. Set up the optimiser
alg0 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0 )
alg1 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.1)
alg2 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 0.5 )
alg3 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 1)
alg4 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 5 )
alg5 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 10 )

alg6 = opt.GBrent(xL, xR, fL, fR, xM, fM, Lipschitz = 1e5 )
algM = opt.GBrentLimit(xL, xR, fL, xR, xM, fM)

algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, delta=1e-6)


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
alg0.naive_solve(f)
alg1.naive_solve(f, maxiter=16)
alg2.naive_solve(f)
alg3.naive_solve(f)
alg4.naive_solve(f)
alg5.naive_solve(f)
alg6.naive_solve(f)

algM.naive_solve(f)
algbrent.solve(f)


#%%
_,_,gsacc,_,_ = GoldenSection.goldenSection(bracket, fbracket, xM, fM, f, eps=1e-5)

# 5. Plot convergence rate
plt.semilogy(alg0.acc[:30], label='M = '+ str(alg0.lip))
plt.semilogy(alg1.acc[:30], label='M = '+ str(alg1.lip))
plt.semilogy(alg2.acc[:30], label='M = '+ str(alg2.lip))
plt.semilogy(alg3.acc[:30], label='M = '+ str(alg3.lip))
plt.semilogy(alg4.acc[:30], label='M = '+ str(alg4.lip))
plt.semilogy(alg5.acc[:30], label='M = '+ str(alg5.lip))
plt.semilogy(alg6.acc[:30], label='M = '+ str(alg6.lip))
plt.semilogy(algM.acc, label=r'$M = \infty$')


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

# 6. Observe plot and note the bottleneck. For AN1, it occurs at iteration 9

#%%

arr0 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg0.guess])
arr1 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg1.guess])
arr2 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg2.guess])
arr3 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg3.guess])
arr4 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg4.guess])
arr5 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg5.guess])
arr6 = np.array([np.fabs(tmp.xtrue - xxx) for xxx in alg6.guess])

arrbrent = np.array([np.fabs(tmp.xtrue - xxx) for xxx in algbrent.guess])


plt.semilogy(arr0[:30], label='M = '+ str(alg0.lip))
plt.semilogy(arr1[:30], label='M = '+ str(alg1.lip))
plt.semilogy(arr2[:30], label='M = '+ str(alg2.lip))
plt.semilogy(arr3[:30], label='M = '+ str(alg3.lip))
plt.semilogy(arr4[:30], label='M = '+ str(alg4.lip))
plt.semilogy(arr5[:30], label='M = '+ str(alg5.lip))
plt.semilogy(arr6[:30], label='M = '+ str(alg5.lip))

plt.semilogy(arrbrent, label='Brent')
#plt.semilogy(gsacc, label='Golden Section')

plt.xlabel('Number of Function Evaluations')
plt.ylabel('Length of Bracket')
plt.legend()

figname = 'testcase'+tcname+'_outcome_alt'

plt.tight_layout()
plt.savefig(figname+'.pdf')
plt.savefig(figname+'.eps')
plt.savefig(figname+'.png', dpi=600)

plt.show()




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
xx = np.arange(-5,5, 0.01)
yy = np.array([f(x) for x in xx])

plt.plot(xx,yy)

#%%


























