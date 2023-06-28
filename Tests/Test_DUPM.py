#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:38:52 2019

@author: Jonathan GP
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-cocoa powder
"""
Created on Wed Mar 27 16:59:55 2019

@author: Jonathan GP
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

import MiffOpt

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 4))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tcname = 'NU1'

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

# Identify the starting bracket

algD = UPM.DUPM(xL, xR, fL, xR, xM, fM)

algMiff = MiffOpt.MiffOpt(xL, xR, fL, xR, xM, fM)
#algD2 = UPM.DUPM2(xL, xR, fL, xR, xM, fM)



#xx = np.arange(xL[-1], xR[-1], 0.1)
#yy = np.array([f(x) for x in xx])
#plt.plot(xx,yy)
#plt.show()

#figname = 'testcase'+tcname

#plt.tight_layout()
#plt.savefig(figname+'.pdf')
#plt.savefig(figname+'.eps')
#plt.savefig(figname+'.png', dpi=600)


algD.solve(f)


plt.semilogy(algD.acc[:30], label='Dynamic')

plt.xlabel('Number of Function Evaluations')
plt.ylabel('Length of Bracket')
plt.legend()

figname = 'testcase'+tcname+'_outcome'
#%%
plt.tight_layout()
plt.savefig(figname+'.pdf')
plt.savefig(figname+'.eps')
plt.savefig(figname+'.png', dpi=600)

plt.show()

# 6. Observe plot and note the bottleneck. For AN1, it occurs at iteration 9
#%%

#%%

#%%


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


























T