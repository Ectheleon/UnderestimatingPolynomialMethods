#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:23 2019

@author: Jonathan GP
"""

import LStstfn as lst
import numpy as np
import matplotlib.pyplot as plt
import UPML as upml
import importlib
from scipy.stats import mstats
#%%
importlib.reload(upml)

#%%

tmp = lst.AN3()
f = tmp.func
ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

xL, xM, xR, fL, fM, fR = lst.setup_grid(f, start, direction, 1e-5)

print(start, ans, direction)
print(xL)
print(xM)
print(xR)
print(lst.check_grid(xL, xM, xR, fL, fM, fR, ans))
i=0


x0 = np.array([xL[1], xL[0], xM, xR[0] , xR[1]])


ymin = 0.4
ymax = 2
xx = np.arange(xL[1], xR[1], 0.0001)
yy = np.array([f(x) for x in xx])

plt.vlines(xL[1], ymin, ymax)
plt.vlines(xL[0], ymin, ymax)
plt.vlines(xR[1], ymin, ymax)
plt.vlines(xR[0], ymin, ymax)
plt.vlines(xM, ymin, ymax)

plt.plot(xx,yy)

tot = 0

#%%itertype

_, _, dat = upml.UPML(x0, fM, f, 1e-8, log=True)



b = dat[:,0]
ratios = np.array([b[i+1]/b[i] for i in range(len(b)-1)])

gmean = mstats.gmean(ratios)

print(gmean)

#%%
for i in range(len(b)-1):
    print(int(dat[i+1, -1]), ratios[i] )









