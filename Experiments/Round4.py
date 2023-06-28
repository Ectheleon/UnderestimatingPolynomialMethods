#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:43:12 2019

@author: Jonathan Grant-Peters
"""


import importlib
import matplotlib.pyplot as plt
import sys
#import scipy

sys.path.append('/home/user/Documents/Projects/ResearchProject/SiemensCode/3dv1d/JonathansCode/ContinuousOptimiser/python/')

import LStstfn as lst
import GoldenSection 
import numpy as np

import Optimize1D as opt
importlib.reload(opt)
import UPM
importlib.reload(UPM)
#import copy
from tabulate import tabulate
import MiffOpt
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
    alg1 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0.01)
    alg2 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 0.1)
    alg3 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1 )
    alg4= UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 10)
    alg5 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 100 )
    algE = UPM.EUPM(xL, xR, fL, fR, xM, fM)
    algD = UPM.DUPM(xL, xR, fL, fR, xM, fM)
    algM = MiffOpt.MiffOpt(xL, xR, fL, fR, xM, fM)
    
    algbrent = opt.Brent(bracket, xM, fbracket, fM, eps=1e-5, min_step=1e-6)
    _,_,gsacc,_,_ = GoldenSection.goldenSection(bracket,xM,  fbracket, fM, f, eps=1e-5)

    alg0.solve(f)
    alg1.solve(f)
    alg2.solve(f)
    alg3.solve(f)
    alg4.solve(f)
    alg5.solve(f)
    #print('Attempting DUPM')
    algD.solve(f)
    #print('Completed DUPM')
    algE.solve(f)
    algbrent.solve(f)
    algM.solve(f)
    
    
    res0 = aveConvergence(alg0.acc) if alg0.status==1 else np.inf
    res1 = aveConvergence(alg1.acc) if alg1.status==1 else np.inf
    res2 = aveConvergence(alg2.acc) if alg2.status==1 else np.inf
    res3 = aveConvergence(alg3.acc) if alg3.status==1 else np.inf
    res4 = aveConvergence(alg4.acc) if alg4.status==1 else np.inf
    res5 = aveConvergence(alg5.acc) if alg5.status==1 else np.inf
    
    resE =  aveConvergence(algE.acc) if algE.status==1 else np.inf
    resD = aveConvergence(algD.acc) if algD.status==1 else np.inf
    resBrent = aveConvergence(algbrent.acc) 
    resGol = aveConvergence(gsacc)
    resMiff = aveConvergence(algM.acc) if algM.status==1 else np.inf
    

    #return xL, xM, xR, fL, fM, fR,
    return res0, res1, res2, res3, res4, res5, resE, resD, resBrent, resGol, resMiff



#%%
N = 1000
#tags = ['NU']#['SU', 'SM', 'NU']
ntests = np.sum(np.array([lst.ns[key] for key in lst.ns]))

#%%
category = 'SU'

table = np.empty([lst.ns[category], 11])
index = 0

for i in range(lst.ns[category]):
    dat = np.empty([ N, 11]) 
    tcname = category+str(i+1)
    print(tcname)
    for j in range(N):
        print(j)
        dat[j] = DataPoint(tcname)
        
    table[index] = np.mean(dat, axis=0)
    index+=1
#%%
        
print(tabulate(table, tablefmt="latex", floatfmt=".2"))
#%%
dat_path = "/home/user/Documents/Projects/ResearchProject/Data/UnivariateOpt/"
header="0, 0.01, 0.1, 1, 10, 100, EUPM, DUPM, Brent, GS, Mifflin"
np.savetxt(dat_path+category+'test'+str(N)+'.csv', table, comments='', header=header,delimiter=', ')


#%%
category = 'SM'

table = np.empty([lst.ns[category], 11])
index = 0

for i in range(lst.ns[category]):
    dat = np.empty([ N, 11]) 
    tcname = category+str(i+1)
    print(tcname)
    for j in range(N):
        print(j)
        dat[j] = DataPoint(tcname)
        
    table[index] = np.mean(dat, axis=0)
    index+=1
#%%
        
print(tabulate(table, tablefmt="latex", floatfmt=".2"))
#%%
dat_path = "/home/user/Documents/Projects/ResearchProject/Data/UnivariateOpt/"
header="0, 0.01, 0.1, 1, 10, 100, EUPM, DUPM, Brent, GS, Mifflin"
np.savetxt(dat_path+category+'test'+str(N)+'.csv', table, comments='', header=header,delimiter=', ')



#%%
category = 'NU'

table = np.empty([lst.ns[category], 11])
index = 0

for i in range(lst.ns[category]):
    dat = np.empty([ N, 11]) 
    tcname = category+str(i+1)
    print(tcname)
    for j in range(N):
        print(j)
        dat[j] = DataPoint(tcname)
        
    table[index] = np.mean(dat, axis=0)
    index+=1
    
'''  

tcname = 'NU3'

table = np.empty([lst.ns[category], 11])
index = 0

for j in range(N):
    print(j)
    dat[j] = DataPoint(tcname)
    
    tmp = getattr(lst, tcname)()
    xL, xM, xR, fL, fM, fR = tmp.setup_grid()
    f = tmp.func
    
    alg3 = UPM.SUPM(xL, xR, fL, fR, xM, fM, alpha = 1 )
    alg3.solve(f)

    if alg3.status!=1:
        break
    
    
#table[index] = np.mean(dat, axis=0)
#index+=1
'''
    
#%%
        
print(tabulate(table, tablefmt="latex", floatfmt=".2"))
#%%
dat_path = "/home/user/Documents/Projects/ResearchProject/Data/UnivariateOpt/"
header="0, 0.01, 0.1, 1, 10, 100, EUPM, DUPM, Brent, GS, Mifflin"
np.savetxt(dat_path+category+'test'+str(N)+'.csv', table, comments='', header=header,delimiter=', ')



#%% Debugger
category = 'NU'

table = np.empty([lst.ns[category], 11])
index = 0

for i in range(lst.ns[category]):
    dat = np.empty([ N, 11]) 
    tcname = category+str(i+1)
    print(tcname)
    for j in range(N):
        print(j)
        
        
        tmp = getattr(lst, tcname)()

        xL, xM, xR, fL, fM, fR = tmp.setup_grid()
        f = tmp.func
    

        algD = UPM.DUPM(xL, xR, fL, fR, xM, fM)
        
    
        algD.solve(f)
        
        
        resD = aveConvergence(algD.acc) if algD.status==1 else np.inf

        if resD == np.inf:
            raise ValueError()
        
        
    index+=1


#%%