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
sys.path.append('/home/user/Documents/Projects/ResearchProject/SiemensCode/3dv1d/JonathansCode/ContinuousOptimiser/python/')

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

plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tcname = 'SU4'

tmp = getattr(lst, tcname)()

f = tmp.func

# 2. Choose starting grids
#xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR


M = tmp.M
# Record the true solution
#ans = tmp.xtrue

i = 0
M = 0

xL, xM, xR , fL, fM ,fR = tmp.setup_grid()

t1 = 0.25/(xR[0] - xL[0])
t2 = (M - xL[0])/(xR[0] - xL[0])**2
t3 = (xR[0] - M)/(xR[0] - xL[0])**2

alpha = 0.1*min(t1, t2, t3)


#xL = np.array([0.65197749, 0.41591187, 0.41590187])
#xM = 1.0339396788250546
#xR = np.array([1.17983625, 1.2700053 , 1.41590187])

#fL = np.array([0.27382425, 0.2927476 , 0.29274846])
#fM = 0.25870589400656835
#fR = np.array([0.30184159, 0.33412614, 0.39850689])

def project_interval(x, a,b):
    #projects the point x onto the interval [a,b]
    
    if x<= a:
        return a
    elif x>= b:
        return b
    else :
        return x

#%% Iterate
#1. Construct quadratic which interpolates xL1, xL2, xM
#   a) gm = derivative of quadratic at xM
#   b) Qm = constrained minimiser of quadratic in [xL1, xR1]
#   c) ghatm = left linear slope. f[xL1, xL2]
#   d) pm = f[xM, xL1] - ghatm, the left linearisation error.

qL = np.poly1d(np.polyfit([xM,xL[0],xL[1]] , [fM,fL[0],fL[1]] ,2))
        
gL = qL.deriv()(xM)
QL = qL.deriv().roots[0] if qL.deriv().deriv()(0)>=0 and qL.deriv().roots[0] <= xR[0] else xR[0]
ghatL = (fL[0] - fL[1])/(xL[0] - xL[1])
pL = fM - fL[0] - ghatL*(xM - xL[0])


#2. Construct quadratic which interpolates xR1, xR2, xM
#   a) gp = derivative of quadratic at xM
#   b) Qp = constrained minimiser of quadratic in [xL1, xR1]
#   c) ghatp = right linear slope. f[xR1, xR2]
#   d) pp = f[xM, xR1] - ghatm, the right linearisation error.

qR = np.poly1d(np.polyfit([xM,xR[0],xR[1]] , [fM,fR[0],fR[1]] ,2))
        
gR = qR.deriv()(xM)
QR = qR.deriv().roots[0] if qR.deriv().deriv()(0) >=0 and qR.deriv().roots[0] >= xL[0] else xL[0]
ghatR = (fR[0] - fR[1])/(xR[0] - xR[1])
pR = fM - fR[0] - ghatR*(xM - xR[0])



#3. Determine Rm and Rp ?
#   a) Set Rp = Qp and Rm = Qm
#   b) If Qm>xM, Qp >= xM and pp < (ghatp - gm)/(Qm - xM)
#           Pm = xM + pp/(ghatp - gm)
#           Rm = Pm
#   c) If Qm<=xM, Qp < xM and pm < (gp - ghatm)/(xM - Qp)
#           Pp = xM - pm/(gp - ghatm)
#           Rp = Pp

RR, RL = QR, QL

if QL>xM and QR>=xM and pR<(ghatR - gR)/(QL-xM):
    PL = xM + pR/(ghatR - gL)
    RL = PL

if QL<=xM and QR<xM and pL<(gR - ghatL)/(xM -QR):
    PR = xM - pL/(gR - ghatL)
    RR = PR

#4. Determine R
#   a) Set L = min(Rp, Rm), U = max(Rm, Rp))
#   b) If ghat+>0 and ghatm<0, Po = xM + (pp-pm)/(ghat+-ghatm), 
#       else Po = local min of quadratic interpolating xL1, xM, xR1
#   c) R = projection of Po onto [L,U]
    
L, U = min(RR, RL), max(RR, RL)

if ghatR>0 and ghatL<0:
    P0 = xM + (pR-pL)/(ghatR - ghatL)
else:
    
    q0 = np.poly1d(np.polyfit([xL[0], xM, xR[0]], [fL[0], fM, fR[0]], 2))
    
    P0 = q0.deriv().roots[0]
    
R = project_interval(P0, L, U)        

#5. Determine x from R via safeguarding
#   a) Set sigma = s(xR1-xL1)^2
#   b) x = projection of R onto [xL1+sigma, xR1-sigma]
#       If |x-xM|<sigma, replace x by:
#       xM+sigma if xM<= 0.5(xR1+xL1) else xM - sigma

sigma = alpha*(xR[0] - xL[0])**2
x = project_interval(R, xL[0]+sigma, xR[0] - sigma)
    
if np.fabs(x - xM) < sigma:
    if xM <= 0.5*(xL[0] + xR[0]):
        xnew = x+ sigma
    else:
        xnew = x - sigma

xnew =  x    
fnew = f(xnew)


print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xL0' , 'xM', 'xR0', 'xnew', 'bracket'))
print("-"*(20+14*5))
print("%10d |%14.10f |%14.10f |%14.10f |%14.10f |%14.10f" %(i,  xL[0], xM, xR[0], xnew, xR[0] - xL[0]))





xmin  = xL[1] 
xmax = xR[1]
#xmax = xR[1]


ymax = max(f(xmax), f(xmin))
ymin = f(xM)

dist = ymax - ymin

ymax += 0.1*dist
ymin -= 0.1*dist


xx = np.arange(xmin, xmax, (xmax-xmin)/500)

ytrue = np.array([f(x) for x in xx])
yqL = np.array([qL(x) for x in xx])
yqR = np.array([qR(x) for x in xx])

ymL = np.array([fL[0] + (x - xL[0])* ghatL for x in xx   ])
ymR = np.array([fR[0] + (x - xR[0])* ghatR for x in xx   ])


plt.plot(xx,ytrue, label='$f(x)$')
plt.plot(xx,yqL, label='$q^L(x)$')
plt.plot(xx,yqR, label='$q^R(x)$')


plt.plot(xx,ymL, color='r')
plt.plot(xx,ymR, color='r')

#plt.ylim(ymin, ymax)
#plt.xlim(xmin, xmax)

plt.vlines(xM, ymin, ymax, color='r', label=r'$x^M$')
plt.vlines(U, ymin, ymax, color='b', label=r'$U$')
plt.vlines(L, ymin, ymax, color='b', label=r'$L$')
#plt.vlines(xR[0], ymin, ymax, color='gr', label=r'$x^R_1$')
#plt.vlines(xR[1], ymin, ymax, color='gr', label=r'$x^R_2$')
#plt.vlines(xM, ymin, ymax, color='r', label=r'$x^M$')

#plt.vlines(xnew, ymin, ymax, color='g', label=r'$\tilde{x}$')

#plt.vlines(xL[0], ymin, ymax)
#plt.vlines(xL[1], ymin, ymax)
#plt.vlines(xR[0], ymin, ymax)
#plt.vlines(xR[1], ymin, ymax)

plt.legend()

xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)

figname = 'SUMP_Failure_'+str(i)
plt.tight_layout()

plt.savefig(figname+'.pdf')
plt.savefig(figname+'.eps')
plt.savefig(figname+'.png', dpi=600)


i+=1

#%%
num = 1
plt.savefig('SUPM_failure_'+str(num)+'pdf', dpi=600)
plt.savefig('SUPM_failure_'+str(num)+'pdf', dpi=600)
plt.savefig('SUPM_failure_'+str(num)+'pdf', dpi=600)

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