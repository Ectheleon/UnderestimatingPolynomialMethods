#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:46:14 2019

@author: Jonathan GP
"""


import autograd.numpy as np
import importlib
import matplotlib.pyplot as plt
import sys
import scipy.optimize as so
from scipy.stats import mstats
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

#%%
def setup_grid(func, x0, direction, eps, step = 1, lstol = 1e-5):
   N = 2
   
   xL = np.zeros([N])
   xR = np.zeros([N])
   fL = np.zeros([N])
   fR = np.zeros([N])

   xpert = x0+direction*eps
   xM = xpert
   fM = func(xM)

   xL[0] = x0
   fL[0] = func(xL[0])
   
   if fM>= fL[0]:
      raise ValueError('direction is not a valid direction of descent')
      
   step_num = 1
   xnew = x0+step_num*step*direction
   fnew = func(xnew)
   
   
   while fnew <= fM:
      xL, fL = np.roll(xL, 1), np.roll(fL, 1)
      
      step_num +=1

      xM, fM, xL[0], fL[0] = xnew, fnew, xM, fM
      xnew = x0+step_num*step*direction
      fnew = func(xnew)
      
   xR[0], fR[0] = xnew, fnew
   
   ratio = 0.5*(-1 + np.sqrt(5))
   
   
   Lfull = step_num
   Rfull = 1
   
   #print('after constructing bracket')
   #print('Rfull', Rfull, xR)
   #print('xM', xM)
   #print('Lfull', Lfull, xL)
   #print('-'*40)

   while Rfull <N and np.fabs(xR[0] - xL[0])>= lstol:
      xnew = xM*(1-ratio) + xR[0]*ratio
      fnew = f(xnew)
      
      if fnew <= fM:
         #print('fnew', 'fM', fnew, fM)
         xL, fL = np.roll(xL, 1), np.roll(fL, 1)
         xM, fM, xL[0], fL[0] = xnew, fnew, xM, fM
         Lfull+=1
      else:
         Rfull +=1
         xR, fR = np.roll(xR, 1), np.roll(fR, 1)
         xR[0], fR[0] = xnew, fnew
         
      #print('check', fM, f(xM))
   #print('after R refinement')
   #print('Rfull', Rfull, xR)
   #print('xM', xM)
   #print('Lfull', Lfull, xL)
   #print('-'*40)
         
   
   if Lfull<N and xM ==xpert:
      xnew = xM*(ratio) + xR[0]*(1-ratio)
      fnew = f(xnew)
      
      if fnew <= fM:
         xL, fL = np.roll(xL, 1), np.roll(fL, 1)
         xM, fM, xL[0], fL[0] = xnew, fnew, xM, fM
         Lfull+=1
      else:
         Rfull +=1
         xR, fR = np.roll(xR, 1), np.roll(fR, 1)
         xR[0], fR[0] = xnew, fnew

      
      
   
   while Lfull <N and np.fabs(xR[0] - xL[0])>=lstol:
      xnew = xM*(1-ratio) + xL[0]*ratio 
      fnew = f(xnew)
      
      if fnew <= fM:
         xR, fR = np.roll(xR, 1), np.roll(fR, 1)
         xM, fM, xR[0], fR[0] = xnew, fnew, xM, fM
         Rfull+=1
      else:
         Lfull +=1
         xL, fL = np.roll(xL, 1), np.roll(fL, 1)
         xL[0], fL[0] = xnew, fnew
   #print('after L refinement')
   #print('Rfull', Rfull, xR)
   #print('xM', xM)
   #print('Lfull', Lfull, xL)
   #print('-'*40)

   return (xL, xM, xR, fL, fM, fR) if direction>0 else (xR, xM, xL, fR, fM, fL)
      

def check_grid(xL, xM, xR, fL, fM, fR, answer):
   #s1 = xL[2] < xL[1]
   s2 = xL[1] < xL[0]
   s3 = xL[0] < xM
   s4 = xM < xR[0]
   s5 = xR[0] < xR[1]
   #s6 = xR[1] < xR[2]
   
   increasing = s2 and s3 and s4 and s5# and s6 and s1
   
   bracket = answer > xL[0] and answer < xR[0]
   
   minimum = fM < fL[0] and fM < fR[0]
   
   return increasing, bracket, minimum

#%% Set up problem
   
tmp = lst.AN3()
f = tmp.func
ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

xL, xM, xR, fL, fM, fR = setup_grid(f, start, direction, 1e-5)

print(start, ans, direction)
print(xL)
print(xM)
print(xR)
print(check_grid(xL, xM, xR, fL, fM, fR, ans))
i=0


tot = 0
 #%%


print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'yL1', 'yM', 'yR1', 'xnew' , 'newbracket'))
print("-"*(20+14*5))


Niters = 20

bracket = np.zeros([Niters])
breduction = np.zeros([Niters])

xMhist = np.zeros([Niters])
xnewhist = np.zeros([Niters])
#ahist = np.zeros([Niters])
#bhist = np.zeros([Niters])
#chist = np.zeros([Niters])

bracket[-1] = xR[0] - xL[0]

for i in range(Niters):   
   #pL = np.poly1d([alpha*1, alpha*(-xL[0] -xL[1]),alpha* xL[0]*xL[1]])
   #pR = np.poly1d([1, (-xR[0] -xR[1]), xR[0]*xR[1]])
   
   yL1 = xL[0] - ans
   yL2 = xL[1] - ans
   #yL3 = xL[2] - ans
   yR1 = xR[0] - ans
   yR2 = xR[1] - ans
   #yR3 = xR[2] - ans
   
   
   pL = np.poly1d([1, -xL[0]-xL[1], xL[0]*xL[1]  ])
   pR = np.poly1d([1, -xR[0]-xR[1], xR[0]*xR[1]  ])
   
   N= 200
   xx=np.array([xL[0] + i*(xR[0]-xL[0])/N for i in range(N)  ])  
   plt.plot(xx, pL(xx))
   plt.plot(xx, pR(xx))
   
   
      
   xnew = (xR[0]*xR[1] - xL[0]*xL[1])/(xR[0]+xR[1]-xL[0]-xL[1])    
   fnew = f(xnew)
   
   xMhist[i] = xM
   xnewhist[i] = xnew
   bracket[i] = xR[0] - xL[0]
   breduction[i] = bracket[i]/bracket[i-1]
   
   fnew = f(xnew)
   ynew = xnew - ans
   
   
   xnew_bnd = ((xnew- xL[0]) if ynew>0 else (xR[0] - xnew))/(xR[0] - xL[0])
   
   ynew = xnew - ans

   newbracket = ((ynew- yL1) if ynew>0 else (yR1 - ynew))/(yR1 - yL1)
   bracketboundL = (yR1- yL1)*(yR1- yL2)/(yR1+yR2- yL1-yL2)
   bracketboundR = (yR1 - yL1)*(yR2 - yL1)/( yR1+yR2-yL1-yL2)
   bracketbound = (bracketboundL if ynew<0 else bracketboundR)/(yR1 - yL1)

   check = 0#check7
   
   oldbracket = xR[0] - xL[0]   
   xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)
   newbracket = xR[0] - xL[0]
   breduction[i] = newbracket/oldbracket

   print("%10d |%14.9f |%14.9f |%14.9f |%14.9f |%14.9f" %(i, yL1, xM-ans, yR1, ynew, breduction[i]))
   tot+= newbracket/oldbracket

#%%

N = len(bracket)
gmean = scipy.stats.mstats.gmean(breduction)
print(gmean)
approx = np.array([gmean**i*bracket[0] for i in range(N)])
plt.semilogy(bracket)
plt.semilogy(approx)


#%%
print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'yL1', 'yM', 'yR1', 'xnew' , 'newbracket'))
print("-"*(20+14*5))

tmp = lst.AN3()
f = tmp.func
ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

xL, xM, xR, fL, fM, fR = setup_grid(f, start, direction, 1e-5)

print(start, ans, direction)
print(xL)
print(xM)
print(xR)
print(check_grid(xL, xM, xR, fL, fM, fR, ans))


bracket = np.zeros([Niters])
breduction = np.zeros([Niters])

xMhist = np.zeros([Niters])
xnewhist = np.zeros([Niters])
#ahist = np.zeros([Niters])
#bhist = np.zeros([Niters])
#chist = np.zeros([Niters])

bracket[-1] = xR[0] - xL[0]
i=0
#%%
#pL = np.poly1d([alpha*1, alpha*(-xL[0] -xL[1]),alpha* xL[0]*xL[1]])
#pR = np.poly1d([1, (-xR[0] -xR[1]), xR[0]*xR[1]])

yL1 = xL[0] - ans
yL2 = xL[1] - ans
#yL3 = xL[2] - ans
yR1 = xR[0] - ans
yR2 = xR[1] - ans
#yR3 = xR[2] - ans

xnew = (xR[0]*xR[1] - xL[0]*xL[1])/(xR[0]+xR[1]-xL[0]-xL[1])    
fnew = f(xnew)

pL = np.poly1d([1, -xL[0]-xL[1], xL[0]*xL[1]  ])
pR = np.poly1d([1, -xR[0]-xR[1], xR[0]*xR[1]  ])


ymax0 = max(f(xL[1]), f(xR[1]))
ymin0 = f(ans)

ymin = min(pL(xR[1]), pR(xL[1]))
ymax = max(pL(xL[1]), pR(xR[1]))



#####
N= 200


############
fig, axs = plt.subplots(2, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)



xx=np.array([xL[1] + i*(xR[1]-xL[1])/N for i in range(N)  ])  

ytrue = np.array([f(x) for x in xx])


axs[0].plot(xx,ytrue)
axs[0].vlines(xL[0], ymin0, ymax0, color='g')
axs[0].vlines(xL[1], ymin0, ymax0, color='g')
axs[0].vlines(xR[0], ymin0, ymax0, color='b')
axs[0].vlines(xR[1], ymin0, ymax0, color='b')
axs[0].vlines(xM, ymin0, ymax0, color='black')
axs[0].vlines(xnew, ymin0, ymax0, color='r')
axs[0].set_ylim(ymin0, ymax0)


axs[1].plot(xx, pL(xx), label='pL')
axs[1].plot(xx, pR(xx), label='pR')
axs[1].vlines(xL[0], ymin, ymax, color='g')
axs[1].vlines(xL[1], ymin, ymax, color='g')
axs[1].vlines(xR[0], ymin, ymax, color='b')
axs[1].vlines(xR[1], ymin, ymax, color='b')
axs[1].vlines(xM, ymin, ymax, color='black')
axs[1].vlines(xnew, ymin, ymax, color='r')
axs[1].set_ylim(ymin, ymax)
axs[1].legend()
   

xMhist[i] = xM
xnewhist[i] = xnew
bracket[i] = xR[0] - xL[0]
breduction[i] = bracket[i]/bracket[i-1]

fnew = f(xnew)
ynew = xnew - ans


xnew_bnd = ((xnew- xL[0]) if ynew>0 else (xR[0] - xnew))/(xR[0] - xL[0])

ynew = xnew - ans

newbracket = ((ynew- yL1) if ynew>0 else (yR1 - ynew))/(yR1 - yL1)
bracketboundL = (yR1- yL1)*(yR1- yL2)/(yR1+yR2- yL1-yL2)
bracketboundR = (yR1 - yL1)*(yR2 - yL1)/( yR1+yR2-yL1-yL2)
bracketbound = (bracketboundL if ynew<0 else bracketboundR)/(yR1 - yL1)

check = 0#check7

oldbracket = xR[0] - xL[0]   
xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)
newbracket = xR[0] - xL[0]
breduction[i] = newbracket/oldbracket

print("%10d |%14.9f |%14.9f |%14.9f |%14.9f |%14.9f" %(i, yL1, xM-ans, yR1, ynew, breduction[i]))
tot+= newbracket/oldbracket

i+=1

#%% Set up problem
   
tmp = lst.AN3()
f = tmp.func
ans = tmp.xtrue
start = np.random.uniform(ans-2,ans+2)
direction = np.sign(ans-start)

xL, xM, xR, fL, fM, fR = setup_grid(f, start, direction, 1e-5)

print(start, ans, direction)
print(xL)
print(xM)
print(xR)
print(check_grid(xL, xM, xR, fL, fM, fR, ans))
i=0
#%% Iterate


#M = 10
A = np.fabs(xR[0] - xR[2])
B = np.fabs(xL[0] - xL[2])
alpha = B/A

pL = np.poly1d([alpha*1, alpha*(-xL[0] -xL[1]),alpha* xL[0]*xL[1]])
pR = np.poly1d([1, (-xR[0] -xR[1]), xR[0]*xR[1]])

yL1 = xL[0] - ans
yL2 = xL[1] - ans
yL3 = xL[2] - ans
yR1 = xR[0] - ans
yR2 = xR[1] - ans
yR3 = xR[2] - ans

a =1-alpha
b = -(yR1 + yR2) + (yL1 + yL2)*alpha
c = yR1*yR2 - alpha*yL1*yL2
xnew = opt.min_max_poly(pL, pR, xL[0], xR[0])
fnew = f(xnew)

ynewtheory = -b/(2*a) - np.sqrt(b**2 - 4*a*c)/(2*a)

xMhist[i] = xM
xnewhist[i] = xnew
bracket[i] = xR[0] - xL[0]
ahist[i] = a
bhist[i] = b
chist[i] = c

fnew = f(xnew)




print('a', 'b', 'c')
print(a, b, c)
print(a*c)

#xnew = opt.min_max_poly(pL, pR, xL[0], xR[0])



fnew = f(xnew)

print("%10s |%14s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xL0' , 'xM', 'xR0', 'xnew', 'bracket', 'alpha'))
print("-"*(20+14*5))
print('{:10d} |{:14.10f} |{:14.10f} |{:14.10f} |{:14.10f} |{:14.10f} |{:14.10f}'.format(i,  xL[0], xM, xR[0], xnew, xR[0] - xL[0], alpha))





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




fig, axs = plt.subplots(2, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
axs[0].plot(xx,ytrue)
#axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
#axs[0].set_ylim(-1, 1)

axs[0].vlines(xM, ymin, ymax, color='r', label=r'$x_{min}$')
axs[0].vlines(xnew, ymin, ymax, color='g', label=r'$x_{test}$')

axs[0].vlines(xL[0], ymin, ymax)
axs[0].vlines(xL[1], ymin, ymax)
axs[0].vlines(xR[0], ymin, ymax)
axs[0].vlines(xR[1], ymin, ymax)
axs[0].set_ylim(ymin, ymax)




axs[1].plot(xx,-ypL, label='pL')
axs[1].plot(xx,-ypR, label='pR')
axs[1].hlines(0, min(xx), max(xx))
#axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
#axs[1].set_ylim(0, 1)



#plt.plot(xx,ytrue)
#plt.plot(xx,-ypL, label='pL')

#plt.plot(xx,-ypR, label='pR')
plt.xlim(xmin, xmax)

plt.legend()

xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)

#figname = 'Stall_Type1'+str(i)
#plt.tight_layout()

#plt.savefig(figname+'.pdf')
#plt.savefig(figname+'.eps')
#plt.savefig(figname+'.png', dpi=600)

plt.show()

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


#%%






#%%

print("%10s |%14s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xnew' , 'newbracket', 'a', 'b', 'c', 'check'))
print("-"*(20+14*5))


Npts  = 1000
alphas = np.array([i/200 for i in range(Npts)])

acs = np.zeros([Npts])

bracket = np.zeros([Niters])
xMhist = np.zeros([Niters])
xnewhist = np.zeros([Niters])
ahist = np.zeros([Niters])
bhist = np.zeros([Niters])
chist = np.zeros([Niters])

for i in range(Npts):
   alpha = alphas[i]
   
   pL = np.poly1d([alpha*1, alpha*(-xL[0] -xL[1]),alpha* xL[0]*xL[1]])
   pR = np.poly1d([1, (-xR[0] -xR[1]), xR[0]*xR[1]])
   
   yL1 = xL[0] - ans
   yL2 = xL[1] - ans
   yL3 = xL[2] - ans
   yR1 = xR[0] - ans
   yR2 = xR[1] - ans
   yR3 = xR[2] - ans
   
   a =1-alpha
   b = -(yR1 + yR2) + (yL1 + yL2)*alpha
   c = yR1*yR2 - alpha*yL1*yL2
   
   acs[i] = a*c
   

#%%
   
   
plt.plot(alphas, acs)
   
   