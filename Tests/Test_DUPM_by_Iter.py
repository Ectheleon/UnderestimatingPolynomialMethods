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
import os
os.chdir('/home/user/Documents/Projects/ResearchProject/SiemensCode/3dv1d/JonathansCode/ContinuousOptimiser/python/Tests/')
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
import NSLS_merit_fctn as mt
importlib.reload(mt)
import copy
plt.rc('text', usetex=True)
plt.rc('figure', figsize=(12, 8))
plt.rc('font', size=18)


#%% Set up problem

# 1. Choose test function 
tcname = 'NU3'

tmp = getattr(lst, tcname)()

f = tmp.func

# 2. Choose starting grids
#xL, fL, xM, fM, xR, fR = tmp.xL, tmp.fL, tmp.xM, tmp.fM, tmp.xR, tmp.fR


M = tmp.M
# Record the true solution

i = 0
alpha = 0

xL, xM, xR , fL, fM ,fR = tmp.setup_grid()

#xL = np.array([0.65197749, 0.41591187, 0.41590187])
#xM = 1.0339396788250546
#xR = np.array([1.17983625, 1.2700053 , 1.41590187])

#fL = np.array([0.27382425, 0.2927476 , 0.29274846])
#fM = 0.25870589400656835
#fR = np.array([0.30184159, 0.33412614, 0.39850689])
xL0 = copy.copy(xL)
xM0 = copy.copy(xM)
xR0 = copy.copy(xR)
fL0 = copy.copy(fL)
fM0 = copy.copy(fM)
fR0 = copy.copy(fR)
xL0 = copy.copy(xL)
xM0 = copy.copy(xM)
xR0 = copy.copy(xR)
fL0 = copy.copy(fL)
fM0 = copy.copy(fM)
fR0 = copy.copy(fR)





delta = 1e-5


#%%

xL = copy.copy(xL0)
xM = copy.copy(xM0)
xR = copy.copy(xR0)
fL = copy.copy(fL0)
fM = copy.copy(fM0)
fR = copy.copy(fR0)
i = 0
alpha = 0

def construct_poly(xx, fx, alpha, hx):
    c = fx[0]
    b = opt.dd(2, xx[:2], fx[:2])
    a = opt.dd(3, xx[:3], fx[:3]) - alpha*hx
    
    return np.poly1d([a, b-a*(xx[0]+xx[1]), c - b*xx[0]+a*xx[0]*xx[1]])
        
def dupm_step(alpha, xL, xR, fL, fR):
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
    pL = construct_poly(xL, fL, alpha, hx)
    pR = construct_poly(xR, fR, alpha, hx)
    
    ans  = opt.min_max_quad(pL, pR, xL[0], xR[0])
    #print('dupm_step', ans, (pL-pR).roots)
    return ans

def dupm_step_inv(xtest, xL, xR, fL, fR):
    #alpha0, x0 = intersect_threshold(xL, fL, xR, fR)
    #x0 = dupm_step(alpha0, xL, xR, fL, fR) if alpha0>-np.inf else 
    xinf = limit_option(xL, xR)
    
    alpha1, x1 = nsmin_threshold(xL, fL, xR, fR)
            
    if ((xtest>=x1 and xtest <= xinf) or (xtest>= xinf and xtest <= x1)):
        #simple inverse case
        a, b1, b2, c1, c2 = shorthand(xL, fL, xR, fR)
        
        return -(c1 + xtest*b1 + xtest**2*a)/(xtest*b2 + c2)
    else:
        #xtest is between x0 and x1
        return alpha1

def is_smooth_min(pL, pR, a,b):
        
    if pR.order==2:
        tR = pR.deriv().roots[0]
        if tR >= a and tR<= b:
            if pR(tR) >= pL(tR):
                return True
        else:
            if pR(a)>= pL(a):
                return True
                
    if pL.order ==2:
        tL = pL.deriv().roots[0] 
        if tL>= a and tL <= b:
            if pL(tL)>= pR(tL):
                return True
        else:
            if pL(b)>=pR(b):
                return True
            
    if pL.order ==1:
        if pL(b)>=pR(b):
            return True 
        
    if pR.order==1:
        if pR(a)>= pL(a):
            return True

    return False

def valid_slope_threshold(xL, xR, fL, fR):
    #computes the minimum value of alpha which ensures that the slope of pL at
    #xL1 is negative and the slope of pR at xR1 is positive.
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
    fL2 = opt.dd(2, xL[:2], fL[:2])
    fR2 = opt.dd(2, xR[:2], fR[:2])
    fR3 = opt.dd(3, xR[:3], fR[:3]) #-  alpha*hx
    fL3 = opt.dd(3, xL[:3], fL[:3]) #- alpha*hx
    
    boundL = fL2/(hx*(xL[0] - xL[1])) + fL3/hx
    boundR = fR2/(hx*(xR[0] - xR[1])) + fR3/hx
    
    return max(boundL, boundR)
    
def underestimation_threshold(xL, xR, xM, fL, fR, fM):
    #The minimum value of alpha such that pL and pR underestimate f at xM.
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
    fR3 = opt.dd(3, xR[:3], fR[:3]) #-  alpha*hx
    fL3 = opt.dd(3, xL[:3], fL[:3]) #- alpha*hx
    
    fMLL = opt.dd(3, [xM, xL[0], xL[1]], [fM, fL[0], fL[1]])
    fMRR = opt.dd(3, [xM, xR[0], xR[1]], [fM, fR[0], fR[1]])
    
    boundR = (fR3 -fMRR)/hx
    boundL = (fL3 - fMLL)/hx
    
    return max(boundL, boundR)

def intersect_threshold(xL, fL, xR, fR):
    #This function computes the value of alpha such that for all values of 
    #alpha greater than this one, pL intersects with pR
   
    a, b1, b2, c1, c2 = shorthand(xL, fL, xR, fR)
    
    disc = a**2 * c2**2 + a*c1*b2**2 - a*b1*b2*c2
    
    if disc<0:
        return 0, (-b1+np.sqrt(b1**2 - 4*a*c1))/(2*a)
    else:
        a0 = (-b1*b2 + 2*a*c2 + 2*np.sqrt(disc))/(b2**2)
        return a0, (-b1-a0*b2)/(2*a)
        
def nsmin_threshold(xL, fL, xR, fR):
    #This function computes the value of alpha such that for all greater values, 
    #The min of the max of pL and pR is found at their intersection.
    bmin, _ = intersect_threshold(xL, fL, xR, fR)
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))

    #print('stage 1')
    #first find bmax
    
    fR3 = opt.dd(3, xR[:3], fR[:3]) #-  alpha*hx
    fL3 = opt.dd(3, xL[:3], fL[:3]) #- alpha*hx

    bmax= max(fR3, fL3)
        
    #Now bmin, bmax bracket the critical value of alpha which is where the 
    #minimum is first nonsmoooth.
    
    #Now bisect!
    while bmax-bmin>1e-6:
        #print(bmax, bmin, fR3, fL3)
        pL = construct_poly(xL, fL, 0.5*(bmin+bmax), hx)
        pR = construct_poly(xR, fR, 0.5*(bmin+bmax), hx)

        if is_smooth_min(pL, pR, xL[0], xR[0]):
            #print('smin')
            bmin = 0.5*(bmin+bmax)
        else:
            #print('nsmin')
            bmax = 0.5*(bmin+bmax)
        #print(bmin, bmax)
    #print('done!', bmin, bmax)
    
    return 0.5*(bmin+bmax), dupm_step(0.5*(bmin+bmax), xL, xR, fL, fR)

def shorthand(xL, fL, xR, fR):
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
    fL1 = fL[0]
    fL2 = opt.dd(2, xL[:2], fL[:2])
    
    fR1 = fR[0]     
    fR2 = opt.dd(2, xR[:2], fR[:2])
    
    fR3 = opt.dd(3, xR[:3], fR[:3]) #-  alpha*hx
    fL3 = opt.dd(3, xL[:3], fL[:3]) #- alpha*hx

    a = fR3 - fL3
    b1 = fR2 - fL2 - (xR[0] + xR[1])*fR3 + (xL[0]+xL[1])*fL3
    b2 = hx*(xR[0] + xR[1] - xL[0] - xL[1])
    c1 = fR1 - fL1 - xR[0]*fR2 + xL[0]*fL2 + xR[0]*xR[1]*fR3 - xL[0]*xL[1]*fL3
    c2 = -hx*(xR[0]*xR[1] - xL[0] * xL[1])
    
    return a, b1, b2, c1, c2

def limit_option(xL, xR):
    return (xR[0]*xR[1]-xL[0]*xL[1])/(xR[0]+xR[1]-xL[0]-xL[1])


def modify_alpha(xR, xL, xM, fR, fL, fM, alphaold, xnew, d1, d2, d3):
    #Check that both sides have been populated recently:

    xinf = (xR[0]*xR[1]-xL[0]*xL[1])/(xR[0]+xR[1]-xL[0]-xL[1])
    if d1==d2 and d1==d3:
        print('Increasing alpha')
        #In this case increase alpha
        
        return dupm_step_inv(0.5*(xnew+xinf), xL, xR, fL, fR), True

    else:
        print('Reducing alpha')

        hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
        pL = construct_poly(xL, fL, alphaold, hx)
        pR = construct_poly(xR, fR, alphaold, hx)
    
        #Now check if alpha should be reduced
        pstar = max(pL(xM), pR(xM))
        fstar = 0.5*(pstar + fM)

        if pstar<fM:
            fL1 = fL[0]
            fL2 = opt.dd(2, xL[:2], fL[:2])
            
            fR1 = fR[0]     
            fR2 = opt.dd(2, xR[:2], fR[:2])
            
            fR3 = opt.dd(3, xR[:3], fR[:3]) #-  alpha*hx
            fL3 = opt.dd(3, xL[:3], fL[:3]) #- alpha*hx

            #Identify the 'dominant' polynomial. And reduce by half the amaount by which it underestimates f at xM
            
            print('before modification', pstar, fM)
            print('after modification', fstar)
            if xnew>xinf:
                    
                tmp1 = (fstar - fL1)/(xM - xL[0])
                tmp2 = (tmp1 - fL2)/(xM - xL[1])
                return max(0, (fL3 - tmp2)/hx), True
            else:
                
                tmp1 = (fstar - fR1)/(xM - xR[0])
                tmp2 = (tmp1 - fR2)/(xM - xR[1])
                return max(0, (fR3 - tmp2)/hx), True

    #Shouldn't happend, but just in case
    print('PROBLEM')
    return alphaold, False



#%% Iterate


print('Default value of alpha', alpha)
alpha0, x0 = intersect_threshold(xL, fL, xR, fR)
alpha1, x1 = nsmin_threshold(xL, fL, xR, fR)

#tmp = first_check(xL, fL, xR, fR, xM, fM)

alpha = max(alpha1, alpha)
print('Alpha modified to ensure nsmin', alpha)

lb_alpha = valid_slope_threshold(xL, xR, fL, fR)
#alpha = max(alpha, lb_alpha)
print('Alpha modified to ensure valid slopes', lb_alpha, alpha)

tmp = underestimation_threshold(xL, xR,xM, fL, fR, fM)
lb_alpha = max(lb_alpha, tmp)
print('Alpha modified to ensure underestimation', lb_alpha, alpha)

alpha = max(alpha, lb_alpha)


#tmp = second_check(xL, fL, xR, fR, xM, fM, alpha)
###print('Alpha', alpha, tmp)
#alpha = max(tmp, alpha)


#print(alpha, tmp)


hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
pL = construct_poly(xL, fL, alpha, hx)

pR = construct_poly(xR, fR, alpha, hx)

#pL, pR = construct_polys(xL, fL, xR, fR, alpha)

xnew = opt.min_max_quad(pL, pR, xL[0], xR[0])
xinf = limit_option(xL, xR)

if i>2:
    alpha , modified= modify_alpha(xR, xL, xM, fR, fL, fM, alpha, xnew, update_side, prev_update, prev_update2)
    
    alpha = max(alpha, lb_alpha)
    
    if modified:
        print('Alpha following additional modification', alpha)
        pL = construct_poly(xL, fL, alpha, hx)
        pR = construct_poly(xR, fR, alpha, hx)
        
        #pL, pR = construct_polys(xL, fL, xR, fR, alpha)
        
        xnew = opt.min_max_quad(pL, pR, xL[0], xR[0])
        
        xalt = dupm_step(alpha, xL, xR, fL, fR)
        print('comparison', xnew, xalt)

    
    
#    if update_side == prev_update and update_side==prev_update2:
#        print('Speed up correction', alpha)
#        x_prev, xnew, alpha = xnew, 0.5*(xnew+xinf), dupm_step_inv(0.5*(xnew+xinf), xL, xR, fL, fR)
#        print('sanity 2', dupm_step_inv(0.5*(xnew+xinf), xL, xR, fL, fR), alpha)



fnew = f(xnew)

print("%10s |%14s |%14s |%14s |%14s |%14s" %('iteration', 'xL0' , 'xM', 'xR0', 'xnew', 'bracket'))
print("-"*(20+14*5))
print("%10d |%14.10f |%14.10f |%14.10f |%14.10f |%14.10f" %(i,  xL[0], xM, xR[0], xnew, xR[0] - xL[0]))





xmin  = xL[1] 
xmax = xR[1]
#xmax = xR[1]


ymax = max(f(xmax), f(xmin))
ymin = min(f(xM), f(xnew))
#print(ymin, ymax)
dist = ymax - ymin

ymax += 0.1*dist
ymin -= 0.1*dist
#print(ymin,ymax)

xx = np.arange(xmin, xmax, (xR[1] - xL[1])/500)

ytrue = np.array([f(x) for x in xx])
ypL = np.array([pL(x) for x in xx])
ypR = np.array([pR(x) for x in xx])



plt.plot(xx,ytrue)
plt.plot(xx,ypL, label='pL')

plt.plot(xx,ypR, label='pR')
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)


plt.vlines(xinf, ymin, ymax, color='y', label=r'$x_\infty$')
plt.vlines(xM, ymin, ymax, color='r', label=r'$x_{min}$')
plt.vlines(xnew, ymin, ymax, color='g', label=r'$x_{test}$')

plt.vlines(xL[0], ymin, ymax)
plt.vlines(xL[1], ymin, ymax)
plt.vlines(xR[0], ymin, ymax)
plt.vlines(xR[1], ymin, ymax)

plt.legend()
#xL, xR, xM, fL, fR, fM = opt.process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew)



if i >1:
    prev_update, prev_update2 = copy.copy(update_side), prev_update
elif i>0:
    prev_update = copy.copy(update_side)

if fnew < fM:
    if xnew < xM:
        update_side = 'R'
        xR, xM, fR, fM= np.concatenate(([xM], xR)), xnew, np.concatenate(([fM], fR)), fnew
    else:
        update_side = 'L'
        xL, xM, fL, fM= np.concatenate(([xM], xL)), xnew, np.concatenate(([fM], fL)), fnew
else:
    if xnew < xM:
        update_side = 'L'
        xL, fL = np.concatenate(([xnew], xL)), np.concatenate(([fnew], fL))
    else:
        update_side = 'R'
        xR, fR = np.concatenate(([xnew], xR)), np.concatenate(([fnew], fR))


#figname = 'Stall_Type1'+str(i)
#plt.tight_layout()

#plt.savefig(figname+'.pdf')
#plt.savefig(figname+'.eps')
#plt.savefig(figname+'.png', dpi=600)

if i==0:
    print('First Iteration', update_side)
elif i==1:
    print('Last two iterations', update_side, prev_update)
else:
    print('Last three iterations', update_side, prev_update, prev_update2)
i+=1


#%%


#%%
xL, xM, xR , fL, fM ,fR = tmp.setup_grid()

alpha = nsmin_threshold(xL, fL, xR, fR, xM, fM)

pL, pR = construct_polys(xL, fL, xR, fR, alpha)


xx = np.arange(xL[0], xR[0], 0.01)
yyL = np.array([pL(x) for x in xx])
yyR =np.array([pR(x) for x in xx])

plt.plot(xx,yyL)
plt.plot(xx,yyR)






#%%
alpha0, x0 = intersect_threshold(xL, fL, xR, fR)
#x0 = dupm_step(alpha0, xL, xR, fL, fR)

alpha1, x1 = nsmin_threshold(xL, fL, xR, fR)
xinf = limit_option(xL, xR)

a, b1, b2, c1, c2 = shorthand(xL, fL, xR, fR)

print('nums')
print(a)
print(b1)

print(b2)
print(c1)
print(c2)


#disc = a**2 * c2**2 + a*c1*b2**2 - a*b1*b2*c2

#alpha0_alt = (-b1*b2 + 2*a*c2 + 2*np.sqrt(disc))/(b2**2)
#x0_alt = -b1/(2*alpha0_alt)-b2/2

#print((b1+b2*alpha0)**2 - 4*a*(c1+c2*alpha0))


pL, pR = construct_polys(xL, fL, xR, fR, alpha0)

xx = np.arange(xL[0], xR[0], 0.01)
yyL = np.array([pL(x) for x in xx])
yyR =np.array([pR(x) for x in xx])

plt.plot(xx,yyL)
plt.plot(xx,yyR)

plt.vlines(xinf, ymin, ymax, color='y', label=r'$x_\infty$')
plt.vlines(x0, ymin, ymax, color='b', label=r'$x_0$')
plt.vlines(x1, ymin, ymax, color='g', label=r'$x_1$')
plt.legend()

p = pR-pL
print(p.roots)

print((-b1 -b2*alpha0)/(2*a))