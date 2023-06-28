#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:09:34 2019

@author: Jonathan GPU
"""

import numpy as np
import Optimize1D

def construct_poly(xx, fx, alpha, hx):
    c = fx[0]
    b = Optimize1D.dd(2, xx[:2], fx[:2])
    a = Optimize1D.dd(3, xx[:3], fx[:3]) - alpha*hx
    
    return np.poly1d([a, b-a*(xx[0]+xx[1]), c - b*xx[0]+a*xx[0]*xx[1]])

def is_smooth_min(pL, pR, a,b):
    #print('polys')
    #print()
    #print(pL)
    #print()
    #print(pR)
    #print('done')
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
            if pL(tL)>= pR(tR):
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

def shorthand(xL, fL, xR, fR):
    hx = max(np.fabs(xL[2] - xR[0]), np.fabs(xR[2] - xL[0]))
    fL1 = fL[0]
    fL2 = Optimize1D.dd(2, xL[:2], fL[:2])
    
    fR1 = fR[0]     
    fR2 = Optimize1D.dd(2, xR[:2], fR[:2])
    
    fR3 = Optimize1D.dd(3, xR[:3], fR[:3]) #-  alpha*hx
    fL3 = Optimize1D.dd(3, xL[:3], fL[:3]) #- alpha*hx

    a = fR3 - fL3
    b1 = fR2 - fL2 - (xR[0] + xR[1])*fR3 + (xL[0]+xL[1])*fL3
    b2 = hx*(xR[0] + xR[1] - xL[0] - xL[1])
    c1 = fR1 - fL1 - xR[0]*fR2 + xL[0]*fL2 + xR[0]*xR[1]*fR3 - xL[0]*xL[1]*fL3
    c2 = -hx*(xR[0]*xR[1] - xL[0] * xL[1])
    
    return a, b1, b2, c1, c2


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





class UPM(Optimize1D.Opt1d):
    '''
    UPM: Underestimating Polynomial Method
    
    A parent class for the three variations of this algorithm.
    '''
    def __init__(self, xL, xR, fL, fR, M, fM, eps=1e-5,min_step = 1e-6, trustUser=True, alpha = 100):
        '''
        xL: The values of xL1, xL2, and xL2
        xR: as above
        fL: The values of f at xL
        xR: as above
        ...
        eps: the algorithm terminates once the bracket is smaller than epsilon
        min_step: the smallest step the algorithm is allowed to take. For numerical stability.
        alpha: The parameter which is the main controllable option for the UPM if applicalbe
        '''

        self.xL0 = np.sort(xL)[::-1]
        self.xR0 = np.sort(xR)
        self.fL0 = np.array([x for _,x in sorted(zip(xL, fL))])[::-1]
        self.fR0 = self.fR = np.array([x for _,x in sorted(zip(xR, fR))])
        self.alpha = alpha
        self.status = 0 #Not run yet
        
        super().__init__([self.xL0[0], self.xR0[0]], M, np.array([self.fL0[0], self.fR0[0]]), fM, eps,min_step, trustUser=trustUser)

    def not_converged(self, maxiter):
        return self.acc[-1]>self.eps and self.niter < maxiter

    def set_up_problem(self):
        self.xR = self.xR0
        self.xL = self.xL0
        self.fR = self.fR0
        self.fL = self.fL0
        super().set_up_problem()

    def update_brackets(self, xtest, ftest):
        #print(self.xL[0], self.xmin, self.xR[0], xtest)
        if ftest < self.fxmin:
            if xtest < self.xmin:
                self.xR = np.insert(self.xR, 0,self.xmin)
                self.fR = np.insert(self.fR, 0,self.fxmin)
            else:
                self.xL = np.insert(self.xL,0,self.xmin)
                self.fL = np.insert(self.fL,0, self.fxmin)
        else:
            if xtest<self.xmin:
                self.xL = np.insert(self.xL, 0,xtest)
                self.fL = np.insert(self.fL, 0,ftest)
            else:
                self.xR = np.insert(self.xR, 0,xtest)
                self.fR = np.insert(self.fR, 0,ftest)
        #print(self.xL[0], self.xmin, self.xR[0], xtest)
        super().update_brackets(xtest, ftest)
        #print(self.xL[0], self.xmin, self.xR[0], xtest)
    
    def solve(self, f, maxiter=60):
        self.set_up_problem()
        
        while self.not_converged(maxiter):
            step = self.determine_step()  #compute new step
            xtest = self.stabilize_step(step)
            #print(np.fabs(xtest-self.xmin), np.fabs(self.xL[0] - xtest), np.fabs(xtest - self.xR[0]), xtest)
            #print(self.xL[:3], self.xmin, self.xR[:3])
            #print(xtest == self.xL[0])


            #Compute f at new xtest
            ftest = f(xtest) 
            
            #record data
            self.update_brackets(xtest, ftest) #update the algorithm parameters
            #print(self.xL[:3], self.xmin, self.xR[:3])
            #print(self.xL[0]==xtest, self.xL[0] == self.xL[1])

            self.niter +=1
            self.acc.append(self.b-self.a)
            self.lbs.append(self.a)
            self.ubs.append(self.b)
            self.guess.append(self.xmin)
            
            
        self.status = 1 if self.niter<maxiter else -1

        return self.xmin
    
    def print_progress(self,xtest, step, oldstep):
        print(' {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(self.xL[1], self.xL[0], self.xmin, self.xR[0], self.xR[1], xtest, step, oldstep))




class SUPM(UPM):
    '''
    SUPM: Static Underestimating Polynomial Method.
    
    The word static refers to the fact that the value of alpha remains constant.
    '''

    def determine_step(self):
        
        hx = max(np.fabs(self.xL[2] - self.xR[0]), np.fabs(self.xR[2] - self.xL[0]))
        #hx = self.xR[0]-self.xL[0]
        
        pL = construct_poly(self.xL, self.fL, self.alpha, hx)
        pR = construct_poly(self.xR, self.fR, self.alpha, hx)

        xnew = Optimize1D.min_max_quad(pL, pR, self.xL[0], self.xR[0])
        return xnew -self.xmin

    
class EUPM(UPM):
    '''
    EUPM: Extremal Underestimating Polynomial Method
    
    The value of alpha is taken to be infinity
    '''
    def determine_step(self):
        xnew = (self.xR[0]*self.xR[1] - self.xL[0]*self.xL[1])/(self.xR[0]+self.xR[1]-self.xL[0]-self.xL[1])
        return xnew -self.xmin

class DUPM(SUPM):
    '''
    DUPM: Dynamic Underestimating Polynomial Method
    
    The value if alpha is dynamically altered.
    '''
    def __init__(self, xL, xR, fL, fR, M, fM, eps=1e-5,min_step = 1e-6, trustUser=True, alpha = 0):
        self.alpha_min = 1e-2
        if np.fabs(M-xL[0])<np.fabs(M - xR[0]):
            self.u1 , self.u2, self.u3 = 'L', 'L', 'R'
        else:
            self.u1 , self.u2, self.u3 = 'R', 'R', 'L'

        super().__init__(xL, xR, fL, fR, M, fM, eps,min_step, trustUser, max(self.alpha_min, alpha))

    def determine_step(self):
        #First ensure alpha is big enough given what it already known
        self.alpha, xnew = self.modify_alpha()
        return xnew- self.xmin
                
    def update_brackets(self, xtest, ftest):
        #print(self.xL[0], self.xmin, self.xR[0], xtest)
        if ftest < self.fxmin:
            if xtest < self.xmin:
                self.u1, self.u2, self.u3 = 'R',self.u1, self.u2
            else:
                self.u1, self.u2, self.u3 = 'L',self.u1, self.u2
        else:
            if xtest<self.xmin:
                self.u1, self.u2, self.u3 = 'L',self.u1, self.u2
            else:
                self.u1, self.u2, self.u3 = 'R',self.u1, self.u2
        #print(self.xL[0], self.xmin, self.xR[0], xtest)
        super().update_brackets(xtest, ftest)
        #print(self.xL[0], self.xmin, self.xR[0], xtest)

    def gE(self):
        return (self.xR[0]*self.xR[1]-self.xL[0]*self.xL[1])/(self.xR[0]+self.xR[1]-self.xL[0]-self.xL[1])
    
    def valid_slope_threshold(self):
        #computes the minimum value of alpha which ensures that the slope of pL at
        #xL1 is negative and the slope of pR at xR1 is positive.
        hx = max(np.fabs(self.xL[2] - self.xR[0]), np.fabs(self.xR[2] - self.xL[0]))
        fL2 = Optimize1D.dd(2, self.xL[:2], self.fL[:2])
        fR2 = Optimize1D.dd(2, self.xR[:2], self.fR[:2])
        fR3 = Optimize1D.dd(3, self.xR[:3], self.fR[:3]) #-  alpha*hx
        fL3 = Optimize1D.dd(3, self.xL[:3], self.fL[:3]) #- alpha*hx
        
        boundL = fL2/(hx*(self.xL[0] - self.xL[1])) + fL3/hx
        boundR = fR2/(hx*(self.xR[0] - self.xR[1])) + fR3/hx
        
        return max(boundL, boundR)
        
    def underestimation_threshold(self):
        #The minimum value of alpha such that pL and pR underestimate f at xM.
        hx = max(np.fabs(self.xL[2] - self.xR[0]), np.fabs(self.xR[2] - self.xL[0]))
        fR3 = Optimize1D.dd(3, self.xR[:3], self.fR[:3]) #-  alpha*hx
        fL3 = Optimize1D.dd(3, self.xL[:3], self.fL[:3]) #- alpha*hx
        
        fMLL = Optimize1D.dd(3, [self.xmin, self.xL[0], self.xL[1]], [self.fxmin, self.fL[0], self.fL[1]])
        fMRR = Optimize1D.dd(3, [self.xmin, self.xR[0], self.xR[1]], [self.fxmin, self.fR[0], self.fR[1]])
        
        boundR = (fR3 -fMRR)/hx
        boundL = (fL3 - fMLL)/hx
        
        return max(boundL, boundR)

    
    def nsmin_threshold(self):
        #This function computes the value of alpha such that for all greater values, 
        #The min of the max of pL and pR is found at their intersection.
        bmin, _ = intersect_threshold(self.xL, self.fL, self.xR, self.fR)
        hx = max(np.fabs(self.xL[2] - self.xR[0]), np.fabs(self.xR[2] - self.xL[0]))
        
        #print('stage 1')
        #first find bmax
        
        fR3 = Optimize1D.dd(3, self.xR[:3], self.fR[:3]) #-  alpha*hx
        fL3 = Optimize1D.dd(3, self.xL[:3], self.fL[:3]) #- alpha*hx
        
        bmax= max(fR3, fL3)
            
        #Now bmin, bmax bracket the critical value of alpha which is where the 
        #minimum is first nonsmoooth.
        
        #Now bisect!
        while bmax-bmin>1e-6:
            pL = construct_poly(self.xL, self.fL, 0.5*(bmin+bmax), hx)
            pR = construct_poly(self.xR, self.fR, 0.5*(bmin+bmax), hx)
        
            if is_smooth_min(pL, pR, self.xL[0], self.xR[0]):
                bmin = 0.5*(bmin+bmax)
            else:
                bmax = 0.5*(bmin+bmax)
            #print(bmin, bmax)
        #print('done!', bmin, bmax)
        
        return 0.5*(bmin+bmax)

    def modify_alpha(self):
        #this function returns a suitable value of alpha given the current one
        hx = max(np.fabs(self.xL[2] - self.xR[0]), np.fabs(self.xR[2] - self.xL[0]))
        #alpha1 = self.valid_slope_threshold()
        alpha2 = self.underestimation_threshold()
        alpha3 = self.nsmin_threshold()
        
        self.alpha = max(self.alpha, alpha1, alpha2, alpha3)
        
        pL = construct_poly(self.xL, self.fL, self.alpha, hx)
        pR = construct_poly(self.xR, self.fR, self.alpha, hx)
        xnew  = Optimize1D.min_max_quad(pL, pR, self.xL[0], self.xR[0])
        
        #Check that both sides have been populated recently:
    
        if self.u1==self.u2 and self.u1==self.u3:
            #print('Increasing alpha')
            #In this case increase alpha
            
            return self.alpha, self.gE()#self.gD_inv(0.5*(xnew+xinf)), self.gE()
        else:
            #Shouldn't happend, but just in case
            #print('PROBLEM, underestimate threshold seems to have failed')
            return self.alpha, xnew
            
