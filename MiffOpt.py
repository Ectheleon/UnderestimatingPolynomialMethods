#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:38:10 2020

@author: Jonathan GPM
"""

import UPM
import Optimize1D
import numpy as np
import scipy.interpolate as si

def project_interval(x, a,b):
    #projects the point x onto the interval [a,b]
    
    if x<= a:
        return a
    elif x>= b:
        return b
    else :
        return x

class MiffOpt(UPM.EUPM):
    '''
    My implementation of the algorithm of Mifflin and Strodiot taken from the paper:
        "A rapidly convergent five point algorithm for univarite minimization"
    
    It uses the same five point structure and update function as the EUPM, hence
    I inherit those functions.
    '''
    def __init__(self, xL, xR, fL, fR, M, fM, eps=1e-5,min_step = 1e-6, trustUser=True, alpha = 100):
        
        super().__init__(xL, xR, fL, fR, M, fM, eps=1e-5,min_step = 1e-6, trustUser=True, alpha = 100)
        
        
        t1 = 0.25/(xR[0] - xL[0])
        t2 = (M - xL[0])/(xR[0] - xL[0])**2
        t3 = (xR[0] - M)/(xR[0] - xL[0])**2
        
        self.alpha = min( min_step, 0.5*min(t1, t2, t3))


    def solve(self, f, maxiter=60):
        #MiffOpt has its own method of ensuring numerical stability. Therefore we
        #remove the stabilise_step from the parent method.
        self.set_up_problem()
        
        while self.not_converged(maxiter):
            step = self.determine_step()  #compute new step
            #xtest = self.stabilize_step(step)
            xtest = self.xmin +step
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

    
    
    def determine_step(self):
        
        #1. Construct quadratic which interpolates xL1, xL2, xM
        #   a) gm = derivative of quadratic at xM
        #   b) Qm = constrained minimiser of quadratic in [xL1, xR1]
        #   c) ghatm = left linear slope. f[xL1, xL2]
        #   d) pm = f[xM, xL1] - ghatm, the left linearisation error.

        qL = np.poly1d(np.polyfit([self.xmin,self.xL[0],self.xL[1]] , [self.fxmin,self.fL[0],self.fL[1]] ,2))
                
        gL = qL.deriv()(self.xmin)
        
        if qL.order==2:
            QL = qL.deriv().roots[0] if qL.deriv().roots[0] <= self.xR[0] else self.xR[0]
        else:
            QL = self.xR[0]
        ghatL = (self.fL[0] - self.fL[1])/(self.xL[0] - self.xL[1])
        pL = self.fxmin - self.fL[0] - ghatL*(self.xmin - self.xL[0])

        
        #2. Construct quadratic which interpolates xR1, xR2, xM
        #   a) gp = derivative of quadratic at xM
        #   b) Qp = constrained minimiser of quadratic in [xL1, xR1]
        #   c) ghatp = right linear slope. f[xR1, xR2]
        #   d) pp = f[xM, xR1] - ghatm, the right linearisation error.
        
        qR = np.poly1d(np.polyfit([self.xmin,self.xR[0],self.xR[1]] , [self.fxmin,self.fR[0],self.fR[1]] ,2))
                
        gR = qR.deriv()(self.xmin)
        
        if qR.order==2:
            QR = qR.deriv().roots[0] if qR.deriv().roots[0] >= self.xL[0] else self.xL[0]
        else:
            QR = self.xL[0]
        
        ghatR = (self.fR[0] - self.fR[1])/(self.xR[0] - self.xR[1])
        pR = self.fxmin - self.fR[0] - ghatR*(self.xmin - self.xR[0])

        

        #3. Determine Rm and Rp ?
        #   a) Set Rp = Qp and Rm = Qm
        #   b) If Qm>xM, Qp >= xM and pp < (ghatp - gm)/(Qm - xM)
        #           Pm = xM + pp/(ghatp - gm)
        #           Rm = Pm
        #   c) If Qm<=xM, Qp < xM and pm < (gp - ghatm)/(xM - Qp)
        #           Pp = xM - pm/(gp - ghatm)
        #           Rp = Pp

        RR, RL = QR, QL
        
        if QL>self.xmin and QR>=self.xmin and pR<(ghatR - gR)/(QL-self.xmin):
            PL = self.xmin + pR/(ghatR - gL)
            RL = PL

        if QL<=self.xmin and QR<self.xmin and pL<(gR - ghatL)/(self.xmin -QR):
            PR = self.xmin - pL/(gR - ghatL)
            RR = PR

        #4. Determine R
        #   a) Set L = min(Rp, Rm), U = max(Rm, Rp))
        #   b) If ghat+>0 and ghatm<0, Po = xM + (pp-pm)/(ghat+-ghatm), 
        #       else Po = local min of quadratic interpolating xL1, xM, xR1
        #   c) R = projection of Po onto [L,U]
            
        L, U = min(RR, RL), max(RR, RL)
        
        if ghatR>0 and ghatL<0:
            P0 = self.xmin + (pR-pL)/(ghatR - ghatL)
        else:
            #print([self.xL[0], self.xmin, self.xR[0]], [self.fL[0], self.fxmin, self.fR[0]])
            q0 = np.poly1d(np.polyfit([self.xL[0], self.xmin, self.xR[0]], [self.fL[0], self.fxmin, self.fR[0]], 2))
            #print(q0)
            
            if q0.order==2:
                P0 = q0.deriv().roots[0]
            else:
                P0 = self.xmin + self.golden_step()
            
        R = project_interval(P0, L, U)        
        
        #5. Determine x from R via safeguarding
        #   a) Set sigma = s(xR1-xL1)^2
        #   b) x = projection of R onto [xL1+sigma, xR1-sigma]
        #       If |x-xM|<sigma, replace x by:
        #       xM+sigma if xM<= 0.5(xR1+xL1) else xM - sigma
        
        sigma = self.alpha*(self.xR[0] - self.xL[0])**2
        x = project_interval(R, self.xL[0]+sigma, self.xR[0] - sigma)
            
        if np.fabs(x - self.xmin) < sigma:
            if self.xmin <= 0.5*(self.xL[0] + self.xR[0]):
                return sigma
            else:
                return - sigma
        
        return x - self.xmin