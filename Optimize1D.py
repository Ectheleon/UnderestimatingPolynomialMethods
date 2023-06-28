#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:50:08 2019

@author: Jonathan Grant-Peters
"""

import numpy as np
import abc
import matplotlib.pyplot as plt
import GoldenSection as gs


def newton_poly(n, x, y):
    '''Constructs the Newton Interpolating Polynomial from the data
    
        This function is only designed for n = 2 or n = 3.
    '''
    if len(x) != n or len(y) != n:
        raise ValueError('The arrays x and y should be of length n')
    else:
        F = np.zeros([n,n])
         
        for i in range(n):
            F[i,0] = y[i]
            
        for j in range(1,n):
            for i in range(n-j):
                F[i,j] = (F[i,j-1]-F[i+1,j-1])/(x[i] - x[i+j])
                
        poly = np.empty([n])
        '''poly[i] is the coefficienty of x^i in the interpolating polynomial'''
        
        
        poly[0] = F[0,0] - x[0]*F[0,1]
        poly[1] = F[0,1]
        
        if n ==2:
            return poly
        else:
            poly[0] += x[0]*x[1]*F[0,2]
            poly[1] += -(x[0] + x[1])*F[0,2]
            poly[2] = F[0,2]

            return poly

def dd(n, x, y):
    '''dd is short for divided difference'''
    if len(x) != n or len(y) != n:
        raise ValueError('The arrays x and y should be of length n')
    else:
        if n==1:
            return y[0]
        else:
            if x[0] == x[1]:
                raise ValueError('Cannot compute a divided difference using the same point twice', x, y)
            return (dd(n-1,x[:-1],y[:-1]) - dd(n-1, x[1:],y[1:]))/(x[0] - x[-1]) 

def process_new_point(xL, xR, xM, fL, fR, fM, xnew, fnew):
    assert(xnew>=xL[0])
    assert(xnew<=xR[0])
    
    if fnew < fM:
        if xnew < xM:
            xR, xM, fR, fM= np.concatenate(([xM], xR)), xnew, np.concatenate(([fM], fR)), fnew
        else:
            xL, xM, fL, fM= np.concatenate(([xM], xL)), xnew, np.concatenate(([fM], fL)), fnew
    else:
        if xnew < xM:
            xL, fL = np.concatenate(([xnew], xL)), np.concatenate(([fnew], fL))
        else:
            xR, fR = np.concatenate(([xnew], xR)), np.concatenate(([fnew], fR))

    
    return xL, xR, xM, fL, fR, fM


def min_max_poly(pL, pR, a,b, eps=1e-8):
    '''Minimises the maximum of the two polynomicals pL and pR on the interval a,b.'''
    g = lambda x: max(pL(x), pR(x))
    xinit = (a+b)/2
    sol, _, _, _, _ = gs.goldenSection([a,b], [g(a), g(b)], xinit, g(xinit), g, eps)
    return sol

def min_max_quad(qL, qR, a,b):
    '''Minimises the maximum of the two quadratics pL and pR on the interval [a,b].'''
    
    '''
    There are between 2 and 6 candidates for where the minimum might be.
    
        -The min might be at the boundary -> 2 options.
        -The min might be a smooth local minimum -> 2 options
        -The min might be a nonsmooth local minimum -> 2 options    
    
    '''
    
    '''
    if qL.order ==2 and qR.order==2:
        tL = qL.deriv().roots[0]
        tR = qR.deriv().roots[0]
        
        if tR >= a and tR<=b:
            if qR(tR) >= qL(tR):
                return tR
        else:
            if qR(a) >= qL(a):
                return a
            
        
        if tL >= a and tL<=b:
            if qL(tL) >= qR(tL):
                return tL
        else:
            if qL(b) >= qR(b):
                return b
            
        #If we get this far, then qL and qR must intersect, and the minimum
        #must be at one of the intersection points.
        rts = (qR-qL).roots
        frts = np.array([max(qR(t), qL(t)) for t in rts])
        
        if rts[0] >= a and rts[0] <= b:
            if frts[0]<= frts[1]:
                return rts[0]
            elif rts[1]<a or rts[1]>b:
                return rts[0]
            else:
                return rts[1]
        
        
        candidates = []
        for rt in rts:
            if rt>=a and rt<=b:
                candidates.append(rt)
    
        
    
    
    '''
    g = lambda x: max(qL(x), qR(x))
    
    #First the boundary points.
    candidates = [a,b]
    fcandidates = [g(a), g(b)]
    
    #print(qL, qL.deriv(), qL.deriv().roots)
    #Next the candidates for the smooth local minimum.
    #print(qL, qR)

    if qL.order ==2:
        if qL.deriv().roots[0]<b and qL.deriv().roots[0]>a:
            candidates.append(qL.deriv().roots[0])
            fcandidates.append(g(qL.deriv().roots[0]))
    
    #print(qR, qR.deriv(), qR.deriv().roots)
    if qR.order ==2:
        if qR.deriv().roots[0]<b and qR.deriv().roots[0]>a:
            candidates.append(qR.deriv().roots[0])
            fcandidates.append(g(qR.deriv().roots[0]))
        
    #Finally the candidates for the nonsmooth local minimum
    rts = (qR-qL).roots
    #print('roots', rts)
    if len(rts)==2:
        if not isinstance(rts[0], complex):
            if a< rts[0] and rts[0]<b:
                candidates.append(rts[0])
                fcandidates.append(g(rts[0]))
                
            if a < rts[1] and rts[1] < b:
                fcandidates.append(g(rts[1]))
                candidates.append(rts[1])
    elif len(rts)==1:
        if a< rts[0] and rts[0]<b:
            candidates.append(rts[0])
            fcandidates.append(g(rts[0]))
        
    #print('ROOTS')
    #print(rts)
    #print(candidates)
    #print(fcandidates)
    return candidates[np.argmin(fcandidates)]


def underestimate_interp(x, f, Lipschitz):
    c = f[0]
    b = dd(2, x[:2], f[:2])
    a = dd(3, x[:3], f[:3])
    adjustment = 0.5*Lipschitz*np.fabs(x[2] - x[0])
    a -= adjustment
    
    return np.poly1d([a, b - a*(x[0]+x[1]), c - b*x[0]+a*x[0]*x[1]])

def interpolating_model(xL, xR, fL, fR, order=1):
    
    if len(xL)<order+1 and len(xR)<order+1:
        raise ValueError('The arrays xL and xR are not big enough for linear interpolation')
        
    pL = newton_poly(order+1, xL[:order+1], fL[:order+1])
    pR = newton_poly(order+1, xR[:order+1], fR[:order+1])

    if order==1:
        if pL[1] > 0 or pR[1] < 0:
            '''The slope at xL[0] should be negative and the slope at
            xR[0] should be positive'''
            print('err1')
            return False, 0
        
        elif pL[1] ==0 and pR[1] ==0:
            print('err2')
            ''' This implies that the linear interpolation is useless '''
            return False, 0
                
        else:
            xnew = -(pR[1] - pL[1])/(pR[0] - pR[0])

            if xnew >= xL[0] and xnew <= xR[0]:   
                '''Check that intersection of lines is within desired region'''
                return True, xnew
            else:
                return False, 0
            
    elif order ==2:
        if pL[1] + 2*pL[2]*xL[0] >= 0 or pR[1] + 2*pR[2]*xR[0]<=0:
            '''The slope at xL[0] should be negative and the slope at
            xR[0] should be positive'''
            print('err1')
            return False, 0
        
        det = (pR[1] - pL[1])**2 - 4*(pR[2] - pL[2])*(pR[0] - pL[0])
        
        if det < 0:
            '''The two polynomial do not intersect'''
            print('err2')
            return False,0
        
        else:
            base = - (pR[1] - pL[1])/(2*(pR[0] - pL[0]))
            r1,r2 = base + np.sqrt(det), base - np.sqrt(det)

            b1, b2 = (r1>xR[0] or r1<xL[0]), (r2>xR[0] or r2<xL[0])        


            if b1 and b2:
                '''roots lie outside required interval'''
                print('err3')
                return False, 0
            elif b1 ^ b2:
                '''Only one root lies in the required interval'''
                xnew = r1 if b1 else r2
                return True, xnew
            else:
                '''both roots lie in the required interval'''
                
            
            

class Opt1d(object):
    def __init__(self, bracket, xinit, fbracket, fxinit, eps=1e-5, min_step = 1e-6, trustUser=True):
        self.bracket0 = bracket
        self.fbracket0 = fbracket
        self.fxinit = fxinit
        self.xinit = xinit
        self.eps = eps
        self.cgold = (3-np.sqrt(5))/2
        self.reset()
        self.min_step = min_step
        
        if not trustUser:
            self.sanity_check()
        
    def golden_step(self, *args):
        return self.cgold*(self.b - self.xmin) if self.xmin<0.5*(self.b+self.a) else self.cgold*(self.a - self.xmin)

        
    def reset(self):
        self.acc = [self.bracket0[1] - self.bracket0[0]]
        self.ubs = [self.bracket0[1]]
        self.lbs = [self.bracket0[0]]
        self.guess = [self.xinit]
        self.step_type = []
        self.a, self.b = self.bracket0
        self.niter = 0
 
    def sanity_check(self):
        if not self.bracket0[0] < self.xinit and self.xinit < self.bracket0[1]:
            if not self.fbracket0[0]>self.fxinit and self.fbracket0[1]>self.fxinit:
                raise ValueError('Invalid Bracket inputted')
       
    @abc.abstractmethod         
    def determine_step(self, *args):
        pass
        
    def stabilize_step(self, step):
        step_size = np.fabs(step)
        step_dir = np.sign(step)
            
        
        
        #Check that the step takes us far enough away from xmin
        if step_size>self.min_step:
            #and also not too close to xR0
            if step_dir>0:
                if step_size < self.b-self.xmin:
                    xtest = self.xmin+step
                    #print(1)
                elif self.b-self.xmin < 2*self.min_step:
                    #print(2)
                    xtest = self.xmin-self.min_step
                else:
                    #print(3)
                    xtest = self.b - self.min_step
                
            else:
            #or too close to xL0
                if step_size < self.xmin-self.a:
                    #print(4)
                    xtest = self.xmin+step
                elif self.xmin - self.a <2*self.min_step:
                    #print(5)
                    xtest = self.xmin + self.min_step
                else:
                    #print(6)
                    xtest = self.a + self.min_step
        
        else:
            if step_dir>0:
                if self.b-self.xmin < 2*self.min_step:
                    #print(7)
                    xtest = self.xmin - self.min_step
                else:
                    #print(8)
                    xtest = self.xmin + self.min_step
                
            else:
                if self.xmin - self.a <2*self.min_step:
                    #print(9)
                    xtest = self.xmin + self.min_step
                else:
                    #print(10)
                    xtest = self.xmin - self.min_step
        
        return xtest
    
    def set_up_problem(self, *args):
        self.xmin = self.xinit
        self.fxmin = self.fxinit
    
    def update_brackets(self, xtest, ftest):
        if ftest < self.fxmin:
            if xtest < self.xmin:
                self.b = self.xmin
            else:
                self.a = self.xmin
            self.xmin, self.fxmin = xtest, ftest
        else:
            if xtest < self.xmin:
                self.a = xtest
            else:
                self.b = xtest
            

    @abc.abstractmethod
    def not_converged(self, maxiter):
        pass

def plot_progress(xL, xR, xnew1, xnew2, xM, xtrue=None, zoom=False):
    plt.figure(figsize=(8, 1))

    if zoom:
        if xtrue is not None:
            tmp = np.array([np.fabs(xtrue - xnew1), np.fabs(xtrue-xnew2), np.fabs(xtrue - xM)])
        else:
            tmp = np.array([np.fabs(xM - xnew1), np.fabs(xM-xnew2)])
        
        scale = np.min(tmp)
        xmin = max(xtrue - 20*scale, np.min(xL))
        xmax = min(xtrue + 20*scale, np.max(xR))
        plt.xlim(xmin, xmax)

    plt.scatter(xL, np.zeros_like(xL), vmin=-2, cmap = "hot_r", marker="|", s=1000,linewidth=3, color='b' )
    plt.scatter(xR, np.zeros_like(xR), vmin=-2, cmap = "hot_r", marker="|", s=1000,linewidth=3, color='b' )
    plt.scatter(xM, np.zeros_like(xM), vmin=-2, cmap = "hot_r", marker="|", s=1000,linewidth=3, color= 'black')
    plt.scatter([xnew1, xnew2], np.zeros(2), vmin=-2, cmap = "hot_r", marker="|", s=1000,linewidth=3, color='g' )
    if xtrue is not None:
        plt.scatter(xtrue, np.zeros(1), vmin=-2, cmap = "hot_r", marker="|", s=1000,linewidth=3, color='r' )


class Brent(Opt1d):
        
    def set_up_problem(self):
        self.w, self.v = self.xinit, self.xinit
        self.fw, self.fv = self.fxinit, self.fxinit
        super().set_up_problem()
    
    def not_converged(self, maxiter):
        return self.acc[-1]>self.eps and self.niter < maxiter


    def print_progress(self,xtest, step):
        print('  {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format( self.a, self.xmin, self.b, xtest, step, self.oldstep))


    def update_brackets(self, xtest, ftest):
        if ftest <= self.fxmin:
            self.v, self.w = self.w, self.xmin
            self.fv, self.fw = self.fw, self.fxmin
        else:
            if ftest <= self.fw or self.w == self.xmin:
                self.v, self.w, self.fv, self.fw = self.w, xtest, self.fw, ftest
            else:
                self.v, self.fv = xtest, ftest

        super().update_brackets( xtest, ftest)

    
    def interpolate_step(self, *args):
        r = (self.xmin-self.w) * (self.fxmin - self.fv)
        #print('r=' , r)
        q = (self.xmin-self.v) * (self.fxmin - self.fw)
        #print('q=' , q)
        p = (self.xmin-self.v) * q - (self.xmin-self.w)*r
        #print('p=' , p)

        q = 2* (q-r)
        
        if q>0:
            p = -p
        q = np.fabs(q)
        #print('p', 'q', p, q)
        #print(self.v-self.w, self.fv- self.fw)
        if q==0:
            return np.inf
        else:
            return p/q #return the minimum of the interpolated quadratic


    def determine_step(self):
        
        if np.fabs(self.oldstep)<=self.min_step:
            return self.golden_step()
        else:
            trial_step = self.interpolate_step() 
            #print(trial_step, self.oldstep)
            
            if self.xmin+trial_step >= self.b or self.xmin+trial_step <= self.a:
                #print('invalid model adjustment')
                return self.golden_step()
            elif np.fabs(trial_step) >= 0.5*np.fabs(self.oldstep):
                #print('Slow convergence adjustment')
                return self.golden_step()
            else:
                #print('valid step')
                return trial_step
            

    def solve(self, f, maxiter=100):
        self.set_up_problem()
        
        
        self.oldstep, step = 0,0 #initialise

        while self.not_converged(maxiter):
            #while the bracket length is not of desired size
                     
            
            self.oldstep, step = step, self.determine_step()
            xtest = self.stabilize_step(step)
            
            ftest = f(xtest) #evaluate the function at xtest
                        
            #self.print_progress(xtest, step)
            
            self.update_brackets(xtest, ftest) #update the algorithm parameters
            
            self.niter +=1
            self.acc.append(self.b-self.a)
            self.lbs.append(self.a)
            self.ubs.append(self.b)
            self.guess.append(self.xmin)
            

        return self.xmin
