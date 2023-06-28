#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:34:58 2019

@author: Jonathan GP
"""
import numpy as np

def setup_grid(f, x0, direction, eps, step = 1, lstol = 1e-5):
   xL = np.zeros([3])
   xR = np.zeros([3])
   fL = np.zeros([3])
   fR = np.zeros([3])

   xpert = x0+direction*eps
   xM = xpert
   fM = f(xM)

   xL[0] = x0
   fL[0] = f(xL[0])
   
   if fM>= fL[0]:
      raise ValueError('direction is not a valid direction of descent')
      
   step_num = 1
   xnew = x0+step_num*step*direction
   fnew = f(xnew)
   
   
   while fnew <= fM:
      xL, fL = np.roll(xL, 1), np.roll(fL, 1)
      
      step_num +=1

      xM, fM, xL[0], fL[0] = xnew, fnew, xM, fM
      xnew = x0+step_num*step*direction
      fnew = f(xnew)
      
   xR[0], fR[0] = xnew, fnew
   
   ratio = 0.5*(-1 + np.sqrt(5))
   
   
   Lfull = step_num
   Rfull = 1
   
   #print('after constructing bracket')
   #print('Rfull', Rfull, xR)
   #print('xM', xM)
   #print('Lfull', Lfull, xL)
   #print('-'*40)

   while Rfull <3 and np.fabs(xR[0] - xL[0])>= lstol:
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
         
   
   if Lfull<3 and xM ==xpert:
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

      
      
   
   while Lfull <3 and np.fabs(xR[0] - xL[0])>=lstol:
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
   s1 = xL[2] < xL[1]
   s2 = xL[1] < xL[0]
   s3 = xL[0] < xM
   s4 = xM < xR[0]
   s5 = xR[0] < xR[1]
   s6 = xR[1] < xR[2]
   
   increasing = s1 and s2 and s3 and s4 and s5 and s6
   
   bracket = answer > xL[0] and answer < xR[0]
   
   minimum = fM < fL[0] and fM < fR[0]
   
   return increasing, bracket, minimum
