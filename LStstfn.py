import numpy as np
import abc

'''
def setup_grid(func, x0, direction, eps, step = 1, lstol = 1e-5):
   xL = np.zeros([3])
   xR = np.zeros([3])
   fL = np.zeros([3])
   fR = np.zeros([3])

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

   while Rfull <3 and np.fabs(xR[0] - xL[0])>= lstol:
      xnew = xM*(1-ratio) + xR[0]*ratio
      fnew = func(xnew)
      
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
      fnew = func(xnew)
      
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
      fnew = func(xnew)
      
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
'''      

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


#First we create an testfunction abstract parent class

class TestFunction(object):
    '''an abstract class for testfunctions'''
    def __init__(self, llim, rlim, *args):
        self.llim = llim
        self.rlim = rlim
        #self.xtrue = xtrue
        if len(args)!=0 :
            self.params = np.array(args[0])
            
        #self.ymin = self.func(self.xtrue)


    @abc.abstractmethod
    def func(self, variable):
        '''A general function call wrapper'''
        pass
    
    def setup_grid(self):
        diam = self.rlim - self.llim
        tmpL = np.random.uniform(self.llim, self.llim+0.2*diam, size=4)
        tmpR = np.random.uniform(self.rlim-0.2*diam, self.rlim, size=4)
        
        xL = -np.sort(-tmpL)
        fL = np.array([self.func(xx) for xx in xL])
        xR = np.sort(tmpR)
        fR = np.array([self.func(xx) for xx in xR])
        
        while ((fL[0] < fR[0]) and (fL[1] < fL[0])) or ((fR[0]<fL[0]) and (fR[1]<fR[0])):
            #print('trying to find initial grid')
            tmpL = np.random.uniform(self.llim, self.llim+0.2*diam, size=4)
            tmpR = np.random.uniform(self.rlim-0.2*diam, self.rlim, size=4)
            
            xL = -np.sort(-tmpL)
            fL = np.array([self.func(xx) for xx in xL])
            xR = np.sort(tmpR)
            fR = np.array([self.func(xx) for xx in xR])
        
        if fL[0] < fR[0]:
            return xL[1:4], xL[0], xR[:3], fL[1:4], fL[0], fR[:3]
        else:
            return xL[:3], xR[0], xR[1:4], fL[:3], fR[0], fR[1:4]
            

ns = {'SU':7, 'SM':7, 'NU':5}  
   
#Smooth Unimodal
class SU1(TestFunction):
    def __init__(self):
        '''blah'''
        self.M = 100
        super().__init__(-1,1)
        
    def func(self, variable):
        return -(1/np.sqrt(np.e))*np.exp(-0.5*variable**2)
            
class SU2(TestFunction):
    '''basic smooth test function'''
    def __init__(self):
        '''blah'''
        self.M = 100
        super().__init__(-1,1)
    
    def func(self, variable):
        return variable**4/24

class SU3(TestFunction):
    '''basic smooth test function'''
    def __init__(self):
        '''blah'''
        self.M = 100
        super().__init__(-2.5,3)
    
    def func(self, x):
        return (-np.sin(2*x - 0.5*np.pi) - 3*np.cos(x) - 0.5*x)/11.0

class SU4(TestFunction):
    def __init__(self):
        self.M = 10*np.pi + 25*np.pi**2
        super().__init__(-10,10)
    
    def func(self, variable):
        return (variable**2/2 - np.cos(5*np.pi*variable)/(25*np.pi**2) - (variable * np.sin(5*np.pi*variable) )/(5*np.pi))/2500.0

class SU5(TestFunction):
    '''the first asymetric smooth testfunction'''
    def __init__(self):
        self.M = 5
        super().__init__(0.1, 0.9)

    def func(self, variable):
        '''the function call'''
        return (-variable**(2.0/3)- (1-variable**2)**(1.0/3))/250.0

class SU6(TestFunction):
    def __init__(self):
        self.M = 6000
        super().__init__(0.1, 3)

    def func(self, variable):
        return (np.exp(variable) + 1/np.sqrt(variable))/6000

class SU7(TestFunction):
    def __init__(self):
        self.M = 6000
        super().__init__(1.3, 3.9)

    def func(self, variable):
        return (-(16*variable**2-24*variable+5)*np.exp(-variable))/13.0


# Smooth multimodal
class SM1(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(-1, 1)

    def func(self, x):
        '''the function call'''
        return x**6 *(2+np.sin(1/x))/300.0
    
class SM2(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(-1, 1)

    def func(self, x):
        '''the function call'''
        return -np.sin(5*np.pi*x)**6/40000.0

class SM3(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(0.1, 1)

    def func(self, x):
        '''the function call'''
        return -np.sin(5*np.pi*(np.sign(x)*np.power(np.fabs(x), 0.75)-0.05))**6/250000.0

class SM4(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(-1, 1)

    def func(self, x):
        '''the function call'''
        return (np.sin(16.0/15.0 * x - 1) + np.sin(16.0/15.0 * x - 1)**2)/5.0

class SM5(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(-100, 100)

    def func(self, x):
        '''the function call'''
        return x**2/4000 - np.cos(x) + 1
    
class SM6(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(2.5, 9.5)

    def func(self, x):
        '''the function call'''
        return (np.log(x-2)**2 + np.log(10-x)**2 - np.power(x, 0.2))/71
    
class SM7(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(0.5,10)

    def func(self, x):
        '''the function call'''
        return (np.sin(x) + np.sin(10/3.0 * x) +np.log(x) - 0.84*x)/40.0

    
# Nonsmooth Unimodal
        
class NU1(TestFunction):
    def __init__(self):
        self.M = 5
        super().__init__(-32, 32)

    def func(self, x):
        '''the function call'''
        return -60000*np.exp(-0.02 * np.fabs(x))

class NU2(TestFunction):
    ''' The first asymetric nonsmooth testfunction'''
    def __init__(self, params=[1,1,1,0]):
        '''blah'''
        self.M = 5
        super().__init__(-2, 10 , params)

    def func(self, variable):
        '''the function call'''
        if variable<=0:
            return 1/((variable+3)* 6)
        else:
            return max(self.params[0]/(variable+3)**self.params[1], self.params[2]*np.log(variable)+self.params[3])/6.0
    
class NU3(TestFunction):
    '''The second asymetric nonsmooth testfunction'''
    def __init__(self, params=[1,1,0,2]):
        '''blah'''
        self.M = 15
        super().__init__(-2, 2, params)
        
    def func(self, variable):
        '''the function call'''
        return max(self.params[0]/(variable + 3)**self.params[1], self.params[2]+1/(variable-3)**self.params[3])*1/24.0
    
    
class NU4(TestFunction):
    '''the third asymetric nonsmooth testfunction'''
    def __init__(self, params=[1,1,1,1]):
        '''blah'''
        self.M = 6
        super().__init__(-2, 5 , params)
        
    def func(self, variable):
        '''function call'''
        return max(self.params[0]/(variable+3)**self.params[1], self.params[2]*np.exp(self.params[3]*variable))/60.0
    
    
class NU5(TestFunction):
    '''the third asymetric nonsmooth testfunction'''
    def __init__(self, params=[1,1,1,1]):
        '''blah'''
        self.M = 6
        super().__init__(-5, 5 , params)
        
    def func(self, variable):
        '''function call'''
        return max(self.params[0]*np.exp(-self.params[1]*variable), self.params[2]*np.exp(self.params[3]*variable))/150.0




#Asymmetric smooth multimodal

class ASM1(TestFunction):
    '''the first asymetric smooth testfunction'''
    def __init__(self):
        self.M = 5
        super().__init__(2.7, 7.5)

    def func(self, variable):
        '''the function call'''
        return np.sin(variable)+np.sin(10*variable/3)


class ASM2(TestFunction):
    def __init__(self):
        self.M = 6000
        super().__init__(0, 1.2)


    def func(self, variable):
        return -(1.4 - 3*variable)*np.sin(18*variable)

class ASM3(TestFunction):
    def __init__(self):
        self.M = 6000
        super().__init__(-10, 10)


    def func(self, variable):
        return -(variable + np.sin(variable))*np.exp(-variable**2)

class ASM4(TestFunction):
    def __init__(self):
        self.M = 6000
        super().__init__(2.7, 7.5)


    def func(self, variable):
        return np.sin(variable) + np.sin(10.0*variable/3.0) + np.log(variable) - 0.84*variable + 3.0




