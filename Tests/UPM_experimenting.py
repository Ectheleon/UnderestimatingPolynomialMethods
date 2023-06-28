#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:54:08 2019

@author: Jonathan Grant-Peters
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats as st

#%%
def B(t):
    return t[1]+t[2]

def R(t):
    return t[1]+t[2]+t[3]

def L(t): 
    return t[0]+t[1]+t[2]

def H(t):
    return t[2]*(t[2]+t[3])-t[1]*(t[1]+t[0])

def X(t):
    return t[::-1]

def S1(t):
    return np.array([t[0], B(t)*R(t)/(L(t)+R(t)), -H(t)/(L(t)+R(t)), t[2]])

def S2(t):
    return np.array([t[1], H(t)/(L(t)+R(t)), B(t)*L(t)/(L(t)+R(t)), t[3]])

def S3(t):
    return np.array([t[0], t[1], H(t)/(L(t)+R(t)), B(t)*L(t)/(L(t)+R(t))])

def S4(t):
    return np.array([B(t)*R(t)/(L(t)+R(t)), -H(t)/(L(t)+R(t)), t[2], t[3]])

def SS(t, i):
    if i==1:
        return S1(t)
    elif i==2:
        return S2(t)
    elif i==3:
        return S3(t)
    else :
        return S4(t)

def g(x): 
    return (x[3]*x[4]-x[0]*x[1])/(-x[0]-x[1]+x[3]+x[4])    

def U1(x, f):
    return np.array([x[0], x[1], g(x), x[2], x[3]]), np.array([f[0], f[1], 0.5*f[2], f[2], f[3]])

def U2(x, f):
    return np.array([x[1], x[2], g(x), x[3], x[4]]), np.array([f[1], f[2], 0.5*f[2], f[3], f[4]])

def U3(x, f):
    return np.array([x[0], x[1], x[2], g(x), x[3]]), np.array([f[0], f[1], f[2], 0.5*f[2], f[3]])

def U4(x, f):
    return np.array([x[1], g(x), x[2], x[3], x[4]]), np.array([f[1], 0.5*f[2], f[2], f[3], f[4]])


def random_grid():
    nums = np.sort(np.random.uniform(0,1,[3]))
    p = nums[0]
    q = nums[1]-nums[0]
    r = nums[2]-nums[1]
    s = 1- nums[2]
    return np.array([p,q,r,s])

def UPM_run(t0, I):
    n = len(I)
    
    ts = np.empty([n+1, 4])
    ts[0] = t0
    
    for i in range(n):
        ts[i+1] = SS(ts[i], I[i])
        
    return ts
    
    
def sequence_from_int(t0, i, n):
    oldt = t0
    seq = []
    
    for j in range(n):
        if (i>>j)%2:
            if H(oldt)<0:
                seq.append(1)
                oldt = S1(oldt)
            else:
                seq.append(2)
                oldt = S2(oldt)
        else:
            if H(oldt)<0:
                seq.append(4)
                oldt = S4(oldt)
            else:
                seq.append(3)
                oldt = S3(oldt)
    
        
    return seq
    
    

def UPM_experiment(t0, tol= 1e-8):
    sequence = []
    xL2_pos = 0
    x = np.array([xL2_pos, xL2_pos + t0[0], xL2_pos + t0[0]+t0[1], xL2_pos + t0[0]+t0[1]+t0[2], xL2_pos+1])
    f = np.array([1, 0.5, 0.25, 0.5, 1])
    Bs = [B(t0)]
    
    xs = [x[0], x[1], x[2], x[3], x[4]]
    fs = [f[0], f[1], f[2], f[3], f[4]]
    
    print(xs)
    print(fs)
    t = t0
    
    while B(t)>tol:
        #randomly determine if the new point evaluated is smaller or larger than
        #the best known point
        
        #print(x, g(x))
        print(np.array([x[1]-x[0], x[2]-x[1], x[3]-x[2], x[4]-x[3]]))
        print(t)
        
        if np.random.binomial(1,0.5):
            if g(x)>x[2]:
                x,f = U2(x,f)
                t = S2(t)
                
                xs.append(x[2])
                fs.append(f[2])
                sequence.append(2)
                
            else:
                x,f = U1(x,f)                
                t = S1(t)
                
                xs.append(x[2])
                fs.append(f[2])
                sequence.append(1)
        else:
            if g(x)>x[2]:
                x,f = U3(x,f)
                t = S3(t)
                
                xs.append(x[3])
                fs.append(f[3])
                sequence.append(3)

            else:
                x,f = U4(x,f)
                t = S4(t)
                
                xs.append(x[1])
                fs.append(f[1])
                sequence.append(4)
        
        Bs.append(B(t))        
        print('B(t) = ', B(t), sequence[-1])
    
    return t, xs, fs, Bs, sequence

        

#%%
    
def comprehensive_scan(t0, n):
    data = np.empty([2**n, n+1, 4])
    
    for j in range(2**n):
        I = sequence_from_int(t0, j, n)
        data[j] = UPM_run(t0, I)
    
    return data
    
def Brackets(data, n):
    return np.array([[B(data[i,j])  for j in range(n+1)] for i in range(2**n)])

def AveRates(data, n):
    Bs = Brackets(data, n)
    Ratios = np.array([[Bs[i,j+1]/Bs[i,j] for j in range(n)] for i in range(2**n)])
    return np.array([ st.mstats.gmean(Ratios[i]) for i in range(2**n)])
    
def Sim(nruns, n):
    data = np.empty([nruns, 10])
    
    for j in range(nruns):
        print('j = ', j)
        t = random_grid()
        tmp = comprehensive_scan(t, n)
        data[j, :4] = t
        data[j, 4:] = refineData(AveRates(tmp, n))
       
        
    return data

def refineData(dat):
    return np.mean(dat), np.quantile(dat, 0.6), np.quantile(dat, 0.7), np.quantile(dat, 0.8), np.quantile(dat, 0.9), np.max(dat)

#%%
    
sequence = []
t0 = random_grid()
print(t0)
print(X(S1(X(t0)))-S2(t0))
print(X(S4(X(t0)))-S3(t0))


n=14
time0 = time.time()
data = comprehensive_scan(t0, n)
Bs = Brackets(data, n)
Ratios = np.array([[Bs[i,j+1]/Bs[i,j] for j in range(n)] for i in range(2**n)])
means = AveRates(data,n)
time1 = time.time()

print(time1-time0)

plt.hist(means, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


print(np.mean(means))
print(refineData(means))
#%%


dataset_14 = Sim(1000, 14)

np.savetxt( 'UPM_convergence_data_n_14.csv', dataset_14)

dataset_15 = Sim(1000, 15)

np.savetxt( 'UPM_convergence_data_n_15.csv', dataset_15)

#%%

dataset_16 = Sim(1000, 16)

np.savetxt( 'UPM_convergence_data_n_16.csv', dataset_16)

dataset_17 = Sim(1000, 17)

np.savetxt( 'UPM_convergence_data_n_17.csv', dataset_15)


#%%

nsteps = 6
eps = 1/nsteps

a = eps
b = 2*eps
c = 3*eps

while a<1-eps:
    b = a+eps
    while b<1-eps:
        c= b+eps
        while c<1-eps:
            print(a,b,c)
            c+=eps
        b+=eps
    a+=eps


#%%

hist, bin_edges = np.histogram(y, bins='auto', density='true')
np.sum(hist*np.diff(bin_edges))


#%%

y = dataset[0]
size = len(y)
x = np.arange(0,1,0.001)
h = plt.hist(y, bins='auto', density='true')
#%%

dist = ss.beta
param = dist.fit(y, floc=0, fscale=1)
pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) 
plt.plot(x, pdf_fitted, label='gamma')
    #plt.xlim(0,47)


plt.legend(loc='upper right')
plt.show()



#%%

#size = 30000
#x = scipy.arange(size)
#y = scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))

y = dataset[0]
size = len(y)
x = np.arange(0,1,0.001)
h = plt.hist(y, bins='auto', density='true')

DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]


DISTRIBUTIONS = [        
        st.beta,st.exponweib,st.weibull_max,st.weibull_min,
    ]

DISTRIBUTIONS = [        
        st.burr,st.exponweib,st.exponpow,st.foldcauchy,st.frechet_r, st.gengamma,st.gompertz,st.johnsonsb , st.loglaplace,st.mielke  ,st.powerlognorm  ,        st.weibull_min
    ]

dist_names = ['beta', 'weibull_min', 'alpha', 'betaprime', 'dweibull']

#for dist_name in dist_names:
    
for dist in DISTRIBUTIONS:
    print(str(dist))
    #dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y, floc=0)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) 
    plt.plot(x, pdf_fitted, label=str(dist))
    print(param)
    #plt.xlim(0,47)
#plt.legend(loc='upper right')
plt.show()


#%%


#%%




#%%




#%%



#%%

I = sequence_from_int(t0, 27, 10)


#%%
UPM_run(t0, I)



#%%

t,xs,fs,Bs,seq = UPM_experiment(t0)

Bratios = np.array([Bs[i+1]/Bs[i] for i in range(len(Bs)-1)])
Bgmean = ss.mstats.gmean(Bratios)
Bfit = np.array([Bs[0]*Bgmean**i for i in range(len(Bs))  ])

plt.semilogy(Bfit, label='Slope = '+str(Bgmean))
plt.semilogy(Bs)
plt.legend()
plt.show()
print(xs)
print(fs)
print(seq)
plt.scatter(xs, fs)
plt.show()

#%%
fmax = 1
fmin = 0
