# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 23:26:44 2016

@author: keithlandry
"""
from __future__ import division
import numpy as np

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
        
def multiNomial(numbers, probs):
    n = sum(numbers)
    denom = 1
    numer = factorial(n)
    for i,p in zip(numbers,probs):
        denom = denom * factorial(i)
        numer = numer * np.power(p,i)
    
    return numer/denom
    

nFT = 100 
pmake = .7
c = 4
makes = range(0,c)

probs = []
diceSides = 0
for i in makes:
    print i
    nPoss = choose(c-1,i)
    diceSides += nPoss
    prob = np.power(pmake,i)*np.power(1-pmake, c-i)

    for d in range(nPoss):
        probs.append(prob)
        
#add in all makes
diceSides += 1
probs.append(np.power(pmake,c))



nums = np.zeros(diceSides)
for i in range(diceSides):
    


while sum(nums) != nFT/c - nums[diceSides-1]:
    
    
n = 10
c = 2
k = 4
    
p = 0
for i in range(c+1):
    p += (n-c+i)*choose(n-c-i,k-c)
    print (n-c+i)*choose(n-c-i,k-c)
    
p = choose(n-c,k-c)
print p/choose(n,k)


n = 10
c = 4
k = 2


for a in range(c+1,n-c):
    print a
    for i in range(c,a-1)
        print i
        combosLeftTot  = choose(a-1,i)
        combosLeftCons = (a-c)*choose(a-c-1,i-c) #left has at least 1 string of c
        combosRight    = choose(n-a-c,k-c-i)
        
        
        
        