# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:56:43 2016

@author: keithlandry
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from random import randint




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
        
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

k = 7
d = 32
maxN = 500
PnMe = []

m = d
Pn = []
N = []

expectedNme = 0
expectedN   = 0

#for standard n >= m+1 if serial numbers start at 1
#for me n>d+1 (n = d results in zero probability)

for n in range(d,maxN):
    probMe = (k-2)/k * choose(d-1,k-2)/choose(n,k)*(n-d)
    prob   = (k-1)/k * choose(m-1,k-1)/choose(n,k)
    PnMe.append(probMe)
    Pn.append(prob)
    N.append(n)
    expectedNme += n*probMe
    expectedN   += (n)*prob
    #print expectedN, n, prob, n*prob


mostProb = d+PnMe.index(max(PnMe))

print " expected number of tanks serial numbers start at 1    : ", expectedN
print " expected number of tanks arbitrary start serial number: ", expectedNme
print "most prob number of tanks arbitrary start serial number: ", mostProb

#plt.plot(N,PnMe,'b^',N,Pn,'r--')
#plt.plot(N,PnMe,'r--')
#xmin = max(0,d-10)
#xmax = d+[ n for n,i in enumerate(Pn) if i<1E-3][0]
#plt.xlim(0,100)
#plt.show()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

abStartNum = randint(0,10000)
nIterations = 300000

narray = []

for i in range(nIterations):
    N2 = randint(d, maxN)
    seenSerials = np.random.choice(np.arange(abStartNum, abStartNum+N2), k, replace=False)
    delta = max(seenSerials) - min(seenSerials)
    #print i, N, delta
    if delta == d:
        narray.append(N2)
        
        
#fig = plt.figure()
theory = np.array(PnMe)
#narray = np.array(narray)
#maxTheory = max(theory)
#maxData = max(y)
#theory = maxData*theory/maxTheory

y, x, _ = plt.hist(narray, bins = maxN, range=[0, maxN], color = "red", alpha = 0.5, label = "simulated probability", normed = 1)


plt.plot(N,theory,'--', linewidth = 4, label = "theoretical probability")
xmin = max(0,d-10)
xmax = d+[ n for n,i in enumerate(Pn) if i<1E-3][0]
plt.xlim(0,100)
plt.ylim(0,.079)

plt.xlabel("number of German tanks")
plt.ylabel("Probability of there \n being N german tanks")
figTitle = "observing the arbitrarily starting serial numbers of " + str(k) + " tanks \n with a maximum difference in observed serial numbers  " + str(d) #+ " \n(assuming maximum number of tanks is 100)"
plt.title(figTitle) 

plt.legend(frameon = False)
plt.savefig("germanTankProblem_5.pdf")
plt.show()
     



    

