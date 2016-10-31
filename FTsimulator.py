# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:50:13 2016

@author: keithlandry
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt




def randsToMakes(rands, prob):
    outcomes = []
    for rand in rands:
        if rand <= prob:
            outcomes.append(1)
        else:
            outcomes.append(0)
    
    return outcomes




    
#--------------------------------------
#--------------------------------------            
#--------------------------------------
#simulation for 4b

pmake = 0.5
nFT = 100
nTrials = 100000
c = 3

totalMakes = []
trialsWithCconsecutive = 0

for i in range(nTrials):

    rands = np.random.random_sample(nFT)
    outcomes = randsToMakes(rands,pmake)
                
    totalMakes.append(sum(outcomes))
    
    #check if c consecutive makes
    for j in range(nFT-c):
        if sum(outcomes[j:j+c]) == 0:
            trialsWithCconsecutive += 1
            break
        
    
print trialsWithCconsecutive/nTrials



    
#--------------------------------------
#--------------------------------------            
#--------------------------------------
#numerical for 4b

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
    
n = 100
#k = 15
c = 3
Prun = 0
p = 0.5

for k in range(n+1):
    
    Pkn = np.power(p,k)*np.power(1-p,n-k)
    
    omega = 0
    for i in range(1,n):
        #print np.power(-1, i+1)*( choose(n-i*c-i+1,k-i*c)* choose(n-i*c, i-1) + choose(n-i*c-i, k-i*c)* choose(n-i*c, i))
        omega += np.power(-1, i+1)*( choose(n-i*c-i+1,k-i*c)* choose(n-i*c, i-1) + choose(n-i*c-i, k-i*c)* choose(n-i*c, i))

    Prun += omega*Pkn

print Prun

    
    
    
#--------------------------------------
#--------------------------------------            
#--------------------------------------
#simulation for 4d
    
pmakeX = 0.5
pmakeY = 0.7
pmakeZ = 0.9
nTrials = 10000
maxFT = 100


probXwins = []
probYwins = []
probZwins = []
probTie = []

for nFT in range(maxFT):

    playerXwins = 0
    playerYwins = 0
    playerZwins = 0
    ties = 0
    
    for i in range(nTrials):
        randsX = np.random.random_sample(nFT)
        outcomesX = randsToMakes(randsX, pmakeX)
        
        randsY = np.random.random_sample(nFT)
        outcomesY = randsToMakes(randsY, pmakeY)
        
        randsZ = np.random.random_sample(nFT)
        outcomesZ = randsToMakes(randsZ, pmakeZ)

        if sum(outcomesX) > sum(outcomesY) and sum(outcomesX) > sum(outcomesZ):
            playerZwins += 1
        elif sum(outcomesY) > sum(outcomesX) and sum(outcomesY) > sum(outcomesZ):
            playerYwins += 1
        elif sum(outcomesZ) > sum(outcomesY) and sum(outcomesZ) > sum(outcomesX):
            playerZwins +=1
        else:
            ties += 1

        
    probXwins.append(playerXwins/nTrials)
    probYwins.append(playerYwins/nTrials)
    probZwins.append(playerZwins/nTrials)
    probTie.append(ties/nTrials)
    
    

plt.plot(range(0,maxFT), probXwins, 'r--', label = 'probability of X winning')
plt.plot(range(0,maxFT), probYwins, 'b--', label = 'probability of Y winning')
plt.plot(range(0,maxFT), probZwins,'g--', label = 'probability of Z winning')
plt.plot(range(0,maxFT), probTie, 'y--', label = 'probability of tie')
plt.legend(frameon = False)
plt.show()

    
#--------------------------------------
#--------------------------------------            
#--------------------------------------
#numerical solution for 4d
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



def pBinom(n,k,p):
    Pr = choose(n,k)*np.power(p,k)*np.power(1-p,n-k)
    return Pr
#############
pmakeX = 0.9
pmakeY = 0.5
pmakeZ = 0.7

maxFT = 100

probXwins2 = []

for nFT in range(maxFT):
    #print nFT
    pwin = 0
    for kx in range(nFT+1):
        pKx = pBinom(nFT,kx,pmakeX)
        pKzlower = 0
        pKylower = 0
        for ky in range(kx):
            pKylower += pBinom(nFT,ky,pmakeY)
        for kz in range(kx):
            pKzlower += pBinom(nFT,kz,pmakeZ)
                
        #print pKzlower, pKylower, pKzlower*pKylower
        pwin += pKx*pKylower*pKzlower       
        
    probXwins2.append(pwin)
    
zNum = probXwins2    
plt.plot(range(0,maxFT), probXwins,'g--', label = 'player Y')
plt.plot(range(0,maxFT), probXwins,'b--', label = 'player Z')
plt.xlabel('number of free throws in contest')
plt.ylabel('probability winning')
plt.legend(frameon = False)
plt.show()

plt.plot(range(0,100), probYwins,'bo', label = 'simulated player Y', alpha = .5)
plt.plot(range(0,100), yNum,'r--', label = 'numerical player Y', linewidth = 3)
plt.plot(range(0,100), probZwins,'yo', label = 'simulated player Z', alpha = .5)
plt.plot(range(0,100), zNum,'g--', label = 'numerical player Z', linewidth = 3)

plt.legend(frameon = False)
plt.xlabel('number of free throws in contest')
plt.ylabel('probability of winning')

plt.show()



n=10
p=0
for k in range(4):
    p += pBinom(n,k,.7)
    

#--------------------------------------
#--------------------------------------            
#--------------------------------------
#numerical solution for 4c


def findQuantile(n,p,percent):
    
    prob = 0
    for k in range(n+1):
        prob += choose(n,k)*np.power(p,k)*np.power(1-p,n-k)
        if prob >= percent:
            return k

    



po = .7
pa = .9
n = 100

alphabeta = []
alpha = []
beta = []

#loop over possible values of number of freethows in contest
for i in range(10,n+1):
    
    q = findQuantile(i,po,.95)
    
    k = np.arange(q+1,i+1)
    a = 0
    for r in k:
        Pro = choose(i,r)*np.power(po,r)*np.power(1-po,i-r)
        a += Pro
    alpha.append(a)
    
    k = np.arange(0,q+1)
    b = 0
    for r in k:
        Pra = choose(i,r)*np.power(pa,r)*np.power(1-pa,i-r)
        b += Pra
        
    if b < 0.05:
        ncrit = i
        #break

    beta.append(b)


print ncrit
#xzBeta = beta


#remove break in loop if you want to plot
#plt.plot(range(10,101), alpha,'r--', label = 'nolabel', linewidth = 3)
#plt.plot(range(10,101), xyBeta,'b--', label = 'players X,Y', linewidth = 3)
#plt.plot(range(10,101), xzBeta,'g--', label = 'players X,Z', linewidth = 3)
plt.plot(range(10,101), beta,'r--', label = 'players Y,Z', linewidth = 3)

#plt.plot(range(10,101), alphabeta,'g--', label = 'nolabel', linewidth = 3)
plt.axhline(y=.05, linewidth=2, color = 'k')
plt.xlabel('number of FTs in contest')
plt.ylabel('$\\beta$')
plt.legend(frameon = False)
plt.show()




    

    
        
        