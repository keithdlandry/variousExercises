# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:56:49 2016

@author: keithlandry
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import scipy.optimize
from scipy.optimize import curve_fit





#--------------------------------
#--------------------------------
#--------------------------------
#season wins linear regression


df = pd.read_csv('/Users/keithlandry/Documents/last4seasonteamstats2.csv')

colNames = ['W','L','WP','MPG','FGM','FGA','FGP','3PM','3PA','3PP','FTM','FTA',
            'FTP','ORB','DRB','RPG','APG','TOV','SPG','BPG','BLKA','PF','PFD','PTS','plusminus']
            
            
df.columns = colNames

df['2PM'] = df['FGM'] - df['3PM']
df['2PA'] = df['FGA'] - df['3PA']
df['2PP'] = df['2PM']/df['2PA']*100


#plt.plot(df['3PA'],df['W'],'bo')
#plt.plot(df['3PM'],df['W'],'bo')


#x = np.array([df['3PP'],df['3PA'],np.power(df['3PP'],1)*np.power(df['3PA'],2)])
#x = np.array([ df['3PP'],df['3PP']*df['3PA']/(df['FGA'] - df['3PA']), (df['FGM']-df['3PM'])/(df['FGA']-df['3PA']), (df['FGM']-df['3PM'])/(df['FGA']-df['3PA'])*(df['FGA']-df['3PA'])/df['3PA'], df['RPG'], df['TOV']  ])
#x = np.array([ df['3PP'],df['3PA'],(df['FGM']-df['3PM'])/(df['FGA']-df['3PA']),(df['FGA']-df['3PA']) ,df['RPG'], df['TOV']])

x = np.array([ df[df['3PA']/df['FGA'] < 0.263]['3PP'] ]) #winner

#x = np.array([  df['3PP'], df['3PA'], df['FGP'], df['FGA']    ])
#x = np.array(df['3PP'])
#x = x.reshape(len(x),1) # X must be shape   (samples, features)
x = np.transpose(x)

#y = np.array(df['W'])
y = np.array(df[df['3PA']/df['FGA'] < 0.263]['W'])

print np.shape(x), np.shape(y)              # Y can be in shape (samples,)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#print np.shape(X_train), np.shape(y_train)
regr = linear_model.LinearRegression()
regr.fit(x, y)
print regr.score(x,y)

print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x) - y) ** 2))






plt.plot(x,y,'bo')
plt.plot(x, regr.predict(x), color='r',linewidth=3)
plt.xlabel('three point field goal percentage')
plt.ylabel('wins')






x = np.array([df['2PA'], df["APG"], df['3PA'], df['2PP'], df['3PP'], df['3PP'] ] )
x = np.array([ df['3PP'],df['3PP']*df['3PA']/(df['FGA'] - df['3PA']), (df['FGM']-df['3PM'])/(df['FGA']-df['3PA']), (df['FGM']-df['3PM'])/(df['FGA']-df['3PA'])*(df['FGA']-df['3PA'])/df['3PA'], df['RPG'], df['TOV']  ])

#x = np.array((df['FGM']-df['3PM'])/(df['FGA']-df['3PA']))
x = np.transpose(x)
#x = x.reshape(len(x),1) # X must be shape   (samples, features)
y = np.array(df['W'])
regr = linear_model.LinearRegression()
regr.fit(x, y)
print '  results'
print('Coefficients: \n', regr.coef_)


residuals = y - regr.predict(x)

plt.plot(df['3PP'],residuals,'bo')

regrN = linear_model.LinearRegression()
x3P = np.array(df['3PP'])
x3P = x3P.reshape(len(x3P),1) # X must be shape   (samples, features)
regrN.fit(x3P, residuals)
print regrN.score(x3P, residuals)


plt.plot(x3P,residuals,'bo')
plt.plot(x3P,regrN.predict(x3P),'b')

print '  results'
print('Coefficients: \n', regrN.coef_)



#---------------
#just regress on 3 point percentage
depVar = 'W'
indVar = '2PA'
regr = linear_model.LinearRegression()
x = np.array(df[indVar])
x = x.reshape(len(x),1) # X must be shape   (samples, features)
#x = np.transpose(x)
y = np.array(df[depVar])

regr.fit(x, y)
print regr.score(x,y)
print 'coefs : ' , regr.coef_

residuals = y - regr.predict(x)
#plt.hist(residuals,15)
msr = np.mean(residuals** 2)
print msr




plt.plot(x,y, 'bo')
plt.plot(x, regr.predict(x), color='r',linewidth=3)
#---------------

#---------------
#regress on 3 point percentage and 2 point percentage
regr = linear_model.LinearRegression()
x = np.array([df['APG'], df['3PP'], df['2PP'], df['2PA'], df['3PA'] ])
x = np.array([ df['3PP']*df['3PA']/df['FGA'], df['2PP']*df['2PA']/df['FGA'] ]) #winner
#x = np.array([ df['3PP'], df['2PP'], df['3PP']*df['3PA']/df['FGA'], df['2PP']*df['2PA']/df['FGA'], df['3PA'], df['2PA'] ]) #winner
x = np.array([ df[df['3PA']/df['FGA'] > 0.263]['3PP'] ]) #winner

#x = np.array([ df['3PP'], df['2PP'], df['3PA'], df['2PA'] ])

#x = np.array([ df['3PP'], df['2PP'] ])
#x = np.array([ df['3PP']/df['3PA']/df['FGA'] ])

# = x.reshape(len(x),1) # X must be shape   (samples, features)
x = np.transpose(x)
#y = np.array(df['W'])

y = np.array(df[df['3PA']/df['FGA'] > 0.263]['W'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)


regr.fit(X_train, y_train)
print regr.score(X_test,y_test)
print 'coefs : ' , regr.coef_

residuals = y_test - regr.predict(X_test)
plt.hist(residuals,15)
msr = np.mean(residuals** 2)
print msr


plt.plot(df['3PP']/df['3PA']*df['FGA'],df['W'], 'bo')
plt.plot(df['3PP'],df['W'], 'ro')

plt.plot(x, regr.predict(x), color='r',linewidth=3)



#---------------
#logistig reg


    
def sigmoid(x, x0, k):
     y = 82 / (1 + np.exp(-k*(x-x0)))
     return y
 
def residuals(x,y):
    return y - sigmoid(x, *popt)
   

pMax = 1
pMin = 0

x = np.array( df[(df['3PA']/df['FGA'] < pMax) & (df['3PA']/df['FGA'] >= pMin)]['3PP'] ) #winner
y = np.array(df[(df['3PA']/df['FGA'] < pMax) & (df['3PA']/df['FGA'] >= pMin)]['W'])
#x = np.array( df['3PP'] ) #winner
#y = np.array(df['W'])

popt, pcov = curve_fit(sigmoid, x, y, p0 = [33,.7] )
print popt

res = residuals(x,y)
msr = np.mean(res**2)
print msr

xp = np.linspace(-1, 50, 500)
yp = sigmoid(xp, *popt)

plt.plot(x, y, 'o', label='data')
plt.plot(xp,yp, label='fit')
plt.ylim(0,85)
plt.legend(loc='best')
plt.show()








import statsmodels.api as sm

pMax = 1
pMin = 0

x = np.array( df[(df['3PA']/df['FGA'] < pMax) & (df['3PA']/df['FGA'] >= pMin)]['3PP'] ) #winner
#x = np.array(df['APG'])
#y = np.array(df['W'])
y = np.array(df[(df['3PA']/df['FGA'] < pMax) & (df['3PA']/df['FGA'] >= pMin)]['W'])

x = sm.add_constant(x)

model = sm.OLS(y,x)
results = model.fit()
print(results.summary())

plt.plot(x, y, 'ro', label="data")
plt.plot(x, results.fittedvalues, 'r--.', label="OLS")


res = y - results.fittedvalues
msr = np.mean(res**2)
print msr




x = np.array([.20097,0.23261675633,0.2664204009,0.2892375,0.31131983,0.3514335])
y = np.array([0,2.4978,4.4156,3.372,6.744,3.54])
y2 = np.array()
x = sm.add_constant(x)

model = sm.OLS(y,x*x)
results = model.fit()
print(results.summary())

plt.plot(x*x, y, 'ro', label="data")
plt.plot(x*x, results.fittedvalues, 'r--.', label="OLS")




#---------------
#partial regression

x = np.array(df['2PP'])
x = x.reshape(len(x),1) # X must be shape   (samples, features)
y = np.array(df['W'])
regr = linear_model.LinearRegression()
regr.fit(x, y)

res2PY = y - regr.predict(x)


x2 = np.array(df['3PP'])
x = x.reshape(len(x),1) # X must be shape   (samples, features)
regr2 = linear_model.LinearRegression()
regr2.fit(x, x2)

res2P3P = x2 - regr.predict(x)
res2P3P = res2P3P.reshape(len(res2P3P),1) # X must be shape   (samples, features)



#plt.plot(x,x2, 'bo')
#plt.plot(x, regr2.predict(x), color='r',linewidth=3)


regr3 = linear_model.LinearRegression()
regr3.fit(res2P3P, res2PY)

plt.plot(res2P3P,res2PY, 'bo')
plt.plot(res2PY, regr3.predict(res2P3P), color='r',linewidth=3)




























#---------------

y = np.array(df['W'])
x2P = np.array((df['FGM']-df['3PM'])/(df['FGA']-df['3PA'])*100)
x2P = x2P.reshape(len(x2P),1) # X must be shape   (samples, features)
regr2P = linear_model.LinearRegression()
regr2P.fit(x2P, y)

x3P = np.array(df['3PP'])
x3P = x3P.reshape(len(x3P),1) # X must be shape   (samples, features)
regr3P = linear_model.LinearRegression()
regr3P.fit(x3P, y)

print '2 point results'
print('Coefficients: \n', regr2P.coef_)
## The mean square error
print("Residual sum of squares: %.2f"
     % np.mean((regr2P.predict(x2P) - y) ** 2))
     
print '3 point results'
print('Coefficients: \n', regr3P.coef_)
## The mean square error
print("Residual sum of squares: %.2f"
     % np.mean((regr3P.predict(x3P) - y) ** 2))





plt.plot(x2P, y,'ro')
plt.plot(x2P, regr2P.predict(x2P), color='r',linewidth=3)
plt.plot(x3P, y,'bo')
plt.plot(x3P, regr3P.predict(x3P), color='b',linewidth=3)





#--------------------------------
#--------------------------------
#--------------------------------
#game outcome logistic regression

colNames = ['nothing','W','MPG','PTS','FGM','FGA','FGP','3PM','3PA','3PP','FTM','FTA',
            'FTP','ORB','DRB','RPG','APG','TOV','SPG','BPG','PF','plusminus']
df2 = pd.read_csv('/Users/keithlandry/Documents/last2ClippersGameStats.csv',header=None)
df2.columns = colNames

df2.loc[df2['W'] == 'W', 'W'] = 1
df2.loc[df2['W'] == 'L', 'W'] = 0



X = np.array(df2['3PP'],df2['3PA'])
X = X.reshape(len(X),1)   # X must be shape   (samples, features)
y = np.array(df2['W']) # Y can be in shape (samples,)
print np.shape(X), np.shape(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


logReg = LogisticRegression(C=1e3)
logReg = logReg.fit(X_train, y_train)
#logReg.score(X_test,y_test)
print (logReg.predict(X_test) - y_test)**2

a = (logReg.predict(X_test) - y_test)**2




plt.scatter(X, y,  color='black')
plotX = np.linspace(0, 80, 300)
def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(plotX * logReg.coef_ + logReg.intercept_).ravel()
plt.plot(plotX, loss, color='blue', linewidth=3)
plt.xlabel('Los Angeles Clipper 3PP')
plt.ylabel('Win probability')




