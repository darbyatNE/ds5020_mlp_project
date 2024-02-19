import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Orthonormalization (Gram-Schmidt)

def gs(Ain):
    A = Ain.copy()
    m,n = np.shape(A)
    A[:,0] = A[:,0]/np.sqrt(np.dot(A[:,0],A[:,0]))
    for k in range(1,n):
        for j in range(k):
            A[:,k] = A[:,k]-np.dot(A[:,j],A[:,k])*A[:,j]
        A[:,k] = A[:,k]/np.sqrt(np.dot(A[:,k],A[:,k]))
    return A

m = 5
n = 3
A = np.round(np.random.normal(0,1,[m,n]),3)
print(A)

Z = gs(A)
print(Z)
np.matmul(np.transpose(Z),Z)

### Code in this section performs and plots Simple Linear Regression
###
def slr(y,x):
    n = len(y)
    xbar = np.mean(x)
    ybar = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    zx = (x-xbar)/sx
    zy = (y-ybar)/sy
    r = np.sum(np.multiply(zy,zx))/(n-1)
    beta = r*sy/sx
    return ybar-beta*xbar,beta

dat = pd.read_csv("data/lin_reg_data/HousingData.csv")
colnames = dat.columns
dat = dat.to_numpy()
x = dat[:,0]
y = dat[:,1]

b0,b1 = slr(y,x)
plt.plot(x,y,'k.')    
plt.plot(x,b0+b1*x,color='blue',linewidth=3)
plt.show()
###


### Code in this section will demonstrate Multiple Linear Regression

# def mlr(y,X,intercept=True):
#     n,p = np.shape(X)
#     if intercept and not np.all(X[:,0]==1):
#         X = np.column_stack((np.ones(n),X))
#     Xt = np.transpose(X)
#     XtX = np.matmul(Xt,X)
#     XtXinv = np.linalg.inv(XtX)
#     Xty = np.matmul(Xt,y)
#     return np.matmul(XtXinv,Xty)

# import sklearn.datasets as data
# X,y = data.load_diabetes(return_X_y=True)
# betahat = mlr(y,X)

###

### Code in this section will demostrate a multilayer perceptron
'''
MLP Code goes in here :)
'''
###