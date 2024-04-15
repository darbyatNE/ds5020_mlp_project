import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Code in this section performs and plots Simple Linear Regression
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

dat = pd.read_csv("data/lin_reg_data/Heights.csv")
colnames = dat.columns
dat = dat.to_numpy()
x = dat[:,0]
y = dat[:,1]

b0,b1 = slr(y,x)
plt.plot(x,y,'k.')    
plt.plot(x,b0+b1*x,color='blue',linewidth=3, label='Linear Regression Best Fit Line')
plt.legend()
plt.title("Simple Linear Regresssion")
plt.xlabel("Mother's Height")
plt.ylabel("Daughter's Height")
filename = 'figs/lin_reg_mother_daughter_heights.png'
plt.savefig(filename)
print('A chart showing the linear relationship of the daughter\'s height to their mother\'s \
    height as set by the linear regression model has been saved to', filename)


### Code in this section will demonstrate Multiple Linear Regression

def mlr(y,X,intercept=True):
    n,p = np.shape(X)
    if intercept and not np.all(X[:,0]==1):
        X = np.column_stack((np.ones(n),X))
    Xt = np.transpose(X)
    XtX = np.matmul(Xt,X)
    XtXinv = np.linalg.inv(XtX)
    Xty = np.matmul(Xt,y)
    betahat = np.matmul(XtXinv,Xty)

    # Predicted values
    y_pred = np.matmul(X, betahat)
    return betahat

import sklearn.datasets as data
X, y = data.load_diabetes(return_X_y=True)
betahat = mlr(y, X)
