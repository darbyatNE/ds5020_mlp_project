import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Orthonormalization (Gram-Schmidt)

def gs(Ain):
    A = Ain.copy()
    m, n = np.shape(A)
    A[:,0] = A[:,0] / np.sqrt(np.dot(A[:,0], A[:,0]))
    for k in range(1, n):
        for j in range(k):
            A[:,k] = A[:,k] - np.dot(A[:,j], A[:,k]) * A[:,j]
        A[:,k] = A[:,k] / np.sqrt(np.dot(A[:,k], A[:,k]))
    return A

### Simple Linear Regression (SLR) for four variables

def slr(x1, x2, x3, y):
    n = len(y)
    x1bar = np.mean(x1)
    x2bar = np.mean(x2)
    x3bar = np.mean(x3)
    ybar = np.mean(y)
    sx1 = np.std(x1)
    sx2 = np.std(x2)
    sx3 = np.std(x3)
    sy = np.std(y)
    zx1 = (x1 - x1bar) / sx1
    zx2 = (x2 - x2bar) / sx2
    zx3 = (x3 - x3bar) / sx3
    zy = (y - ybar) / sy
    r_x1y = np.sum(np.multiply(zy, zx1)) / (n - 1)
    r_x2y = np.sum(np.multiply(zy, zx2)) / (n - 1)
    r_x3y = np.sum(np.multiply(zy, zx3)) / (n - 1)
    beta1 = r_x1y * sy / sx1
    beta2 = r_x2y * sy / sx2
    beta3 = r_x3y * sy / sx3
    b0 = ybar - beta1 * x1bar - beta2 * x2bar - beta3 * x3bar
    return b0, beta1, beta2, beta3

# Read data from CSV
dat = pd.read_csv("data/lin_reg_data/HousingData_Lmtd.csv")
colnames = dat.columns
dat = dat.to_numpy()
x1 = dat[:,0]  # First independent variable
x2 = dat[:,1]  # Second independent variable
x3 = dat[:,2]  # Third independent variable
y = dat[:,3]   # Dependent variable

# Perform Simple Linear Regression
b0, b1, b2, b3 = slr(x1, x2, x3, y)

# Plotting the data and the regression surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, y, color='k', marker='.')
x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))
y_pred = b0 + b1 * x1_grid + b2 * x2_grid + b3 * x3.mean()
ax.plot_surface(x1_grid, x2_grid, y_pred, alpha=0.5)
ax.set_xlabel(colnames[0])
ax.set_ylabel(colnames[1])
ax.set_zlabel(colnames[2])
ax.set_zlabel(colnames[3])
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