import numpy as np
from sklearn.preprocessing import StandardScaler

def logistic_gd (y ,X , alp =0.1 , eps =1e-4):
    # alpha : step size
    # epsilon : used in haling criterion
    def logistic ( z ):
        return np . exp ( z )/(1+ np . exp ( z ))
    n , p = np . shape ( X )
    if not np . all ( X [: ,0]==1):
        X = np . column_stack (( np . ones ( n ) , X ))
    Xt = np . transpose ( X )
    vbeta = np . random . normal (0 ,1 , p+1)
    vbeta0 = vbeta +1
    while np . max ( np . abs ( vbeta - vbeta0 )) > eps :
        vbeta0 = vbeta
        linpred = np . matmul (X , vbeta )
        yhat = logistic ( linpred )
        grad = np . matmul ( Xt , yhat - y )/ n
        vbeta = vbeta - alp * grad
    return vbeta , yhat


def logistic_newton (y ,X , eps =1e-4):
    # alpha : step size
    # epsilon : used in haling criterion
    def logistic ( z ):
        return np . exp ( z )/(1+ np . exp ( z ))
    n , p = np . shape ( X )
    if not np . all ( X [: ,0]==1):
        X = np . column_stack (( np . ones ( n ) , X ))
    Xt = np . transpose ( X )
    vbeta = np . random . normal (0 ,1 , p+1)
    vbeta0 = vbeta +1
    while np . max ( np . abs ( vbeta - vbeta0 )) > eps :
        vbeta0 = vbeta
        linpred = np . matmul (X , vbeta )
        yhat = logistic ( linpred )
        grad = np . matmul ( Xt , yhat - y )
        H = np . matmul ( np . matmul ( Xt , np . diag ( np . multiply ( yhat ,1 - yhat ))) , X )
        Hinv = np . linalg . inv ( H )
        vbeta = vbeta - Hinv * grad
    return vbeta , yhat