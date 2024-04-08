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

# Load the CSV file into a NumPy array
data = np.genfromtxt('data/water_potability.csv', delimiter=',', skip_header=1)

# Calculate column-wise means ignoring NaN values
column_means = np.nanmean(data, axis=0)

# Find indices of missing values (NaNs) in the array
missing_indices = np.isnan(data)

# Replace missing values with column-wise means
for i in range(data.shape[1]):  # Iterate over columns
    col = data[:, i]
    col_mean = column_means[i]
    col[np.isnan(col)] = col_mean
# Check for missing values
missing_values = np.isnan(data).sum(axis=0)

# feature_columns = ['ph', 'Hardness','Solids','Chloramines','Solids', 'Sulfate', 'Conductivity',
#                    'Organic_carbon', 'Trihalomethanes', 'Turbidity']  
X = data[:, :-1]  # All rows, all columns except the last column

# define target
# target_column = 'Potability' 

# split data by characteristics and target
y = data[:,-1]  # All rows, only the last column
# normalize data sets mean to a value of zero and var. To a value of one.
scaler = StandardScaler()
X = scaler.fit(X).transform(X)

# preprocessing data check 
# check for zero mean and unit variance
np.mean(X,0)
np.var(X,0)

vbeta, yhat = logistic_gd (y ,X , alp =0.1 ,eps=1e-4)
print(f'The prediction rate using logistic regression with gradient descent was: {yhat}')
print(f'The estimated coefficients of the logistic regression model were: {vbeta}')

vbeta, yhat = logistic_newton(y ,X , eps=1e-4)
print(f'The prediction rate using logistic regression with Newton\'s method was: {yhat}')
print(f'The estimated coefficients of the logistic regression model were: {vbeta}')
