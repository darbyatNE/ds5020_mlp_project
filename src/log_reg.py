import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def logistic_gd(y, X, alp=0.1, eps=1e-4):
    def logistic(z):        
        return np.exp(z) / (1 + np.exp(z))    
    n, p = np.shape(X)
    if not np.all(X[:, 0] == 1):
        X = np.column_stack((np.ones(n), X))
    Xt = np.transpose(X)
    vbeta = np.random.normal(0, 1, p + 1)
    vbeta0 = vbeta + 1
    vbeta_history = []  # List to store the changes in vbeta values
    while np.max(np.abs(vbeta - vbeta0)) > eps:
        vbeta0 = vbeta
        linpred = np.matmul(X, vbeta)
        yhat = logistic(linpred)
        grad = np.matmul(Xt, yhat - y) / n
        delta_vbeta = alp * grad  # Change in vbeta for this iteration
        vbeta = vbeta - delta_vbeta  # Update vbeta
        vbeta_history.append(np.sum(delta_vbeta))  # Append the change in vbeta to history
    
    # Convert vbeta_history to numpy array for plotting
    vbeta_history = np.array(vbeta_history)
    
    # Plot and save vbeta history
    plt.plot(vbeta_history)
    plt.xlabel('Iteration')
    plt.ylabel('Change in vbeta')
    plt.title('Change in vbeta vs. Iteration in the logistic regression gradient descent model')
    filename = 'fig/log_grad_dec_vbeta_change_plot.png'
    plt.savefig(filename)
    plt.show()
    print('A chart showing the change in vbeta through the iterations of the model has been saved to', filename)
    return vbeta, yhat


def logistic_newton(y, X, alpha=0.1, eps=1e-4):
    def logistic(z):
        return np.exp(z) / (1 + np.exp(z))    
    n, p = np.shape(X)
    if not np.all(X[:, 0] == 1):
        X = np.column_stack((np.ones(n), X))
    Xt = np.transpose(X)
    vbeta = np.random.normal(0, 1, p + 1)
    vbeta0 = vbeta + 1
    vbeta_history = []  # List to store the changes in vbeta values
    while np.max(np.abs(vbeta - vbeta0)) > eps:
        vbeta0 = vbeta
        linpred = np.matmul(X, vbeta)
        # Use linear predictor directly for predictions
        yhat = linpred
        grad = np.matmul(Xt, yhat - y)
        H = np.matmul(np.matmul(Xt, np.diag(np.multiply(yhat, 1 - yhat))), X)
        Hinv = np.linalg.inv(H)
        delta_vbeta = alpha * np.matmul(Hinv, grad)  # Change in vbeta for this iteration
        vbeta = vbeta - delta_vbeta  # Update vbeta
        vbeta_history.append(np.sum(delta_vbeta))  # Append the change in vbeta to history
    
    # Convert vbeta_history to numpy array for plotting
    vbeta_history = np.array(vbeta_history)
    
    # Plot and save vbeta history
    plt.plot(vbeta_history)
    plt.xlabel('Iteration')
    plt.ylabel('Change in vbeta')
    plt.title('Change in vbeta vs. Iteration')
    filename = 'fig/log_newton_vbeta_change_plot.png'
    plt.savefig(filename)
    plt.show()
    print('A chart showing the change in vbeta through the iterations of the model has been saved to', filename)
    return vbeta, yhat


# Load the CSV file into a NumPy array
def load_dataset():
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
    # set the target variable
    y = data[:,-1]  # All rows, only the last column
    # normalize data sets mean to a value of zero and var. To a value of one.
    scaler = StandardScaler()
    X = scaler.fit(X).transform(X)
    return X, y

#Load dataset and execute logistic regression gradient descent model 
X, y = load_dataset()
vbeta, yhat = logistic_gd (y ,X , alp =0.1 ,eps=5e-5)

print(f'vhat: {yhat}')
print(f'vbeta: {vbeta}')

plt.clf()  # Clear the current figure

#Load dataset and execute logistic regression Newton model 
X, y = load_dataset()
vbeta, yhat = logistic_newton(y ,X , alpha=0.02, eps=2e-4)
print('Listed below are the final parameter values:/n')
print(f'vhat: {yhat}')
print(f'vbeta: {vbeta}')
