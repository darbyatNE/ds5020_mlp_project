import numpy as np  # Line 1: Import NumPy library as np
from sklearn import datasets  # Line 2: Import datasets module from scikit-learn library
import matplotlib.pyplot as plt  # Line 3: Import pyplot module from matplotlib library
from sklearn.preprocessing import StandardScaler  # Line 4: Import StandardScaler class from scikit-learn


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
#target_column = 'Potability' 

# split data by characteristics and target
y = data[:,-1]  # All rows, only the last column
# normalize data sets mean to a value of zero and var. To a value of one.
scaler = StandardScaler()
X = scaler.fit(X).transform(X)

# Check for zero mean and unit variance
print('Checking mean of scaled features (Value should be near 0): ', np.mean(X, 0))  # Line 9: Compute mean of each feature in scaled feature matrix X
print('Checking variance of scaled features (Value should equal 1): ', np.var(X, 0))   # Line 10: Compute variance of each feature in scaled feature matrix X

def MLP(y, X, m=10, gam=0.1, T=1e4):  # Line 11: Define MLP function with parameters
    n, p = np.shape(X)  # Line 12: Get number of samples (n) and number of features (p) in X
    beta0 = 0  # Line 13: Initialize bias term beta0
    vbeta = np.zeros(m)  # Line 14: Initialize weight vector vbeta
    valp0 = np.zeros(m)  # Line 15: Initialize intermediate variable valp0
    A = np.zeros((m, p))  # Line 16: Initialize weight matrix A

    def phi(z):  # Line 17: Define sigmoid activation function phi
        return 1 / (1 + np.exp(-z))  # Line 18: Sigmoid activation function definition

    t = 0  # Line 19: Initialize iteration counter t
    err = np.zeros(int(T))  # Line 20: Initialize array to store errors during training
    while t < T-1:  # Line 21: Start main training loop
        t += 1  # Line 22: Increment iteration counter
        # Sample a random index (data point)
        ix = np.random.choice(n, 1)  # Line 24: Randomly select a data point index
        # Forward part of back propagation
        vh = phi(valp0 + np.matmul(A, X[ix].flatten()))  # Line 26: Compute hidden layer activations
        yhat = phi(beta0 + np.dot(vbeta, vh))  # Line 27: Compute output prediction
        # Update parameters
        E = yhat - y[ix]  # Line 29: Compute error
        beta0 = beta0 - gam * E  # Line 30: Update bias term
        temp = vbeta * vh * (1 - vh)  # Line 31: Compute temporary variable for weight update
        vbeta = vbeta - gam * E * vh  # Line 32: Update weight vector
        valp0 = valp0 - gam * E * temp  # Line 33: Update intermediate variable
        A = A - gam * E * np.outer(temp, X[ix].flatten())  # Line 34: Update weight matrix

        # Compute mean error
        for i in range(n):  # Line 36: Iterate over all data points
            vh = phi(valp0 + np.matmul(A, X[i]))  # Line 37: Compute hidden layer activations for each data point
            yhat = phi(beta0 + np.dot(vbeta, vh))  # Line 38: Compute output predictions
            err[t] += np.sum(np.abs(yhat - y[i]))  # Line 39: Compute absolute error
        err[t] = err[t] / n  # Line 40: Compute mean error for the iteration
        # if np.mod(t, 100) == 0:  # Line 41: Plot error curve every 100 iterations
        # err_plot(0) =err[t] 
    return beta0, vbeta, valp0, A, err  # Line 44: Return learned parameters after training

beta0, vbeta, valp0, A, err = MLP(y, X, m=10, gam=0.005, T=600)
# Plot the error curve
plt.figure(figsize=(8, 6))  # Set the size of the figure
plt.plot(err[1:])
plt.xlabel('Iteration')
plt.ylabel('Total Error')
plt.title('Error Curve over # of Iterations')
plt.tight_layout()  # Adjust layout to prevent clipping of labels
filename = 'figs/mlp_error_curve.png'
plt.savefig(filename)  # Save the plot as a PNG file in the fig directory
print('Plot of the error curve was saved to ', filename)

