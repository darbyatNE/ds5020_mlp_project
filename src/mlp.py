# import dependencies
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

# preprocessing data check 
# check for zero mean and unit variance
np.mean(X,0)
np.var(X,0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# defines the MLP function with parameters y, X, m, gam & T
# y is the target variable array
# X contains all of the input features from the dataset
# m holds the number of neurons in hidden layer
# gam is a hyperparameter which holds the learning rate
# T is a hyperparameter which holds the number of epochs to run the model
def MLP(y,X,m=9,gam=.1,T=5e3):
    # defines the number of samples (n) and the number of features (p) in the dataset
    n, p = np.shape(X) 
    # beta0 is the bias term
    beta0 = 0
    # vbeta is the weighting vector
    vbeta = np.zeros(m)
    # valp0 is the hidden layer value array
    valp0 = np.zeros(m)
    # init weight matrix A with params m and p
    # p holds the number of features in the dataset, typically this matches the 
    #	 number of neurons in the input layer
    A = np.zeros((m,p))
    # defines the sigmoid activation function
    def phi(z):
        return 1/(1+np.exp(-z))
    # initializes the training loop iteration count
    t = 0
    # initializes the mean error array of all epochs(T)
    err = np.zeros(int(T))
    
    # Gradient Descent Process: implemented to optimize the parameters (beta0, vbeta, valp0, and A) 
    while t<T-1:
        t += 1
        # selects one index from n possible samples
        ix = np.random.choice(n,1)
        # performs a matrix multiplication of input features with the weight matrix
        vh = phi(valp0+np.matmul(A,X[ix].flatten()))
        # yhat holds the predicted output
        yhat = phi(beta0+np.dot(vbeta,vh))
        # error (E) between the predicted output (yhat) and the target label (y[ix]) 
        E = yhat-y[ix]
        #E = (yhat - y[ix]) * np.ones_like(vbeta)        
        # updates the bias term
        beta0 = beta0-gam*E
        # holds temp value for weight update
        temp = vbeta*vh*(1-vh)
        # updates the weight vector
        vbeta = vbeta-gam*E*vh
        # updates the hidden layer neuron values
        valp0 = valp0-gam*E*temp
        # updates the weight matrix
        A = A-gam*E*np.outer(temp,X[ix].flatten())
        # compute mean error: Ei=ŷi-yi
        for i in range(n):
            # calculate individual neuron activations in the hidden layer
            vh = phi(valp0+np.matmul(A,X[i]))
            # calculates the output predictions
            yhat = phi(beta0+np.dot(vbeta,vh))
            # calculates the absolute error/mean for each sample (water source)
            err[t] += np.abs(yhat-y[i])
            err[t] = err[t]/n
            # Print epoch number and total error
            print(f"Epoch {t}: Total Error = {err[t]:.6f}")
    
    # Plot the error curve
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    plt.plot(err[1:t])
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title('Error Curve over Epochs')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('fig/error_curve.png')  # Save the plot as a PNG file in the fig directory

         
    # Function to predict labels
    def predict(beta0, vbeta, valp0, A, X):
        vh = phi(valp0[:, np.newaxis] + np.matmul(A, X.T))
        yhat = phi(beta0 + np.dot(vbeta, vh))
        return yhat

    # Make predictions on training and test data
    y_train_pred = predict(beta0, vbeta, valp0, A, X_train)
    y_test_pred = predict(beta0, vbeta, valp0, A, X_test)

    # Calculate accuracy
    def calculate_accuracy(y_true, y_pred):
        correct = (y_true == np.round(y_pred)).sum()
        total = len(y_true)
        accuracy = correct / total
        return accuracy

    # Report accuracy
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)         
    return train_accuracy, test_accuracy

   
# Call the MLP function
train_accuracy, test_accuracy= MLP(y, X, m=6, gam=0.05, T=160)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)  
