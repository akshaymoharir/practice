
# This is a basic example of multiple variable linear regression.

# Goal: Implement multiple variable linear regression from scratch.

# Description: Use example of house prices as that of taught in course work by Prof Andrew Ng in courses


# Features are 
# 1. Area of house in sqft
# 2. number of bedrooms
# 3. Age of house
# 4. Number of floors

# f_wb = w1x1 + w2x2 + w3x3 + w4x4 + b

import numpy as np
import matplotlib.pyplot as plt

# Training data
X_train = np.array([[2104, 5, 45, 1], [1416, 3, 40, 2], [852, 2, 35, 1]])
Y_train = np.array([460, 232, 178])

print(f"Shape of X_train:{X_train[0].shape}")

def initialize_weights_and_biases(X_train):
    W = np.ones((X_train[0].shape)) * 0.000001
    b = np.array([1.0]) * 0.000001
    print(f"Number of Input features:{X_train[0].shape}, Shape of W:(n_x, n_h):{X_train[0].shape, 1} ")
    return W, b

# Compute Cost
# Cost is defined by mean squarred error. 
# J(w,b) = (1/2m)*summation( (f_wb(x[i]) - y[i])^2 )
def compute_cost(X_train, Y_train, W, b, m):
    total_cost = 0
    cost = 0
    f_wb = 0
    for i in range(m):
        f_wb = np.dot(W.T,X_train[i]) + b
        cost = cost + (f_wb - Y_train[i])**2
    total_cost = ((1/(2*m))*cost).astype(float)
    print(total_cost)
    return total_cost

# Compute Gradient
# Partial derivative of cost wrt W is (1/2m)* summation(2(WX[i]+b - Y[i])X[i])
# Partial derivative of cost wrt b is (1/2m)* summation(2(WX[i]+b - Y[i]))
def compute_gradient(X_train, Y_train, W, b, m):
    dJ_dW_i = 0
    dJ_db_i = 0
    # Find the summation of the partial derivate of cost
    for i in range(m):
        f_wb = np.dot(W.T,X_train[i]) + b
        dJ_dW_i += 2*(f_wb - Y_train[i])*X_train[i]
        dJ_db_i += 2*(f_wb - Y_train[i])
    # Divide the summation by 2m
    dJ_dW = (1/(2*m))*dJ_dW_i
    dJ_db = (1/(2*m))*dJ_db_i
    return dJ_dW.astype(float), dJ_db.astype(float)


# Iterate training process
m = X_train.shape[0]
print(f"Number of training examples:{m}")

# Number of iterations to train model on training data
num_iterations = 500

# Learning rate
learning_rate = 1e-8

# Initialize weights and biases
W, b = initialize_weights_and_biases(X_train=X_train)

# Iterate training
history_of_cost = np.zeros((num_iterations))
history_of_W1 = []
for n in range(num_iterations):
    # Compute cost
    total_cost = compute_cost(X_train=X_train, Y_train=Y_train, W=W, b=b, m=m)
    history_of_cost[n] = total_cost
    # Compute gradient
    dJ_dW, dJ_db = compute_gradient(X_train=X_train, Y_train=Y_train, W=W, b=b, m=m)
    # Update weights and biases
    W = W - learning_rate*dJ_dW
    b = b - learning_rate*dJ_db


# Plot
fig = plt.figure()
# Plot cost
ax1 = fig.add_subplot(2,1,1)
ax1.plot(np.arange(num_iterations),history_of_cost)
ax1.set_title("MSE Cost")
ax1.set_xlabel("Num of iterations")
ax1.set_ylabel("Value of cost")
#ax2 = fig.add_subplot(2,1,2, projection="3d")
plt.show()
