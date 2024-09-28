
# This is very beginning of ML basics.

# Goal: Implement a simple linear regression with one variable.

# Objective-1: Practice to strengthen basic concepts.
# Objective-2: Learning opportunity from scratch. 
#               This flow from basic applications will provide opportunity to sync basic concepts.
# Objective-3: Designing systems from scratch with own constraints.

# Approach: Implement a hypothetical model for prices of house given area of house
# 1. All data is hypothetical and created based on intuition.
# 2. Input feature is size of a house in sqft and output(prediction) is price is in thousands of dollars.
# 3. Create a skeleton for implementation with training data with enough number of examples and prices of houses.
# 4. Create skeleton for implemetation of the problem as taught in course videos. This includes hypothesis relation between
#       input feature and output. Skeleton also includes defining cost function, defining gradient descent.
# 5. Define number of training examples, set nominal learning rate, define number of iterations for training.
# 6. Compute cost and compute gradient
# 7. Update weigths and biases simultaneously
# 8. Tune learning rate and number of iterations
# 9. Ensure convergence. Use different training data samples as needed. Observe cost and ensure it converges. If it doesnt,
#       find root cause.


import numpy as np
import matplotlib.pyplot as plt

# Define training data
X_train = np.array([3277, 2394, 3900, 1150, 2200])
Y_train = np.array([530, 470, 670, 270, 430])

# Following training data represents a straight line y = x, (f_wb = 1*x + 0)
# Following training data is used for debugging as it is easier for calculations and has easily predictable w and b
# X_train = np.array([2.3, 3.2, 3.9, 5, 6.9])
# Y_train = np.array([2, 3, 4, 5, 7])


# Initialize weights and biases
def initialize_weights_and_biases():
    W = 0.000000001
    b = 0.000000002
    return W,b

# Model
# Compute cost
def compute_cost(X_train, Y_train, m, W, b):
    cost = 0
    f_wb = 0
    for i in range(m):
        f_wb = W*X_train[i] + b
        cost += (f_wb - Y_train[i])**2
    total_cost = (1/(2*m))*cost
    #print(f"total_cost:{total_cost:.2f}")
    return total_cost

# Compute gradient
def compute_gradient(X_train, Y_train, m, W, b):
    dJ_dW = 0
    dJ_db = 0
    for i in range(m):
        y = Y_train[i]
        x = X_train[i]
        f_wb = W*x + b
        #print(f"compute_gradient: f_wb:{f_wb:.3f}")
        dJ_dW_i = 2*(f_wb - y)*x
        dJ_db_i = 2*(f_wb - y)
        #print(f"compute_gradinet: dJ_dW_i:{dJ_dW_i:.3f}, dJ_db_i:{dJ_db_i:.3f}")
        dJ_dW += dJ_dW_i
        dJ_db += dJ_db_i

    dJ_dW = (1/m)*dJ_dW
    dJ_db = (1/m)*dJ_db
    print(f"\ndJ_dW:{dJ_dW:.3f}, dJ_db:{dJ_db:.3f}")
    
    return dJ_dW, dJ_db


# Evaluate hypothesis and run simple linear regression model
# Number of training examples
m = X_train.shape[0]
print(f"m:{m}")
learning_rate = 1e-9
num_iter = 250

W, b = initialize_weights_and_biases()
cost_history = []
for n in range(num_iter):
    total_cost = compute_cost(X_train=X_train, Y_train=Y_train, m=m, W=W, b=b)
    cost_history.append(total_cost)
    dJ_dW, dJ_db = compute_gradient(X_train=X_train, Y_train=Y_train, m=m, W=W, b=b)
    print(f"Total cost:{total_cost:.3f}")
    print(f"W:{W:.3f}, dJ_dW:{dJ_dW:.3f}, Updated W:{(W+(learning_rate*dJ_dW)):.3f}")
    W = W - (learning_rate * dJ_dW)
    b = b - (learning_rate * dJ_db)
    print(f"Iteration:({n}), Cost:{total_cost}")#, (w,b):({W:.3f,b:.3f})")

print(f"\nFinal values of W:{W:.3f} and b:{b:.3f}")

# Visualize training data for better understanding
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,5))
ax1.plot(X_train, Y_train, 'ro')
ax1.set_title("Pricing of houses in Michigan")
ax1.set_xlabel("Area of house in sqft")
ax1.set_ylabel("Price of house in 1000s of dollars")
ax1.set_xlim(0,max(X_train)*1.25)
ax1.set_ylim(0, max(Y_train)*1.25)
# Plot prediction
Y_pred = np.zeros((len(X_train)))
for k in range(len(X_train)):
    Y_pred[k] = W*X_train[k] + b
ax1.plot(X_train, Y_pred, 'b-')

# Plot cost history showing convergence
ax2.plot(np.arange(num_iter),cost_history)
ax2.set_title("Cost vs number of iterations")
ax2.set_xlabel("number of iterations")
ax2.set_ylabel("Value of cost function")
ax2.set_xlim(0,num_iter*1.25)
ax2.set_ylim(0, max(cost_history)*1.25)
plt.show()
