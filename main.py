import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('cancer_data.csv', delimiter=',')

# Extract X and y
X = data[:,:-1]
y = data[:,-1]

# Normalize X
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Check that mean is 0 and standard deviation is 1
print(np.mean(X, axis=0))  # should print array([0., 0., 0., 0., 0.])
print(np.std(X, axis=0))   # should print array([1., 1., 1., 1., 1.])

X = np.hstack((np.ones((X.shape[0], 1)), X))

def predict_linear_regression(x, theta):
    return np.dot(x, theta)

def cost_linear_regression(X, y, theta):
    m = len(y)
    h = predict_linear_regression(X, theta)
    return np.sum((h - y) ** 2) / (2 * m)

def gradient_linear_regression( X, y, theta):
    m = len(y)
    h = predict_linear_regression(X, theta)
    return np.dot(X.T, (h - y)) / m

def gradient_descent_linear_regression(X, y, alpha, num_iters):
    m, n = X.shape
    theta = np.zeros(n)
    J_history = []
    for i in range(num_iters):
        theta -= alpha * gradient_linear_regression(X, y, theta)
        J_history.append(cost_linear_regression(X, y, theta))
    return theta, J_history

alpha_values = [0.5, 0.1, 0.01, 0.001]
for alpha in alpha_values:
    theta, J_history = gradient_descent_linear_regression(X, y, alpha, 100)
    plt.plot(J_history, label='alpha = {}'.format(alpha))
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent for Linear Regression')
plt.legend()
plt.show()


def mini_batch_gradient_descent_linear_regression(X, y, alpha, num_iters, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    J_history = []
    for i in range(num_iters):
        idx = np.random.choice(m, batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        theta -= alpha * gradient_linear_regression(X_batch, y_batch, theta)
        J_history.append(cost_linear_regression(X, y, theta))
    return theta, J_history

batch_sizes = [10, 50, 100, 500]
for batch_size in batch_sizes:
    theta, J_history = mini_batch_gradient_descent_linear_regression(X, y, 0.01, 100, batch_size)
    plt.plot(J_history, label='batch_size = {}'.format(batch_size))
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Mini-Batch Gradient Descent for Linear Regression')
plt.legend()
plt.show()

def adam(theta, X, y, alpha, num_iters, epsilon=1e-8, beta1=0.9, beta2=0.999):
    m = y.size
    grad_squared = 0
    grad_momentum = 0
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        grad = gradient_linear_regression( X, y, theta)
        grad_momentum = beta1 * grad_momentum + (1 - beta1) * grad
        grad_squared = beta2 * grad_squared + (1 - beta2) * grad**2
        grad_momentum_corrected = grad_momentum / (1 - beta1**(i+1))
        grad_squared_corrected = grad_squared / (1 - beta2**(i+1))
        theta = theta - alpha * grad_momentum_corrected / (np.sqrt(grad_squared_corrected) + epsilon)
        J_history[i] = cost_linear_regression(X, y, theta)
    return theta, J_history

theta = np.zeros(X.shape[1])
alpha = 0.01
num_iters = 1000
theta_final, J_history = adam(theta, X, y, alpha, num_iters)
plt.plot(np.arange(num_iters), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()