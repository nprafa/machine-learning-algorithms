import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def cal_cost(theta,X,y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost

def batch_gradient_descent( x, y, theta , max_itr, alpha = 1e-3):
    theta_history = np.zeros((iterations, 2))
    cost_history = np.zeros(iterations)
    m = len(y)

    for it in range(max_itr):
        predictions = np.dot(x, theta) - y
        theta = theta - (1 / m) * alpha * x.T.dot(predictions)
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, x, y)
    return theta, theta_history, cost_history

def stocastic_gradient_descent(x, y, theta , alpha = 1e-3):
    m = len(y)
    theta_history = np.zeros((m, 2))
    cost_history = np.zeros(m)
    it = 0
    for x_t, y_t in zip(x, y):
        predictions = np.dot(theta, x_t) - y_t
        gradient = x_t * predictions
        theta += alpha * gradient / m
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, x, y)
        it += 1
    return theta, theta_history, cost_history




filename = 'ex1data.txt'
data = open(filename, 'rt')
data = list(csv.reader(data, delimiter=',', quoting=csv.QUOTE_NONE))

#data variable
theta_bg = np.array([0.01, 0.01])
theta_sg = np.array([0.2, 1])
x = np.array([elem[0] for elem in data], dtype=float)
X = np.c_[np.ones(x.shape[0]), x]
Y = np.array([elem[1] for elem in data], dtype=float)
#split data as test and train
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

#m - size of training samples
#n - no of parameters to train
m = x_train.shape[0]
n = len(data[0])
iterations = 1000



#batch gradient decsent output
(theta_n, theta_history, cost_history) = batch_gradient_descent(x_train, y_train, theta_bg, iterations, 0.001)
#stocastic gradient descent output
(theta_s, theta_history_s, cost_history_s) = stocastic_gradient_descent(x_train, y_train, theta_sg)

#plot
plt.figure()
plt.scatter(X[:,1], Y)
plt.plot(x_test[:,1], theta_n[0] + theta_n[1]*x_test[:,1], 'g', label='batch')
plt.plot(x_test[:,1], theta_s[0] + theta_s[1]*x_test[:,1], '-r', label='stochastic')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()






















