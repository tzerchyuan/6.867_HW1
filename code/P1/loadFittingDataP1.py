import pylab as pl
import numpy as np


def getData():

    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y)

X, y = getData()

def gradient_descent(thresh, learning_rate):
    old_theta = np.array([0]*len(X[0]))
    new_theta = old_theta - learning_rate*(2*(np.dot(np.transpose(X), np.dot(X, old_theta) - y)))
    step = 0
    while np.linalg.norm(new_theta - old_theta) > thresh:
        old_theta = new_theta
        new_theta = old_theta - learning_rate*(2*(np.dot(np.transpose(X), np.dot(X, old_theta) - y)))
        step += 1
    return (new_theta, step)

def sgd(thresh, learning_rate):
    old_theta = np.array([0]*len(X[0]))
    new_theta = old_theta - 2*learning_rate*(np.dot(X[0], (np.dot(X[0], old_theta) - y[0])))
    i = 0
    step = 0
    while np.linalg.norm(new_theta-old_theta) > thresh:
        step += 1
        old_theta = new_theta
        ind = i%len(X)
        new_theta = old_theta - 2*learning_rate*(np.dot(X[ind], (np.dot(X[ind], old_theta) - y[ind])))
        i += 1
    return (new_theta, step)

gd_ans = gradient_descent(0.0000011, 0.000000025)
print('gradient_descent: ', gd_ans[0], 'steps: ',gd_ans[1] )

real_answer = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
print('real_answer--->', real_answer)



sgd_ans = sgd(0.000000011, 0.00000028)
print('sgd: ', sgd_ans[0], 'steps: ', sgd_ans[1])
print('difference_____ >', real_answer - sgd_ans[0])
