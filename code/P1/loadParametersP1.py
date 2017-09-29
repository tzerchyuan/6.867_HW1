import pylab as pl
import numpy as np
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def getData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean,gaussCov,quadBowlA,quadBowlb)

mean, cov, A, b = getData()

def gauss(mean, cov, x):
    det = np.linalg.det(cov)
    first_term = -10**4/(((2*math.pi)**len(x))*det)
    v = x - mean
    second_term = np.exp(-.5*np.dot(np.dot(np.transpose(v),np.linalg.inv(cov)),v))
    return first_term*second_term

def gauss_derivative(f, mean, cov, x):
    return -(np.dot(np.dot(f(mean, cov, x), np.linalg.inv(cov)), (x-mean)))


def gauss_grad_descent(f, df, start, thresh, step_size):
    norms = []
    ws = []
    old_w = start
    new_w = old_w - step_size*df(f, mean, cov, old_w)
    norms.append(np.linalg.norm(df(f, mean, cov, old_w)))
    ws.append(old_w)
    while(abs(f(mean, cov, new_w) - f(mean, cov, old_w)) >= thresh and np.linalg.norm(df(f, mean, cov, old_w)) >= thresh):
        # print("New, Old: ", new_w, old_w)
        old_w = new_w
        new_w = old_w - step_size*df(f, mean, cov, old_w)
        norms.append(np.linalg.norm(df(f, mean, cov, old_w)))
        ws.append(old_w)
    norms.append(np.linalg.norm(df(f, mean, cov, new_w)))
    ws.append(new_w)
    return new_w, norms, ws

def do_grad_gauss():
    start = [5,5]
    thresh = .000001
    step_size = .25
    end, norms, ws = gauss_grad_descent(gauss, gauss_derivative, start, thresh, step_size)
    print("GD Gauss: ", end)
    plt.plot(norms)
    plt.show()

def bowl(A, b, x):
    return 0.5*np.dot(np.dot(np.transpose(x), A), x) - np.dot(np.transpose(x), b)

def bowl_derivative(A, b, x):
    return np.dot(A, x) - b

def bowl_grad_descent(f, df, start, thresh, step_size):
    norms = []
    ws = []
    old_w = start
    new_w = old_w - step_size*df(A, b, old_w)
    norms.append(np.linalg.norm(df(A, b, old_w)))
    ws.append(old_w)
    while(abs(f(A, b, new_w) - f(A, b, old_w)) >= thresh and np.linalg.norm(df(A, b, old_w)) >= thresh):
        # print("New, Old: ", new_w, old_w)
        old_w = new_w
        new_w = old_w - step_size*df(A, b, old_w)
        norms.append(np.linalg.norm(df(A, b, old_w)))
        ws.append(old_w)
    norms.append(np.linalg.norm(df(A, b, new_w)))
    ws.append(new_w)
    return new_w, norms, ws

def do_grad_bowl():
    start = [5,5]
    thresh = .000001
    step_size = .0025
    end, norms, ws = bowl_grad_descent(bowl, bowl_derivative, start, thresh, step_size)
    print("GD Bowl: ", end)
    plt.plot(norms)
    plt.show()

# do_grad_bowl()
# do_grad_gauss()

"""
P1.2
"""

# d = [ss,ss,ss,ss,...]
def approx_grad_gauss(f, x, d):
    # return (f(mean, cov, x + d) - f(mean, cov, x))/np.linalg.norm(d)
    approx_grad = []
    for i in range(len(x)):
        # for x, looks like [d0,0]
        # for y, looks like [0,d1]
        y = np.array([0 if j != i else d[i] for j in range(len(x))])
        approx_grad.append((f(mean, cov, x+y)-f(mean, cov, x))/d[i])
    return np.array(approx_grad)

delta = 0.001
for i in np.arange(9,11.25,0.25):
    for j in np.arange(9,11.25,0.25):
        point = np.array([i,j])
        print("APPROX: ", approx_grad_gauss(gauss, point, np.array([delta]*len(point))))
        print("ACTUAL: ", gauss_derivative(gauss, mean, cov, np.array([i,j])))
        print("------------------------------------")