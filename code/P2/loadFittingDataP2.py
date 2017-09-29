import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import math

def arr(x):
    return np.array(x)
def inv(x):
    return np.linalg.inv(x)
def dot(x,y):
    return np.dot(x,y)
def T(x):
    return np.transpose(x)

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)

def poly_basis(M):
    return [(lambda y: (lambda x: x**y))(i) for i in range(M+1)]

def max_likelihood_weight_vector(X, y, M):
    N = len(X)
    phi = np.array([[poly_basis(M)[i](X[j]) for i in range(M+1)] for j in range(N)])
    phiT = T(phi)
    return dot(dot(inv(dot(phiT, phi)),phiT),y)

X, y = getData(False)

# for M in [0, 1, 2, 3, 6, 10]:
#     mlwv = max_likelihood_weight_vector(X, y, M)
#
#     plt.plot(X, y, 'o')
#     plt.plot(X, [dot(mlwv, arr([poly_basis(M)[i](X[j]) for i in range(M + 1)])) for j in range(len(X))], 'x')
#     plt.show()

def plot_poly_regression(X, y, M):
    mlwv = max_likelihood_weight_vector(X, y, M)

    plt.plot(X, y, 'o')
    plt.plot(X, [dot(mlwv, arr([poly_basis(M)[i](X[j]) for i in range(M + 1)])) for j in range(len(X))], 'x')
    plt.show()

# plot_poly_regression(X, y, 10)

def poly_SSE(X, y, M, w):
    N = len(X)
    # w = max_likelihood_weight_vector(X, y, M)
    # print([(y[i] - dot(T(w), [poly_basis(M)[j](X[i]) for j in range(M+1)]))**1 for i in range(N)])
    return sum([(y[i] - dot(T(w), [poly_basis(M)[j](X[i]) for j in range(M+1)]))**2 for i in range(N)])

def poly_SSE_derivative(X, y, M, w):
    # w = max_likelihood_weight_vector(X, y, M)
    N = len(X)
    phi = np.array([[poly_basis(M)[i](X[j]) for i in range(M + 1)] for j in range(N)])
    phiT = T(phi)
    return -2 * dot(phiT, (y - dot(phi, w)))

def approx_grad_poly(X, y, M, w, d):
    approx_grad = []
    for i in range(len(w)):
        # for x, looks like [d0,0]
        # for y, looks like [0,d1]
        z = np.array([0 if j != i else d[i] for j in range(len(w))])
        approx_grad.append((poly_SSE(X, y, M, w+z)-poly_SSE(X, y, M, w-z))/(2*d[i]))
    return np.array(approx_grad)

def poly_grad_descent(start, thresh, step_size, X, y, M):
    norms = []
    ws = []
    old_w = start
    new_w = old_w - step_size*poly_SSE_derivative(X, y, M, old_w)
    norms.append(np.linalg.norm(poly_SSE_derivative(X, y, M, old_w)))
    ws.append(old_w)
    while(abs(poly_SSE(X, y, M, new_w) - poly_SSE(X, y, M, old_w)) >= thresh and np.linalg.norm(poly_SSE_derivative(X, y, M, new_w)) >= thresh):
        # print("New, Old: ", new_w, old_w)
        old_w = new_w
        new_w = old_w - step_size*poly_SSE_derivative(X, y, M, old_w)
        norms.append(np.linalg.norm(poly_SSE_derivative(X, y, M, old_w)))
        ws.append(old_w)
    norms.append(np.linalg.norm(poly_SSE_derivative(X, y, M, new_w)))
    ws.append(new_w)
    return new_w, norms, ws

def poly_sgd(start, thresh, learning_rate, X, y, M):
    norms = []
    ws = []
    old_w = start
    new_w = old_w - learning_rate*poly_SSE_derivative([X[0]], y[0], M, old_w)
    norms.append(np.linalg.norm(poly_SSE_derivative([X[0]], y[0], M, old_w)))
    ws.append(old_w)
    i = 1
    while (abs(np.linalg.norm(old_w - new_w)) >= thresh and np.linalg.norm(poly_SSE_derivative([X[0]], y[0], M, new_w)) >= thresh):
        ind = i % len(X)
        old_w = new_w
        new_w = old_w - learning_rate*poly_SSE_derivative([X[ind]], y[ind], M, old_w)
        i += 1
    return new_w, norms, ws



M = 3
w = max_likelihood_weight_vector(X, y, M)
print max_likelihood_weight_vector(X, y, 3)
# print poly_SSE(X, y, M, w)
# plot_poly_regression(X, y, 10)
# print poly_SSE_derivative(X, y, M, 1.2*w)
# print approx_grad_poly(X, y, M, 1.2*w, [0.001]*len(w))
# print w
#
# print("SOL: ", w)
print("GD converged to: ", poly_grad_descent(1.2*w, 0.00000001, 0.01, X, y, M)[0])
print(poly_sgd(1.2*w, 0.000001, 0.001, X, y, M)[0])

def poly_cos_basis(M):
    return [(lambda y: (lambda x: math.cos(y*math.pi*x)))(i) for i in range(M + 1)]

def max_cos_likelihood_weight_vector(X, y, M):
    N = len(X)
    phi = np.array([[poly_cos_basis(M)[i](X[j]) for i in range(1, M+1)] for j in range(N)])
    phiT = T(phi)
    return dot(dot(inv(dot(phiT, phi)),phiT),y)

# M=8
# print(max_cos_likelihood_weight_vector(X, y, M))
#
# for M in [8]:
#     mlwv = max_cos_likelihood_weight_vector(X, y, M)
#
#     plt.plot(X, y, 'o')
#     plt.plot(X, [dot(mlwv, arr([poly_cos_basis(M)[i](X[j]) for i in range(1, M + 1)])) for j in range(len(X))], 'x')
#     plt.show()

def rr_max_likelihood_weight_vector(X, y, M, L):
    N = len(X)
    phi = np.array([[poly_basis(M)[i](X[j]) for i in range(M+1)] for j in range(N)])
    phiT = T(phi)
    return dot(dot(inv(L*np.identity(phiT.shape[0])+dot(phiT, phi)),phiT),y)

# M = 2
# L = .1
# print(rr_max_likelihood_weight_vector(X, y, M, L))
#
# for L in [0, 0.01, 0.1, 1, 10]:
#     for M in [2]:
#         mlwv = rr_max_likelihood_weight_vector(X, y, M, L)
#
#         plt.plot(X, y, 'o')
#         plt.plot(X, [dot(mlwv, arr([poly_basis(M)[i](X[j]) for i in range(M + 1)])) for j in range(len(X))], 'x')
#         plt.show()