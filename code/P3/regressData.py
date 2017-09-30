import pdb
import random
import numpy as np
import pylab as pl

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

AX, Ay = regressAData()
BX, By = regressBData()
VX, Vy = validateData()

# print(VX)
# AX = AX[0]
# Ay = Ay[0]
# BX = BX[0]
# By = By[0]
# VX = VX[0]
# Vy = Vy[0]
# print(VX)

AX = np.array([i[0] for i in AX])
BX = np.array([i[0] for i in BX])
VX = np.array([i[0] for i in VX])
Ay = np.array([i[0] for i in Ay])
By = np.array([i[0] for i in By])
Vy = np.array([i[0] for i in Vy])

# print(VX)

def arr(x):
    return np.array(x)
def inv(x):
    return np.linalg.inv(x)
def dot(x,y):
    return np.dot(x,y)
def T(x):
    return np.transpose(x)

def poly_basis(M):
    return [(lambda y: (lambda x: x**y))(i) for i in range(M+1)]

def poly_SSE(X, y, M, w):
    N = len(X)
    # print("X0!!!!: ", X[0])
    return sum([(y[i] - dot(T(w), [poly_basis(M)[j](X[i]) for j in range(M+1)]))**2 for i in range(N)])

def rr_max_likelihood_weight_vector(X, y, M, L):
    N = len(X)
    phi = np.array([[poly_basis(M)[i](X[j]) for i in range(M+1)] for j in range(N)])
    phiT = T(phi)
    return dot(dot(inv(L*np.identity(phiT.shape[0])+dot(phiT, phi)),phiT),y)


# A train, B test

results = []

for L in [0, 0.001, 0.01, 0.1, 1, 10]:
    for M in [1, 2, 4, 8]:
        # train using A
        w = rr_max_likelihood_weight_vector(AX, Ay, M, L)
        # find error on validation set
        sse = poly_SSE(VX, Vy, M, w)
        results.append((M, L, sse))

print("Best A validation results: (M, L, SSE) = " + str(sorted(results, key = lambda x: x[2])[0]))

# print("ALL RESULTS: ", sorted(results, key = lambda x: x[2]))

### test on B now

M_test_b, L_test_b = sorted(results, key = lambda x: x[2])[0][0], sorted(results, key = lambda x: x[2])[0][1]
w_test_b = rr_max_likelihood_weight_vector(AX, Ay, M_test_b, L_test_b)
sse_test_b = poly_SSE(BX, By, M_test_b, w_test_b)

print("Test B SSE: " + str(sse_test_b))

####################################################################################

# B train, A test

results2 = []

for L in [0, 0.001, 0.01, 0.1, 1, 10]:
    for M in [1, 2, 4, 8]:
        # train using B
        w = rr_max_likelihood_weight_vector(BX, By, M, L)
        # find error on validation set
        sse = poly_SSE(VX, Vy, M, w)
        results2.append((M, L, sse))

print("Best B validation results: (M, L, SSE) = " + str(sorted(results2, key = lambda x: x[2])[0]))

print("ALL RESULTS: ", sorted(results, key = lambda x: x[2]))

### test on A now

M_test_a, L_test_a = sorted(results2, key = lambda x: x[2])[0][0], sorted(results2, key = lambda x: x[2])[0][1]
w_test_a = rr_max_likelihood_weight_vector(BX, By, M_test_a, L_test_a)
sse_test_a = poly_SSE(AX, Ay, M_test_a, w_test_a)

print("Test A SSE: " + str(sse_test_a))