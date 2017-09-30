import pdb
import random
import pylab as pl
from sklearn import linear_model
import math
import numpy as np
import matplotlib.pyplot as plt

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def getTrueW(name):
    return pl.loadtxt(name)

def arr(x):
    return np.array(x)
def inv(x):
    return np.linalg.inv(x)
def dot(x,y):
    return np.dot(x,y)
def T(x):
    return np.transpose(x)

def lasso_basis(M):
    basis = [(lambda y: (lambda x: math.sin(0.4*math.pi*y*x)))(i) for i in range(1, M)]
    basis.insert(0, lambda x: x)
    return basis

def lasso_SSE(X, Y, M, w):
    # print(X)
    # print("------------------------")
    # print(Y)
    # print("------------------------")
    # print(M)
    # print("------------------------")
    # print(w)
    # N = len(X)

    # s = [(Y[i] - dot(T(w), [lasso_basis(M)[j](X[i]) for j in range(M)])) ** 2 for i in range(N)]
    # print("X0!!:", X[0])
    # s = [lasso_basis(M)[j](X[0]) for j in range(M)]
    # print(s)

    return sum([(Y[i] - dot(T(w), [lasso_basis(M)[j](X[i]) for j in range(M)]))**2 for i in range(N)])

# def ridge_SSE(X, y, M, w):
#     N = len(X)
#     return sum([(y[i] - dot(T(w), [lasso_basis(M)[j](X[i]) for j in range(M+1)]))**2 for i in range(N)])

def rr_max_likelihood_weight_vector(X, y, M, L):
    N = len(X)
    phi = np.array([[lasso_basis(M)[i](X[j]) for i in range(M)] for j in range(N)])
    phiT = T(phi)
    return dot(dot(inv(L*np.identity(phiT.shape[0])+dot(phiT, phi)),phiT),y)

trueW = getTrueW("lasso_true_w.txt")
trainX, trainY, = lassoTrainData()
validateX, validateY, = lassoValData()
testX, testY = lassoTestData()

# print(trainX)
# print("--------------------------------")
# print(trainY)
# print("--------------------------------")
# print("--------------------------------")
trainX = np.array([i[0] for i in trainX])
validateX = np.array([i[0] for i in validateX])
testX = np.array([i[0] for i in testX])
trainY = np.array([i[0] for i in trainY])
validateY = np.array([i[0] for i in validateY])
testY = np.array([i[0] for i in testY])

# print(validateX)


#
# print(trainX)
# print("--------------------------------")
# print(trainY)

# given
M = 13

results = []
N = len(trainX)
N2 = len(validateX)
N3 = len(testX)
phi_trainX = np.array([[lasso_basis(M)[i](trainX[j]) for i in range(M)] for j in range(N)])
phi_validateX = np.array([[lasso_basis(M)[i](validateX[j]) for i in range(M)] for j in range(N2)])

for L in [.01, .025, .05, .1, .25, .5, .75, 1, 1.5, 2, 5]:
    clf = linear_model.Lasso(alpha=L, fit_intercept=False)
    clf.fit(phi_trainX, trainY)
    sse = lasso_SSE(validateX, validateY, M, clf.coef_)

    results.append((L, sse))

print("Best lasso validation results: (L, SSE) = " + str(sorted(results, key=lambda x: x[1])[0]))
# print(sorted(results, key = lambda x: x[1]))

L_test = sorted(results, key = lambda x: x[1])[0][0]
clf = linear_model.Lasso(alpha=L_test, fit_intercept=False)

N = len(trainX)
phi_trainX = np.array([[lasso_basis(M)[i](trainX[j]) for i in range(M)] for j in range(N)])
clf.fit(phi_trainX, trainY)
w_test = clf.coef_
sse_test = lasso_SSE(testX, testY, M, w_test)

print("Test SSE: " + str(sse_test))

"""#######################################################################################"""

results_ridge = []

for L in [.01, .025, .05, .1, .25, .5, .75, 1, 1.5, 2, 5]:
    # train using train data
    w = rr_max_likelihood_weight_vector(trainX, trainY, M, L)
    # find error on validation set
    sse = lasso_SSE(validateX, validateY, M, w)
    results_ridge.append((L, sse))

print("Best ridge validation results: (L, SSE) = " + str(sorted(results_ridge, key = lambda x: x[1])[0]))

# print("ALL RESULTS: ", sorted(results, key = lambda x: x[2]))

### test on test data now

L_test_b = sorted(results_ridge, key = lambda x: x[1])[0][0]
w_test_b = rr_max_likelihood_weight_vector(trainX, trainY, M, L_test_b)
sse_test_b = lasso_SSE(testX, testY, M, w_test_b)

print("Test B SSE: " + str(sse_test_b))


plt.bar([i for i in range(1,M+1)],trueW+[.000001]*M)
plt.xlabel("Parameter Number")
plt.ylabel("Parameter Value")
plt.title("True Parameter Values")
plt.show()

plt.bar([i for i in range(1,M+1)],w_test+[.000001]*M)
plt.xlabel("Parameter Number")
plt.ylabel("Parameter Value")
plt.title("LASSO Parameter Values")
plt.show()

plt.bar([i for i in range(1,M+1)],w_test_b+[.000001]*M)
plt.xlabel("Parameter Number")
plt.ylabel("Parameter Value")
plt.title("Ridge Regression Parameter Values")
plt.show()


l_strs = [math.log(i) for i in [.01, .025, .05, .1, .25, .5, .75, 1, 1.5, 2, 5]]
results_sses = [results[i][1] for i in range(len(results))]
results_ridge_sses = [results_ridge[i][1] for i in range(len(results_ridge))]

plt.bar(l_strs,results_sses)
plt.xlabel("log(Lambda)")
plt.ylabel("LASSO SSE")
plt.title("Lasso SSE vs log(Lambda)")
plt.show()

plt.bar(l_strs,results_ridge_sses)
plt.xlabel("log(Lambda)")
plt.ylabel("Ridge SSE")
plt.title("Ridge SSE vs log(Lambda)")
plt.show()