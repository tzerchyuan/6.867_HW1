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

def do_grad_gauss(start, thresh, step_size, plotNorm = False):
    # start = [5,5]
    # thresh = .000001
    # step_size = .25
    end, norms, ws = gauss_grad_descent(gauss, gauss_derivative, start, thresh, step_size)
    print("GD Gauss: ", end)
    if plotNorm:
        plt.plot(norms)
        plt.xlabel("Iterations")
        plt.ylabel("Norm of Gradient")
        plt.title("Evolution of Gaussian Gradient Norm (S = " + str(start) + ")")
        plt.show()
        # plt.plot(ws)
        # plt.show()
    return end, norms, ws

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

# delta = 0.001
# for i in np.arange(9,11.25,0.25):
#     for j in np.arange(9,11.25,0.25):
#         point = np.array([i,j])
#         print("APPROX: ", approx_grad_gauss(gauss, point, np.array([delta]*len(point))))
#         print("ACTUAL: ", gauss_derivative(gauss, mean, cov, np.array([i,j])))
#         print("------------------------------------")



# print()

#
# l1 = []
# l2 = []
# lD = []
# l0 = []
# for i in np.arange(80/3-5,80/3+8.1,0.1):
#     v = np.array([i,i])
#     l1.append(v)
#     l2.append(bowl(A,b, v))
#     l0.append(0)
#     lD.append(bowl_derivative( A,b, v))
#
# # plt.plot(l1, l2, "bo")
# plt.plot(l1, lD, "r^")
# plt.plot(l1, l0, "g-")
# plt.show()
#
#

#

def gen_gauss_plots():
    # start = [5,5]
    # thresh = .000001
    # step_size = .25
    #
    # end, norms, ws = do_grad_gauss(start, thresh, step_size, False)

    ends_vary_start = []
    ends_vary_step = []
    ends_vary_thresh = []

    for i,j in [[k]*2 for k in range(4, 10)]:
        start = [i,j]
        thresh = .0001
        step_size = .25
        end, norms, ws = do_grad_gauss(start, thresh, step_size, False)
        ends_vary_start.append((end, np.linalg.norm([10-i,10-j])))

    ends_only_start_dist_end = [(10-ends_vary_start[i][0][0])*1.414 for i in range(len(ends_vary_start))]
    ends_only_start_dist_start = [ends_vary_start[i][1] for i in range(len(ends_vary_start))]
    for i in range(len(ends_only_start_dist_start)):
        print("-----")
        print(ends_only_start_dist_start)
        print("-----")
        print(ends_only_start_dist_end)
        print("-----")
        plt.plot(ends_only_start_dist_start[i], ends_only_start_dist_end[i], ls = "None", color = str((i+1.0)/len(ends_only_start_dist_start)), marker = "o")
    plt.xlabel("Initial Distance from Minimum")
    plt.ylabel("End Distance from Minimum")
    plt.title("Distance from Minimum Over Starting Distances " + "(thresh = " + str(thresh) + ", step size = " + str(step_size) + ")")
    plt.show()

    for thresh in [.01, .03, .001, .003, .0001, .003, .00001]:
        start = [7,7]
        # thresh = .0001
        step_size = .25

        end, norms, ws = do_grad_gauss(start, thresh, step_size, False)
        ends_vary_thresh.append((end, -math.log(thresh)))

    ends_only_thresh_dist_end = [(10-ends_vary_thresh[i][0][0])*1.414 for i in range(len(ends_vary_thresh))]
    ends_only_thresh_val = [ends_vary_thresh[i][1] for i in range(len(ends_vary_thresh))]
    for i in range(len(ends_only_thresh_dist_end)):
        plt.plot(ends_only_thresh_val[i], ends_only_thresh_dist_end[i], ls = "None", color = str((i+1.0)/len(ends_only_thresh_val)), marker = "o")
    plt.xlabel("-log(threshold)")
    plt.ylabel("End Distance from Minimum")
    plt.title("Distance from Minimum Over Thresholds " + "(step size = " + str(step_size) + ", start= " + str(start) + ")")
    plt.show()

    for step_size in [1, .25, .1, .05, .025, .01]:
        start = [7,7]
        thresh = .0001
        # step_size = .25

        end, norms, ws = do_grad_gauss(start, thresh, step_size, False)
        ends_vary_step.append((end, step_size))

    ends_only_step_dist_end = [(10-ends_vary_step[i][0][0])*1.414 for i in range(len(ends_vary_step))]
    ends_only_step_val = [ends_vary_step[i][1] for i in range(len(ends_vary_step))]
    for i in range(len(ends_only_step_dist_end)):
        plt.plot(ends_only_step_val[i], ends_only_step_dist_end[i], ls = "None", color = str((i+1.0)/len(ends_only_step_val)), marker = "o")
    plt.xlabel("Step Size")
    plt.ylabel("End Distance from Minimum")
    plt.title("Distance from Minimum Over Step Sizes " + "(thresh = " + str(thresh) + ", start= " + str(start) + ")")
    plt.show()

    # ends_only_start_x = [ends_vary_start[i][0][0] for i in range(len(ends_vary_start))]
    # ends_only_start_y = [ends_vary_start[i][0][1] for i in range(len(ends_vary_start))]
    # for i in range(len(ends_only_start_x)):
    #     plt.plot(ends_only_start_x[i], ends_only_start_y[i], ls = "None", color = str((i+1.0)/len(ends_only_start_x)), marker = "o")
    # plt.xlabel("End x Value")
    # plt.ylabel("End y Value")
    # plt.title("Ending Point Based on Varying Start (darker = closer)")
    # plt.show()

    # ends_only_start_dist_end = [(10-ends_vary_start[i][0][0])*1.414 for i in range(len(ends_vary_start))]
    # ends_only_start_dist_start = [ends_vary_start[i][1] for i in range(len(ends_vary_start))]
    # for i in range(len(ends_only_start_dist_start)):
    #     print("-----")
    #     print(ends_only_start_dist_start)
    #     print("-----")
    #     print(ends_only_start_dist_end)
    #     print("-----")
    #     plt.plot(ends_only_start_dist_start[i], ends_only_start_dist_end[i], ls = "None", color = str((i+1.0)/len(ends_only_start_dist_start)), marker = "o")
    # plt.xlabel("Initial Distance from Minimum")
    # plt.ylabel("End Distance from Minimum")
    # plt.title("Distance from Minimum Over Starting Distances " + "(thresh = " + str(thresh) + ", step size = " + str(step_size) + ")")
    # plt.show()


# gen_gauss_plots()

# print(gauss(mean, cov, [0,0]))
# print('derivative: ', gauss_derivative(gauss, mean, cov, [10,10]))
# print("MEAN: ", mean)
#
# # Plot the gaussian curve and its derivative
# l1 = []
# l2 = []
# lD = []
# l0 = []
# lw = []
# for i in np.arange(5,15.25,0.25):
#     v = np.array([i,i])
#     l1.append(v)
#     l2.append(gauss(mean, cov, v))
#     l0.append(0)
#     lD.append(gauss_derivative(gauss, mean, cov, v))
#
# plt.plot(l1, l2, "bo")
# plt.plot(l1, lD, "r^")
# plt.plot(l1, l0, "g-")
# plt.show()
