__author__ = 'amilkov3'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(x):
    y = (x - 1.5)**2 + .5
    print "x = {}, y = {}".format(x, y)
    return y

def error(line, data):

    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err

def test_minima_run():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print "Minima found at:"
    print "x = {}, y = {}".format(min_result.x, min_result.fun)

    Xplot = np.linspace(0.5, 2.5, 21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title('Minima of an objective function')
    plt.show()

def fit_line(data, error_func):
    l = np.float32([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends * l[1], 'm--', linewidth=2.0)

    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp': True})
    return result.x

def test_error_run():
    l_orig = np.float32([4,2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig,  'b ', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.array([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label='Data points')

    l_fit = fit_line(data, error)
    print "Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1])
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0)
if __name__ == "__main__":
    test_run()
