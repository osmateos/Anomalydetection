

import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy
from pylab import *



def hp_filter(x, lamb=5000):
    w = len(x)
    b = [[1]*w, [-2]*w, [1]*w]
    D = sparse.spdiags(b, [0, 1, 2], w-2, w)
    I = sparse.eye(w)
    B = (I + lamb*(D.transpose()*D))
    return sparse.linalg.dsolve.spsolve(B, x)


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def AnomalyDetection(x, alpha=0.2, lamb=5000):

    xhat = hp_filter(x, lamb=lamb)
    resid = x - xhat
    ds = pd.Series(resid)
    ds = ds.dropna()
    md = np.median(x)
    data = ds - md
    ares = (data - data.median()).abs()
    data_sigma = data.mad() + 1e-12
    ares = ares/data_sigma
    p = 1. - alpha
    R = stats.expon.interval(p, loc=ares.mean(), scale=ares.std())
    threshold = R[1]
    r_id = ares.index[ares > threshold]

    return r_id



def run(your_list, alpha_p=0.2):
 

    s = [float(x[1]) for x in your_list]
    t_date = [x[0] for x in your_list]

    t = arange(0.0, len(s), 1)
    t2_arr= numpy.array(s)
    total=[]
    r_idx = AnomalyDetection(t2_arr, alpha=alpha_p) 
    for k in r_idx.values:
        total= total + your_list[k]

    t2 = [float(x[1]) for x in your_list]
    t_date = [x[0] for x in your_list]
    t = arange(0.0, len(t2), 1)

    s = t2
    markers_on = r_idx.values.tolist()
    plot(t,s, markevery=markers_on,marker='*', color='b',markeredgecolor='r')
    grid(True)
    plt.show()
    return (markers_on)

import csv
reader = csv.reader(open('TextFile1.txt', 'r'))
your_list = list(reader)
points =run(your_list, alpha_p=0.04)
list_result = [your_list[i] for i in points]
print(list_result)

