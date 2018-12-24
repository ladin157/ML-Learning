"""
Link here: http://greenteapress.com/thinkstats/correlation.py
"""

import math
import random
from statistics import thinkstats


def cov(xs, ys, mux=None, muy=None):
    """
    Compute Cov(X,Y)
    :param xs:
    :param ys:
    :param mux:
    :param muy:
    :return:
    """
    if mux is None:
        mux = thinkstats.Mean(xs)
    if muy is None:
        muy = thinkstats.Mean(ys)
    total = 0.0
    for x, y in zip(xs, ys):
        total += (x - mux) * (y - muy)
    return total / len(xs)


def corr(xs, ys):
    """
    Compute Corr(X, Y)
    :param xs:
    :param ys:
    :return:
    """
    xbar, varx = thinkstats.MeanVar(xs)
    ybar, vary = thinkstats.MeanVar(ys)
    corr = cov(xs, ys, xbar, ybar) / math.sqrt(varx * vary)

    return corr

def serial_corr(xs):
    """
    Computes the serial correlation of a sequence
    :param xs:
    :return:
    """
    return corr(xs[:-1], xs[1:])

