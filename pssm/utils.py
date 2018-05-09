from math import exp


def ilogit(x):
    return exp(x) / (1.0 + exp(x))
