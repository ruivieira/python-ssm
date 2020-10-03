"""Utility methods"""
from math import exp
import numpy as np


def ilogit(x):
    """Return the inverse logit"""
    return exp(x) / (1.0 + exp(x))


def block_diag(*arrays):
    """Create a block diagonal matrix from a set of matrices"""
    if arrays == ():
        arrays = ([],)
    arrays = [np.atleast_2d(a) for a in arrays]

    bad_args = [k for k in range(len(arrays)) if arrays[k].ndim > 2]
    if bad_args:
        raise ValueError(
            "arguments in the following positions have dimension "
            "greater than 2: %s" % bad_args
        )

    shapes = np.array([a.shape for a in arrays])
    out_dtype = np.find_common_type([arr.dtype for arr in arrays], [])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r : r + rr, c : c + cc] = arrays[i]
        r += rr
        c += cc
    return out
