from __future__ import division
import numpy as np
import inspect
from cStringIO import StringIO
from matplotlib.colors import rgb2hex


def num_args(func):
    return len(inspect.getargspec(func).args)


def permute_matrix(A,perm=None):
    if perm is None:
        perm = np.random.permutation(A.shape[0])
    return A[np.ix_(perm,perm)]


def get_usages(A):
    return np.maximum(A.sum(0),A.sum(1))


def get_usage_order(A):
    return np.argsort(get_usages(A))[::-1]


def permute_by_usage(A):
    return permute_matrix(A,get_usage_order(A))


def get_agraph_pngstr(agraph):
    sio = StringIO()
    agraph.draw(sio,format='png')
    sio.seek(0)
    return sio


def rgb2hexa(rgb):
    return rgb2hex(rgb[:-1]) + ('%02x' % int(255.*rgb[-1]))

