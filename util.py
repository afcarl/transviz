from __future__ import division
import numpy as np
import inspect
from cStringIO import StringIO
from matplotlib.colors import rgb2hex


def num_args(func):
    return len(inspect.getargspec(func).args)


def get_transmat(labels,N=None,counts=False):
    labels = labels[~np.isnan(labels)]
    N = np.max(labels)+1 if not N else N
    out = np.zeros((N,N))
    for i, j in zip(labels[:-1],labels[1:]):
        out[i,j] += 1
    if counts:
        return out
    else:
        return out / out.sum(1)[:,None]


def pad_zeros(mat,shape):
    out = np.zeros(shape,dtype=mat.dtype)
    m, n = mat.shape
    out[:m,:n] = mat
    return out


def permute_matrix(A,perm=None):
    if perm is None:
        perm = np.random.permutation(A.shape[0])
    return A[np.ix_(perm,perm)]


def get_usages(A):
    out = np.maximum(A.sum(0),A.sum(1))
    return out / out.sum()


def get_usage_order(A):
    return np.argsort(get_usages(A))[::-1]


def permute_by_usage(A,return_perm=False):
    perm = get_usage_order(A)
    if return_perm:
        return permute_matrix(A,perm), perm
    else:
        return permute_matrix(A,perm)


def get_agraph_pngstr(agraph):
    sio = StringIO()
    agraph.draw(sio,format='png')
    sio.seek(0)
    return sio


def rgb2hexa(rgb):
    return rgb2hex(rgb[:-1]) + ('%02x' % int(255.*rgb[-1]))

