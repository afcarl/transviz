from __future__ import division
import numpy as np
import inspect
from cStringIO import StringIO
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt


##########
#  misc  #
##########


def num_args(func):
    return len(inspect.getargspec(func).args)


###############
#  sequences  #
###############


def relabel_by_usage(seq):
    good = ~np.isnan(seq)
    usages = np.bincount(seq[good].astype('int32'))
    perm = np.argsort(np.argsort(usages)[::-1])

    out = np.empty_like(seq)
    out[good] = perm[seq[good].astype('int32')]
    if np.isnan(seq).any():
        out[~good] = np.nan

    return out


def count_transitions(labels,N=None,ignore_self=True):
    labelset = np.unique(labels)
    labelset = labelset[~np.isnan(labelset)]
    N = int(max(labelset))+1 if not N else N

    out = np.zeros((N,N))
    for i,j in zip(labels[:-1],labels[1:]):
        if i == i and j == j:
            out[i,j] += 1

    if ignore_self:
        out -= np.diag(np.diag(out))

    return out


def get_transmats(labelss, N):
    usages = sum(np.bincount(l[~np.isnan(l)].astype('int32'),minlength=N)
                 for l in labelss)
    perm = np.argsort(usages)[::-1]
    n = (usages > 0).sum()
    return [permute_matrix(count_transitions(l,N),perm)[:n,:n]
            for l in labelss]


##############
#  matrices  #
##############


def normalize_transmat(A):
    norm = A.sum(1)
    return A / np.where(norm,norm,1.)[:,None]


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


def show_cmap(cmap):
    plt.imshow(np.tile(np.linspace(0,1,256),(10,1)),cmap=cmap,aspect='equal')
    plt.grid('off')
    plt.yticks([])

