from __future__ import division
import numpy as np
import inspect
import operator
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

from pyhsmm.util.general import rle, cumsum
from pyhsmm.util.cstats import count_transitions as _count_transitions

##########
#  misc  #
##########


def num_args(func):
    return len(inspect.getargspec(func).args)


###############
#  sequences  #
###############


def get_labelset(labelss):
    if isinstance(labelss,np.ndarray):
        labelset = np.unique(labelss)
        return set(labelset[~np.isnan(labelset)])
    else:
        return reduce(operator.or_,(get_labelset(l) for l in labelss))


def get_N(labelss):
    return int(max(get_labelset(labelss)))+1


def relabel_by_usage(labelss, N=None):
    if isinstance(labelss,np.ndarray):
        backwards_compat = True
        labelss = [labelss]
    else:
        backwards_compat = False

    N = get_N(labelss) if not N else N
    usages = sum(np.bincount(l[~np.isnan(l)].astype('int32'),minlength=N)
                 for l in labelss)
    perm = np.argsort(np.argsort(usages)[::-1])

    outs = []
    for l in labelss:
        out = np.empty_like(l)
        good = ~np.isnan(l)
        out[good] = perm[l[good].astype('int32')]
        if np.isnan(l).any():
            out[~good] = np.nan
        outs.append(out)

    return outs if not backwards_compat else outs[0]


def count_transitions(labels,N=None,ignore_self=True):
    N = get_N(labels) if not N else N
    out = sum(_count_transitions(l.astype('int32'),N) for l in split_on_nans(labels))

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


def split_on_nans(seq):
    return [seq[sl] for sl in slices_from_indicators(~np.isnan(seq))]


def slices_from_indicators(indseq):
    indseq = np.asarray(indseq)
    if not indseq.any():
        return []
    else:
        vals, durs = rle(indseq)
        starts, ends = cumsum(durs,strict=True), cumsum(durs,strict=False)
        return [slice(start,end)
                for val,start,end in zip(vals,starts,ends) if val]


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


def topk_per_row(A,k):
    return np.array([np.where(row >= sorted(row)[-k],row,0.) for row in A])


#############
#  drawing  #
#############


def rgb2hexa(rgb):
    return rgb2hex(rgb[:-1]) + ('%02x' % int(255.*rgb[-1]))


def show_cmap(cmap):
    plt.imshow(np.tile(np.linspace(0,1,256),(10,1)),cmap=cmap,aspect='equal')
    plt.grid('off')
    plt.yticks([])


