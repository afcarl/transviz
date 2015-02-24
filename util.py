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



# def cumsum(v,strict=False):
#     if not strict:
#         return np.cumsum(v,axis=0)
#     else:
#         out = np.zeros_like(v)
#         out[1:] = np.cumsum(v[:-1],axis=0)
#         return out


# def rle(stateseq):
#     pos, = np.where(np.diff(stateseq) != 0)
#     pos = np.concatenate(([0],pos+1,[len(stateseq)]))
#     return stateseq[pos[:-1]], np.diff(pos)


# def slices_from_indicators(indseq):
#     indseq = np.asarray(indseq)
#     if not indseq.any():
#         return []
#     else:
#         vals, durs = rle(indseq)
#         starts, ends = cumsum(durs,strict=True), cumsum(durs,strict=False)
#         return [slice(start,end) for val,start,end in zip(vals,starts,ends) if val]


# def get_labelset(labels):
#     # NOTE: set difference doesn't work with nans
#     if isinstance(labels,np.ndarray):
#         return set(l for l in set(labels) if not np.isnan(l))
#     else:
#         return reduce(operator.or_, (get_labelset(l) for l in labels))


# def split_on_nans(seq):
#     return [seq[sl] for sl in slices_from_indicators(~np.isnan(seq))]


# def get_transmats(labels,counts=True,ignore_self=True):
#     if isinstance(labels,np.ndarray):
#         labels = [labels]

#     labelset = get_labelset(labels)
#     N = max(labelset)+1

#     mats = [sum(count_transitions(seq,N)
#             for seq in split_on_nans(l)) for l in labels]

#     if ignore_self:
#         mats = [m - np.diag(np.diag(m)) for m in mats]

#     if not counts:
#         def normalize(m):
#             norms = m.sum(1)
#             return m / np.where(norms > 0, norms, 1.)
#         mats = [normalize(m) for m in mats]

#     if len(mats) > 1:
#         return mats
#     else:
#         return mats[0]


##############
#  matrices  #
##############


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

