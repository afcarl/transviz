from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import transviz as tv

np.random.seed(0)
algs = ['dot','circo','circular','neato','sfdp','fruchterman_reingold','spring']
nmax = 20

A = np.load('trans-mats-for-matts.npz')['tmt_transmat']

colors = [(0.0, 0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0, 1.0)]
cmap = LinearSegmentedColormap.from_list('blue', colors)

for alg in algs:
    G = tv.TransGraph(A,nmax=nmax)
    G.edge_attrs(
        lambda i,j,v: {
            'color':cmap(v),
            'penwidth':1+4*v})\
     .layout(alg)\
     .draw(outfile='{alg}.{nmax}.png'.format(alg=alg,nmax=nmax))

