from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import transviz as tv

np.random.seed(0)
# algs = ['dot','circo','circular','neato','sfdp','fruchterman_reingold',
#         'spring','spectral']
# algs = ['circo','fruchterman_reingold','spectral']
alg = 'circo'
nmax = 20

f = np.load('trans-mats-for-matts.npz')
A = f['tmt_transmat']
B = f['blank_transmat']

colors = [(0.0, (1.0, 0.0, 0.0, 1.0)),
          (0.4, (1.0, 0.0, 0.0, 1.0)),
          (0.5, (1.0, 1.0, 1.0, 0.0)),
          (0.6, (0.0, 0.0, 1.0, 1.0)),
          (1.0, (0.0, 0.0, 1.0, 1.0))]
cmap = LinearSegmentedColormap.from_list('myrwb', colors)

# cmap = plt.get_cmap('bwr')

G = tv.TransDiff((A,B),nmax=nmax,normalize=False)

G.edge_attrs(
    lambda i, j, aij, bij: {
        'color':cmap(0.5+0.5*(bij-aij)),
        'penwidth':3+2*(bij-aij),
        'weight':aij})\
 .node_attrs(
    lambda i, ai, bi: {'width': 0.5 + 0.5*ai})\
 .layout(alg)\
 .draw()

plt.show()

# TODO usage diff: maybe red/blue, color+modulate thickness of edge or
# color+modulate opacity of background
# TODO no label?
# TODO more extreme sizes
# TODO histogram aij and bij, ai and bi, probably want to rescale those to (0,1)

