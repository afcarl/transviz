from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import transviz as tv

np.random.seed(0)
f = np.load('trans-mats-for-matts.npz')

from matplotlib.colors import LinearSegmentedColormap
colors = [(0.0, 0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0, 1.0)]
cmap = LinearSegmentedColormap.from_list('blue', colors)

g = tv.TransGraph(f['tmt_transmat'],nmax=20)
g.edge_attrs(
    lambda i,j,v: {
        'color':cmap(v),
        'penwidth':1+4*v})\
 .layout('circo').draw()

# dot, circo, circular, neato, sfdp, fruchterman_reingold, spring

plt.show()
