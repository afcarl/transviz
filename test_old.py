from __future__ import division
import numpy as np
import transviz as tv

np.random.seed(0)
f = np.load('trans-mats-for-matts.npz')

m = tv.TransMat(f['tmt_transmat'])

m.matshow(nmax=60)

m.network(nmax=40,layout='dot')
m.network(nmax=40,layout='circo')
m.network(nmax=40,layout='neato')
m.network(nmax=40,layout='sfdp')
m.network(nmax=40,layout='fruchterman_reingold')
m.network(nmax=40,layout='spring')
m.network(nmax=40,layout='circular')


