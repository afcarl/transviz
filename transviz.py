from __future__ import division
import numpy as np
import networkx as nx
from collections import defaultdict

from util import rgb2hexa, permute_by_usage, permute_matrix, \
    pad_zeros, num_args, get_agraph_pngstr, get_usages

# TODO take count matrices?

# TODO add highlighting of nodes/neighborhoods
# TODO add igraph kk layout
# TODO circo bend through middle?

# default graphviz attributes


graphdefaults = dict(
    dpi='72',
    outputorder='edgesfirst',
)

nodedefaults = dict(
    shape='circle',
    fillcolor='white',
    style='filled',
    fixedsize='true',
)

edgedefaults = dict()

# default arguments to graphviz layout routines

graphviz_layouts = {
    'twopi':{},
    'gvcolor':{},
    'wc':{},
    'ccomps':{},
    'tred':{},
    'sccmap':{},
    'fdp':{},
    'circo':{},
    'neato':{'overlap':'false','sep':'+8'},
    'acyclic':{},
    'nop':{},
    'gvpr':{},
    'dot':{},
    'sfdp':{},
}

# default arguments to networkx layout routines

networkx_layouts = {
    'circular':{'scale':120},
    'shell':{'scale':120},
    'spring':{'scale':120},
    'spectral':{'scale':250},
    'fruchterman_reingold':{'scale':120},
}


# converters from my attribute formats to graphviz formats

converters = defaultdict(
    lambda: str,
    {
        'pos': lambda xy: '%f,%f!' % xy,
        'color': lambda rgba: rgb2hexa(rgba),
        'weight': lambda x: x,
    }
)


def convert(dct):
    try:
        return {attr:converters[attr](val) for attr, val in dct.items()}
    except:
        return dct


class TransGraph(nx.DiGraph):
    def __init__(self,A,relabel=None,normalize=True,nmax=None):
        # preprocess the trans matrix/matrices
        if relabel == 'usage':
            A, self.perm = permute_by_usage(A,return_perm=True)
        A = A[:nmax,:nmax]
        self.usages = get_usages(A)
        if normalize:
            A = A / A.max()
        self.A = A

        # initialize as a nx.DiGraph
        super(TransGraph,self).__init__(A)

        # set defaults
        self.graph['graph'] = graphdefaults
        self.graph['node'] = nodedefaults
        self.graph['edge'] = edgedefaults

    def graph_attrs(self,**kwargs):
        self.graph['graph'].update(convert(kwargs))
        return self

    def node_attrs(self,func):
        nargs = num_args(func)

        if nargs == 1:
            for i, node in self.nodes_iter(data=True):
                node.update(convert(func(i)))
        elif nargs == 2:
            for i, node in self.nodes_iter(data=True):
                node.update(convert(func(i,self.usages[i])))
        else:
            raise ValueError('func must take 1 or 2 arguments')

        return self

    def edge_attrs(self,func):
        nargs = num_args(func)

        if nargs == 1:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func((i,j))))
        elif nargs == 2:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func(i,j)))
        elif nargs == 3:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func(i,j,self.A[i,j])))
        else:
            raise ValueError('func must take 1, 2, or 3 arguments')

        return self

    def layout(self,algname,**kwargs):
        if algname in graphviz_layouts:
            self.graph['graph'].update(dict(graphviz_layouts[algname],**kwargs))
            posdict = nx.graphviz_layout(self,algname)
        elif algname in networkx_layouts:
            func = nx.__dict__[algname+'_layout']
            kwargs = dict(networkx_layouts[algname],**kwargs)
            kwargs['scale'] *= np.sqrt(self.order())
            posdict = func(self,**kwargs)
        else:
            raise ValueError(
                'algname must be one of %s' %
                (graphviz_layouts.keys() + networkx_layouts.keys()))

        nx.set_node_attributes(
            self,'pos',
            {k:('%f,%f!' % tuple(v)) for k,v in posdict.items()})

        self.has_layout = True

        return self

    def draw(self,outfile=None,matplotlib=True,notebook=False):
        agraph = nx.to_agraph(self)
        agraph.has_layout = self.has_layout

        if outfile is None:
            pngstr = get_agraph_pngstr(agraph)

            if matplotlib:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                plt.figure()
                plt.imshow(mpimg.imread(pngstr),aspect='equal')
                plt.axis('off')

            if notebook:
                from IPython.display import Image, display
                display(Image(data=pngstr))
        else:
            agraph.draw(outfile)


class TransDiff(TransGraph):
    def __init__(self,(A,B),normalize=True,nmax=None):
        raise NotImplementedError  # TODO

    def edge_attrs(self,func):
        nargs = num_args(func)

        if nargs == 1:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func((i,j))))
        elif nargs == 2:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func(i,j)))
        elif nargs == 4:
            for i, j, edge in self.edges_iter(data=True):
                edge.update(convert(func(i,j,self.A[i,j],self.B[i,j])))
        else:
            raise ValueError('func must take 1, 2, or 4 arguments')

        return self

    def node_attrs(self,func):
        nargs = num_args(func)

        if nargs == 1:
            for i, node in self.nodes_iter(data=True):
                node.update(convert(func(i)))
        elif nargs == 3:
            for i, node in self.nodes_iter(data=True):
                node.update(convert(func(i,self.usages[i],self.B_usages[i])))
        else:
            raise ValueError('func must take 1 or 3 arguments')

        return self

