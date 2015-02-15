from __future__ import division
import numpy as np
import networkx as nx
from cStringIO import StringIO
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

# TODO more network layout options
# TODO maybe make it hang on to layout, since that might take time to generate.
# then we can iterate faster.
# TODO a little weird to use graphviz_layout on a networkx object, since that
# converts to a pygraphviz agraph and then converts back. maybe i should only
# use pygraphviz and not networkx at all, but networkx is better documented.

#########
#  viz  #
#########


nx_graph_opts = dict(
    graph=dict(dpi='300',outputorder='edgesfirst'),
    node=dict(shape='circle',fillcolor='white',style='filled'),
    edge=dict(penwidth='1.5',arrowsize='1',arrowhead='normal'),
)

cmap = plt.get_cmap('Blues')


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

networkx_layouts = {
    'circular':{},
    'shell':{},
    'spring':{},
    'spectral':{},
    'fruchterman_reingold':{},
}


class DiGraph(nx.DiGraph):
    def layout(self,algstr):
        if algstr in graphviz_layouts:
            self.graph['graph'].update(graphviz_layouts[algstr])
            posdict = nx.graphviz_layout(self,algstr)
            nx.set_node_attributes(
                self,'pos',
                {k:('%f,%f!' % v) for k,v in posdict.items()})
        elif algstr in networkx_layouts:
            func = nx.__dict__[algstr+'_layout']
            posdict = func(self,scale=120*np.sqrt(self.order()),
                           **networkx_layouts[algstr])
            nx.set_node_attributes(
                self,'pos',
                {k:('%f,%f!' % tuple(v)) for k,v in posdict.items()})
        else:
            raise ValueError(
                'algstr must be one of %s' %
                (graphviz_layouts.keys() + networkx_layouts.keys()))

        return self

    def draw(self):
        return draw_graph(self)


class TransMat(object):
    def __init__(self,mat,nx_graph_opts=nx_graph_opts):
        self.mat = mat
        self.nx_graph_opts = nx_graph_opts

    def matshow(self,label_order='usage',nmax=None):
        assert label_order in ('raw','usage')
        if label_order == 'raw':
            plt.matshow(self.mat[:nmax,:nmax])
        else:
            plt.matshow(permute_by_usage(self.mat)[:nmax,:nmax])

    def network(self,layout='dot',label_order='usage',
                nmax=None,use_alpha=True,adjust_width=True):
        assert label_order in ('raw','usage')
        if label_order == 'raw':
            mat = self.mat[:nmax,:nmax]
        else:
            mat = permute_by_usage(self.mat)[:nmax,:nmax]

        nx_graph = DiGraph(mat,**self.nx_graph_opts)

        if not use_alpha:
            cmap = plt.get_cmap('Blues')
            rgba_colors = cmap(mat / mat.max() if mat.max() > 0 else mat)
            for i,j in nx_graph.edges():
                nx_graph.edge[i][j]['color'] = rgb2hex(rgba_colors[i,j])
        else:
            cmap = constant_colormap('blue')
            rgba_colors = cmap(np.ones_like(mat))
            for i,j in nx_graph.edges():
                nx_graph.edge[i][j]['color'] = rgb2hex(rgba_colors[i,j,:3]) \
                    + ('%02x' % int(255.*mat[i,j]/mat.max()))

        if adjust_width:
            nx.set_edge_attributes(
                nx_graph,'penwidth',
                {e:str(1.+3*mat[e]/mat.max()) for e in nx_graph.edges_iter()})

        nx_graph.layout(layout).draw()


####################
#  numerical util  #
####################


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


##############
#  viz util  #
##############


def draw_pngstr(png_str):
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    plt.imshow(mpimg.imread(sio),aspect='equal')



def constant_colormap(color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('constant',[(0,color),(1,color)])


def draw_graph(nx_graph):
    fig = plt.figure()

    # use pygraphviz, pydot is alternative
    agraph = nx.to_agraph(nx_graph)
    agraph.has_layout = True

    sio = StringIO()
    agraph.draw(sio,format='png')
    sio.seek(0)

    if running_in_notebook():
        from IPython.display import Image, display
        display(Image(data=sio))
    else:
        import matplotlib.image as mpimg
        plt.imshow(mpimg.imread(sio),aspect='equal')
        plt.axis('off')


def running_in_notebook():
    # TODO
    return False

