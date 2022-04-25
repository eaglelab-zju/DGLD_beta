import torch
from ogb.nodeproppred import DglNodePropPredDataset
from scipy.stats import rankdata
import sys
import scipy.sparse as sp
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import dgl
import inspect,os
current_file_name = inspect.getfile(inspect.currentframe())
data_path = os.path.abspath(os.path.dirname(os.path.dirname(current_file_name)))+'/data/'

def ranknorm(input_arr):
    r"""
    input_arr: np.ndarray like object.
    """
    return rankdata(input_arr, method='min') / len(input_arr)

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return torch.allclose(a.float().cpu(),
            b.float().cpu(), rtol=rtol, atol=atol)

def move_start_node_fisrt(pace, start_node):
    """
    return a new pace in which the start node is in the first place.
    """
    if pace[0] == start_node:return pace
    for i in range(1, len(pace)):
        if pace[i] == start_node:
            pace[i] = pace[0]
            break
    pace[0] = start_node
    return pace

def is_bidirected(g):
    """Return whether the graph is a bidirected graph.
    A graph is bidirected if for any edge :math:`(u, v)` in :math:`G` with weight :math:`w`,
    there exists an edge :math:`(v, u)` in :math:`G` with the same weight.
    """
    src, dst = g.edges()
    num_nodes = g.num_nodes()

    # Sort first by src then dst
    idx_src_dst = src * num_nodes + dst
    perm_src_dst = torch.argsort(idx_src_dst, dim=0, descending=False)
    src1, dst1 = src[perm_src_dst], dst[perm_src_dst]

    # Sort first by dst then src
    idx_dst_src = dst * num_nodes + src
    perm_dst_src = torch.argsort(idx_dst_src, dim=0, descending=False)
    src2, dst2 = src[perm_dst_src], dst[perm_dst_src]

    return allclose(src1, dst2) and allclose(src2, dst1)

def load_ogbn_arxiv():
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    graph, _ = data[0]
    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]


def load_mat_data2dgl(data_path,verbose=True):
    """load data from .mat file

    Parameters
    ----------
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    list
        [graph]
    """
    mat_path =data_path
    data_mat = sio.loadmat(mat_path)
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    # feat = preprocessing.normalize(feat, axis=0)
    truth = data_mat['Label']
    truth = truth.flatten()
    graph=dgl.from_scipy(adj)
    graph.ndata['feat']=torch.from_numpy(feat.toarray()).to(torch.float32)
    graph.ndata['label']=torch.from_numpy(truth).to(torch.float32)
    num_classes=len(np.unique(truth))

    if verbose:
        print('  NumNodes: {}'.format(graph.number_of_nodes()))
        print('  NumEdges: {}'.format(graph.number_of_edges()))
        print('  NumFeats: {}'.format(graph.ndata['feat'].shape[1]))
        print('  NumClasses: {}'.format(num_classes))
    
    # # add reverse edges
    # srcs, dsts = graph.all_edges()
    # graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]

def load_BlogCatalog():
    """load BlogCatalog dgl graph

    Returns
    -------
    list
        [graph]
    
    Using
    -------
    >>> graph=load_BlogCatalog()[0]
    """
    return load_mat_data2dgl(data_path=data_path+'BlogCatalog.mat')

def load_Flickr():
    """load Flickr dgl graph

    Returns
    -------
    list
        [graph]

    Using
    -------
    >>> graph=load_Flickr()[0]
    """
    return load_mat_data2dgl(data_path=data_path+'Flickr.mat')

def load_ACM():
    raise NotImplementedError
    

r"""
cd CoLA
python main.py --dataset ACM
"""
if __name__ == "__main__":
    load_ogbn_arxiv()
