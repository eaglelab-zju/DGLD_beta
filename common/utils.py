import torch
from ogb.nodeproppred import DglNodePropPredDataset
from scipy.stats import rankdata
import sys
import scipy.sparse as sp
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
# from torch_geometric.datasets import Planetoid,HGBDataset,AttributedGraphDataset
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import Data
import dgl
import os
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
data_path =current_dir +'/data/'
print('data_path:',data_path)


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
    print('ogbn-arxiv datapath:',current_dir+'/data/')
    data = DglNodePropPredDataset(name="ogbn-arxiv",root=current_dir+'/data/')
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


def load_raw_pyg_dataset(data_name='',verbose=True):
    """Read raw data from pyg.
        If data not in pyg,read from .mat files.
    Parameters
    ----------
    data_name : str, optional
        name of dataset, by default ''
    """
    assert data_name in ['Cora','Citeseer','Pubmed','BlogCatalog','Flickr','ogbn-arxiv'],\
        'datasets do not have this data!!!'
    if data_name in ['Cora','Citeseer','Pubmed','BlogCatalog']:
        data = AttributedGraphDataset(data_path+'attr-graphs/', data_name,transform=T.NormalizeFeatures())[0]
        # generate to bidirectional,if has,not genreate. Maybe add self loop on isolated nodes.
        data.edge_index=to_undirected(data.edge_index)
    elif data_name in ['Flickr']:
        data = AttributedGraphDataset(data_path+'attr-graphs/', data_name)[0]
        data.x=data.x.to_dense()
    elif data_name in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(name='ogbn-arxiv',root=data_path)[0]
        # generate to bidirectional,if has,not genreate. Maybe add self loop on isolated nodes.
        data.edge_index=to_undirected(data.edge_index)
    
    if verbose:
        print('  PyG dataset: {}'.format(data_name))
        print('  NumNodes: {}'.format(data.num_nodes))
        print('  NumEdges: {}'.format(data.num_edges))
        print('  NumFeats: {}'.format(data.x.shape[1]))
    return data

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
        print('  DGL dataset')
        print('  NumNodes: {}'.format(graph.number_of_nodes()))
        print('  NumEdges: {}'.format(graph.number_of_edges()))
        print('  NumFeats: {}'.format(graph.ndata['feat'].shape[1]))
        print('  NumClasses: {}'.format(num_classes))
    if 'ACM' in data_path:
        print('ACM')
        return [graph]
    # # add reverse edges
    # srcs, dsts = graph.all_edges()
    # graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]


def load_mat_data2PyG(data_path,verbose=True):
    """load data from .mat file

    Parameters
    ----------
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    list
        PyG data
    """
    mat_path =data_path
    data_mat = sio.loadmat(mat_path)
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    feat=torch.FloatTensor(feat.toarray())
    truth=torch.Tensor(truth)
    row=torch.Tensor(adj.tocoo().row)
    col=torch.Tensor(adj.tocoo().col) 
    data=Data(feat,torch.vstack([row,col]).to(torch.int64),y=truth)

    if verbose:
        print('  NumNodes: {}'.format(data.num_nodes))
        print('  NumEdges: {}'.format(data.num_edges))
        print('  NumFeats: {}'.format(data.x.shape[1]))
    
    # graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {data.num_edges}")
    data.edge_index,_ = remove_self_loops(data.edge_index)
    data.edge_index,_ = add_self_loops(data.edge_index)
    
    print(f"Total edges after adding self-loop {data.num_edges}")
    assert data.is_undirected() == True,'not a bidirected graph'
    return data


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
    """load ACM dgl graph

    Returns
    -------
    list
        [graph]
    
    Using
    -------
    >>> graph=load_ACM()[0]
    """
    return load_mat_data2dgl(data_path=data_path+'ACM.mat')


r"""
cd CoLA
python main.py --dataset ACM
"""
if __name__ == "__main__":
    load_ogbn_arxiv()