import torch

from scipy.stats import rankdata
import dgl
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset
import os,wget,ssl,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
data_path =current_dir +'/data/'
# print('data_path:',data_path)

def ranknorm(input_arr):
    """
    return the 1-norm of rankdata of input_arr

    Parameters
    ----------
    input_arr: list
        the data to be ranked

    Returns
    -------
    rank : numpy.ndarray
        the 1-norm of rankdata
    """
    return rankdata(input_arr, method='min') / len(input_arr)

def allclose(a, b, rtol=1e-4, atol=1e-4):
    """
    This function checks if a and b satisfy the condition:
    |a - b| <= atol + rtol * |b|

    Parameters
    ----------
    input : Tensor
        first tensor to compare
    other : Tensor
        second tensor to compare
    atol : float, optional
        absolute tolerance. Default: 1e-08
    rtol : float, optional
        relative tolerance. Default: 1e-05

    Returns
    -------
    res : bool
        True for close, False for not
    """
    return torch.allclose(a.float().cpu(),
            b.float().cpu(), rtol=rtol, atol=atol)

def move_start_node_fisrt(pace, start_node):
    """
    return a new pace in which the start node is in the first place.

    Parameters
    ----------
    pace : list
        the subgraph of start node
    start_node: int
        target node

    Returns
    -------
    pace : list
        subgraph whose first value is start_node
    """
    if pace[0] == start_node:return pace
    for i in range(1, len(pace)):
        if pace[i] == start_node:
            pace[i] = pace[0]
            break
    pace[0] = start_node
    return pace

def is_bidirected(g):
    """
    Return whether the graph is a bidirected graph.
    A graph is bidirected if for any edge :math:`(u, v)` in :math:`G` with weight :math:`w`,
    there exists an edge :math:`(v, u)` in :math:`G` with the same weight.

    Parameters
    ----------
    g : DGL.graph

    Returns
    -------
    res : bool
        True for bidirected, False for not
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


def load_raw_pyg_dataset(data_name='',verbose=True):
    """
    Read raw data from pyg.
        If data not in pyg, read from .mat files.
    Parameters
    ----------
    data_name : str, optional
        name of dataset, by default ''

    returns
    -------
    data : dgl.graph
        the graph of dataset
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
    
    #save to mat
    adj=to_scipy_sparse_matrix(data.edge_index).tocsr()
    feat=sp.csr_matrix(data.x)

    mat_dict={}
    mat_dict['Network']=adj
    mat_dict['Attributes']=feat
    
    save_path=data_path+data_name+'_pyg.mat'
    sio.savemat(save_path,mat_dict)
    print('save mat data to:',save_path)
    print('-'*60,'\n\n')

    return data

def load_mat_data2dgl(data_path,verbose=True):
    """
    load data from .mat file

    Parameters
    ----------
    data_path : str
        the file to read in
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    graph : [DGL.graph]
        the graph read from data_path
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
        print()
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



def load_ogbn_arxiv(raw_dir=data_path):
    """
    Read ogbn-arxiv from dgl.

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.  

    returns
    -------
    graph : [dgl.graph]
        the graph of ogbn-arxiv
    """
    data = DglNodePropPredDataset(name="ogbn-arxiv",root=raw_dir)
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


#create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def load_BlogCatalog(raw_dir=data_path):
    """
    load BlogCatalog dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.  

    Returns
    -------
    graph : [DGL.graph]

    Examples
    -------
    >>> graph=load_BlogCatalog()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir,'BlogCatalog.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/BlogCatalog/BlogCatalog.mat?raw=true'
        wget.download(url,out=data_file,bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)

def load_Flickr(raw_dir=data_path):
    """
    load Flickr dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.  

    Returns
    -------
    graph : [DGL.graph]

    Examples
    -------
    >>> graph=load_Flickr()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir,'Flickr.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/Flickr/Flickr.mat?raw=true'
        wget.download(url,out=data_file,bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)


def load_ACM(raw_dir=data_path):
    """load ACM dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.  

    Returns
    -------
    graph : [DGL.graph]

    Examples
    -------
    >>> graph=load_ACM()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir,'ACM.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/dataset/ACM.mat?raw=true'
        wget.download(url,out=data_file,bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)

r"""
cd CoLA
python main.py --dataset ACM
"""
if __name__ == "__main__":
    load_ACM()
