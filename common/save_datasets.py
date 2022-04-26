from cProfile import label
import imp
from re import A, S
from numpy import dtype
import scipy.io as sio
import scipy.sparse as sp
from sklearn.covariance import graphical_lasso
from dataset import GraphNodeAnomalyDectionDataset, split_auc
from torch_geometric.utils import to_dense_adj,to_scipy_sparse_matrix
from torch_geometric.data import Data
import os
from common.utils import load_raw_pyg_dataset,load_mat_data2PyG
import torch
from pygod.utils import gen_attribute_outliers, gen_structure_outliers


current_file_name = __file__
data_path = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))+'/data/'
# print('data_path:',data_path)


def gen_and_save_pygod_dataset(data_name=''):
    """injecting anomaly nodes to pyg datasets and save to mat file.
    The return datasets distinguish between two kinds of 
    anomaly nodes by structural anomaly==1,attribute anomaly==2.

    Parameters
    ----------
    data_name : str, optional
        name of dataset, by default ''
    """
    pyg_data=load_raw_pyg_dataset(data_name)
    print('pyg_data:',pyg_data)
    p=15
    q_map = {
            "BlogCatalog": 10,
            "Flickr": 15,
            "ACM": 20,
            "Cora": 5,
            "Citeseer": 5,
            "Pubmed": 20,
            "ogbn-arxiv": 200,
        }
    data, ya = gen_attribute_outliers(pyg_data, n=p*q_map[data_name], k=50)
    data, ys = gen_structure_outliers(pyg_data, m=p, n=q_map[data_name])

    #distinguish outliers types
    ya=torch.where(ya==1,2,0)
    y=torch.vstack([ya,ys])
    y,_=y.max(dim=0)
    data.y=y

    #save to mat
    adj=to_scipy_sparse_matrix(data.edge_index).tocsr()
    feat=sp.csr_matrix(data.x)
    label=data.y.numpy()
    mat_dict={}
    mat_dict['Network']=adj
    mat_dict['Attributes']=feat
    mat_dict['Label']=label
    
    save_path=data_path+data_name+'_pygod.mat'
    sio.savemat(save_path,mat_dict)
    print('save mat data to:',save_path)
    print('-'*60,'\n\n')
    

def gen_and_save_dgld_dataset(data_name=''):
    """Injecting anomaly nodes to dgl datasets and save to mat file.
    The return datasets distinguish between two kinds of 
    anomaly nodes by structural anomaly==1,attribute anomaly==2.
    
    Parameters
    ----------
    data_name : str, optional
        name of dataset, by default ''
    """
    graph = GraphNodeAnomalyDectionDataset(data_name)[0]
    print(graph)
    print("anomaly_label", graph.ndata['anomaly_label'])
    adj=graph.adj(scipy_fmt='csr')
    feat=graph.ndata['feat']
    feat=sp.csr_matrix(feat)
    label=graph.ndata['anomaly_label'].numpy()
    print('adj shape:',adj.shape)
    print('feat shape:',feat.shape)
    print('label shape:',label.shape)

    #save to mat
    mat_dict={}
    mat_dict['Network']=adj
    mat_dict['Attributes']=feat
    mat_dict['Label']=label
    
    save_path=data_path+data_name+'_dgld.mat'
    sio.savemat(save_path,mat_dict)
    print('save mat data to:',save_path)
    print('-'*60,'\n\n')

if __name__=='__main__':
    datasets=["Cora", "Citeseer","Pubmed","BlogCatalog","Flickr","ogbn-arxiv"]
    for data_name in datasets:
        print('>'*30,data_name)
        gen_and_save_dgld_dataset(data_name)
        gen_and_save_pygod_dataset(data_name)



