from re import A
import scipy.io as sio
import scipy.sparse as sp
from sklearn.covariance import graphical_lasso
from dataset import GraphNodeAnomalyDectionDataset
import os
current_file_name = __file__
data_path = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))+'/data/'
# print('data_path:',data_path)


def read_raw_pyg_dataset(data_name=''):
    """Read raw data from pyg.
        If data not in pyg,read from .mat files.
    Parameters
    ----------
    data_name : str, optional
        name of dataset, by default ''
    """
    pass


def gen_and_save_pygod_dataset(data_name=''):
    pass

def gen_and_save_dgld_dataset(data_name=''):
    graph = GraphNodeAnomalyDectionDataset(data_name)[0]
    print(graph)
    print("anomaly_label", graph.ndata['anomaly_label'])
    adj=graph.adj(scipy_fmt='csr')
    feat=graph.ndata['feat']
    feat=sp.csc_matrix(feat)
    label=graph.ndata['anomaly_label'].numpy()
    print('adj shape:',adj.shape)
    print('feat shape:',feat.shape)
    print('label shape:',label.shape)

    mat_dict={}
    mat_dict['Network']=adj
    mat_dict['Attributes']=feat
    mat_dict['Label']=label
    
    save_path=data_path+data_name+'_dgld.mat'
    sio.savemat(save_path,mat_dict)
    print('save mat data to:',save_path)
    print('-'*60,'\n\n')

if __name__=='__main__':
    datasets=["Cora","Pubmed", "Citeseer","BlogCatalog","Flickr"]
    for data_name in datasets:
        print('>'*30,data_name)
        gen_and_save_dgld_dataset(data_name)



