import dgl
import torch
import numpy as np
import pandas as pd

from dgl.data import DGLDataset
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean
import scipy.sparse as sp
import  os
from dgl.data.utils import download

import scipy.io as sio
from dgl import backend as F
import sys
sys.path.append('..')
from common.utils import is_bidirected, load_ogbn_arxiv, load_BlogCatalog, load_Flickr,load_ACM

#'BlogCatalog'  'Flickr' 'cora'  'citeseer' 'pubmed' 'ACM' 'ogbn-arxiv'
# TODO: add all datasets above.

def split_auc(groundtruth, prob):
    r"""
    print the scoring(AUC) of the two types of anomalies separately.
    Parameter:
    ----------
    groundtruth: np.ndarray, Indicates whether this node is an injected anomaly node.
            0: normal node
            1: structural anomaly
            2: contextual anomaly

    prob: np.ndarray-like array saving the predicted score for every node
    Return:
    -------
    None
    """
    str_pos_idx = groundtruth == 1
    attr_pos_idx = groundtruth == 2
    norm_idx = groundtruth == 0

    str_data_idx = str_pos_idx | norm_idx
    attr_data_idx = attr_pos_idx | norm_idx

    str_data_groundtruth = groundtruth[str_data_idx]
    str_data_predict = prob[str_data_idx]

    attr_data_groundtruth = np.where(groundtruth[attr_data_idx] != 0, 1, 0)
    attr_data_predict = prob[attr_data_idx]

    s_score = roc_auc_score(str_data_groundtruth, str_data_predict)
    a_score = roc_auc_score(attr_data_groundtruth, attr_data_predict)
    final_score = roc_auc_score(np.where(groundtruth == 0, 0, 1), prob)

    print("structural anomaly score:", s_score)
    print("attribute anomaly score:", a_score)
    print("final anomaly score:", final_score)
    return final_score, a_score, s_score


class GraphNodeAnomalyDectionDataset(DGLDataset):
    r"""
    follow [CoLA](https://arxiv.org/abs/2103.00113)，inject Anomaly to the oral graph
    We fix p = 15
    and set q to 10, 15, 20, 5, 5, 20, 200 for
    BlogCatalog, Flickr, ACM, Cora, Citeseer, Pubmed and ogbn-arxiv, respectively.
    """

    def __init__(self, name="Cora", p=15, k=50, cola_preprocess_features=True, g_data=None, y_data=None):
        r"""
        Parameter
        ---------
        name:
        when name == 'custom', using custom data and please Specify custom data by g_data.
        and Specify label by y_data. [BlogCatalog, Flickr, Cora, Citeseer, Pubmed and ogbn-arxiv] is supported default follow CoLA.
        
        p and k :
        and anomaly injection hyperparameter follow CoLA

        cola_preprocess_features: 
        follow the same preprocess as CoLA, default:True

        g_data:
        Specify custom data by g_data.

        y_data:
        Specify custom label by g_data.

        """
        super().__init__(name=name)
        self.dataset_name = name
        self.cola_preprocess_features = cola_preprocess_features
        if name in ['Flickr', 'BlogCatalog', 'Pubmed']:
            self.cola_preprocess_features = False
        self.p = p
        self.q_map = {
            "BlogCatalog": 10,
            "Flickr": 15,
            "ACM": 20,
            "Cora": 5,
            "Citeseer": 5,
            "Pubmed": 20,
            "ogbn-arxiv": 200,
        }
        self.dataset_map = {
            "Cora": "dgl.data.CoraGraphDataset()",
            "Citeseer": "dgl.data.CiteseerGraphDataset()",
            "Pubmed": "dgl.data.PubmedGraphDataset()",
            "ogbn-arxiv":"load_ogbn_arxiv()",
            "ACM":"load_ACM()",
            "BlogCatalog":"load_BlogCatalog()",
            "Flickr":"load_Flickr()"
        }
        if self.dataset_name != 'custom':
            assert self.dataset_name in self.q_map and self.dataset_name in self.dataset_map, self.dataset_name
            self.q = self.q_map[name]
            self.k = k
            self.seed = 42
            self.dataset = eval(self.dataset_map[self.dataset_name])[0]
        else:
            self.dataset = g_data

        assert is_bidirected(self.dataset) == True
        self.init_anomaly_label(label=y_data)

        if self.dataset_name != 'custom' and y_data == None:
            print('inject_contextual_anomalies and inject_structural_anomalies')
            self.inject_contextual_anomalies()
            self.inject_structural_anomalies()

        if self.cola_preprocess_features:
            print('preprocess_features as CoLA')
            self.dataset.ndata['feat'] = self.preprocess_features(self.dataset.ndata['feat'])
    @property
    def num_anomaly(self):
        r"""
        anomaly_label: Indicates whether this node is an injected anomaly node.
            0: normal node
            1: structural anomaly
            2: contextual anomaly
        """
        return sum(self.dataset.ndata["anomaly_label"] != 0)

    @property
    def num_nodes(self):
        return self.dataset.num_nodes()

    @property
    def anomaly_label(self):
        return self.dataset.ndata["anomaly_label"]

    @property
    def anomalies_idx(self):
        anomalies = torch.where(self.anomaly_label != 0)[0].numpy()
        return anomalies

    @property
    def structural_anomalies_idx(self):
        anomalies = torch.where(self.anomaly_label == 1)[0].numpy()
        return anomalies

    @property
    def contextual_anomalies_idx(self):
        anomalies = torch.where(self.anomaly_label == 2)[0].numpy()
        return anomalies

    @property
    def normal_idx(self):
        nodes = torch.where(self.anomaly_label == 0)[0].numpy()
        return nodes

    @property
    def node_attr(self):
        return self.dataset.ndata["feat"]

    def set_node_attr(self, attr):
        self.dataset.ndata["feat"] = attr

    def set_anomaly_label(self, label):
        self.dataset.ndata["anomaly_label"] = label

    def init_anomaly_label(self, label=None):
        number_node = self.dataset.num_nodes()
        if label != None:
            self.dataset.ndata["anomaly_label"] = label# torch.zeros(number_node)
        else:
            self.dataset.ndata["anomaly_label"] = torch.zeros(number_node)

    def reset_anomaly_label(self):
        self.init_anomaly_label()

    def process(self):
        pass

    def inject_structural_anomalies(self):
        np.random.seed(self.seed)
        src, dst = self.dataset.edges()
        labels = self.anomaly_label
        p, q = self.p, self.q
        number_nodes = self.dataset.num_nodes()
        anomalies = set(torch.where(labels != 0)[0].numpy())

        new_src, new_dst = [], []
        # q cliques
        for i in range(q):
            q_list = []
            # selet p nodes
            for j in range(p):
                a = np.random.randint(number_nodes)
                while a in anomalies:
                    a = np.random.randint(number_nodes)
                q_list.append(a)
                anomalies.add(a)
                labels[a] = 1
            # make full connected
            for n1 in range(p):
                for n2 in range(n1 + 1, p):
                    new_src.extend([q_list[n1], q_list[n2]])
                    new_dst.extend([q_list[n2], q_list[n1]])

        src, dst = list(src.numpy()), list(dst.numpy())
        src.extend(new_src)
        dst.extend(new_dst)
        # update edges
        self.dataset.remove_edges(torch.arange(self.dataset.num_edges()))
        self.dataset.add_edges(src, dst)
        # print(self.dataset.num_edges())
        # BUG
        r"""
        dgl.DGLGraph.to_simple is not supported inplace
        """
        # self.dataset.to_simple()
        self.dataset = dgl.to_simple(self.dataset)
        # print(self.dataset.num_edges())
        self.set_anomaly_label(labels)
        print(
            "inject structural_anomalies numbers:", len(self.structural_anomalies_idx)
        )
        print("anomalies numbers:", len(self.anomalies_idx))

    def inject_contextual_anomalies(self):
        np.random.seed(self.seed)
        k = self.k
        attribute_anomalies_number = self.p * self.q
        normal_nodes_idx = self.normal_idx
        attribute_anomalies_idx = np.random.choice(
            normal_nodes_idx, size=attribute_anomalies_number, replace=False
        )
        all_attr = self.node_attr
        all_nodes_idx = list(range(self.dataset.num_nodes()))
        for aa_i in attribute_anomalies_idx:
            # random sample k nodes
            random_k_idx = np.random.choice(all_nodes_idx, size=k, replace=False)
            # cal the euclidean distance and replace the node attribute with \
            biggest_distance = 0
            biggest_attr = 0
            for i in random_k_idx:
                dis = euclidean(all_attr[aa_i], all_attr[i])
                if dis > biggest_distance:
                    biggest_distance, biggest_attr = dis, all_attr[i]
            # the node which has biggest one euclidean distance
            all_attr[aa_i] = biggest_attr

        self.set_node_attr(all_attr)
        labels = self.anomaly_label
        labels[attribute_anomalies_idx] = 2
        self.set_anomaly_label(labels)
        print(
            "inject contextual_anomalies numbers:", len(self.contextual_anomalies_idx)
        )
        print("anomalies numbers:", len(self.anomalies_idx))

    def preprocess_features(self, features):
        r"""Row-normalize feature matrix and convert to tuple representation
        copy from [CoLA]()
        """
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return torch.Tensor(features).float()

    def evalution(self, prediction):
        r"""
        print the scoring(AUC) of the two types of anomalies separately.
        Parameter:
        ----------
        prediction: np.ndarray-like array save the predicted score for every node
        Return:
        -------
        None
        """
        return split_auc(self.anomaly_label, prediction)

    def evaluation_multiround(self, predict_score_arr):
        '''
        Description
        -----------
        mulit-round result(as CoLA) evaluation.

        Parameter
        ---------
        predict_score_arr: node * num_round
        '''
        mean_predict_result = predict_score_arr.mean(1)
        std_predict_result = predict_score_arr.std(1)
        max_predict_result = predict_score_arr.max(1)
        min_predict_result = predict_score_arr.min(1)
        median_predict_result = np.median(predict_score_arr, 1)

        descriptions = {
            "mean": mean_predict_result,
            "std": std_predict_result,
            "-std": - std_predict_result,
            "mean+std": mean_predict_result + std_predict_result,
            "mean-std": mean_predict_result - std_predict_result,
            "mean-2std": mean_predict_result - 2*std_predict_result,
            "mean-3std": mean_predict_result - 3*std_predict_result,
            "mean+median": mean_predict_result + median_predict_result,
            "max": max_predict_result,
            "min": min_predict_result,
            "min-std": min_predict_result-std_predict_result,
            "median": median_predict_result,
        }
        for stat in descriptions:
            print("=" * 10 + stat + "=" * 10)
            self.evalution(descriptions[stat])

    def __getitem__(self, idx):
        return self.dataset

    def __len__(self):
        return 0




def test_cutom_dataset():
    my_g = dgl.data.CoraGraphDataset()[0]
    label = torch.ones(my_g.num_nodes())
    dataset = GraphNodeAnomalyDectionDataset('custom', g_data=my_g, y_data=label)
    print("num_anomaly:", dataset.num_anomaly)
    print("anomaly_label", dataset.anomaly_label)

if __name__ == "__main__":
    test_cutom_dataset()
    data_path = '../data/'
    well_test_dataset = ["Cora", "Pubmed", "Citeseer","BlogCatalog","Flickr","ACM", "ogbn-arxiv"]
    num_nodes_list = []
    num_edges_list = []
    num_anomaly_list = []
    num_attributes_list = []
    random_evaluation_list = []
    
    for data_name in well_test_dataset:
        print("\ndataset:", data_name)
        dataset = GraphNodeAnomalyDectionDataset(data_name)
        print("num_anomaly:", dataset.num_anomaly)
        print("anomaly_label", dataset.anomaly_label)
        rand_ans = np.random.rand(dataset.num_nodes)
        _, _, final_score = dataset.evalution(rand_ans)
        num_nodes_list.append(dataset.num_nodes)
        num_edges_list.append(dataset.dataset.num_edges())
        num_anomaly_list.append(dataset.num_anomaly.item())
        num_attributes_list.append(dataset.dataset.ndata['feat'].shape[1])
        random_evaluation_list.append(final_score)


    dataset_info = pd.DataFrame({
        "well_test_dataset":well_test_dataset,
        "num_nodes":num_nodes_list,
        "num_edges":num_edges_list,
        "num_anomaly": num_anomaly_list,
        "num_attributes": num_attributes_list,
        "random_evaluation":random_evaluation_list
    })
    print(dataset_info)