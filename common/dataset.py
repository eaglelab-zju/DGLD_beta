import dgl
import torch
import numpy as np
from dgl.data import DGLDataset
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean

from .utils import is_bidirected
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

    print(
        "structural anomaly score:",
        roc_auc_score(str_data_groundtruth, str_data_predict),
    )
    print(
        "attribute anomaly score:",
        roc_auc_score(attr_data_groundtruth, attr_data_predict),
    )
    print(
        "final anomaly score:",
        roc_auc_score(np.where(groundtruth==0, 0, 1), prob),
    )
    


class GraphNodeAnomalyDectionDataset(DGLDataset):
    r"""
    follow [CoLA](https://arxiv.org/abs/2103.00113)，inject Anomaly to the oral graph
    We fix p = 15
    and set q to 10, 15, 20, 5, 5, 20, 200 for
    BlogCatalog, Flickr, ACM, Cora, Citeseer, Pubmed and ogbn-arxiv, respectively.
    """

    def __init__(self, name="Cora", p=15, k=50):
        super().__init__(name=name)
        self.dataset_name = name
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
        }

        assert self.dataset_name in self.q_map and self.dataset_name in self.dataset_map
        self.q = self.q_map[name]
        self.k = k
        self.seed = 42
        self.dataset = eval(self.dataset_map[self.dataset_name])[0]
        assert is_bidirected(self.dataset) == True
        self.init_anomaly_label()
        self.inject_contextual_anomalies()
        self.inject_structural_anomalies()

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

    def init_anomaly_label(self):
        number_node = self.dataset.num_nodes()
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
        split_auc(self.anomaly_label, prediction)

    def __getitem__(self, idx):
        return self.dataset

    def __len__(self):
        return 0


if __name__ == "__main__":
    well_test_dataset = ["Cora", "Pubmed", "Citeseer"]
    for data_name in well_test_dataset:
        print(well_test_dataset)
        dataset = GraphNodeAnomalyDectionDataset("Pubmed")
        print(dataset[0].num_nodes())
        print(dataset.num_anomaly)
        print(dataset.anomaly_label)
        rand_ans = np.random.rand(dataset.num_nodes)
        dataset.evalution(rand_ans)
