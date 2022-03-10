import os
from os import path as osp

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl
from dgl.data import DGLDataset

import sys 
sys.path.append('../../')
from common.dataset import GraphNodeAnomalyDectionDataset
from common.sample import CoLASubGraphSampling

class CoLADataSet(DGLDataset):
    def __init__(self, base_dataset_name='Cora', subgraphsize=4):
        super(CoLADataSet).__init__()
        self.dataset_name = base_dataset_name
        self.subgraphsize = subgraphsize
        self.dataset = GraphNodeAnomalyDectionDataset(name=self.dataset_name)[0]
        self.colasubgraphsampler = CoLASubGraphSampling(length=self.subgraphsize)
        self.paces = []
        self.random_walk_sampling()

    def random_walk_sampling(self):
        self.paces = self.colasubgraphsampler(self.dataset, list(range(self.dataset.num_nodes())))

    def __getitem__(self, i):
        pos_subgraph = dgl.node_subgraph(self.dataset, self.paces[i])
        neg_idx = np.random.randint(self.dataset.num_nodes())
        neg_subgraph = dgl.node_subgraph(self.dataset, self.paces[neg_idx])
        return pos_subgraph, neg_subgraph

    def __len__(self):
        return self.dataset.num_nodes()

    def process(self):
        pass

if __name__ == '__main__':

    dataset = CoLADataSet()
    # print(dataset[0].edges())
    ans = []
    for i in range(100):
        dataset.random_walk_sampling()
        ans.append(dataset[502][1].ndata[dgl.NID].numpy().tolist())
    print(set([str(t) for t in ans]))
    # graph, label = dataset[0]
    # print(graph, label)
