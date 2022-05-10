import os
from os import path as osp
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import EdgeWeightNorm

import sys 
sys.path.append('../../')
from common.dataset import GraphNodeAnomalyDectionDataset
from common.sample import CoLASubGraphSampling, UniformNeighborSampling
from common.dglAug import ComposeAug,NodeShuffle,AddEdge,RandomMask
from colautils import get_parse
args = get_parse()

def safe_add_self_loop(g):
    newg = dgl.remove_self_loop(g)
    newg = dgl.add_self_loop(newg)
    return newg

class CoLADataSet(DGLDataset):
    def __init__(self, base_dataset_name='Cora', subgraphsize=4):
        super(CoLADataSet).__init__()
        self.dataset_name = base_dataset_name
        self.subgraphsize = subgraphsize
        self.oraldataset = GraphNodeAnomalyDectionDataset(name=self.dataset_name)
        self.dataset = self.oraldataset[0]
        self.colasubgraphsampler = CoLASubGraphSampling(length=self.subgraphsize)
        self.paces = []
        self.normalize_feat()
        self.random_walk_sampling()
    def normalize_feat(self):
        self.dataset.ndata['feat'] = F.normalize(self.dataset.ndata['feat'], p=1, dim=1)
        norm = EdgeWeightNorm(norm='both')
        self.dataset = safe_add_self_loop(self.dataset)
        norm_edge_weight = norm(self.dataset, edge_weight=torch.ones(self.dataset.num_edges()))
        self.dataset.edata['w'] = norm_edge_weight
        # print(norm_edge_weight)

    def random_walk_sampling(self):
        self.paces = self.colasubgraphsampler(self.dataset, list(range(self.dataset.num_nodes())))

    def graph_transform(self, g):
        if args.aug_type=='add_edge':
            augmentor = ComposeAug([AddEdge(args.aug_ratio)])
        elif args.aug_type=='random_mask':
            augmentor = ComposeAug([RandomMask(args.aug_ratio)])
        elif args.aug_type=='node_shuffle':
            augmentor = ComposeAug([NodeShuffle()])
        elif args.aug_type=='none':
            augmentor = lambda x:x
        newg = augmentor(g)
        return newg

    def __getitem__(self, i):
        pos_subgraph = dgl.node_subgraph(self.dataset, self.paces[i])

        neg_idx = np.random.randint(self.dataset.num_nodes()) 
        while neg_idx == i:
            neg_idx = np.random.randint(self.dataset.num_nodes()) 
        neg_subgraph = dgl.node_subgraph(self.dataset, self.paces[neg_idx])
        neg_aug_subg = self.graph_transform(neg_subgraph)

        return pos_subgraph, neg_subgraph, neg_aug_subg

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
