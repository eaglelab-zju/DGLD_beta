# Author: Peng Zhang <zzhangpeng@zju.edu.cn>
# License: BSD 2 clause
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.common.evaluation import split_auc
from DGLD.AAGNN import AAGNN
from DGLD.AAGNN import AAGNN_batch
from DGLD.utils.utils import seed_everything
from DGLD.AAGNN import get_parse
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    # """
    # sklearn-like API for most users.
    # """
    # """
    # using GraphNodeAnomalyDectionDataset 
    # """
    # gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    # graph = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # in_feats = graph.ndata['feat'].shape[1]
    # model = AAGNN(in_feats=in_feats, out_feats=300)
    # model.fit(graph, num_epoch=100, device='cuda:0')
    # result = model.predict(graph, device='cuda:0')
    # print(split_auc(label, result))

    """[command line mode]
    test command line mode
    """
    args = get_parse()
    # seed_everything(args['seed'])
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = AAGNN(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)
    print(args)

