from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.DOMINANT import Dominant
from DGLD.DOMINANT import get_parse
from DGLD.common.evaluation import split_auc
from DGLD.common.utils import load_ACM
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    """
    sklearn-like API for most users.
    """
    """
    using GraphNodeAnomalyDectionDataset 
    """
    gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = Dominant(feat_size=1433, hidden_size=64, dropout=0.3)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g)
    print(split_auc(label, result))

    """
    custom dataset
    """
    g=load_ACM()[0]
    label = g.ndata['label']
    model = Dominant(feat_size=8337, hidden_size=64, dropout=0.3)
    model.fit(g, num_epoch=1, device='2')
    result = model.predict(g,device='2')
    print(split_auc(label, result,'custom'))
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    print(args)
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = Dominant(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)

