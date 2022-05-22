from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.DOMINANT import Dominant
from DGLD.DOMINANT import get_parse
from DGLD.common.evaluation import split_auc

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
    g = dgl.graph((torch.tensor([0, 1, 2, 4, 6, 7]), torch.tensor([3, 4, 5, 2, 5, 2])))
    g.ndata['feat'] = torch.rand((8, 4))
    g=dgl.add_self_loop(g)
    label = np.array([1, 2, 0, 0, 0, 0, 0, 0])
    model = Dominant(feat_size=4, hidden_size=64, dropout=0.3)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g)
    print(split_auc(label, result))
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

