import sys
sys.path.append('../../')

import scipy.sparse as sp

import torch
from torch.utils.tensorboard import SummaryWriter

from common.dataset import GraphNodeAnomalyDectionDataset
from collections import Counter
from AAGNN_A import AAGNN_A
from AAGNN_M import AAGNN_M
from get_parse import get_parse
if __name__ == '__main__':
    args = get_parse()
    print(args)
    # load dataset
    dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    model = AAGNN_M()
    model.fit(dataset, args)
