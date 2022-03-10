import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader

import sys

sys.path.append("../../")
from common.dataset import GraphNodeAnomalyDectionDataset
from dataset import CoLADataSet
from colautils import get_parse, train_epoch, test_epoch
from model import CoLAModel
if __name__ == "__main__":
    args = get_parse()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # dataset
    dataset = CoLADataSet(args.dataset)
    train_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=True
    )
    test_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False
    )

    # model optimizer loss
    model = CoLAModel(in_feats=dataset[0][0].ndata['feat'].shape[1]).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # train
    for epoch in range(args.num_epoch):
        train_loader.dataset.random_walk_sampling()
        train_epoch(epoch, args, train_loader, model, device, criterion, optimizer)
        predict_score = test_epoch(epoch, args, test_loader, model, device, criterion, optimizer)
        dataset.oraldataset.evalution(predict_score)
    
    # multi-round test
    predict_score_final = 0
    for rnd in range(args.auc_test_rounds):
        test_loader.dataset.random_walk_sampling()
        predict_score = test_epoch(epoch, args, test_loader, model, device, criterion, optimizer)
        predict_score_final += np.array(predict_score)
    predict_score_final /= args.auc_test_rounds
    dataset.oraldataset.evalution(predict_score_final)
