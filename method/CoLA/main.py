import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
import sys

sys.path.append("../../")
from common.dataset import GraphNodeAnomalyDectionDataset
from dataset import CoLADataSet
from utils import get_parse, train_epoch, valid_epoch
from model import CoLAModel
if __name__ == "__main__":
    args = get_parse()

    # dataset
    dataset = CoLADataSet(args.dataset)
    train_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    # model optimizer loss
    model = CoLAModel(in_feats=dataset[0][0].ndata['feat'].shape[1])
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # train
    for epoch in range(args.num_epoch):
        train_epoch(epoch, args, train_loader, model, device, criterion, optimizer)