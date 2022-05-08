import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append("../../")
from common.dataset import GraphNodeAnomalyDectionDataset
from utils.utils import seed_everything
from utils.print import lcprint, cprint

from dataset import CoLADataSet
from colautils import get_parse, train_epoch, test_epoch, train_model, multi_round_test, get_staticpseudolabel, get_staticpseudolabel_mask
from model import CoLAModel

if __name__ == "__main__":
    args = get_parse()
    seed_everything(args.seed)
    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # dataset
    dataset = CoLADataSet(args.dataset)
    train_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True,
    )
    test_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )
    # model optimizer loss
    model = CoLAModel(
        in_feats=dataset[0][0].ndata["feat"].shape[1],
        out_feats=args.embedding_dim,
        global_adg=args.global_adg,
        tau=args.tau,
        generative_loss_w=args.generative_loss_w
    ).to(device)
    print(model)
    
    # train
    writer = SummaryWriter(log_dir=args.logdir)

    if not args.continue_train:
        cprint("training", color='info')
        train_model(model, args, train_loader, test_loader, writer, device, pseudo_label_type='none')
        cprint("after training===> multi-round test", color='info')
        # multi-round test to create pseudo label
        pseudo_labels = multi_round_test(args, test_loader, model, device)
        cprint("self labeling", color='info')
        torch.save({"model": model, "result": pseudo_labels}, 'pretrain.pt')
    else:
        cprint("reload model", color='info')
        pt = torch.load("pretrain.pt")
        model, pseudo_labels = pt["model"], pt["result"]
    # self labeling
    # optimizer.param_groups[0]['lr'] *= 0.1
    train_loader.dataset.dataset.ndata['pseudo_label'] = torch.Tensor(pseudo_labels)
    train_loader.dataset.dataset.ndata['fix_pseudo_label'] = get_staticpseudolabel(pseudo_labels, keep_ratio=args.keep_ratio)
    # if args.reinit:
    #     model = CoLAModel(
    #         in_feats=dataset[0][0].ndata["feat"].shape[1],
    #         out_feats=args.embedding_dim,
    #         global_adg=args.global_adg,
    #         tau=args.tau,
    #         generative_loss_w=args.generative_loss_w
    #     ).to(device)
    # train_model(model, args, train_loader, test_loader, writer, device, pseudo_label_type=args.pseudotype)
    # pseudo_labels = multi_round_test(args, test_loader, model, device)
