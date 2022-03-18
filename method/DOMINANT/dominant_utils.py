import shutil
import sys
import os
sys.path.append('../../')

import argparse
from tqdm import tqdm
import numpy as np
import torch

from models import Dominant
from utils.print import cprint, lcprint


def get_parse():
    parser = argparse.ArgumentParser(
        description='Deep Anomaly Detection on Attributed Networks')
    # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=1)
    # max min avg  weighted_sum
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--num_epoch', type=int,
                        default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='balance parameter')
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.lr is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed', 'Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        elif args.dataset == 'ogbn-arxiv':
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 10

    return args


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + \
        (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def train_step(args, model, optimizer, graph, features, adj):

    model.train()
    optimizer.zero_grad()
    A_hat, X_hat = model(graph, features)
    loss, struct_loss, feat_loss = loss_func(
        adj, A_hat, features, X_hat, args.alpha)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    return l, struct_loss, feat_loss
    # print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))


def test_step(args, model, graph, features, adj):
    model.eval()
    A_hat, X_hat = model(graph, features)
    loss, _, _ = loss_func(adj, A_hat, features, X_hat, args.alpha)
    score = loss.detach().cpu().numpy()
    # print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
    return score
