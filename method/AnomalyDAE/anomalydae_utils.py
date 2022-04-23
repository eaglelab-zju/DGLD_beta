import shutil
import sys
import scipy.sparse as sp
import os
sys.path.append('../../')

import argparse
from tqdm import tqdm
import numpy as np
import torch



def get_parse():
    parser = argparse.ArgumentParser(
        description='AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks')
    # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=42)
    # max min avg  weighted_sum
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='dimension of hidden embedding (default: 256)')
    parser.add_argument('--out_dim', type=int, default=128,
                        help='dimension of output embedding (default: 128)')
    parser.add_argument('--num_epoch', type=int,
                        default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='balance parameter')
    parser.add_argument('--eta', type=float, default=5.0,
                        help='Attribute penalty balance parameter')
    parser.add_argument('--theta', type=float, default=40.0,
                        help='structure penalty balance parameter')
    parser.add_argument('--no_cuda', action='store_true')
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


def loss_func(adj, A_hat, attrs, X_hat, alpha,eta, theta,device):
    # structure penalty
    # reversed_adj=torch.ones(adj.shape).to(device)-adj
    # thetas=torch.where(reversed_adj>0,reversed_adj,torch.full(adj.shape,theta).to(device))
    thetas=adj*(theta-1)+1

    # Attribute penalty
    # reversed_attr=torch.ones(attrs.shape).to(device)-attrs
    # etas=torch.where(reversed_attr==1,reversed_attr,torch.full(attrs.shape,eta).to(device))
    etas=attrs*(eta-1)+1

    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2) * etas
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    thetas = adj * (theta-1) + 1 
    diff_structure = torch.pow(A_hat - adj, 2) * thetas
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + \
        (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def train_step(args, model, optimizer, graph, features, adj,adj_label,device):

    model.train()
    optimizer.zero_grad()
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, struct_loss, feat_loss = loss_func(
        adj_label, A_hat, features, X_hat, args.alpha,args.eta, args.theta,device)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    return l, struct_loss, feat_loss


def test_step(args, model, graph, features, adj,adj_label,device):
    model.eval()
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, _, _ = loss_func(adj_label, A_hat, features, X_hat, args.alpha,args.eta, args.theta,device)
    score = loss.detach().cpu().numpy()
    # print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
    return score

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()