import argparse
from tqdm import tqdm
import numpy as np 
import torch

import shutil
import sys
import os
sys.path.append('../../')
from utils.print import cprint, lcprint

def loss_fun_BPR(pos_scores, neg_scores, criterion, device):
    batch_size = pos_scores.shape[0]
    labels = torch.ones(batch_size).to(device)
    return criterion(pos_scores-neg_scores, labels)

def loss_fun_BCE(pos_scores, neg_scores, criterion, device):
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    batch_size = pos_scores.shape[0]
    pos_label = torch.ones(batch_size).to(device)
    neg_label = torch.zeros(batch_size).to(device)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return criterion(scores, labels)

loss_fun = loss_fun_BCE
def get_parse():
    # parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser = argparse.ArgumentParser(description = 'Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    # parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='tmp')  
    parser.add_argument('--global_adg', type=bool, default=True)  
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--beta', type = float, default = 0.6)
    parser.add_argument('--act_function', type = str, default= "PReLU")
    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    else:
        os.makedirs(args.logdir)


    if args.lr is None:
        if args.dataset in ['Cora','Citeseer','Pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        elif args.dataset == 'ogbn-arxiv':
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora','Citeseer','Pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 10
    
    if args.auc_test_rounds is None:
        if args.dataset != 'ogbn-arxiv':
            args.auc_test_rounds = 256
        else:
            args.auc_test_rounds = 20
                
    return args

def train_epoch(epoch, args, loader, net, device, criterion, optimizer):
    loss_accum = 0
    net.train()
    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        posfeat = [posfeat_1, posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        
        # pos_scores_1, pos_scores_2, neg_scores = net(pos_subgraph_1, pos_subgraph_2, posfeat_1, posfeat_2, neg_subgraph, negfeat)
        # pos_scores, neg_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        
        # loss = loss_fun(pos_scores_1, pos_scores_2, neg_scores, criterion, device)
        # loss = loss_fun(pos_scores, neg_scores, criterion, device)
        loss, single_predict_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.2f}'.format(loss_accum), color='blue')
    return loss_accum

def test_epoch(epoch, args, loader, net, device, criterion, optimizer):
    loss_accum = 0
    net.eval()
    predict_scores = []
    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        posfeat = [posfeat_1, posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)

        loss, single_predict_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        predict_scores.extend(list(single_predict_scores))
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.2f}'.format(loss_accum), color='blue')
    return np.array(torch.tensor(predict_scores, device = 'cpu'))

    
