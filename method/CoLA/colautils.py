import argparse
from tqdm import tqdm
import numpy as np 
import torch
import sys
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
    parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='tmp')  #max min avg  weighted_sum

    args = parser.parse_args()
    return args

def train_epoch(epoch, args, loader, net, device, criterion, optimizer):
    loss_accum = 0
    net.train()
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        pos_scores, neg_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        loss = loss_fun(pos_scores, neg_scores, criterion, device)
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
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)

        pos_scores, neg_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        predict_scores.extend(list((neg_scores-pos_scores).detach().cpu().numpy()))
        loss = loss_fun(pos_scores, neg_scores, criterion, device)
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.2f}'.format(loss_accum), color='blue')
    return np.array(predict_scores)

    
