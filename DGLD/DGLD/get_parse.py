import argparse
from tqdm import tqdm
import numpy as np 
import torch

import shutil
import sys
import os
from DGLD.utils.print import cprint, lcprint
def get_parse():
    parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    #parser.add_argument('--model', type=str, default='AAGNN_A')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='tmp')  #max min avg  weighted_sum
    parser.add_argument('--global_adg', type=bool, default=True)  #max min avg  weighted_sum
    parser.add_argument('--save_path', type=str, default='./model.pt') #the path where you want to save your model
    args = parser.parse_args()
    
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.lr is None:
        if args.dataset in ['Cora','Citeseer','Pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        else:
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora','Citeseer','Pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 10
            
    if args.dataset != 'ogbn-arxiv':
        args.auc_test_rounds = 256
    else:
        args.auc_test_rounds = 20
            
    return args
