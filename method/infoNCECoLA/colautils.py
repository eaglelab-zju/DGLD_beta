import argparse
from tqdm import tqdm
import numpy as np 
import torch
import torch.optim as optim
import dgl
import joblib
import shutil
import sys
import os
sys.path.append('../../')
from utils.print import cprint, lcprint


def get_parse():
    parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--generative_loss_w', type=float, default=0.0)
    
    parser.add_argument('--keep_ratio', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--selflabeling_epcohs', type=int, default=100)

    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--subgraph_size', type=int, default=4)
    # parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='tmp')  
    parser.add_argument('--pseudotype', type=str, default='pseudo', help='none pseudo fix_pseudo')  
    
    parser.add_argument('--global_adg', type=bool, default=True)  
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--reinit', type=bool, default=True)
    
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
            args.num_epoch = 20
    
    if args.auc_test_rounds is None:
        if args.dataset != 'ogbn-arxiv':
            args.auc_test_rounds = 256
        else:
            args.auc_test_rounds = 20
                
    return args


def get_staticpseudolabel(pseudolabel, keep_ratio=0.95):
    masks = torch.Tensor(pseudolabel)
    sorted_scores = torch.sort(masks)[0]
    keep_number = int(len(sorted_scores)*keep_ratio)
    threshold_score = sorted_scores[keep_number]
    mask = (masks < threshold_score).float()
    return mask

def get_staticpseudolabel_mask(batch_graph):
    unbatch_g = dgl.unbatch(batch_graph)
    masks = []
    for g in unbatch_g:
        anchor_label = g.ndata['fix_pseudo_label'][0].item()
        masks.append(anchor_label)
    return torch.Tensor(masks).float()


def get_pseudolabel_mask(batch_graph, keep_ratio=0.95):
    unbatch_g = dgl.unbatch(batch_graph)
    masks = []
    for g in unbatch_g:
        anchor_pseudo_label = g.ndata['pseudo_label'][0].item()
        masks.append(anchor_pseudo_label)
    masks = torch.Tensor(masks)
    sorted_scores = torch.sort(masks)[0]
    keep_number = int(len(sorted_scores)*keep_ratio)
    threshold_score = sorted_scores[keep_number]
    mask = (masks < threshold_score).float()
    return mask

def get_label_mask(batch_graph):
    unbatch_g = dgl.unbatch(batch_graph)
    masks = []
    for g in unbatch_g:
        anchor_label = g.ndata['anomaly_label'][0].item()
        mask = 1 if anchor_label == 0 else 0
        masks.append(mask)
    masks = masks
    return torch.Tensor(masks).float()

def train_epoch(epoch, args, loader, net, device, criterion, optimizer, pseudo_label_type='gt'):
    loss_accum = 0
    net.train()
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc=pseudo_label_type)):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        if pseudo_label_type == 'gt':
            mask = get_label_mask(pos_subgraph).to(device)
        elif pseudo_label_type == 'pseudo':
            mask = get_pseudolabel_mask(pos_subgraph, keep_ratio=args.keep_ratio).to(device)
        elif pseudo_label_type == 'fix_pseudo':
            mask = get_staticpseudolabel_mask(pos_subgraph).to(device)
        else:
            mask = 1
        loss, pos_score, neg_score = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        loss = loss * mask
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.2f}'.format(loss_accum), color='blue')
    return loss_accum


def test_epoch(epoch, args, loader, net, device):
    loss_accum = 0
    net.eval()
    predict_scores = []
    pos_scores = []
    neg_scores = []
    losses = []
    
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        
        loss, pos_score, neg_score = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        losses.extend(loss.detach().cpu().numpy())
        pos_scores.extend(pos_score.detach().cpu().numpy())
        neg_scores.extend(neg_score.detach().cpu().numpy())
        
        predict_scores.extend(list((neg_score-pos_score).detach().cpu().numpy()))
        loss = loss.mean()
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.2f}'.format(loss_accum), color='blue')
    return np.array(predict_scores), np.array(pos_scores), np.array(neg_scores), np.array(losses)

    
def train_model(model, args, train_loader, test_loader, writer, device, pseudo_label_type='gt'):
    optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epoch = args.num_epoch if pseudo_label_type == 'none' else args.selflabeling_epcohs
    save_path = os.path.join(f"{args.logdir}", f"{args.dataset}_score")
    if not os.path.exists(save_path):
        print(f"create {save_path}")
        os.makedirs(save_path)
    for epoch in range(num_epoch):
        
        
        train_loader.dataset.random_walk_sampling()
        loss_accum = train_epoch(
            epoch, args, train_loader, model, device, criterion, optimizer, pseudo_label_type=pseudo_label_type
        )
        writer.add_scalar("loss-{}".format(pseudo_label_type), float(loss_accum), epoch)
        predict_score, pos_scores, neg_scores, losses = test_epoch(
            epoch, args, test_loader, model, device
        )
        save_content = {
            "label": train_loader.dataset.oraldataset.anomaly_label.numpy(),
            "scores":predict_score,
            "pos_scores":pos_scores,
            "neg_scores":neg_scores,
            "losses":losses,
        }
        epcoh_save_path = os.path.join(save_path, f"epoch{epoch}.pkl")
        joblib.dump(save_content, epcoh_save_path)
        final_score, a_score, s_score = train_loader.dataset.oraldataset.evalution(predict_score)
        writer.add_scalars(
            "auc-{}".format(pseudo_label_type),
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )
        writer.flush()
    return model

# multi-round test
def multi_round_test(args, test_loader, model, device):
    predict_score_arr = []
    for rnd in range(args.auc_test_rounds):
        test_loader.dataset.random_walk_sampling()
        predict_score, _, _, _ = test_epoch(
            rnd, args, test_loader, model, device
        )
        predict_score_arr.append(list(predict_score))

    predict_score_arr = np.array(predict_score_arr).T
    test_loader.dataset.oraldataset.evaluation_multiround(predict_score_arr)
    return np.mean(predict_score_arr, axis=1)