import argparse
from tqdm import tqdm
import numpy as np 
import torch
import sys
sys.path.append('../../')
from utils.utils import cprint, lcprint

def get_parse():
    parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    return args

def train_epoch(epoch, args, loader, net, device, criterion, optimizer):
    loss_accum = 0
    net.train()
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        # cprint(labels.shape, 'debug') # torch.Size([16, 1, 5])
        feat = bg.ndata['feat'].to(device)
        efeat = bg.edata['feat'].long().to(device)
        predict = net(bg, feat, efeat)# len(x)=5(max_seq_len) x[0].shape=torch.Size([16, 5000])
        optimizer.zero_grad()
        # loss
        # lcprint('len_predict', len(predict), 'predict[0].shape', predict[0].shape, color='debug')
        # lcprint('labels[:, 0, i].shape', labels[:, 0, 0].shape, color='debug')
        loss = 0
        # print(labels[:, 0, 0].max())
        for i in range(args.max_seq_len):
            loss += criterion(predict[i].float(), labels[:, 0, i]) / args.max_seq_len
        # print(loss.item())
        loss.backward()
        optimizer.step()
        loss_accum += loss.item() 
    loss_accum /= (step + 1)
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.2f}'.format(loss_accum), color='blue')
        

def valid_epoch(epoch, args, loader, net, device, criterion, optimizer, evaluator, info='TEST'):
    loss_accum = 0
    net.eval()
    all_groundtruth = []
    all_prediction = []
    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg, labels = bg.to(device), labels.to(device)
        # cprint(labels.shape, 'debug') # torch.Size([16, 1, 5])
        feat = bg.ndata['feat'].to(device)
        efeat = bg.edata['feat'].long().to(device)
        predict = net(bg, feat, efeat)# len(x)=5(max_seq_len) x[0].shape=torch.Size([16, 5000])
        # loss
        # lcprint('len_predict', len(predict), 'predict[0].shape', predict[0].shape, color='debug')
        # lcprint('labels[:, 0, i].shape', labels[:, 0, 0].shape, color='debug')
        loss = 0
        groundtruth = [list(y[0].cpu().detach().numpy())for y in labels]
        prediction = np.zeros((len(groundtruth), args.max_seq_len))
        for i in range(args.max_seq_len):
            prediction[:, i] = torch.argmax(predict[i], dim=1).cpu().detach().long().numpy()

        prediction = prediction.astype(int).tolist()
        # print(prediction)
        # print(groundtruth)

        all_groundtruth.extend(groundtruth)
        all_prediction.extend(prediction)
        
        for i in range(args.max_seq_len):
            loss += criterion(predict[i].float(), labels[:, 0, i]) / args.max_seq_len
        # print(loss.item())
        loss_accum += loss.item() 
    
    loss_accum /= (step + 1)
    input_dict = {"seq_ref": all_groundtruth, "seq_pred": all_prediction}
    result = evaluator.eval(input_dict)
    lcprint(f'{info}==>epoch', epoch, 'Ave loss: {:.2f}'.format(loss_accum), 'f1 values::{:.3f}'.format(result["F1"]))
    cprint(result)
    return result

    
