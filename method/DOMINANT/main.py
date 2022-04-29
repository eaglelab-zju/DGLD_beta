
import imp
import sys

from sklearn.metrics import roc_auc_score
sys.path.append('../../')

import scipy.sparse as sp
import scipy.io as sio

import torch
from torch.utils.tensorboard import SummaryWriter
import dgl
from dominant_utils import get_parse, train_step, test_step,normalize_adj
from models import Dominant
from common.dataset import GraphNodeAnomalyDectionDataset

if __name__ == '__main__':
    args = get_parse()
    print(args)
    # load dataset
    dataset = GraphNodeAnomalyDectionDataset(args.dataset,cola_preprocess_features=False)
    graph = dataset[0]
    features = graph.ndata['feat']
    print(features)
    adj = graph.adj(scipy_fmt='csr')

    # # 构造原github数据为graph，进行测试
    # data_mat = sio.loadmat('/home/data/zp/ygm/GCN_AnomalyDetection_pytorch/data/BlogCatalog.mat')
    # adj = data_mat['Network']
    # feat = data_mat['Attributes']
    # truth = data_mat['Label']
    # truth = truth.flatten()
    # graph = dgl.from_scipy(adj)
    # features = torch.FloatTensor(feat.toarray())
    
    
    adj_norm = normalize_adj(adj)#has added self loop in load data
    adj_norm = torch.FloatTensor(adj_norm.toarray())

    import numpy as np
    print(np.sum(adj))
    adj_label = torch.FloatTensor(adj.toarray())
    
    print(graph)
    print('adj_label shape:', adj_label.shape)
    print('features shape:', features.shape)

    model = Dominant(feat_size=features.shape[1], hidden_size=args.hidden_dim,
                     dropout=args.dropout)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:" + str(args.device))
        model = model.to(device)
        graph = graph.to(device)
        features = features.to(device)
        adj_label = adj_label.to(device)
        adj_norm = adj_norm.to(device)
        print('Using gpu!!!')
    else:
        device = torch.device("cpu")
        print('Using cpu!!!')

    writer = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.num_epoch):
        loss, struct_loss, feat_loss = train_step(
            args, model, optimizer, graph, features, adj_norm,adj_label)
        predict_score = test_step(args, model, graph, features,adj_norm, adj_label)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
        )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
        writer.add_scalars(
            "loss",
            {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
            epoch,
        )
        # print('auc:',roc_auc_score(truth,predict_score))
        final_score, a_score, s_score = dataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )

        writer.flush()