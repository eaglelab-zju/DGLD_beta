
from ast import arg
import imp
import sys
import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, roc_auc_score
sys.path.append('../../')

import scipy.sparse as sp
import scipy.io as sio
import torch
torch.set_printoptions(precision=8)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl
from method.ComGA.comga_utils import get_parse, train_step, test_step,normalize_adj
from models import ComGA_Base
from common.dataset import GraphNodeAnomalyDectionDataset

if __name__ == '__main__':
    args = get_parse()
    print(args)
    test_mode=False
    # # # load dataset
    if not test_mode:
        dataset = GraphNodeAnomalyDectionDataset(args.dataset,cola_preprocess_features=False)
        graph = dataset[0]
        features = graph.ndata['feat']
        print(features)
        adj = graph.adj(scipy_fmt='csr')
        print('loop graph:\n',graph)

        A=adj.toarray()
        print('BlogCatalog.edgelist: A -->',np.sum(A))
        k1 = np.sum(A, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        num_loop=0
        for i in range(adj.shape[0]):
            if adj[i,i]==1:
                num_loop+=1
        m=(np.sum(A)-num_loop)/2+num_loop
        Eij = k1k2 / (2 * m)
        B =np.array(A - Eij)
        print('B -->',B.shape)

    else:
        # # 构造原github数据为graph，进行测试
        network = nx.read_weighted_edgelist('/home/data/zp/ygm/ComGA/data/BlogCatalog/BlogCatalog.edgelist')
        A = np.asarray(nx.adjacency_matrix(network, nodelist=None, weight='None').todense())
        print('BlogCatalog.edgelist: A -->',np.sum(A))
        k1 = np.sum(A, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        Eij = k1k2 / (2 * args.m)
        B =np.array(A - Eij)
        print('B -->',B.shape)

        data_mat = sio.loadmat('/home/data/zp/ygm/ComGA/data/BlogCatalog/BlogCatalog.mat')
        adj = data_mat['Network']
        feat = data_mat['Attributes']
        truth = data_mat['Label']
        truth = truth.flatten()
        print('BlogCatalog.mat: adj',np.sum(adj))
        graph = dgl.from_scipy(adj)
        graph=dgl.add_self_loop(graph)
        print('loop graph:\n',graph)
        features = torch.FloatTensor(feat.toarray())
        print('features[1]',features[1])

        A=adj.toarray()
        print('BlogCatalog.edgelist: A -->',np.sum(A))
        k1 = np.sum(A, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        num_loop=0
        for i in range(adj.shape[0]):
            if adj[i,i]==1:
                num_loop+=1
        m=(np.sum(A)-num_loop)/2+num_loop
        Eij = k1k2 / (2 * m)
        B =np.array(A - Eij)
        print('B -->',B.shape)

        adj = adj + sp.eye(adj.shape[0])

        
    
    # data preprocess
    adj_label = torch.FloatTensor(adj.toarray())
    B = torch.FloatTensor(B)
    print('B shape:', B.shape)
    print('adj_label shape:', adj_label.shape)
    print('features shape:', features.shape)
    feat_dim=features.shape[1]
    num_nodes=features.shape[0]

    model = ComGA_Base(num_nodes,feat_dim,args.n_enc_1,args.n_enc_2,args.n_enc_3,args.dropout)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:" + str(args.device))
        print('Using gpu!!!')
    else:
        device = torch.device("cpu")
        print('Using cpu!!!')

    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device)
    B = B.to(device)

    writer = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.num_epoch):
        loss,struct_loss, feat_loss,kl_loss,re_loss,_ = train_step(
            args, model, optimizer, graph, features,B,adj_label,device)
        predict_score = test_step(args, model, graph, features, B, adj_label,device)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
        )),"train/kl_loss=", "{:.5f}".format(kl_loss.item()),
         "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()),
         )
        writer.add_scalars(
            "loss",
            {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
            epoch,
        )
        if test_mode:
            print('auc:',roc_auc_score(truth,predict_score))
        else:
            final_score, a_score, s_score = dataset.evalution(predict_score)
            writer.add_scalars(
                "auc",
                {"final": final_score, "structural": s_score, "attribute": a_score},
                epoch,
            )

        writer.flush()
