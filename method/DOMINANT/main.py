
import sys
sys.path.append('../../')

import scipy.sparse as sp

import torch
from torch.utils.tensorboard import SummaryWriter

from dominant_utils import get_parse, train_step, test_step
from models import Dominant
from common.dataset import GraphNodeAnomalyDectionDataset

if __name__ == '__main__':
    args = get_parse()
    print(args)
    # load dataset
    dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    graph = dataset[0]
    features = graph.ndata['feat']
    adj = graph.adj()
    adj_label = sp.csr_matrix(adj.to_dense()) + sp.eye(adj.shape[0])  # A+I
    adj_label = torch.FloatTensor(adj_label.toarray())
    print(graph)
    print('adj_label shape:', adj_label.shape)
    print('features shape:', features.shape)

    model = Dominant(feat_size=features.shape[1], hidden_size=args.hidden_dim,
                     dropout=args.dropout)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
        model = model.to(device)
        graph = graph.to(device)
        features = features.to(device)
        adj_label = adj_label.to(device)
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.num_epoch):
        loss, struct_loss, feat_loss = train_step(
            args, model, optimizer, graph, features, adj_label)
        predict_score = test_step(args, model, graph, features, adj_label)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
        )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
        writer.add_scalars(
            "loss",
            {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
            epoch,
        )
        final_score, a_score, s_score = dataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )

        writer.flush()
