import sys
sys.path.append('../../')

import scipy.sparse as sp

import torch
from torch.utils.tensorboard import SummaryWriter

from common.dataset import GraphNodeAnomalyDectionDataset
from collections import Counter
from AAGNN_A import AAGNN_A
from AAGNN_M import AAGNN_M
from get_parse import get_parse
if __name__ == '__main__':
    args = get_parse()
    print(args)
    # load dataset
    dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    graph = dataset[0]
    features = graph.ndata['feat']
    print(graph)
    print('features shape:', features.shape)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    features = features.to(device)
    #初始化定义模型结构
    if args.model == 'AAGNN_A':    
        print('model = AAGNN_A')
        model = AAGNN_A(graph, features.shape[1], 256, device)
    else:
        print('model = AAGNN_M')
        model = AAGNN_M(graph, features.shape[1], 256, device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    #获取伪标签下的正常样本
    mask = model.mask_label(features, 0.5)
    writer = SummaryWriter(log_dir=args.logdir)
    model.train()
    for epoch in range(args.num_epoch):
        out = model(features)
        loss = model.loss_fun(out, mask, model, 0.0001, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        predict_score = model.anomaly_score(out)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
        )))
        writer.add_scalars(
            "loss",
            {"loss": loss},
            epoch,
        )
        final_score, a_score, s_score = dataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )

        writer.flush()