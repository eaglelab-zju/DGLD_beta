
import sys
sys.path.append('.\\.\\')
# print(sys.path)
import scipy.sparse as sp

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy

# from dominant_utils import get_parse, train_step, test_step
# from models import Dominant
from SL_GAD_utlis import get_parse
from common.dataset import GraphNodeAnomalyDectionDataset
from dgl.dataloading import GraphDataLoader
from utils.utils import seed_everything
from dataset import SL_GAD_DataSet
from model import SL_GAD_Model
from SL_GAD_utlis import get_parse, train_epoch, test_epoch

if __name__ == '__main__':
    args = get_parse()
    seed_everything(args.seed)
    print(args)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # load dataset
    dataset = SL_GAD_DataSet(args.dataset)
    # dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    # dataset = dataset[0]
    print(dataset)
    print(len(dataset))
    
    train_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True,
        # num_samples = 0,
    )
    test_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        # num_samples = 0,
    )
    graph = dataset[0]
    graph = graph[0]
    print(graph)
    features = graph.ndata['feat']
    print(features)
    print(features.shape)
    print(graph.ndata['label'])

    print(graph.ndata['label'].shape)
    
    print(graph.ndata['anomaly_label'])
    print(graph.ndata['anomaly_label'].shape)
    
    model = SL_GAD_Model(
        in_feats=dataset[0][0].ndata["feat"].shape[1],
        out_feats=args.embedding_dim,
        global_adg=args.global_adg,
        args = args,
    ).to(device)

    print(model)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # train
    writer = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.num_epoch):
        train_loader.dataloader.dataset.random_walk_sampling()
        # exit()
        loss_accum = train_epoch(
            epoch, args, train_loader, model, device, criterion, optimizer
        )
        writer.add_scalar("loss", float(loss_accum), epoch)
        predict_score = test_epoch(
            epoch, args, test_loader, model, device, criterion, optimizer
        )
        
        final_score, a_score, s_score = dataset.oraldataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )
        writer.flush()

    # multi-round test
    predict_score_arr = []
    for rnd in range(args.auc_test_rounds):
        test_loader.dataloader.dataset.random_walk_sampling()
        predict_score = test_epoch(
            rnd, args, test_loader, model, device, criterion, optimizer
        )
        predict_score_arr.append(list(predict_score))

    predict_score_arr = numpy.array(predict_score_arr).T
    dataset.oraldataset.evaluation_multiround(predict_score_arr)
    
    # mean_predict_result = predict_score_arr.mean(1)
    # std_predict_result = predict_score_arr.std(1)
    # max_predict_result = predict_score_arr.max(1)
    # min_predict_result = predict_score_arr.min(1)
    # median_predict_result = numpy.median(predict_score_arr, 1)

    # descriptions = {
    #     "mean": mean_predict_result,
    #     "std": std_predict_result,
    #     "mean+std": mean_predict_result + std_predict_result,
    #     "mean-std": mean_predict_result - std_predict_result,
    #     "mean+median": mean_predict_result + median_predict_result,
    #     "max": max_predict_result,
    #     "min": min_predict_result,
    #     "median": median_predict_result,
    # }
    # for stat in descriptions:
    #     print("=" * 10 + stat + "=" * 10)
    #     dataset.oraldataset.evalution(descriptions[stat])
    