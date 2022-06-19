from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.AnomalyDAE import AnomalyDAE
from DGLD.AnomalyDAE import get_parse
from DGLD.common.evaluation import split_auc
from DGLD.common.utils import load_ACM
from DGLD.utils.utils import seed_everything,Multidict2dict,ExpRecord
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    # """
    # sklearn-like API for most users.
    # """
    # """
    # using GraphNodeAnomalyDectionDataset 
    # """
    # gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    # g = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # model = AnomalyDAE(feat_size=1433,num_nodes=2708,embed_dim=256,out_dim=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g,device='cpu')
    # print(split_auc(label, result))

    # """
    # custom dataset
    # """
    # g=load_ACM()[0]
    # label = g.ndata['label']
    # model = AnomalyDAE(feat_size=8337,num_nodes=16484,embed_dim=256,out_dim=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='0')
    # result = model.predict(g,device='0')
    # print(split_auc(label, result))
    """[command line mode]
    test command line mode
    """
    tool = Multidict2dict()
    exprecord = ExpRecord("anomalydae.csv")
    args = get_parse()
    seeds = [42]
    embed_dims = [256]
    out_dims = [128]
    dropouts = [0.2]
    lrs = [1e-3,1e-4,1e-5]
    num_epochs = [100,200,300,400]
    alphas = [0.01,0.1,0.2,0.4,0.6,0.8]
    etas = [1.0,3.0,5.0,7.0,10.0]
    thetas = [10.0,20.0,40.0,60.0,90.0]


    for seed in seeds:
        args['seed']=seed 
        for embed_dim in embed_dims:
            args["model"]["embed_dim"] = embed_dim
            for out_dim in out_dims:
                args["model"]["out_dim"] = out_dim
                for p_drop in dropouts:
                    args["model"]["dropout"] = p_drop
                    for lr in lrs:
                        args["fit"]["lr"] = lr
                        for num_epoch in num_epochs:
                            args["fit"]["num_epoch"] = num_epoch
                            for alpha in alphas:
                                args["fit"]["alpha"] = alpha
                                args["predict"]["alpha"] = alpha
                                for eta in etas:
                                    args["fit"]["eta"] = eta
                                    args["predict"]["eta"] = eta
                                    for theta in thetas:
                                        args["fit"]["theta"] = theta
                                        args["predict"]["theta"] = theta

                                        # print(args)
                                        seed_everything(args['seed'])
                                        gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
                                        g = gnd_dataset[0]
                                        label = gnd_dataset.anomaly_label
                                        model = AnomalyDAE(**args["model"])
                                        model.fit(g, **args["fit"])
                                        result = model.predict(g, **args["predict"])

                                        args_dict = tool.solve(args)
                                        final_score, a_score, s_score = split_auc(label, result)
                                        args_dict["auc"] = final_score
                                        args_dict["attribute_auc"] = a_score
                                        args_dict["structure_auc"] = s_score
                                        exprecord.add_record(args_dict)



