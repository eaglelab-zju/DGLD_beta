from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.DOMINANT import Dominant
from DGLD.DOMINANT import get_parse
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
    # model = Dominant(feat_size=1433, hidden_size=64, dropout=0.3)
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g)
    # print(split_auc(label, result))

    # """
    # custom dataset
    # """
    # gnd_dataset = GraphNodeAnomalyDectionDataset("ACM")
    # g = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # model = Dominant(feat_size=8337, hidden_size=64, dropout=0.3)
    # model.fit(g, num_epoch=1, device='4')
    # result = model.predict(g,device='4')
    # print(split_auc(label, result))
    """[command line mode]
    test command line mode
    """
    tool = Multidict2dict()
    exprecord = ExpRecord("dominant.csv")
    args = get_parse()
    seeds = [4096]
    alphas = [0.1,0.01,0.02]
    num_epochs = [300,400]
    lrs = [5e-4,1e-4,1e-3]
    hidden_sizes = [8,16,32,64,128]
    dropouts = [0.1,0.2,0.3,0.5]

    for seed in seeds:
        args['seed']=seed
        for alpha in alphas:
            args["fit"]["alpha"] = alpha
            args["predict"]["alpha"] = alpha
            for n_epoch in num_epochs:
                args["fit"]["num_epoch"] = n_epoch
                for lr in lrs:
                    args["fit"]["lr"] = lr
                    for hidden_size in hidden_sizes:
                        args["model"]["hidden_size"] = hidden_size
                        for p_drop in dropouts:
                            args["model"]["dropout"] = p_drop
                            # print(args)
                            seed_everything(args['seed'])
                            gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
                            g = gnd_dataset[0]
                            label = gnd_dataset.anomaly_label
                            model = Dominant(**args["model"])
                            model.fit(g, **args["fit"])
                            result = model.predict(g, **args["predict"])

                            args_dict = tool.solve(args)
                            final_score, a_score, s_score = split_auc(label, result)
                            args_dict["auc"] = final_score
                            args_dict["attribute_auc"] = a_score
                            args_dict["structure_auc"] = s_score
                            exprecord.add_record(args_dict)

