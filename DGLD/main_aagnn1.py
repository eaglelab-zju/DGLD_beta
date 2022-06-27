# Author: Peng Zhang <zzhangpeng@zju.edu.cn>
# License: BSD 2 clause
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.common.evaluation import split_auc
from DGLD.AAGNN import AAGNN
from DGLD.AAGNN import AAGNN_batch
from DGLD.utils.utils import seed_everything,Multidict2dict,ExpRecord

from DGLD.AAGNN import get_parse
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    """[command line mode]
    test command line mode
    """
    tool = Multidict2dict()
    exprecord = ExpRecord("aagnn.csv")
    args = get_parse()
    seeds = [1,42,1024,4096]
    lrs = [1e-3,1e-4,1e-5]
    num_epochs = [100,200,300,400]

    for seed in seeds:
        args['seed']=seed 
        for lr in lrs:
            args["fit"]["lr"] = lr
            for num_epoch in num_epochs:
                args["fit"]["num_epoch"] = num_epoch

                # print(args)
                seed_everything(args['seed'])
                gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
                g = gnd_dataset[0]
                label = gnd_dataset.anomaly_label
                model = AAGNN_batch(**args["model"])
                model.fit(g, **args["fit"])
                result = model.predict(g, **args["predict"])
                
                args_dict = tool.solve(args)
                final_score, a_score, s_score = split_auc(label, result)
                args_dict["auc"] = final_score
                args_dict["attribute_auc"] = a_score
                args_dict["structure_auc"] = s_score
                exprecord.add_record(args_dict)








