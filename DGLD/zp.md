# 模型接口
 接口的设计上，我们考虑两个问题
## Q1:如何让喜欢使用sklearn的用户🤔适应我们的接口
```python
classlightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, ...)[source]
fit(X, y, sample_weight=None, ...)[source]
predict(X, raw_score=False, ...)[source]
predict_proba(X, raw_score=False, ...)[source]

# train a dominant detector
from pygod.models import DOMINANT
model = DOMINANT(num_layers=4, epoch=20)  # hyperparameters can be set here
model.fit(data)  # data is a Pytorch Geometric data object
# get outlier scores on the input data
outlier_scores = model.decision_scores # raw outlier scores on the input data
# predict on the new data in the inductive setting
outlier_scores = model.decision_function(test_data) # raw outlier scores on the input data  # predict raw outlier scores on test
```
## Q2:如何researcher喜欢基于我们的工作，继续探索
科研工作者需要做什么
1. 快速复现这个任务下的benchmark
2. 基于现有代码快速实验，验证自己的想法
思路是，保证代码耦合度低，可以由命令行快速进行实验, 例如
```bash
for data in Cora Citeseer ACM Pubmed ogbn-arxiv
do
  for tau in 0.5
  do
    expname=$data'bs4096info_=tau'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=5 python main.py --score_type scorelossfixbatch --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 
  done
done
```

```python # CoLA demo
# Author: Peng Zhang <zzhangpeng@zju.edu.cn>
# License: BSD 2 clause
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.CoLA import CoLA
from DGLD.CoLA import get_parse
from DGLD.common.evaluation import split_auc

import dgl
import torch
import numpy as np

if __name__ == '__main__':
    """
    sklearn-like API for most users.
    """
    """
    using GraphNodeAnomalyDectionDataset 
    """
    gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = CoLA(in_feats=1433)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g, auc_test_rounds=2)
    print(split_auc(label, result))

    """
    custom dataset
    """
    g = dgl.graph((torch.tensor([0, 1, 2, 4, 6, 7]), torch.tensor([3, 4, 5, 2, 5, 2])))
    g.ndata['feat'] = torch.rand((8, 4))
    label = np.array([1, 2, 0, 0, 0, 0, 0, 0])
    model = CoLA(in_feats=4)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g, auc_test_rounds=2)
    print(split_auc(label, result))
    
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    print(args)
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = CoLA(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)
```
## reference
1. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html