# GraphAnomalyDection Benchmarking
GraphAnomalyDection Benchmarking using DGL
## GraphNodeAnomalyDectionDataset
### Related Paper
| Paper                                                                                                                 | Method |  From   |                        Code                         |
| :-------------------------------------------------------------------------------------------------------------------- | :----: | :-----: | :-------------------------------------------------: |
| [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113) |  CoLA  | TNNLS21 | [Pytorch+DGL0.3](https://github.com/GRAND-Lab/CoLA) |
### Reproduced results 
Reported/Reproduced
| Reproducer | Method | BlogCatalog | Flickr  |  cora   | citeseer | pubmed  |   ACM   | ogbn-arxiv |
| :--------: | :----: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :--------: |
| @miziha-zp |  CoLA  |   0.7854/   | 0.7513/ | 0.8237/ | 0.8779/  | 0.8968/ | 0.9512/ |  0.8073/   |
## GraphEdgeAnomalyDectionDataset