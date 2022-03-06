# GraphAnomalyDection Benchmarking
GraphAnomalyDection Benchmarking using DGL
## GraphNodeAnomalyDectionDataset
### Related Paper
| Paper                                                                                                                 | Method |  From   |                        Code                         |
| :-------------------------------------------------------------------------------------------------------------------- | :----: | :-----: | :-------------------------------------------------: |
| [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113) |  CoLA  | TNNLS21 | [Pytorch+DGL0.3](https://github.com/GRAND-Lab/CoLA) |
### Dataset Description
$$
\begin{array}{c|c|c|c|c}
\hline \text { Dataset } & \sharp \text { nodes } & \sharp \text { edges } & \sharp \text { attributes } & \sharp \text { anomalies } \\
\hline \text { BlogCatalog } & 5,196 & 171,743 & 8,189 & 300 \\
\text { Flickr } & 7,575 & 239,738 & 12,407 & 450 \\
\text { ACM } & 16,484 & 71,980 & 8,337 & 600 \\
\text { Cora } & 2,708 & 5,429 & 1,433 & 150 \\
\text { Citeseer } & 3,327 & 4,732 & 3,703 & 150 \\
\text { Pubmed } & 19,717 & 44,338 & 500 & 600 \\
\text { ogbn-arxiv } & 169,343 & 1,166,243 & 128 & 6000 \\
\hline
\end{array}
$$
### Reproduced results 
Reported/Reproduced
| Reproducer | Method | BlogCatalog | Flickr  |  cora   | citeseer | pubmed  |   ACM   | ogbn-arxiv |
| :--------: | :----: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :--------: |
| [@miziha-zp](https://github.com/miziha-zp/) |  CoLA  |   0.7854/   | 0.7513/ | 0.8237/ | 0.8779/  | 0.8968/ | 0.9512/ |  0.8073/   |
## GraphEdgeAnomalyDectionDataset


## Thanks
Thank you to everyone who contributed anything to this repository.