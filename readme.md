# GraphAnomalyDection Benchmarking
GraphAnomalyDection Benchmarking using DGL
## GraphNodeAnomalyDectionDataset
### Related Paper
| Paper                                                                                                                 | Method |  From   |                        Code                         |
| :-------------------------------------------------------------------------------------------------------------------- | :----: | :-----: | :-------------------------------------------------: |
| [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113) |  CoLA  | TNNLS21 | [Pytorch+DGL0.3](https://github.com/GRAND-Lab/CoLA) |
| [Deep Anomaly Detection on Attributed Networks](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67) |  Dominant  | SDM19 | [Pytorch](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch) |

### Dataset Description
<img src="http://latex.codecogs.com/svg.latex?\begin{array}{c|c|c|c|c}\hline&space;\text&space;{&space;Dataset&space;}&space;&&space;\sharp&space;\text&space;{&space;nodes&space;}&space;&&space;\sharp&space;\text&space;{&space;edges&space;}&space;&&space;\sharp&space;\text&space;{&space;attributes&space;}&space;&&space;\sharp&space;\text&space;{&space;anomalies&space;}&space;\\\hline&space;\text&space;{&space;BlogCatalog&space;}&space;&&space;5,196&space;&&space;171,743&space;&&space;8,189&space;&&space;300&space;\\\text&space;{&space;Flickr&space;}&space;&&space;7,575&space;&&space;239,738&space;&&space;12,407&space;&&space;450&space;\\\text&space;{&space;ACM&space;}&space;&&space;16,484&space;&&space;71,980&space;&&space;8,337&space;&&space;600&space;\\\text&space;{&space;Cora&space;}&space;&&space;2,708&space;&&space;5,429&space;&&space;1,433&space;&&space;150&space;\\\text&space;{&space;Citeseer&space;}&space;&&space;3,327&space;&&space;4,732&space;&&space;3,703&space;&&space;150&space;\\\text&space;{&space;Pubmed&space;}&space;&&space;19,717&space;&&space;44,338&space;&&space;500&space;&&space;600&space;\\\text&space;{&space;ogbn-arxiv&space;}&space;&&space;169,343&space;&&space;1,166,243&space;&&space;128&space;&&space;6000&space;\\\hline\end{array}" title="http://latex.codecogs.com/svg.latex?\begin{array}{c|c|c|c|c}\hline \text { Dataset } & \sharp \text { nodes } & \sharp \text { edges } & \sharp \text { attributes } & \sharp \text { anomalies } \\\hline \text { BlogCatalog } & 5,196 & 171,743 & 8,189 & 300 \\\text { Flickr } & 7,575 & 239,738 & 12,407 & 450 \\\text { ACM } & 16,484 & 71,980 & 8,337 & 600 \\\text { Cora } & 2,708 & 5,429 & 1,433 & 150 \\\text { Citeseer } & 3,327 & 4,732 & 3,703 & 150 \\\text { Pubmed } & 19,717 & 44,338 & 500 & 600 \\\text { ogbn-arxiv } & 169,343 & 1,166,243 & 128 & 6000 \\\hline\end{array}" />

### Reproduced results 

Reported/Reproduced

|                 Reproducer                  |   Method   | BlogCatalog | Flickr  |  cora   | citeseer | pubmed  |   ACM   | ogbn-arxiv |
| :-----------------------------------------: | :--------: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :--------: |
| [@miziha-zp](https://github.com/miziha-zp/) |    CoLA    |   0.7854/   | 0.7513/ | 0.8779/ | 0.8968/  | 0.9512/ | 0.8237/ |  0.8073/   |
|                                             | SL-GAD |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
|                                             | ANEMONE |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
| [@GavinYGM](https://github.com/GavinYGM/) |  DOMINANT  |  0.7813/  |   0.7490/    |    /    |    /     |    /    |    0.7494/    |     /      |
| [@GavinYGM](https://github.com/GavinYGM/) |   ComGA    |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
|                                             | AnomalyDAE |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
|                                             | ALARM |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
|                                             | AAGNN |      /      |    /    |    /    |    /     |    /    |    /    |     /      |



## GraphEdgeAnomalyDectionDataset


## Thanks
Thank you to everyone who contributed anything to this repository.
