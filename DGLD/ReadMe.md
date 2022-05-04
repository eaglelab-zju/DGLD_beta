You can use the GraphAnomalyDectionBenchmarking directly in this way:

```
pip install DGLD
```

​	

Until now, DGLD includes many models such as: AAGNN, AnomaluDAE, CoLA, ComGA, DOMINANT, SL-GAD, infoNCECoLA。



These models are used as follows:

```python
from DGLD.AAGNN import AAGNN_A
from DGLD.AAGNN import AAGNN_M
model = AAGNN_A.model()
model = AAGNN_M.model()
```



Below is an example of a complete trained model and predicted data:

```python
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD import get_parse
from DGLD.AAGNN import AAGNN_A
from DGLD.AAGNN import AAGNN_M
if __name__ == '__main__':
    args = get_parse.get_parse()
    print(args)
    dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    #use the model AAGNN_A
    model = AAGNN_A.model()
    #fit the model and save it
    model.fit(dataset[0], args)
    #load the model you save and to infer the graph dataset
    pred_score = model.infer(dataset[1], args)
    print(pred_score)

```

