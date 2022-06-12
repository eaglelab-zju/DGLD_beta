# Installation
## Dependencies
In order to ensure a lightweight installation package, only the necessary components of DGLD are included in our installation packages, other installation dependencies need to be installed first, as specified below.

Required Dependencies:

```bash
joblib==1.1.0
networkx==2.7.1
numpy==1.21.2
ogb==1.3.3
pandas==1.4.1
scikit_learn==1.1.1
scipy==1.8.0
termcolor==1.1.0
texttable==1.6.4
tqdm==4.63.0
```
In addition to the above dependency packages, PyTorch and DGL is needed, we really recommend you to follow the instructions from [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) official website. The minimum version required is as follows(and GPU version is recommended):
```bash
dgl_cu113==0.8.1 %(GPU)
torch==1.11.0+cu113 %(GPU)

dgl==0.8.1 %(CPU)
torch==1.11.0 %(CPU)
```

## pip or conda:
