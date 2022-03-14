1. Instance Pair Sampling
1.1 Target Node Selection
1.2 Anonymization
2. GNN-based Contrastive Learning Model
3. Anomaly Score Computation

result:
BCE: 

python main.py --dataset ogbn-arxiv --lr 0.001 --logdir=log/ogbn-arxiv-ga-l2




#  CoLASubGraphSampling
CoLASubGraphSampling 
paces = dgl.sampling.random_walk(g, start_nodes, **length=self.length*3**, restart_prob=0)[0]

## exp1
structural anomaly score: 0.9341256189731562
attribute anomaly score: 0.8516757883763357
final anomaly score: 0.8929007036747458
## exp2
structural anomaly score: 0.9166379984362785
attribute anomaly score: 0.8629137346885588
final anomaly score: 0.8897758665624186
## exp3
==========mean==========
structural anomaly score: 0.9360854834506125
attribute anomaly score: 0.8448214751107636
final anomaly score: 0.8904534792806881
==========std==========
structural anomaly score: 0.5422986708365911
attribute anomaly score: 0.3623507948918425
final anomaly score: 0.4523247328642168
==========mean+std==========
structural anomaly score: 0.9382486317435497
attribute anomaly score: 0.8238050560333594
final anomaly score: 0.8810268438884544
==========mean-std==========
structural anomaly score: 0.906260099035705
attribute anomaly score: 0.8457753453218662
final anomaly score: 0.8760177221787855
==========max==========
structural anomaly score: 0.7517122752150117
attribute anomaly score: 0.5916653635652854
final anomaly score: 0.6716888193901487
==========min==========
structural anomaly score: 0.7517748240813135
attribute anomaly score: 0.7793328120927807
final anomaly score: 0.7655538180870471
==========median==========
structural anomaly score: 0.9193015376596299
attribute anomaly score: 0.8428824602554079
final anomaly score: 0.8810919989575188
# UniformNeighborSampling
structural anomaly score: 0.777847276518113
attribute anomaly score: 0.7157857701329163
final anomaly score: 0.7468165233255146

# without Bilinear


# CoLA-Citeseer
==========mean==========
structural anomaly score: 0.8712328192214879
attribute anomaly score: 0.8464757108383172
final anomaly score: 0.8588542650299025
==========std==========
structural anomaly score: 0.8653404679467002
attribute anomaly score: 0.48982478228937154
final anomaly score: 0.6775826251180359
==========mean+std==========
structural anomaly score: 0.9134067778826985
attribute anomaly score: 0.8230699821634664
final anomaly score: 0.8682383800230826
==========mean-std==========
structural anomaly score: 0.7188248872101564
attribute anomaly score: 0.8464841045011017
final anomaly score: 0.7826544958556289
==========mean+median==========
structural anomaly score: 0.8599727205959501
attribute anomaly score: 0.8510754380442765
final anomaly score: 0.8555240793201134
==========max==========
structural anomaly score: 0.8881418529010596
attribute anomaly score: 0.6856531318854264
final anomaly score: 0.786897492393243
==========min==========
structural anomaly score: 0.48441086979330605
attribute anomaly score: 0.7324939670548736
final anomaly score: 0.6084524184240898
==========median==========
structural anomaly score: 0.841783653341727
attribute anomaly score: 0.8499003252544329
final anomaly score: 0.84584198929808