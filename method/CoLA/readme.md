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

# UniformNeighborSampling
structural anomaly score: 0.777847276518113
attribute anomaly score: 0.7157857701329163
final anomaly score: 0.7468165233255146

# without Bilinear


