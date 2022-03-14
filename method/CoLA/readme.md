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
paces = dgl.sampling.random_walk(g, start_nodes, **length=self.length**, restart_prob=0)[0]
