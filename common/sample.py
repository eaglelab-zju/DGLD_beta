import dgl
import torch
import numpy as np
from dgl import transform

from .utils import move_start_node_fisrt

# TODO CoLA Sample
class BaseSubGraphSampling:
    r"""An abstract class for writing transforms."""

    def __call__(self, g, start_node):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"

class UniformNeighborSampling(BaseSubGraphSampling):
    r"""
    Uniform sampling Neighbors to generate subgraph.
    """
    def __init__(self, length=4):
        self.length = 4

    def __call__(self, g, start_nodes):
        rwl = []
        for node in start_nodes:
            pace = [node]
            successors = g.successors(node).numpy().tolist()
            # remove node and shuffle
            successors.remove(node)
            np.random.shuffle(successors)
            pace += successors[:self.length-1]
            pace += [pace[0]] * max(0, self.length - len(pace))
            rwl.append(pace)
        return rwl

class CoLASubGraphSampling(BaseSubGraphSampling):
    r"""
    ''
    we adopt random walk with restart (RWR)
    as local subgraph sampling strategy due to its usability and efficiency.
    we fixed the size 𝑆 of the sampled subgraph (number of nodes in the subgraph) to 4.
    For isolated nodes or the nodes which belong to a community with a size smaller than
    the predetermined subgraph size, we sample the available nodes repeatedly until an
    overlapping subgraph with the set size is obtained."
    described in [CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)
    Parameters:
    -----------
    g: DGLGraph object

    start_nodes: a Tensor or array contain start node.
    Return:
    -------
    rwl: List[List]
    examples:
    ---------
        ```python
        cola_sampler = CoLASubGraphSampling()
        g = dgl.graph(([0, 1, 2, 3, 6], [1, 2, 3, 4, 0]))
        g = dgl.add_reverse_edges(g)
        g = dgl.add_self_loop(g)
        ans = cola_sampler(g, [1, 2, 3, 5])
        print(ans)
        # [[1, 0, 2, 3], [2, 1, 0, 6], [3, 1, 2, 0], [5, 5, 5, 5]]
        ```
    """

    def __init__(self, length=4):
        self.length = 4

    def __call__(self, g, start_nodes):
        """
        add self_loop to handle isolated nodes as soon as
        the nodes which belong to a community with a size smaller than
        it is a little different from author's paper.
        """
        # newg = dgl.remove_self_loop(g)
        # newg = dgl.add_self_loop(newg)
        # length is Very influential to the effect of the model, maybe caused "remote" neighbor is 
        # not "friendly" to Rebuild Properties.
        paces = dgl.sampling.random_walk(g, start_nodes, length=self.length * 3, restart_prob=0)[0]
        rwl = []
        for start, pace in zip(start_nodes, paces):
            pace = pace.unique().numpy()[: self.length].tolist()
            pace = move_start_node_fisrt(pace, start)
            pace += [pace[0]] * max(0, self.length - len(pace))
            rwl.append(pace)
        return rwl


if __name__ == "__main__":
    cola_sampler = CoLASubGraphSampling()
    g = dgl.graph(([0, 1, 2, 3, 6], [1, 2, 3, 4, 0]))
    g = dgl.add_reverse_edges(g)
    # g = dgl.add_self_loop(g) for isolated nodes 
    ans = cola_sampler(g, [1, 2, 3, 5])
    print(ans)
    # [[1, 0, 2, 3], [2, 1, 0, 6], [3, 1, 2, 0], [5, 5, 5, 5]]
    subg = dgl.node_subgraph(g, ans[0])
    print(subg)