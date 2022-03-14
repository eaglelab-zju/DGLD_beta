import torch
from ogb.nodeproppred import DglNodePropPredDataset

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return torch.allclose(a.float().cpu(),
            b.float().cpu(), rtol=rtol, atol=atol)

def move_start_node_fisrt(pace, start_node):
    """
    return a new pace in which the start node is in the first place.
    """
    if pace[0] == start_node:return pace
    for i in range(1, len(pace)):
        if pace[i] == start_node:
            pace[i] = pace[0]
            break
    pace[0] = start_node
    return pace

def is_bidirected(g):
    """Return whether the graph is a bidirected graph.
    A graph is bidirected if for any edge :math:`(u, v)` in :math:`G` with weight :math:`w`,
    there exists an edge :math:`(v, u)` in :math:`G` with the same weight.
    """
    src, dst = g.edges()
    num_nodes = g.num_nodes()

    # Sort first by src then dst
    idx_src_dst = src * num_nodes + dst
    perm_src_dst = torch.argsort(idx_src_dst, dim=0, descending=False)
    src1, dst1 = src[perm_src_dst], dst[perm_src_dst]

    # Sort first by dst then src
    idx_dst_src = dst * num_nodes + src
    perm_dst_src = torch.argsort(idx_dst_src, dim=0, descending=False)
    src2, dst2 = src[perm_dst_src], dst[perm_dst_src]

    return allclose(src1, dst2) and allclose(src2, dst1)

def load_ogbn_arxiv():
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    graph, _ = data[0]
    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]

def load_BlogCatalog():
    pass

def load_Flickr():
    pass

def load_ACM():
    pass
r"""
cd CoLA
python main.py --dataset ACM
"""
if __name__ == "__main__":
    load_ogbn_arxiv()