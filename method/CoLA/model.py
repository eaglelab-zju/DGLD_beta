import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  dgl.nn.pytorch import EdgeWeightNorm
class Discriminator(nn.Module):
    def __init__(self, out_feats):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(out_feats, out_feats, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, readout_emb, anchor_emb):
        logits = self.bilinear(readout_emb, anchor_emb)
        return logits


class OneLayerGCNWithGlobalAdg(nn.Module):
    r"""
    a onelayer subgraph GCN can use global adjacent metrix.
    """
    def __init__(self, in_feats, out_feats=64, global_adg=True):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.conv = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        self.act = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, bg, in_feat):
        # Anonymization
        unbatchg = dgl.unbatch(bg)
        unbatchg_list = []
        anchor_feat_list = []
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            g.ndata['feat'][0, :] = 0
            unbatchg_list.append(g)
            anchor_feat_list.append(anchor_feat)
        
        # anchor_out
        anchor_embs = torch.stack(anchor_feat_list, dim=0)
        anchor_out = torch.matmul(anchor_embs, self.weight) + self.bias

        bg = dgl.batch(unbatchg_list)
        in_feat = bg.ndata['feat']
        in_feat = torch.matmul(in_feat, self.weight) 
        # GCN
        if self.global_adg:
            h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
        else:
            h = self.conv(bg, in_feat)
        h += self.bias
        h = self.act(h)
        with bg.local_scope():
            # pooling        
            bg.ndata["h"] = h
            subgraph_pool_emb = []
            unbatchg = dgl.unbatch(bg)
            for g in unbatchg:
                subgraph_pool_emb.append(torch.mean(g.ndata["h"], dim=0))
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
        # return subgraph_pool_emb, anchor_out
        return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class OneLayerGCN(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, bias=True):
        super(OneLayerGCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, bias=bias)
        self.act = nn.PReLU()

    def forward(self, bg, in_feat):
        h = self.conv(bg, in_feat)
        h = self.act(h)
        with bg.local_scope():
            bg.ndata["h"] = h
            # subgraph_pool_emb = dgl.mean_nodes(bg, "h")
            subgraph_pool_emb = []
            # get anchor embedding
            unbatchg = dgl.unbatch(bg)
            anchor_out = []
            for g in unbatchg:
                subgraph_pool_emb.append(torch.mean(g.ndata["h"][:-1], dim=0))
                anchor_out.append(g.ndata["h"][-1])
            anchor_out = torch.stack(anchor_out, dim=0)
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
        return subgraph_pool_emb, anchor_out
        # return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class CoLAModel(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=True):
        super(CoLAModel, self).__init__()
        self.gcn = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        self.discriminator = Discriminator(out_feats)

    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat):
        pos_pool_emb, anchor_out = self.gcn(pos_batchg, pos_in_feat)
        neg_pool_emb, _ = self.gcn(neg_batchg, neg_in_feat)
        pos_scores = self.discriminator(pos_pool_emb, anchor_out)
        neg_scores = self.discriminator(neg_pool_emb, anchor_out)
        return pos_scores[:, 0], neg_scores[:, 0]


if __name__ == "__main__":
    # sample
    model = CoLAModel(5)
    g1 = dgl.graph(([1, 2, 3], [2, 3, 1]))
    g1 = dgl.add_self_loop(g1)
    g1.ndata["feat"] = torch.rand((4, 5))
    g2 = dgl.graph(([3, 2, 4], [2, 3, 1]))
    g2 = dgl.add_self_loop(g2)
    g2.ndata["feat"] = torch.rand((5, 5))
    bg = dgl.batch([g1, g2])
    bg2 = dgl.batch([g2, g1])

    ans = model(bg, bg.ndata["feat"], bg2, bg2.ndata["feat"])
    print(ans)
    # g.ndata['feat'] = torch.rand((4, 5))
    # print(g.ndata['feat'])
    # subg = dgl.node_subgraph(g, [1,2])
    # print(subg.ndata['feat'])
    # subg.ndata['feat'] = torch.zeros((2, 5))
    # print(subg.ndata['feat'])
    # print(g.ndata['feat'])
