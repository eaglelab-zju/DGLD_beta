import dgl
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class OneLayerGCN(nn.Module):
    def __init__(self, in_feats, out_feats=300, bias=True):
        super(OneLayerGCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, bias=bias, norm='none')
        self.act = nn.PReLU()

    def forward(self, bg, in_feat):
        h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
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
        # return subgraph_pool_emb, anchor_out
        return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class CoLAModel(nn.Module):
    def __init__(self, in_feats, out_feats=64, bias=True):
        super(CoLAModel, self).__init__()
        self.gcn = OneLayerGCN(in_feats, out_feats, bias)
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
