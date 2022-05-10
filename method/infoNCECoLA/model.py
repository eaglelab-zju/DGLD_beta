import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  dgl.nn.pytorch import EdgeWeightNorm
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling

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
    def __init__(self, in_feats, out_feats=64, global_adg=True, bias_term=True):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias_term:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
            self.bias_term = bias_term
        self.conv = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        self.subg2anchor = torch.nn.Sequential(
            nn.Linear(out_feats, out_feats),
            # nn.PReLU(),
            # nn.Linear(out_feats, out_feats)
        )
        self.gcn2anchor = torch.nn.Sequential(nn.Linear(out_feats, out_feats))
        self.anchormlp = torch.nn.Sequential(nn.Linear(out_feats, out_feats))
        
        self.act = nn.PReLU()
        self.reset_parameters()
        self.pool = AvgPooling()

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
        anchor_embs = bg.ndata['feat'][::4, :].clone()
        # Anonymization
        bg.ndata['feat'][::4, :] = 0
        # anchor_out
        anchor_out = torch.matmul(anchor_embs, self.weight) 
        if self.bias_term:
            anchor_out = anchor_out + self.bias

        anchor_out = self.act(anchor_out)
        
        in_feat = bg.ndata['feat']
        in_feat = torch.matmul(in_feat, self.weight) 
        # GCN
        if self.global_adg:
            h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
        else:
            h = self.conv(bg, in_feat)

        if self.bias_term:
            h += self.bias

        h = self.act(h)

        with bg.local_scope():
            # pooling        
            bg.ndata["h"] = h
            subgraph_pool_emb = self.pool(bg,h)
            gcn_emb = bg.ndata['h'][::4, :].clone()
        
        # return subgraph_pool_emb, anchor_out
        subgraph_pool_emb = self.subg2anchor(subgraph_pool_emb)
        gcn_emb = self.gcn2anchor(gcn_emb)
        # anchor_out = self.anchormlp(anchor_out)
        return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1),\
            F.normalize(gcn_emb, p=2, dim=1)

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def dropout_feature(x: torch.FloatTensor, drop_prob: float = 0.2) -> torch.FloatTensor:
    return F.dropout(x, p=1. - drop_prob)

class CoLAModel(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=True, tau=0.5, generative_loss_w=0):
        super(CoLAModel, self).__init__()
        self.gcn = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        self.discriminator = Discriminator(out_feats)
        self.tau = tau
        self.beta = generative_loss_w

        self.attr_mlp = torch.nn.Sequential(nn.Linear(out_feats, out_feats),
        nn.ReLU(),
        nn.Linear(out_feats, in_feats),
        nn.ReLU())
        
    def infonceloss(self, pos_emb, neg_emb, anchor_emb):
        tau = self.tau
        epsison = 0.00001
        pos_score = torch.exp(torch.sum(pos_emb*anchor_emb, dim=1) / tau)
        # neg_score = torch.exp(torch.sum(neg_emb*anchor_emb, dim=1) / tau)
        # key_emb = torch.cat([pos_emb, neg_emb], dim=0)
        key_emb = neg_emb
        neg_score_all = torch.exp(anchor_emb @ key_emb.T / tau).sum(1)
        
        neg_score = neg_score_all / neg_score_all.shape[0]
        loss = -torch.log(pos_score / (neg_score_all+epsison))#torch.Size([n_nodes])
        return loss, pos_score, neg_score

    def aug_infonceloss(self, pos_emb, neg_emb,neg_aug_emb, anchor_emb):
        tau = self.tau
        epsison = 0.00001
        pos_score = torch.exp(torch.sum(pos_emb*anchor_emb, dim=1) / tau)
        # neg_score = torch.exp(torch.sum(neg_emb*anchor_emb, dim=1) / tau)
        key_emb = torch.cat([neg_emb, neg_aug_emb], dim=0)
        # key_emb = neg_emb
        neg_score_all = torch.exp(anchor_emb @ key_emb.T / tau).sum(1)
        
        neg_score = neg_score_all / neg_score_all.shape[0]
        loss = -torch.log(pos_score / (neg_score_all+epsison))#torch.Size([n_nodes])
        return loss, pos_score, neg_score

    def generative_loss(self, anchors_oral, anchors_pred_pos, anchors_pred_neg):
        # tau = self.tau
        # epsison = 0.00001
        loss = torch.norm(anchors_oral - anchors_pred_pos) / anchors_oral.shape[1]
        pos_score = -loss
    
        return loss, pos_score, 0

    def get_anchor_oral_features(self, bg, feat):
        unbatchg = dgl.unbatch(bg)
        anchor_feat_list = []
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            anchor_feat_list.append(anchor_feat)        
        # anchor_input
        anchor_embs = torch.stack(anchor_feat_list, dim=0)
        return anchor_embs

    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat,neg_aug_batchg, neg_aug_feat):
        pos_in_feat = F.dropout(pos_in_feat, 0.2, training=self.training)
        neg_in_feat = F.dropout(neg_in_feat, 0.2, training=self.training)
        anchor_inputs = self.get_anchor_oral_features(pos_batchg, pos_in_feat)


        pos_pool_emb, pos_anchor_out, pos_gcn_emb = self.gcn(pos_batchg, pos_in_feat)
        neg_pool_emb, neg_anchor_out, neg_gcn_emb = self.gcn(neg_batchg, neg_in_feat)      
        neg_aug_pool_emb, neg_aug_anchor_out, neg_aug_gcn_emb = self.gcn(neg_aug_batchg, neg_aug_feat)  
        
        loss_pool, pos_score_pool, neg_score_pool = self.infonceloss( pos_pool_emb,  \
            neg_pool_emb, pos_anchor_out)
        # loss_gcn, pos_score_gcn, neg_score_gcn = self.infonceloss(pos_gcn_emb, \
        #     neg_gcn_emb, pos_anchor_out)


        loss_gen, pos_score_gen, neg_score_gen = self.generative_loss(anchor_inputs, self.attr_mlp(pos_gcn_emb),  self.attr_mlp(neg_gcn_emb))
        # gcn_mapself.attr_mlp 
        beta = self.beta
        loss = loss_pool + loss_gen*beta# + loss_gcn
        # print(loss_pool.mean().item(), loss_gen.mean().item()*beta)
        pos_score = pos_score_pool + pos_score_gen*beta# + pos_score_gcn
        neg_score = neg_score_pool + neg_score_gen*beta# + neg_score_gcn
        return loss, pos_score, neg_score
    # def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat):
    #     pos_in_feat = F.dropout(pos_in_feat, 0.2, training=self.training)
    #     pos_pool_emb, anchor_out, pos_gcn_emb = self.gcn(pos_batchg, pos_in_feat)
    #     neg_pool_emb, _, pos_gcn_emb = self.gcn(neg_batchg, neg_in_feat)      
    #     # pos_pool_emb, anchor_out
    #     # batch * embeddingsz, batch * embeddingsz
    #     # print(pos_pool_emb.shape, anchor_out.shape)
    #     tau = self.tau
    #     epsison = 0.00001
    #     pos_score = torch.exp(torch.sum(pos_pool_emb*anchor_out, dim=1) / tau)
    #     neg_score = torch.exp(torch.sum(neg_pool_emb*anchor_out, dim=1) / tau)

    #     neg_score_all = torch.exp(anchor_out @ pos_pool_emb.T / tau).sum(1)
    #     neg_score_all2 = torch.exp(pos_pool_emb @ anchor_out.T / tau).sum(1)
    #     loss1 = -torch.log(pos_score / (neg_score_all+epsison))
    #     loss2 = -torch.log(pos_score / (neg_score_all2+epsison))
    #     loss = loss1
    #     return loss, pos_score, neg_score


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
