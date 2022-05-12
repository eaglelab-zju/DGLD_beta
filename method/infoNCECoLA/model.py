import dgl
import math
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
            nn.Linear(out_feats, out_feats)
        )
        self.gcn2anchor = torch.nn.Sequential(nn.Linear(out_feats, out_feats))
        self.anchormlp = torch.nn.Sequential(nn.Linear(out_feats, out_feats))
        self.pool = AvgPooling()
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

class CoLAModel(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=True, tau=0.5, generative_loss_w=0, score_type="score1"):
        super(CoLAModel, self).__init__()
        self.gcn = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        self.discriminator = Discriminator(out_feats)
        self.tau = tau
        self.beta = generative_loss_w
        self.score_type = score_type
        self.attr_mlp = torch.nn.Sequential(nn.Linear(out_feats, out_feats),
        nn.ReLU(),
        nn.Linear(out_feats, in_feats),
        nn.ReLU())
        
    def infonceloss(self, pos_emb, neg_emb, anchor_emb, score_type='score1'):
        r"""
        Parameter:
        ----------
        \begin{align}
            x = \sum^{B}_{j=1}{e^{\widetilde{eg}_{j}^{T}z_{i}/\tau}}, y=e^{eg_{i}^{l}z_{i}}
        \end{align}

        score1:
        -------
        \begin{align}
            score1 = x / B - y
        \end{align}

        scoreloss:
        ----------
        \begin{align}
            score_{loss} = -log(y/x) = logx - logy
        \end{align}

        scorelossfixbatch:
        ------------------
        \begin{align}
            score_{loss2} = -log(y/(\frac{1}{B}x)) = logx - logy - logB
        \end{align}

        score2:
        -------
        \begin{align}
            score_{loss3} = \frac{logx}{B} - logy
        \end{align}

        scorelossfixbatch:
        ------------------
        \begin{align}
            score_{loss4} = log(\frac{x}{B}) - logy = logx - logy - logB 
        \end{align}
        """
        tau = self.tau
        epsison = 0.00001
        y = torch.exp(torch.sum(pos_emb*anchor_emb, dim=1) / tau)
        key_emb = neg_emb
        x = torch.exp(anchor_emb @ key_emb.T / tau).sum(1)
        B = x.shape[0]
        loss = -torch.log(y / (x + epsison))
        if self.score_type == "score1":
            return loss, y, x / B 
        elif self.score_type == "scoreloss":
            return loss, torch.log(y), torch.log(x)
        elif self.score_type == "scorelossfixbatch":
            return loss, torch.log(y), torch.log(x)-math.log(B)
        elif self.score_type == "score2":
            return loss, torch.log(y), torch.log(x) / B 
        else:
            raise NotImplementedError

    def get_anchor_oral_features(self, bg, feat):
        unbatchg = dgl.unbatch(bg)
        anchor_feat_list = []
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            anchor_feat_list.append(anchor_feat)        
        # anchor_input
        anchor_embs = torch.stack(anchor_feat_list, dim=0)
        return anchor_embs

    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat):
        # pos_in_feat = F.dropout(pos_in_feat, 0.2, training=self.training)
        # neg_in_feat = F.dropout(neg_in_feat, 0.2, training=self.training)
        anchor_inputs = self.get_anchor_oral_features(pos_batchg, pos_in_feat)
        pos_pool_emb, pos_anchor_out, pos_gcn_emb = self.gcn(pos_batchg, pos_in_feat)
        neg_pool_emb, neg_anchor_out, neg_gcn_emb = self.gcn(neg_batchg, neg_in_feat)      
        loss_pool, pos_score_pool, neg_score_pool = self.infonceloss(pos_pool_emb,  \
            neg_pool_emb, pos_anchor_out)

        loss = loss_pool
        pos_score = pos_score_pool 
        neg_score = neg_score_pool
        return loss, pos_score, neg_score

