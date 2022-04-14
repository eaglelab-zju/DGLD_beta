import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  dgl.nn.pytorch import EdgeWeightNorm
import math
import numpy

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
    def __init__(self, in_feats, out_feats=64, global_adg=True, args = None):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.conv = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        if args.act_function == "PReLU":
            self.act = nn.PReLU()
        elif args.act_function == "ReLU":
            self.act = nn.ReLU()
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
        # bg.ndata['feat'] = in_feat
        unbatchg = dgl.unbatch(bg)
        unbatchg_list = []
        anchor_feat_list = []
        u = 0
        for g in unbatchg:
            # u = u + 1
            # print(g)
            # print(g.nodes())
            # exit()
            anchor_feat = g.ndata['feat'][0, :].clone()
            g.ndata['feat'][0, :] = 0
            unbatchg_list.append(g)
            anchor_feat_list.append(anchor_feat)
        # print(unbatchg_list)
        # print(len(unbatchg_list))
        # print(u)
        # print(in_feat.size())
        # print(len(in_feat))
        # exit()
        # anchor_out
        anchor_embs = torch.stack(anchor_feat_list, dim=0)
        anchor_out = torch.matmul(anchor_embs, self.weight) + self.bias
        anchor_out = self.act(anchor_out)
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
        return F.normalize(h, p =2, dim = 1) ,F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class OneLayerGCN(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, bias=True, args = None):
        super(OneLayerGCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, bias=bias)
        self.global_adg = args.global_adg
        if args.act_function == "PReLU":
            self.act = nn.PReLU()
        elif args.act_function == "ReLU":
            self.act = nn.ReLU()
        self.conv.set_allow_zero_in_degree(1)

    def forward(self, bg, in_feat): 
        if self.global_adg:
            h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
        else:
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
                anchor_out.append(g.ndata["h"][0])
            anchor_out = torch.stack(anchor_out, dim=0)
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
        return F.normalize(h, p = 2, dim = 1), F.normalize(subgraph_pool_emb, p = 2, dim = 1), F.normalize(anchor_out, p = 2, dim = 1)
        # return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class SL_GAD_Model(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=True, args = None):
        super(SL_GAD_Model, self).__init__()
        self.enc = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg, args)
        self.dec = OneLayerGCN(out_feats, in_feats, global_adg, args)
        self.discriminator = Discriminator(out_feats)
        # print('in_feats', in_feats)
        # print('out_feats', out_feats)


    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat, args):
        pos_batchg_1 = pos_batchg[0]
        pos_batchg_2 = pos_batchg[1]
        
        ori_feat_1 = []
        ori_feat_2 = []
        unbatchg = dgl.unbatch(pos_batchg_1)
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            ori_feat_1.append(anchor_feat)

        unbatchg = dgl.unbatch(pos_batchg_2)
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            ori_feat_2.append(anchor_feat)
        pos_in_feat_1 = pos_in_feat[0]
        pos_in_feat_2 = pos_in_feat[1]
        
        feat_1, pos_pool_emb_1, anchor_out_1 = self.enc(pos_batchg_1, pos_in_feat_1)
        feat_2, pos_pool_emb_2, anchor_out_2 = self.enc(pos_batchg_2, pos_in_feat_2)
        feat_3, neg_pool_emb, _ = self.enc(neg_batchg, neg_in_feat)
        pos_scores_1 = self.discriminator(pos_pool_emb_1, anchor_out_1)
        pos_scores_2 = self.discriminator(pos_pool_emb_2, anchor_out_2)
        neg_scores_1 = self.discriminator(neg_pool_emb, anchor_out_1)
        neg_scores_2 = self.discriminator(neg_pool_emb, anchor_out_2)
        
        pos_scores_1 = torch.sigmoid(pos_scores_1)
        pos_scores_2 = torch.sigmoid(pos_scores_2)
        neg_scores_1 = torch.sigmoid(neg_scores_1)
        neg_scores_2 = torch.sigmoid(neg_scores_2)

        # print(pos_in_feat_1.shape)
        # print(pos_pool_emb_1.shape)
        # print(anchor_out_1.shape)
        # print(feat_1.shape)
        # exit()
        with pos_batchg_1.local_scope():
            # pos_batchg_1.ndata['feat'] = feat_1
            X_hat_1, subgraph_pool_emb_1, anchor_out_1 = self.dec(pos_batchg_1, feat_1)
        with pos_batchg_2.local_scope():
            # pos_batchg_2.ndata['feat'] = feat_2
            X_hat_2, subgraph_pool_emb_2, anchor_out_2 = self.dec(pos_batchg_2, feat_2)
        
        # return pos_scores[:, 0], neg_scores[:, 0]
        
        # alpha = 1
        # beta = [0.2, 0.4, 0.6, 0.8, 1.0][2]
        alpha = args.alpha
        beta = args.beta

        L_con = 0
        # pos_scores_1 = pos_scores_1.squeeze(1).cpu().detach().numpy()
        # pos_scores_2 = pos_scores_2.squeeze(1).cpu().detach().numpy()
        # neg_scores_1 = neg_scores_1.squeeze(1).cpu().detach().numpy()
        # neg_scores_2 = neg_scores_2.squeeze(1).cpu().detach().numpy()

        # print(pos_scores_1.shape)
        # print(neg_scores_1.shape)
        
        L_con = 0
        # L_con = L_con + numpy.mean(math.log(pos_scores_1) + math.log(1 - neg_scores_1)) / 2
        # L_con = L_con + numpy.mean(math.log(pos_scores_2) + math.log(1 - neg_scores_2)) / 2
        L_con = L_con + torch.mean(torch.log(pos_scores_1) + torch.log(1 - neg_scores_1)) / 2
        L_con = L_con + torch.mean(torch.log(pos_scores_2) + torch.log(1 - neg_scores_2)) / 2
        L_con = L_con / 2
        L_con = -L_con
        L_gen = 0
        # L_gen = L_gen + numpy.mean(numpy.square(X_hat_1 - pos_in_feat_1))
        # L_gen = L_gen + numpy.mean(numpy.square(X_hat_2 - pos_in_feat_2))

        # print(ori_feat_1[0])
        # print(ori_feat_1)
        # exit()

        # print(anchor_out_1.shape)
        # print()
        # print(ori_feat_1)
        
        ori_feat_1 = torch.stack(ori_feat_1, dim = 0)
        ori_feat_2 = torch.stack(ori_feat_2, dim = 0)
        
        # ori_feat_1 = torch.tensor(ori_feat_1)#.to('cuda:0')
        # ori_feat_2 = torch.tensor(ori_feat_2)#.to('cuda:0')
        # print(ori_feat_1.shape)
        L_gen = L_gen + torch.mean(torch.square(anchor_out_1 - ori_feat_1))
        L_gen = L_gen + torch.mean(torch.square(anchor_out_2 - ori_feat_2))
        L_gen = L_gen / 2
        # print(pos_scores_1)
        # print(neg_scores_1)
        # print(pos_scores_2)
        # print(neg_scores_2)
        # print(L_con)
        # print(L_gen)
        # exit()
        L = alpha * L_con + beta * L_gen
        # print(pos_scores_1.shape)
        # print((anchor_out_1 - ori_feat_1).shape)
        # exit()
        single_predict_scores = 0
        attributes_num = math.sqrt(len(anchor_out_1[0]))
        # print(attributes_num)
        # exit()
        contrastive_predict_scores = 0
        generative_predict_scores = 0

        contrastive_predict_scores = contrastive_predict_scores + (neg_scores_1 - pos_scores_1 + 1) / 2
        contrastive_predict_scores = contrastive_predict_scores + (neg_scores_2 - pos_scores_2 + 1) / 2
        generative_predict_scores = generative_predict_scores + torch.norm(anchor_out_1 - ori_feat_1) / attributes_num / 2
        generative_predict_scores = generative_predict_scores + torch.norm(anchor_out_2 - ori_feat_2) / attributes_num / 2
        
        single_predict_scores = alpha * contrastive_predict_scores + beta * generative_predict_scores
        # single_predict_scores = single_predict_scores + torch.norm(anchor_out_1 - ori_feat_1) / attributes_num / 2
        # single_predict_scores = single_predict_scores + torch.norm(anchor_out_2 - ori_feat_2) / attributes_num / 2
        # single_predict_scores = single_predict_scores + (neg_scores_1 - pos_scores_1 + 1) / 2
        # single_predict_scores = single_predict_scores + (neg_scores_2 - pos_scores_2 + 1) / 2
        

        
        return L, single_predict_scores






if __name__ == "__main__":
    # sample
    model = SL_GAD_Model(5)
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
