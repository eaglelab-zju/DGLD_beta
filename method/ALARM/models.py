import torch.nn as nn
import torch.nn.functional as F
import torch
# from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, view_num, agg_type, agg_vec):
        super(Encoder, self).__init__()
        
        self.view_num = view_num
        self.agg_type = agg_type
        self.nhid = nhid
        self.agg_vec = agg_vec
        
        self.single_view = int(nfeat / view_num)
        self.view_feat = [self.single_view for i in range(view_num - 1)]
        self.view_feat.append(int(nfeat / view_num) + int(nfeat % view_num))
        
        self.gc1 = nn.ModuleList(GraphConv(self.view_feat[i], nhid, norm="none") for i in range(view_num)) 
        self.gc2 = GraphConv(nhid, nhid, norm="none") 
        self.dropout = dropout

    def forward(self, g, h):
        x = []
        for i in range(self.view_num):
            if i is self.view_num - 1:
                 x.append(h[:,i * (self.single_view):])
            else:
                x.append(h[:,i * self.single_view:(i + 1) * self.single_view])
        
        for i, gc in enumerate(self.gc1):
            if i != self.view_num - 1:
                x[i] = F.relu(self.gc1[i](g, x[i]))
            else:
                x[i] = F.relu(self.gc1[i](g, x[i]))
        
        for i in range(self.view_num):
            x[i] = F.dropout(x[i], self.dropout, training=self.training)
            x[i] = F.relu(self.gc2(g, x[i]))
        
        if self.agg_type is 1:
            rand_weight = torch.rand(1, self.view_num)
            for i in range(0, self.view_num):
                x[i] = rand_weight * x[i]
            x = torch.cat([i for i in x], 1)
        elif self.agg_type is 2:
            manual_weight = torch.tensor(self.agg_vec)
            for i in range(0, self.view_num):
                x[i] = rand_weight * x[i]
        
        x = torch.cat([i for i in x], 1)
        return x

class AttributeDecoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, view_num):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GraphConv(nhid * view_num, nhid,norm="none")
        self.gc2 = GraphConv(nhid, nfeat,norm="none")
        self.dropout = dropout

    def forward(self, g, h):
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))
        return x

class StructureDecoder(nn.Module):
    def __init__(self, nhid, dropout, view_num):
        super(StructureDecoder, self).__init__()

        self.gc1 = GraphConv(nhid * view_num, nhid,norm="none")
        self.dropout = dropout

    def forward(self, g, h):
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

# hidden_size是每个view的size，总的hidden_size要每个view的hidden_size * view_num
class ALARM(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, view_num, agg_type, agg_vec):
        super(ALARM, self).__init__()
        if len(agg_vec) != view_num:
            raise KeyError('Aggregator vector size is {}, but the number of view is {}'.format(len(agg_vec), view_num))
        self.view_num = view_num
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout, view_num, agg_type, agg_vec)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout, view_num)
        self.struct_decoder = StructureDecoder(hidden_size, dropout, view_num)

    def forward(self, g, h):
        # encode
        x = self.shared_encoder(g, h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, x)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(g, x)
        # return reconstructed matrices
        return struct_reconstructed, x_hat