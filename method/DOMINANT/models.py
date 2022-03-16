import torch.nn as nn
import torch.nn.functional as F
import torch
# from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, g, h):
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConv(nhid, nhid)
        self.gc2 = GraphConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, g, h):
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, g, h):
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    def forward(self, g, h):
        # encode
        x = self.shared_encoder(g, h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, x)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(g, x)
        # return reconstructed matrices
        return struct_reconstructed, x_hat