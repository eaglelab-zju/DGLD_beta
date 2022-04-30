"""
AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks
reference:https://github.com/pygod-team/pygod/blob/main/pygod/models/anomalydae.py
"""
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
import dgl
import scipy.sparse as sp
import scipy.io as sio
from sklearn.metrics import precision_score, roc_auc_score
import networkx as nx
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from method.ComGA.comga_utils import get_parse



class ComGA_Base(nn.Module):
    """
    Description
    -----------
    ComGA is an anomaly detector consisting of a Community Detection Module,
    a tGCN Module and an Anomaly Detection Module

    Parameters
    ----------
    in_feat_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim: int
         Dimension of the embedding after the first reduced linear layer (D1)   
    out_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    act: F, optional
         Choice of activation function
    """

    def __init__(self,
                 num_nodes,
                 num_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout,
                 ):
        super(ComGA_Base, self).__init__()
        self.commAE=CommunityAE(num_nodes,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.tgcnEnc=tGCNEncoder(num_feats,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.attrDec=AttrDecoder(num_feats,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.struDec=StruDecoder(dropout)
        

    def forward(self,g,x,B):
        B_enc1,B_enc2,z_a,B_hat = self.commAE(B)
        z = self.tgcnEnc(g,x,B_enc1,B_enc2,z_a)
        X_hat = self.attrDec(g,z)
        A_hat = self.struDec(z)
        
        return A_hat,X_hat,B_hat,z,z_a

def init_weights(module: nn.Module) -> None:
    """Init Module Weights
    ```python
        for module in self.modules():
            init_weights(module)
    ```
    Args:
        module (nn.Module)
    """
    if isinstance(module, nn.Linear):
        # TODO: different initialization
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Bilinear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class CommunityAE(nn.Module):
    """
    Description
    -----------
    Community Detection Module:
        The modularity matrix B is reconstructed by autoencode to obtain a representation 
        of each node with community information.
    
    Parameters
    ----------
    

    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function   

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    embed_x : torch.Tensor
              Embedd nodes after the attention layer
    """

    def __init__(self,
                 num_nodes,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(CommunityAE, self).__init__()
        self.dropout=dropout
        #encoder
        self.enc1 = nn.Linear(num_nodes, n_enc_1)
        self.enc2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc3 = nn.Linear(n_enc_2, n_enc_3)
        #decoder
        self.dec1 = nn.Linear(n_enc_3, n_enc_2)
        self.dec2 = nn.Linear(n_enc_2, n_enc_1)
        self.dec3 = nn.Linear(n_enc_1, num_nodes)

        for module in self.modules():
            init_weights(module)

    def forward(self,B):
        # encoder
        x = torch.relu(self.enc1(B))
        hidden1 = F.dropout(x, self.dropout)
        x=torch.relu(self.enc2(hidden1))
        hidden2 = F.dropout(x, self.dropout)
        x=torch.relu(self.enc3(hidden2))
        z_a = F.dropout(x, self.dropout)
        
        # decoder
        x=torch.relu(self.dec1(z_a))
        se1 = F.dropout(x, self.dropout)
        x=torch.relu(self.dec2(se1))
        se2 = F.dropout(x, self.dropout)
        x=torch.sigmoid(self.dec3(se2))
        community_reconstructions = F.dropout(x, self.dropout)
        
        return hidden1,hidden2,z_a,community_reconstructions




class tGCNEncoder(nn.Module):
    """
    Description
    -----------
    tGCNEncoder:
        To effectively fuse community structure information to GCN model for structure anomaly,
    and learn more distinguishable anomalous node representations for local, global, and 
    structure anomalies.

    Parameters
    ----------
    

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(tGCNEncoder, self).__init__()
        self.enc1=GraphConv(in_feats,n_enc_1,activation=F.relu)
        self.enc2=GraphConv(n_enc_1,n_enc_2,activation=F.relu)
        self.enc3=GraphConv(n_enc_2,n_enc_3,activation=F.relu)
        self.enc4=GraphConv(n_enc_3,n_enc_3,activation=F.relu)
        

        self.dropout = dropout
        for module in self.modules():
            init_weights(module)

    def forward(self,g,x,B_enc1,B_enc2,B_enc3):
        # encoder
        x1=F.dropout(self.enc1(g,x), self.dropout)
        x=x1+B_enc1
        x2=F.dropout(self.enc2(g,x), self.dropout)
        x=x2+B_enc2
        x3=F.dropout(self.enc3(g,x), self.dropout)
        x=x3+B_enc3
        z=F.dropout(self.enc4(g,x), self.dropout)

        return z


class AttrDecoder(nn.Module):
    """
    Description
    -----------
    AttrDecoder:
        utilize attribute decoder to take the learned latent representation
    Z as input to decode them for reconstruction of original nodal attributes.

    Parameters
    ----------



    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(AttrDecoder, self).__init__()
        self.dropout=dropout

        self.dec1=GraphConv(n_enc_3,n_enc_3,activation=F.relu)
        self.dec2=GraphConv(n_enc_3,n_enc_2,activation=F.relu)
        self.dec3=GraphConv(n_enc_2,n_enc_1,activation=F.relu)
        self.dec4=GraphConv(n_enc_1,in_feats,activation=F.relu)
        
        for module in self.modules():
            init_weights(module)

    def forward(self,g,z):
        # decoder
        x1=F.dropout(self.dec1(g,z), self.dropout)
        x2=F.dropout(self.dec2(g,x1), self.dropout)
        x3=F.dropout(self.dec3(g,x2), self.dropout)
        attribute_reconstructions=F.dropout(self.dec4(g,x3), self.dropout)
        
        return attribute_reconstructions    


class StruDecoder(nn.Module):
    """
    Description
    -----------
    StruDecoder:
        utilize structure decoder to take the learned latent representation 
    Z as input to decode them for reconstruction of original graph structure.

    Parameters
    ----------


    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 dropout):
        super(StruDecoder, self).__init__()
        self.dropout=dropout

    def forward(self,z):
        # decoder
        x=F.dropout(z, self.dropout)
        x=z@x.T
        structure_reconstructions=torch.sigmoid(x)

        return structure_reconstructions    






