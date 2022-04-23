"""
AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks
reference:https://github.com/pygod-team/pygod/blob/main/pygod/models/anomalydae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch import nn
import dgl


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

class AnomalyDAE(nn.Module):
    """
    Description
    -----------
    AdnomalyDAE is an anomaly detector consisting of a structure autoencoder,
    and an attribute reconstruction autoencoder. 
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
                 in_feat_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AnomalyDAE, self).__init__()
        self.structure_AE = StructureAE(in_feat_dim, embed_dim,
                                        out_dim, dropout)
        self.attribute_AE = AttributeAE(in_num_dim, embed_dim,
                                        out_dim, dropout, act)

    def forward(self,g,x):
        A_hat, embed_x = self.structure_AE(g, x)
        X_hat = self.attribute_AE(x, embed_x)
        return A_hat, X_hat


class StructureAE(nn.Module):
    """
    Description
    -----------
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node 
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.
    See :cite:`fan2020anomalydae` for details.
    Parameters
    ----------
    in_dim: int
        input dimension of node data
    embed_dim: int
        the latent representation dimension of node
       (after the first linear layer)
    out_dim: int 
        the output dim after the graph attention layer
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
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim) #(n,feat_dim)->(n,emd_dim)
        self.attention_layer = GATConv(embed_dim, out_dim,num_heads=1)
        self.dropout = dropout
        for module in self.modules():
            init_weights(module)

    def forward(self,g,x):
        # encoder
        x = torch.relu(self.dense(x))
        x = F.dropout(x, self.dropout)
        embed_x = self.attention_layer(g, x).squeeze(1) #(n,emd_dim)
        # decoder
        x = torch.sigmoid(embed_x @ embed_x.T) #(n,n)
        return x, embed_x



class AttributeAE(nn.Module):
    """
    Description
    -----------
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to 
    reconstruct the original node attribute.
    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function   
    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim) #(feat_dim,n)->(feat_dim,emd_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act
        for module in self.modules():
            init_weights(module)

    def forward(self,
                x,
                struct_embed):
        # encoder
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)  #(feat_dim,emd_dim)
        # decoder
        x = struct_embed @ x.T  #(n,emd_dim) * (emd_dim,feat_dim)
        return x    #(n,feat_dim)





