import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean
import scipy.sparse as spp
from tqdm import tqdm


class AAGNN_M(nn.Module):
    def __init__(self, g, in_feats, out_feats, device):
        super().__init__()
        self.line = nn.Linear(in_feats, out_feats).to(device)
        #生成邻接矩阵，A的尺寸为(n_node, n_node)
        self.A = torch.zeros((len(g.nodes().numpy()), len(g.nodes().numpy()))).to(device)
        us = g.edges()[0].numpy()
        vs = g.edges()[1].numpy()
        for u, v in zip(us, vs):
            self.A[u][v] = self.A[v][u] = 1.0
            self.A[u][u] = self.A[v][v] = 1.0
        #得到每个节点的度数矩阵，B的尺寸为(n_node, hid_feats)
        self.B = torch.sum(self.A, dim=1).reshape(-1, 1)

    def forward(self, inputs):
        #进行线性映射到低维空间，z的尺寸为(n_node, hid_feats)
        z = self.line(inputs)
        #A * z的结果矩阵尺寸为(n_node, hid_feats)，和度矩阵B对应点除，得到平均值
        h = z - (torch.mm(self.A, z)/ self.B)
        #最后非线性激活函数映射输出
        return F.relu(h)
    #计算伪标签，伪标签为正样本的mask下
    def mask_label(self, inputs, p):
        with torch.no_grad():
            z = self.line(inputs)
            #得到所有节点特征的均值矩阵
            c = torch.mean(z, dim=0)
            #计算距离
            dis = torch.sum((z - c) * (z - c), dim=1)
            best_min_dis = list(dis.cpu().data.numpy())
            #从小到大排序
            best_min_dis.sort()
            #得到距离阈值
            threshold = best_min_dis[int(len(best_min_dis) * p)]
            mask = (dis <= threshold)
            #返回mask，为true表示伪标签是正样本
            return mask
            
    #损失函数
    def loss_fun(self, out, mask, model, super_param, device):
        #得到所有节点特征的均值矩阵
        c = torch.mm(torch.ones(out.shape[0], 1).to(device), torch.mean(out, dim=0).reshape(1, -1))
        #计算所有节点的误差
        loss_matrix = torch.sum((out - c) * (out - c), dim=1)[mask]
        #取均值误差
        loss = torch.mean(loss_matrix, dim=0)

        l2_reg = torch.tensor(0.).to(device)#L2正则项
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return loss + (super_param * l2_reg/2)

    #计算异常分数
    def anomaly_score(self, out):
        s = torch.sum(out * out, dim=1)
        return s.cpu().data.numpy()

