import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean
import scipy.sparse as spp
from tqdm import tqdm
from torch.autograd import Variable # torch 中 Variable 模块
from torch.utils.tensorboard import SummaryWriter
from get_parse import get_parse

def split_auc(groundtruth, prob):
    r"""
    print the scoring(AUC) of the two types of anomalies separately.
    Parameter:
    ----------
    groundtruth: np.ndarray, Indicates whether this node is an injected anomaly node.
            0: normal node
            1: structural anomaly
            2: contextual anomaly

    prob: np.ndarray-like array saving the predicted score for every node
    Return:
    -------
    None
    """
    str_pos_idx = groundtruth == 1
    attr_pos_idx = groundtruth == 2
    norm_idx = groundtruth == 0

    str_data_idx = str_pos_idx | norm_idx
    attr_data_idx = attr_pos_idx | norm_idx

    str_data_groundtruth = groundtruth[str_data_idx]
    str_data_predict = prob[str_data_idx]

    attr_data_groundtruth = np.where(groundtruth[attr_data_idx] != 0, 1, 0)
    attr_data_predict = prob[attr_data_idx]

    s_score = roc_auc_score(str_data_groundtruth, str_data_predict)
    a_score = roc_auc_score(attr_data_groundtruth, attr_data_predict)
    final_score = roc_auc_score(np.where(groundtruth == 0, 0, 1), prob)

    print("structural anomaly score:", s_score)
    print("attribute anomaly score:", a_score)
    print("final anomaly score:", final_score)
    return final_score, a_score, s_score

class AAGNN_A(nn.Module):
    def __init__(self,):
        super().__init__()
    def fit(self, graph, args):
        features = graph.ndata['feat']
        print(graph)
        print('features shape:', features.shape)
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(args.device))
        else:
            device = torch.device("cpu")
        subgraph_size = args.subgraph_size

        model = AAGNN_A_base(features.shape[1], 256, device)
        model = model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        #获取伪标签下的正常样本节点
        node_ids = model.get_normal_nodes(features, 0.5)

        writer = SummaryWriter(log_dir=args.logdir)
        model.train()

        #现在修改成，每次计算误差时的中心向量，都是上一次的
        for epoch in range(args.num_epoch):
            center = self.cal_center(graph, model, 256, device, subgraph_size)

            for index in range(0, len(node_ids), subgraph_size):
                L = index
                R = index + subgraph_size
                subgraph_node_ids = node_ids[L: R]
                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device)

                #只会对应输出subgraph_node_ids的表征
                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)
                loss = self.loss_fun(out, center, model, 0.0001, device)

                opt.zero_grad()
                #loss = loss / accumulation_steps
                loss.backward()
                opt.step()

                print("Epoch:", '%04d' % (epoch), " train_loss=", "{:.10f}".format(loss.item(
                )))

            #opt.step()  # update parameters of net
            #opt.zero_grad()  # reset gradient


            #infer....
            predict_score = []
            for index in range(0, features.shape[0], subgraph_size):
                L = index
                R = min(index + subgraph_size, features.shape[0])

                subgraph_node_ids = np.arange(L, R)

                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device)
                #只会对应输出subgraph_node_ids的表征
                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)

                score = model.anomaly_score(out)
                predict_score += list(score)

            predict_score = np.array(predict_score)
            #print(predict_score.shape)
            #print(features.shape[0])
            #print(graph.ndata["anomaly_label"].shape[0])
            final_score, a_score, s_score = split_auc(graph.ndata["anomaly_label"], predict_score)
            writer.add_scalars(
                "auc",
                {"final": final_score, "structural": s_score, "attribute": a_score},
                epoch,
            )
            writer.flush()


    #进行图采样的函数
    def Graph_batch_sample(self, graph, node_ids, device):
        feats = graph.ndata['feat']
        us = graph.edges()[0].numpy()
        vs = graph.edges()[1].numpy()
        sample_u = []
        sample_v = []
        subgraph_feats = []

        for u, v in zip(us, vs):
            if v in node_ids:
                sample_u.append(u)
                sample_v.append(v)
        sample_nodes = list(set(sample_u + sample_v))


        adj_matrix = np.zeros((len(sample_nodes), len(sample_nodes)))
        node_id_dic = {}

        for i in range(len(sample_nodes)):
            node_id_dic[sample_nodes[i]] = i
            subgraph_feats.append(feats[sample_nodes[i]].cpu().data.numpy())

        for u, v in zip(sample_u, sample_v):
            adj_matrix[node_id_dic[u]][node_id_dic[v]] = 1
            adj_matrix[node_id_dic[u]][node_id_dic[u]] = 1
            adj_matrix[node_id_dic[v]][node_id_dic[v]] = 1

        eye_matrix = torch.eye(len(sample_nodes))
        node_mask = []
        for id in node_ids:
            node_mask.append(node_id_dic[id])
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        eye_matrix = torch.tensor(eye_matrix, dtype=torch.float32).to(device)
        subgraph_feats = torch.tensor(np.array(subgraph_feats), dtype=torch.float32).to(device)
        node_mask = torch.tensor(np.array(node_mask), dtype=torch.long).to(device)

        return adj_matrix, eye_matrix, subgraph_feats, node_mask

    def cal_center(self, graph, model, out_feats, device, subgraph_size):
        center = torch.zeros(out_feats)
        with torch.no_grad():
            features = graph.ndata['feat']
            node_ids = np.arange(features.shape[0])
            for index in range(0, features.shape[0], subgraph_size):
                L = index
                R = index + subgraph_size
                subgraph_node_ids = node_ids[L: R]
                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device)
                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)

                center += torch.sum(out, dim=0)

            #计算平均
            center = center/features.shape[0]

        return center

    #损失函数
    def loss_fun(self, out, center, model, super_param, device):
        #计算所有节点的误差
        loss_matrix = torch.sum((out - center) * (out - center), dim=1)
        #取均值误差
        loss = torch.mean(loss_matrix, dim=0)
        # L2正则项
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return loss + (super_param * l2_reg/2)




class AAGNN_A_base(nn.Module):
    def __init__(self, in_feats, out_feats, device):
        super().__init__()
        self.line = nn.Linear(in_feats, out_feats)
        #定义计算注意力权重的可训练的向量a
        self.a_1 = nn.Parameter(torch.rand(1, out_feats))
        self.a_2 = nn.Parameter(torch.rand(1, out_feats))
        self.LeakyReLU = nn.LeakyReLU()
    def forward(self, adj_matrix, eye_matrix, subgraph_feats, node_mask):
        #进行线性映射到低维空间，z的尺寸为(n_node, hid_feats)
        z = self.line(subgraph_feats)
        zi = torch.sum(self.a_1 * z, dim=1).reshape(-1, 1)
        zj = torch.sum(self.a_2 * z, dim=1).reshape(-1, 1)

        attention_A = adj_matrix * zi
        attention_B = adj_matrix * (eye_matrix * zj)
        #激活函数激活
        attention_matrix = self.LeakyReLU(attention_A + attention_B)
        #乘self.A的意义是把原来为0的地方还原，相当于mask
        attention_matrix = torch.exp(attention_matrix) * adj_matrix
        #这一步等同于做softmax，将所有的数据映射到0-1
        attention_matrix = attention_matrix / torch.sum(attention_matrix, dim=1).reshape(-1, 1)
        h = z - torch.mm(attention_matrix, z)
        #最后非线性激活函数映射输出
        return F.relu(h[node_mask])

    # 得到正样本节点
    def get_normal_nodes(self, node_feats, p):
        with torch.no_grad():
            z = self.line(node_feats)
            # 得到所有节点特征的均值矩阵
            c = torch.mean(z, dim=0)
            # 计算距离
            dis = torch.sum((z - c) * (z - c), dim=1)
            best_min_dis = list(dis.cpu().data.numpy())
            # 从小到大排序
            best_min_dis.sort()
            # 得到距离阈值
            threshold = best_min_dis[int(len(best_min_dis) * p)]

            node_ids = []
            for node_id in range(node_feats.shape[0]):
                if dis[node_id] <= threshold:
                    node_ids.append(node_id)

            return node_ids

    #计算异常分数
    def anomaly_score(self, out):
        s = torch.sum(out * out, dim=1)
        return s.cpu().data.numpy()
