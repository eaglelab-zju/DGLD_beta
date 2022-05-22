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
from dataset import split_auc
from dgl.nn import GATConv
import random
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
#from dgl.nn.pytorch.conv import SAGEConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,  bias=False, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class OCGNN(nn.Module):
    def __init__(self, ):
        super().__init__()

    def fit(self, graph, args):
        features = graph.ndata['feat']
        print(graph)
        print('features shape:', features.shape)
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(args.device))
        else:
            device = torch.device("cpu")
        features = features.to(device)
        anomaly_label = graph.ndata["anomaly_label"].cpu().numpy()
        train_mask = np.zeros(len(anomaly_label))
        val_mask = np.zeros(len(anomaly_label))
        for i in range(len(anomaly_label)):
            if anomaly_label[i] == 0:
                key = random.randint(0, 100)
                if key <= 60:
                    train_mask[i] = True
                else:
                    val_mask[i] = True
            else:
                val_mask[i] = True

        graph.ndata['train_mask'] = torch.tensor(np.array(train_mask), dtype=torch.bool)
        graph.ndata['val_mask'] = torch.tensor(np.array(val_mask), dtype=torch.bool)
        model = GCN(None,
                    features.shape[1],
                    256 * 2,
                    256 ,
                    2,
                    F.relu,
                    0.3)
        #model = OCGNN_base(features.shape[1], 256)
        model = model.to(device)

        data_center = self.init_center(graph, features, model, device)
        radius = torch.tensor(0, device=device)  # radiu

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(log_dir=args.logdir)
        model.train()
        for epoch in range(args.num_epoch):
            outputs = model(graph, features)
            #loss = model.loss_fun(out)
            loss, dist, _ = self.loss_function(data_center, outputs, radius, graph.ndata['train_mask'])

            #print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            radius.data = torch.tensor(self.get_radius(dist), device=device)

            dist, predict_score = self.anomaly_score(data_center, outputs)
            predict_score = predict_score.cpu().data.numpy()

            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
            )))
            writer.add_scalars(
                "loss",
                {"loss": loss},
                epoch,
            )
            #print(predict_score.shape)
            final_score, a_score, s_score = split_auc(graph.ndata["anomaly_label"], predict_score)
            writer.add_scalars(
                "auc",
                {"final": final_score, "structural": s_score, "attribute": a_score},
                epoch,
            )
            writer.flush()


    def init_center(self, input_g, input_feat, model, device, eps=0.001):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""

        n_samples = 0
        c = torch.zeros(256, device=device)

        model.eval()
        with torch.no_grad():

            outputs= model(input_g,input_feat)

            # get the inputs of the batch

            n_samples = outputs.shape[0]
            c =torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def anomaly_score(self, data_center, outputs, radius=0, mask=None):
        if mask == None:
            dist = torch.sum((outputs - data_center) ** 2, dim=1)
        else:
            dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
        scores = dist - radius ** 2
        return dist,scores

    def loss_function(self, data_center, outputs, radius=0, mask=None):
        nu = 0.2
        dist, scores = self.anomaly_score(data_center, outputs, radius, mask)
        loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        return loss, dist, scores

    def get_radius(self, dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        nu = 0.2
        radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
        return radius
