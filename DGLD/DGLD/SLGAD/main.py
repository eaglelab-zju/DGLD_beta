import sys

sys.path.append('.\\.\\')
# print(sys.path)
import scipy.sparse as sp

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy
import scipy.io as scio
from datetime import datetime

# from dominant_utils import get_parse, train_step, test_step
# from models import Dominant

from SL_GAD_utils import get_parse
from common.dataset import GraphNodeAnomalyDectionDataset
from dgl.dataloading import GraphDataLoader
from utils.utils import seed_everything
from dataset import SL_GAD_DataSet
from model import SL_GAD_Model
from SL_GAD_utils import get_parse, train_epoch, test_epoch
import dgl
import time
from common.save_datasets import gen_and_save_dgld_dataset

# torch.set_default_tensor_type(torch.DoubleTensor)

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    print(time.time())
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.sampling.random_walk(dgl_graph, all_idx, length=subgraph_size*3, restart_prob=0.99999)[0]
    # traces = dgl.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        # subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        subv.append(torch.unique(trace,sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            # cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            cur_trace = dgl.sampling.random_walk(dgl_graph, [i], length=subgraph_size*5, restart_prob=0.9)[0]
            subv[i] = torch.unique(cur_trace[0],sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    print(time.time())
    print(subv)
    return subv

import dataset as Dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
def plt_show(args, sum, length, name, plt_idx):
    '''
    target_label = ori_graph.ndata["label"]
    target_label = torch.unsqueeze(target_label, 1)
    group_label = ori_graph.ndata["label"][shuffle_idx]
    # print(target_label)
    # print(group_label)
    res = group_label == target_label
    sum = torch.sum(res, 1)
    # print(res[0:5])
    # print(sum)
    '''
    # print(sum[0:2])
    # print(sum == 1)
    range_index = np.arange(1, length + 0.1, 0.1)
    shuffle_same_label_distribution = [torch.sum(torch.tensor(sum) == _).item() for _ in range_index]
    print(shuffle_same_label_distribution)
    print(torch.sum(torch.tensor(shuffle_same_label_distribution)))
    x_index = np.arange(1, length + 0.1, 0.1)
    print(x_index)
    plt.figure(plt_idx)
    plt.xticks(x_index)
    plt.plot(x_index, shuffle_same_label_distribution)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    subgraph_attr = "_Subgraph"
    if args.positive_subgraph_cor:
        subgraph_attr = subgraph_attr + "_P_T"
    else:
        subgraph_attr = subgraph_attr + "_P_F"

    if args.negative_subgraph_cor:
        subgraph_attr = subgraph_attr + "_N_T"
    else:
        subgraph_attr = subgraph_attr + "_N_F"
        
    img_name = args.dataset + name + subgraph_attr + ".jpg"
    plt.title(img_name)
    plt.savefig(img_name)

import networkx as nx

def ShowGraph(graph, nodeLabel, EdgeLabel = None):
    # plt.figure(4)
    plt.figure(figsize=(8, 8))
    # G=graph.to_networkx(node_attrs=nodeLabel.split(),edge_attrs=EdgeLabel.split())  #转换 dgl graph to networks
    G=graph.to_networkx(node_attrs=nodeLabel.split())  #转换 dgl graph to networks
    pos=nx.spring_layout(G)
    nx.draw(G, pos,edge_color="grey", node_size=500,with_labels=True) # 画图，设置节点大小
    node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
    node_labels = { index:"N:"+ str(data)  for index,data in enumerate(node_data) }  #重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
    pos_higher = {}
    
    for k, v in pos.items():  #调整下顶点属性显示的位置，不要跟顶点的序号重复了
        if(v[1]>0):
            pos_higher[k] = (v[0]-0.04, v[1]+0.04)
        else:
            pos_higher[k] = (v[0]-0.04, v[1]-0.04)
    nx.draw_networkx_labels(G, pos_higher, labels=node_labels,font_color="brown", font_size=12)  # 将desc属性，显示在节点上
    # edge_labels = nx.get_edge_attributes(G, EdgeLabel) # 获取边的weights属性，
 
    # edge_labels= {  (key[0],key[1]): "w:"+str(edge_labels[key].item())  for key in edge_labels } #重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
    nx.draw_networkx_edges(G,pos, alpha=0.5 )
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=12) # 将Weights属性，显示在边上
 
    print(G.edges.data())
    plt.show()

if __name__ == '__main__':
    args = get_parse()
    seed_everything(args.seed)
    print(args)
    # gen_and_save_dgld_dataset(data_name=args.dataset)
    # exit()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # load dataset
    name = 'custom_' + args.dataset + '.mat'
    print('dataset name : ', name)
    data = scio.loadmat(name)
    # print(data)
    temp_label = data['Label'] if ('Label' in data) else data['gnd']
    temp_attr = data['Raw_Attributes'] if ('Raw_Attributes' in data) else data['Attributes']
    # print(temp_attr[:20, :5])
    # exit()
    temp_network = data['Network'] if ('Network' in data) else data['A']
    # print(data)
    # print(type(temp_label))
    # print(type(temp_attr))
    # print(type(temp_network))
    # print(temp_label.shape)
    temp_label = torch.from_numpy(temp_label).squeeze()
    # print(temp_label.shape)
    temp_graph = dgl.from_scipy(sp.coo_matrix(temp_network), eweight_name = 'w')
    # print(temp_network)
    # exit()
    # temp_graph = dgl.from_scipy(temp_network, eweight_name = 'w')
    # print(temp_attr)
    if args.dataset != 'ogbn-arxiv':
        temp_graph.ndata['feat'] = torch.from_numpy(temp_attr.todense())#.double()
    else:
        temp_graph.ndata['feat'] = torch.from_numpy(temp_attr)#.double()
    # print(temp_graph.ndata['feat'].dtype)
    # print(temp_attr.dtype)
    # exit()
    # print(temp_graph.ndata['feat'].shape)
    temp_graph.ndata['label'] = temp_label
    print(temp_graph)
    # print(temp_graph.edata['w'])
    # temp_attr = torch.from_numpy(temp_attr)
    # temp_network = torch.from_numpy(temp_network)
    # print(torch.sum(temp_network, dim = 1))
    # print(torch.sum(temp_attr, dim = 1))
    
    
    # print(type(temp_label))
    # print(type(temp_attr))
    # print(type(temp_network))
    # print(temp_label.shape)
    # print(temp_attr.shape)
    # print(temp_network.shape)
    
    # exit()
    # temp_graph = dataset.dataset
    # print(temp_graph)
    # temp_graph = dgl.remove_self_loop(temp_graph)
    # print(temp_graph)
    # exit()

    dataset = SL_GAD_DataSet('custom', g_data = temp_graph, y_data = temp_label)
    # print(torch.sum(dataset.dataset.ndata['feat'], dim = 1))
    # exit()
    # print(dataset.dataset.ndata['feat'][:20, :5])
    # exit()
    print(dataset.dataset.edata['w'][:20])
    print(dataset.dataset.edges()[0][:20])
    print(dataset.dataset.edges()[1][:20])
    
    # exit()
    train_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        # shuffle=True,
        shuffle = False,
        # num_samples = 0,
    )
    test_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        # shuffle=True,
        shuffle = False,
        # num_samples = 0,
    )
    graph = dataset[0]
    graph = graph[0]
    name = 'nonsense.mat'
    print(dataset.oraldataset.dataset)
    # print(dataset.oraldataset.dataset.ndata['anomaly_label'])
    Label = dataset.oraldataset.dataset.ndata['anomaly_label'] != 0
    Attributes = dataset.oraldataset.dataset.ndata['feat']
    Raw_Attributes = (Attributes > 0) + 0.0
    Class = dataset.oraldataset.dataset.ndata['label']
    str_anomaly_label = dataset.oraldataset.dataset.ndata['anomaly_label'] == 1
    attr_anomaly_label = dataset.oraldataset.dataset.ndata['anomaly_label'] == 2
    edges = dataset.oraldataset.dataset.edges()
    src = edges[0]
    dst = edges[1]
    # print(edges)
    print(src.shape)
    print(dst.shape)
    nodes = dataset.oraldataset.dataset.number_of_nodes()
    print(nodes)
    Network = torch.zeros(nodes, nodes)
    Network[src, dst] = 1.0
    print(torch.sum(Network))
    dict = {}
    dict['Network'] = Network.numpy()
    dict['Label'] = dataset.oraldataset.dataset.ndata['anomaly_label'].numpy()
    dict['Attributes'] = Attributes.numpy()
    dict['Raw_Attributes'] = Raw_Attributes.numpy()
    dict['Class'] = Class.numpy()
    dict['str_anomaly_label'] = str_anomaly_label.numpy()
    dict['attr_anomaly_label'] = attr_anomaly_label.numpy()
    print(Attributes.shape)
    print(torch.sum(Attributes, dim = 1))
    print(torch.sum(Attributes, dim = 1).shape)
    
    # exit()
    # scio.savemat(name, dict)
    # exit()
    # data = scio.loadmat(name)
    # torch.set_printoptions(profile="full")
    # print(torch.tensor(data['Attributes'][0, :]))
    # print(data)

    # print(ori_graph.out_degrees())
    # print(ori_graph.in_degrees() == ori_graph.out_degrees())
    start_time = datetime.now()
    print('start_time : ', start_time)

    SLGAD_subgraphsampler = Dataset.COLASubGraphSampling(length=args.subgraph_size)
    model = SL_GAD_Model(
        in_feats=dataset[0][0].ndata["feat"].shape[1],
        out_feats=args.embedding_dim,
        global_adg=args.global_adg,
        args = args,
    ).to(device)

    # print(model)
    # print(model.enc.weight.weight[:5, :5])
    # print(model.discriminator_1.bilinear.weight[0, :5, :5])
    # print(model.discriminator_1.bilinear.weight.shape)
    # exit()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    # exit()
    # train
    writer = SummaryWriter(log_dir=args.logdir)
    best_loss = 1e9
    best_t = 0

    for epoch in range(args.num_epoch):
        start_time = datetime.now()
        train_loader.dataset.random_walk_sampling()
        end_time = datetime.now()
        print('process time : ', end_time - start_time)
        # exit()
        loss_accum = train_epoch(
            epoch, args, train_loader, model, device, criterion, optimizer
        )
        writer.add_scalar("loss", float(loss_accum), epoch)
        predict_score = test_epoch(
            epoch, args, test_loader, model, device, criterion, optimizer
        )
        
        final_score, a_score, s_score = dataset.oraldataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )
        writer.flush()

        if loss_accum < best_loss:
            best_loss = loss_accum
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(args.expid))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!', flush=True)
            break

    if args.num_epoch == 0:
        torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(args.expid))
    
    # multi-round test
    print('Loading {}th epoch'.format(best_t), flush=True)
    model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(args.expid)))
    predict_score_arr = []
    for rnd in range(args.auc_test_rounds):
        test_loader.dataset.random_walk_sampling()
        predict_score = test_epoch(
            rnd, args, test_loader, model, device, criterion, optimizer
        )
        predict_score_arr.append(list(predict_score))
    # print(len(predict_score_arr))
    # print(len(predict_score_arr[0]))
    print(args)
    predict_score_arr = numpy.array(predict_score_arr).T
    dataset.oraldataset.evaluation_multiround(predict_score_arr)

    end_time = datetime.now()
    print('start_time : ', start_time)
    print('end_time : ', end_time)
    print('process_time : ', end_time - start_time)
    # mean_predict_result = predict_score_arr.mean(1)
    # std_predict_result = predict_score_arr.std(1)
    # max_predict_result = predict_score_arr.max(1)
    # min_predict_result = predict_score_arr.min(1)
    # median_predict_result = numpy.median(predict_score_arr, 1)
    
    # descriptions = {
    #     "mean": mean_predict_result,
    #     "std": std_predict_result,
    #     "mean+std": mean_predict_result + std_predict_result,
    #     "mean-std": mean_predict_result - std_predict_result,
    #     "mean+median": mean_predict_result + median_predict_result,
    #     "max": max_predict_result,
    #     "min": min_predict_result,
    #     "median": median_predict_result,
    # }
    # for stat in descriptions:
    #     print("=" * 10 + stat + "=" * 10)
    #     dataset.oraldataset.evalution(descriptions[stat])
