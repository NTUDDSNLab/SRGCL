# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch_sparse import SparseTensor
from aug_gcl import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
from torch import optim
from torch.nn.parameter import Parameter
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from model import *
from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
from aug_output import get_augmentation
import torch_geometric.transforms as T
from torch_geometric.transforms import Constant, BaseTransform
import pdb
import logging
from torch.autograd import Variable
from copy import deepcopy

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge, add_random_edge, mask_feature, dropout_node
import math
import random
import collections


class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)
        
        y = self.proj_head(y)
        
        return y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def second_loss_cal(self, x, x_aug, indices_list):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        # Diagonal should be positive
        # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # Use index instead of diagnol to determine positive pairs, loss of each graph shoud be mean
        pos_mask = torch.zeros([batch_size, batch_size]).cuda()
        pos_pair_count = torch.zeros([batch_size]).cuda()
        neg_pair_count = torch.ones([batch_size]).cuda()
        neg_pair_count = neg_pair_count * batch_size
        for graphidx in range(batch_size):
            my_index = indices_list[graphidx]
            for i in range(batch_size):
                if (indices_list[i] == my_index): 
                    pos_mask[graphidx][i] = 1
                    pos_pair_count[i] += 1
                    neg_pair_count[i] -= 1
        # print(pos_pair_count)
        # print(neg_pair_count)
        # print(indices_list)
        pos_sim = sim_matrix * pos_mask
        pos_sim = pos_sim.sum(dim=1)
        pos_sim = pos_sim
        #print(pos_sim)
        
        loss = (pos_sim / pos_pair_count) / (batch_size * ((sim_matrix.sum(dim=1) - pos_sim)) / neg_pair_count)
        loss = - torch.log(loss).mean()
        return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class Add_Indices(BaseTransform):
    def __call__(self, data):
        data.indices = torch.tensor([0])
        return data

if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = 10
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    selector = args.d
    log_file = open(f'./logs/log_gcl_gcl_{DS}.txt', 'w')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # add transform to add indices
    dataset = TUDataset(path, name=DS, aug=args.aug, transform=T.Compose([Add_Indices()]))
    dataset_eval = TUDataset(path, name=DS, aug='none')

    # dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    # dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    # print(len(dataset))
    # print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    # Multi Phase Training Hyperparameter Setting
    patience = 2
    min_delta = 0.01
    loss_min = float('inf')
    counter = 0
    aug_data_ratio = 0.2
    dataset_len_multiple = 3
    loss_list = []
    stage_finish_epochs = []

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    anchor_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')
    log_file.write('================\n')
    log_file.write('lr: {}\n'.format(lr))
    log_file.write('num_features: {}\n'.format(dataset_num_features))
    log_file.write('hidden_dim: {}\n'.format(args.hidden_dim))
    log_file.write('num_gc_layers: {}\n'.format(args.num_gc_layers))
    log_file.write('================\n')

    # model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """
    best_test_acc = 0
    best_test_std = 0

    for epoch in range(1, epochs+1):
        loss_all = 0
        anchor_model.load_state_dict(model.state_dict())
        anchor_model.eval()
        clone_data_list = []
        good_aug_data_list = []
        new_origin_data_list = []
        tensor_list = []
        tmp_list = []
        
        # Determined datasets can't add feature, clone a dataset to add index on every graph
        tmp_dataloader = DataLoader(dataset, batch_size = batch_size)
        for data in tmp_dataloader:
            data, data_aug = data
            data = data.to(device)
            data_list = data.to_data_list()
            for i, graph in enumerate(data_list):
                #print(len(data_list))
                #print(graph)
                # graph.indices[0] = i
                clone_data_list.append(graph)
                # good_aug_data_list.append(graph)
                #print(graph)
        #print(clone_data_list[0])
        
        select_dataloader = DataLoader(clone_data_list, batch_size=batch_size, shuffle=False)
        # Collect good augmented data for new dataset
        while True:
            #print(tmp_list)
            total_data = 0
            for data in select_dataloader:
                # print(data.indices)
                # print(data.indices[0])
                # tmp_index_list = []
                # for i in range(len(data)):
                #     tmp_index_list.append(data[i].indices)
                # print(tmp_index_list)
                aug_data_num = math.ceil(aug_data_ratio*len(data))
                data_ori = data.clone()
                data = data.to(device)
                data_ori = data_ori.to(device)

                # Augmentation in data list format to prevent can't use function to_data_list
                data_list = data.to_data_list()
                aug_data_list = []
                aug_ratio = 0.2
                for graph in data_list:
                    #node
                    # graph.edge_index, _, graph_node_mask = dropout_node(graph.edge_index, aug_ratio)
                    # graph_node_mask = graph_node_mask.cpu().detach().numpy().astype(int)
                    # feature_size = graph.x.shape[1]
                    # graph_node_mask_2d = np.tile(graph_node_mask, (feature_size, 1))
                    # graph_node_mask_2d = torch.from_numpy(np.transpose(graph_node_mask_2d)).to(device)
                    # graph.x = graph.x * graph_node_mask_2d
                    # aug_data_list.append(graph)
                    # Edge
                    # graph.edge_index, _ = dropout_edge(graph.edge_index, aug_ratio)
                    # aug_data_list.append(graph)
                    # graph.edge_index, _ = add_random_edge(graph.edge_index, aug_ratio)
                    # Attr
                    graph.x, _ = mask_feature(graph.x, p=aug_ratio, mode='all')
                    aug_data_list.append(graph)
                
                data = Batch.from_data_list(aug_data_list)

                # Anchor model calculate distance
                anchor_x = anchor_model(data_ori.x, data_ori.edge_index, data_ori.batch, data_ori.num_graphs)
                aug_x = anchor_model(data.x, data.edge_index, data.batch, data.num_graphs)
                distances = torch.diag(torch.cdist(anchor_x, aug_x, p=2))
                if(selector == 'cosine'):
                    cosine_sim = F.cosine_similarity(anchor_x, aug_x)
                    distances = 1 - cosine_sim  # Convert similarity to distance
                elif(selector == 'l2_norm'):
                    distances = torch.diag(torch.cdist(anchor_x, aug_x, p=2))
                else:
                    raise ValueError('Invalid selector')

                values, _ = torch.topk(-distances, aug_data_num)
                is_used_data = (distances <= -values[-1])
                #print(values[-1])
                #print(f'Used data {is_used_data}')
                tmp_list.append(is_used_data)
                
                #print(len(is_used_data))
                for i in range(len(is_used_data)):
                    if (is_used_data[i] == True):
                        good_aug_data_list.append(aug_data_list[i]) # include original data and good augmented data
                        new_origin_data_list.append(clone_data_list[i])

            if (len(good_aug_data_list) >= dataset_len_multiple * batch_size):
                break

        second_dataloader = DataLoader(good_aug_data_list, batch_size=batch_size, shuffle=False)
        new_origin_dataloader = DataLoader(new_origin_data_list, batch_size=batch_size, shuffle=False)
        for data, data_ori in zip(second_dataloader, new_origin_dataloader):
            model.train()
            optimizer.zero_grad()
            # data_ori, data_aug = data_ori
            # node_num, _ = data.x.size()
            data = data.to(device)
            data_ori = data_ori.to(device)
            # x : embeddings of the node features (input and output of the encoder)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)
            x_aug = model(data_ori.x, data_ori.edge_index, data_ori.batch, data_ori.num_graphs)


            loss = model.loss_cal(x, x_aug)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        # print('batch')
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        log_file.write('Epoch {}, Loss {}\n'.format(epoch, loss_all / len(dataloader.dataset)))
        loss_list.append(loss_all / len(dataloader.dataset))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            test_acc, test_std = evaluate_embedding(emb, y)
            accuracies['val'].append(test_acc)
            accuracies['test'].append(test_std)
            print('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epoch, test_acc*100, test_std*100))
            log_file.write('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epochs, test_acc*100, test_std*100))

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std
    log_file.write('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))
    print('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))
    log_file.close()

    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        f.write('{},{:.2f},{:.2f}\n'.format(args.DS, best_test_acc*100, best_test_std*100))