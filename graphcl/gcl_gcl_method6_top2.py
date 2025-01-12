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
import random
import datetime
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
from check_dim_collapse import check_dimensional_collapse
from ortho_loss import l2_reg_ortho
# from datetime import datetime


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

def generate_aug_data_batch(data_batch, anchor_model, topk_views_cl=2, generated_views_num=50, augmentation_type='dnodes', total_augmentation_counts=None):
    aug_ratio = args.r
    aug_data_list_1 = []
    aug_data_list_2 = []
    
    # 記錄各種 augmentation 的統計
    augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0}
    
    for graph in data_batch.to_data_list():        
        original_graph = graph.clone()
        aug_data_list = []
        
        if augmentation_type == 'hybrid':
            hybrid_count = round(generated_views_num / 3)
            aug_types = ['dnodes', 'pedges', 'mask_nodes']
            hybrid_augmentation_list = aug_types * hybrid_count
        else:
            hybrid_augmentation_list = [augmentation_type] * generated_views_num
        
        for aug_type in hybrid_augmentation_list:
            graph = original_graph.clone()
            if aug_type == 'dnodes':
                graph.edge_index, _, graph_node_mask = dropout_node(graph.edge_index, aug_ratio)
                graph_node_mask = graph_node_mask.cpu().detach().numpy().astype(int)
                feature_size = graph.x.shape[1]
                graph_node_mask_2d = np.tile(graph_node_mask, (feature_size, 1))
                graph_node_mask_2d = torch.from_numpy(np.transpose(graph_node_mask_2d)).to(device)
                graph.x = graph.x * graph_node_mask_2d
                aug_data_list.append((graph, 'dnodes'))
            elif aug_type == 'pedges':
                graph.edge_index, _ = dropout_edge(graph.edge_index, aug_ratio)
                graph.edge_index = graph.edge_index.to(torch.long)
                aug_data_list.append((graph, 'pedges'))
            elif aug_type == 'mask_nodes':
                graph.x, _ = mask_feature(graph.x, p=aug_ratio, mode='all')
                aug_data_list.append((graph, 'mask_nodes'))
        
        distances = []
        for aug_graph, aug_type in aug_data_list:
            distance = calculate_distance(original_graph, aug_graph, anchor_model, selector)
            distances.append((distance, aug_graph, aug_type))
        distances.sort(key=lambda x: x[0])
        
        aug_data_list_1.append(distances[0][1])
        augmentation_counts[distances[0][2]] += 1
        aug_data_list_2.append(distances[1][1])
        augmentation_counts[distances[1][2]] += 1
    
    # total_augmentations = sum(augmentation_counts.values())
    # ratio = {k: v / total_augmentations for k, v in augmentation_counts.items()}
    # print("Augmentation Ratios", ratio)
    # log_file.write('Augmentation Ratios: {}\n'.format(ratio))
    if total_augmentation_counts is not None:
        for k in augmentation_counts:
            total_augmentation_counts[k] += augmentation_counts[k]


    augmented_data_batch_1 = Batch.from_data_list(aug_data_list_1)
    augmented_data_batch_2 = Batch.from_data_list(aug_data_list_2)

    return augmented_data_batch_1, augmented_data_batch_2

def calculate_distance(original_graph, aug_graph, anchor_model, selector):
    anchor_model.eval()
    original_batch = Batch.from_data_list([original_graph])
    aug_batch = Batch.from_data_list([aug_graph])
    with torch.no_grad():
        original_embedding = anchor_model(original_batch.x, original_batch.edge_index, original_batch.batch, original_batch.num_graphs)
        aug_embedding = anchor_model(aug_batch.x, aug_batch.edge_index, aug_batch.batch, aug_batch.num_graphs)
    # original_embedding = original_embedding[0]
    # aug_embedding = aug_embedding[0]

    if(selector == 'cosine'):
        cosine_sim = F.cosine_similarity(original_embedding, aug_embedding)
        distance = 1 - cosine_sim  # Convert similarity to distance
    elif(selector == 'l2_norm'):
        distance = torch.dist(original_embedding, aug_embedding, p=2)
    else:
        raise ValueError('Invalid selector')
    # return distances.mean().item()
    return distance

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
    accuracies_before = {'val':[], 'test':[]}
    epochs = args.epochs
    generated_views_num = args.v
    topk_views_cl = args.k
    ordecay = args.ordecay

    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    selector = args.d
    augmentation_type = args.aug
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path = osp.join('/disk_195a/qiannnhui/data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # add transform to add indices
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('Start Time: {}'.format(start_time))
    log_file = open(f'./logs/log_gcl_gcl_{DS}_{augmentation_type}_{start_time}.txt', 'w')
    log_file.write('Start Time: {}\n'.format(start_time))
    dataset = TUDataset(path, name=DS, aug=augmentation_type, transform=T.Compose([Add_Indices()])).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    anchor_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # print('================')
    # print('lr: {}'.format(lr))
    # print('num_features: {}'.format(dataset_num_features))
    # print('hidden_dim: {}'.format(args.hidden_dim))
    # print('num_gc_layers: {}'.format(args.num_gc_layers))
    # print('================')
    # log_file.write('================\n')
    # log_file.write('lr: {}\n'.format(lr))
    # log_file.write('num_features: {}\n'.format(dataset_num_features))
    # log_file.write('hidden_dim: {}\n'.format(args.hidden_dim))
    # log_file.write('num_gc_layers: {}\n'.format(args.num_gc_layers))
    # log_file.write('================\n')

    best_test_acc_before = 0
    best_test_std_before = 0

    best_test_acc = 0
    best_test_std = 0

    total_augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0}

    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            anchor_model.load_state_dict(model.state_dict())
            anchor_model.eval()
            # data, data_aug = data
            data, _ = data
            # data_aug = get_augmentation(data, args.aug)
            data = data.to(device)

            aug_data_batch_1,  aug_data_batch_2 = generate_aug_data_batch(data, anchor_model, topk_views_cl, generated_views_num, augmentation_type, total_augmentation_counts)
            aug_data_batch_1 = aug_data_batch_1.to(device)
            aug_data_batch_2 = aug_data_batch_2.to(device)

            optimizer.zero_grad()
            x_aug1 = model(aug_data_batch_1.x, aug_data_batch_1.edge_index, aug_data_batch_1.batch, aug_data_batch_1.num_graphs)
            x_aug2 = model(aug_data_batch_2.x, aug_data_batch_2.edge_index, aug_data_batch_2.batch, aug_data_batch_2.num_graphs)

            
            loss = model.loss_cal(x_aug1, x_aug2)
            loss_all += loss.item() * data.num_graphs
            loss_all += l2_reg_ortho(model)
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
       
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            test_acc, test_std = evaluate_embedding(emb, y)
            check_dimensional_collapse(emb)
            accuracies['val'].append(test_acc)
            accuracies['test'].append(test_std)
            print('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epoch, test_acc*100, test_std*100))

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std
    print('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('End Time: {}'.format(end_time))

    start_time_obj = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time_obj = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time_obj - start_time_obj

    print('Elapsed Time: {}\n'.format(elapsed_time))
    log_file.write('Elapsed Time: {}\n'.format(elapsed_time))

    total_augmentations = sum(total_augmentation_counts.values())
    final_ratio = {k: v / total_augmentations for k, v in total_augmentation_counts.items()}
    print("Final Augmentation Ratios:", final_ratio)
    log_file.write(f'Final Augmentation Ratios: {final_ratio}\n')
    log_file.close()

    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        f.write('{},{:.2f},{:.2f}\n'.format(args.DS, best_test_acc*100, best_test_std*100))