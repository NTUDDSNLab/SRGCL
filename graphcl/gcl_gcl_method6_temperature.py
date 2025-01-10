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
import shutil
import datetime
from torch_sparse import SparseTensor
from aug_gcl import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
from torch import optim
from torch.nn.parameter import Parameter
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

def generate_aug_data_batch(data_batch, anchor_model, topk_views_cl=2, generated_views_num=50, augmentation_type='dnodes'):
    aug_ratio = args.r
    aug_data_list_1 = []
    aug_data_list_2 = []
    for graph in data_batch.to_data_list():        
        original_graph = graph.clone()
        aug_data_list = []
        for _ in range(generated_views_num):
            graph = original_graph.clone()
            if augmentation_type == 'dnodes':
                graph.edge_index, _, graph_node_mask = dropout_node(graph.edge_index, aug_ratio)
                graph_node_mask = graph_node_mask.cpu().detach().numpy().astype(int)
                feature_size = graph.x.shape[1]
                graph_node_mask_2d = np.tile(graph_node_mask, (feature_size, 1))
                graph_node_mask_2d = torch.from_numpy(np.transpose(graph_node_mask_2d)).to(device)
                graph.x = graph.x * graph_node_mask_2d
                aug_data_list.append(graph)
            elif augmentation_type == 'pedges':
                graph.edge_index, _ = dropout_edge(graph.edge_index, aug_ratio)
                graph.edge_index = graph.edge_index.to(torch.long)
                aug_data_list.append(graph)
            elif augmentation_type == 'mask_nodes':
                graph.x, _ = mask_feature(graph.x, p=aug_ratio, mode='all')
                aug_data_list.append(graph)
        distances = []
        for aug_graph in aug_data_list:
            distance = calculate_distance(original_graph, aug_graph, anchor_model, selector)
            distances.append((distance, aug_graph))
        distances.sort(key=lambda x: x[0])
        # print(distances)
        aug_data_list_1.append(distances[0][1])
        aug_data_list_2.append(distances[1][1])


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

def calculate_temperature(init_temp,cosine_factor,exp_factor,current_epoch, max_epoch, start_deterministic, decay_method='exponential'):
    # 計算進度比例
    progress = current_epoch / start_deterministic
    
    if decay_method == 'exponential':
        # 在 start_deterministic 時達到接近 0 的溫度
        temperature = (exp_factor*init_temp) ** (current_epoch * (max_epoch/start_deterministic))
    elif decay_method == 'cosine':
        # 確保在 start_deterministic 時溫度接近 0
        if current_epoch >= start_deterministic:
            temperature = 0
        else:
            # 使用餘弦函數，在 start_deterministic 時達到最低點
            temperature = (cosine_factor*init_temp)*(math.cos(progress * math.pi) + 1)
    else:
        raise ValueError("Invalid decay method. Use 'exponential' or 'cosine'")
    
    return max(0, min(init_temp, temperature))

def generate_views_with_temperature(init_temp, cosine_factor, exp_factor,
                                data_batch, anchor_model, current_epoch=0, max_epoch=30, 
                                  start_deterministic=20, decay_method='exponential',
                                  generated_views_num=50, augmentation_type='dnodes'):
    aug_ratio = args.r
    aug_data_list_1 = []
    aug_data_list_2 = []
    
    # 計算當前溫度
    temperature = calculate_temperature(init_temp,cosine_factor,exp_factor,current_epoch, max_epoch, start_deterministic, decay_method)
    
    for graph in data_batch.to_data_list():        
        original_graph = graph.clone()
        aug_data_list = []
        
        # 生成增強視圖
        for _ in range(generated_views_num):
            graph = original_graph.clone()
            if augmentation_type == 'dnodes':
                graph.edge_index, _, graph_node_mask = dropout_node(graph.edge_index, aug_ratio)
                graph_node_mask = graph_node_mask.cpu().detach().numpy().astype(int)
                feature_size = graph.x.shape[1]
                graph_node_mask_2d = np.tile(graph_node_mask, (feature_size, 1))
                graph_node_mask_2d = torch.from_numpy(np.transpose(graph_node_mask_2d)).to(device)
                graph.x = graph.x * graph_node_mask_2d
                aug_data_list.append(graph)
            elif augmentation_type == 'pedges':
                graph.edge_index, _ = dropout_edge(graph.edge_index, aug_ratio)
                graph.edge_index = graph.edge_index.to(torch.long)
                aug_data_list.append(graph)
            elif augmentation_type == 'mask_nodes':
                graph.x, _ = mask_feature(graph.x, p=aug_ratio, mode='all')
                aug_data_list.append(graph)
                
        # 計算距離
        distances = []
        for aug_graph in aug_data_list:
            distance = calculate_distance(original_graph, aug_graph, anchor_model, selector)
            distances.append((distance, aug_graph))
            
        # 根據溫度進行選擇
        if temperature > 0:
            # 使用溫度控制的隨機選擇
            weights = torch.softmax(-torch.tensor([d[0] for d in distances]) / temperature, dim=0)
            indices = torch.multinomial(weights, 2, replacement=False)
            selected_graphs = [distances[i.item()][1] for i in indices]
        else:
            # 完全確定性選擇
            distances.sort(key=lambda x: x[0])
            selected_graphs = [distances[0][1], distances[1][1]]
            
        aug_data_list_1.append(selected_graphs[0])
        aug_data_list_2.append(selected_graphs[1])

    augmented_data_batch_1 = Batch.from_data_list(aug_data_list_1)
    augmented_data_batch_2 = Batch.from_data_list(aug_data_list_2)

    return augmented_data_batch_1, augmented_data_batch_2



def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)        
        os.mkdir(os.path.join(path, 'model'))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

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
    decay_method = args.decay_type
    start_deterministic = args.start_deterministic
    log_interval = 1
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    selector = args.d
    isSaveckpt = args.ckpt
    augmentation_type = args.aug
    init_temp = args.init_temp
    cosine_factor = args.cosine_factor
    exp_factor = args.exp_factor
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('Start Time: {}'.format(start_time))
    save_name = args.save
    args.save = '{}-{}-{}-{}-{}'.format(args.dataset, args.save, start_time)
    args.save = os.path.join('unsupervised_exp', save_name, args.dataset, args.save)
    create_exp_dir(args.save, None)
    # create_exp_dir(args.save, glob.glob('*.py'))
    log_file = open(f'./logs/{args.save}.txt', 'w')
    log_file.write('Start Time: {}\n'.format(start_time))
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # add transform to add indices
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

    best_test_acc_before = 0
    best_test_std_before = 0

    best_test_acc = 0
    best_test_std = 0

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

            aug_data_batch_1,  aug_data_batch_2= generate_views_with_temperature(
                                                    init_temp,
                                                    cosine_factor, 
                                                    exp_factor,
                                                    data,
                                                    anchor_model,
                                                    current_epoch=epoch,
                                                    max_epoch=epochs,
                                                    start_deterministic=start_deterministic,
                                            decay_method=decay_method,  # 或 'exponential'
                                            generated_views_num=generated_views_num,
                                            augmentation_type=augmentation_type
            )

            optimizer.zero_grad()
            x_aug1 = model(aug_data_batch_1.x, aug_data_batch_1.edge_index, aug_data_batch_1.batch, aug_data_batch_1.num_graphs)
            x_aug2 = model(aug_data_batch_2.x, aug_data_batch_2.edge_index, aug_data_batch_2.batch, aug_data_batch_2.num_graphs)

            
            loss = model.loss_cal(x_aug1, x_aug2)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        log_file.write('Epoch {}, Loss {}\n'.format(epoch, loss_all / len(dataloader.dataset)))
       
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
                if isSaveckpt:
                    torch.save(model.state_dict(), os.path.join(args.save, 'model', 'model_best.pth'))
    log_file.write('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))
    print('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))
    if isSaveckpt:
        torch.save(model.state_dict(), os.path.join(args.save, 'model', 'model_final.pth'))

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('End Time: {}'.format(end_time))
    log_file.write('End Time: {}\n'.format(end_time))

    start_time_obj = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time_obj = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time_obj - start_time_obj

    print('Elapsed Time: {}\n'.format(elapsed_time))
    log_file.write('Elapsed Time: {}\n'.format(elapsed_time))
    log_file.close()

    with open('logs/log_' + args.DS + '_' + args.aug + '_decay_method_' + args.decay_type + '_selector_'+args.d , 'a+') as f:
        f.write('{},{:.2f},{:.2f}\n'.format(args.DS, best_test_acc*100, best_test_std*100))