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
import time
from torch.autograd import Variable
from copy import deepcopy

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge, add_random_edge, mask_feature, dropout_node, subgraph
from aug import subgraph
import math
import random
import collections
# from datetime import datetime
import scipy

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

import math

def calculate_temperature(exp_factor, current_epoch, max_epoch, start_deterministic):
    if current_epoch >= start_deterministic:
        temperature = 0.0
    else:
        temperature = math.exp(-exp_factor * current_epoch * (max_epoch/start_deterministic))
    
    return max(0, min(1, temperature))



def optimized_subgraph_sampling(original_graph, aug_ratio):
    graph = original_graph.clone()
    
    node_num, _ = graph.x.size()
    sub_num = int(node_num * aug_ratio)
    
    # 使用 CSR 矩陣預處理邊
    edge_index = graph.edge_index.cpu().numpy()
    adj_matrix = scipy.sparse.csr_matrix(
        (np.ones(edge_index.shape[1]), 
        (edge_index[0], edge_index[1])), 
        shape=(node_num, node_num)
    )
    
    idx_sub = {np.random.randint(node_num)}
    idx_neigh = set(adj_matrix[list(idx_sub)[0]].indices)
    
    while len(idx_sub) <= sub_num:
        if not idx_neigh:
            break
            
        sample_node = np.random.choice(list(idx_neigh))
        idx_sub.add(sample_node)
        idx_neigh.update(adj_matrix[sample_node].indices)
        idx_neigh -= idx_sub
    
    idx_sub = list(idx_sub)
    idx_drop = list(set(range(node_num)) - set(idx_sub))
    
    # 使用稀疏操作
    adj_matrix[idx_drop, :] = 0
    adj_matrix[:, idx_drop] = 0
    
    new_edge_index = torch.tensor(np.array(adj_matrix.nonzero()), 
                                device=original_graph.x.device, dtype=torch.int64)
    
    # 批量更新節點特徵
    mask = torch.ones(node_num, device=original_graph.x.device)
    mask[idx_drop] = 0
    graph.x = graph.x * mask.view(-1, 1)
    graph.edge_index = new_edge_index
    
    return graph



def generate_views_with_temperature(exp_factor,data_batch, 
                                anchor_model, current_epoch=0, max_epoch=30, 
                                  start_deterministic=20, 
                                  generated_views_num=50, augmentation_type='dnodes', total_augmentation_counts=None):
    aug_ratio = args.r
    aug_data_list_1 = []
    aug_data_list_2 = []

    augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0, 'subgraph': 0}
    
    # 計算當前溫度
    temperature = calculate_temperature(exp_factor,current_epoch, max_epoch, start_deterministic)
    
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
            elif augmentation_type == 'subgraph':
                graph = optimized_subgraph_sampling(graph, aug_ratio)
                aug_data_list.append((graph, 'subgraph'))
        
        distances = []
        for aug_graph, aug_type in aug_data_list:
            distance = calculate_distance(original_graph, aug_graph, anchor_model, selector)
            distances.append((distance, aug_graph, aug_type))
            
        # 根據溫度進行選擇
        if temperature > 0:
            # 使用溫度控制的隨機選擇
            weights = torch.softmax(-torch.tensor([d[0] for d in distances]) / temperature, dim=0)
            indices = torch.multinomial(weights, 2, replacement=False)
            selected_graphs = [distances[i.item()][1] for i in indices]
            selected_graphs_aug_counts = [distances[i.item()][2] for i in indices]
            selected_graphs_aug_counts = selected_graphs_aug_counts[:2]
        else:
            # 完全確定性選擇
            distances.sort(key=lambda x: x[0])
            selected_graphs = [distances[0][1], distances[1][1]]
            selected_graphs_aug_counts = [distances[0][2], distances[1][2]]
        
        for aug_count in selected_graphs_aug_counts:
            augmentation_counts[aug_count] += 1
        aug_data_list_1.append(selected_graphs[0])
        aug_data_list_2.append(selected_graphs[1])
    if total_augmentation_counts is not None:
        for k in augmentation_counts:
            total_augmentation_counts[k] += augmentation_counts[k]

    augmented_data_batch_1 = Batch.from_data_list(aug_data_list_1)
    augmented_data_batch_2 = Batch.from_data_list(aug_data_list_2)

    return augmented_data_batch_1, augmented_data_batch_2

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
    save_name = args.save
    args.save = '{}-{}-{}-{}'.format(args.DS, args.seed, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('unsupervised_exp', save_name, args.DS, args.save)
    # create_exp_dir(args.save, glob.glob('*.py'))
    create_exp_dir(args.save, None)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    accuracies = {'val':[], 'test':[]}
    accuracies_before = {'val':[], 'test':[]}
    epochs = args.epochs
    isSaveckpt = args.ckpt
    generated_views_num = args.v
    topk_views_cl = args.k
    decay_method = args.decay_type
    start_deterministic = args.start_deterministic
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    selector = args.d
    augmentation_type = args.aug
    exp_factor = args.exp_factor

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('Start Time: {}'.format(start_time))
    log_file_path = os.path.join(args.save, f'log_{augmentation_type}.txt')
    log_file = open(log_file_path, 'w')
    log_file.write('Start Time: {}\n'.format(start_time))
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
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
    test_acc = 0
    test_std = 0 
    total_augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0, 'subgraph': 0}
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
                                                    exp_factor,
                                                    data,
                                                    anchor_model,
                                                    current_epoch=epoch,
                                                    max_epoch=epochs,
                                                    start_deterministic=start_deterministic,
                                            generated_views_num=generated_views_num,
                                            augmentation_type=augmentation_type, 
                                            total_augmentation_counts=total_augmentation_counts
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
            emb, y = model.encoder.get_embeddings(dataloader_eval, device)
            test_acc, test_std = evaluate_embedding(emb, y)
            accuracies['val'].append(test_acc)
            accuracies['test'].append(test_std)
            print('Epoch: {}, Test Acc: {:.2f} '.format(epoch, test_acc*100))
            log_file.write('Epoch: {}, Test Acc: {:.2f} '.format(epoch, test_acc*100))
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std
                if isSaveckpt:
                    torch.save(model.state_dict(), os.path.join(args.save, 'model', 'model_best.pth'))
            
    log_file.write('Final Test Acc: {:.2f} '.format(test_acc*100))
    print('Final Test Acc: {:.2f} '.format(test_acc*100))
    log_file.write('Best Test Acc: {:.2f} '.format(best_test_acc*100))
    print('Best Test Acc: {:.2f} '.format(best_test_acc*100))
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
    total_augmentations = sum(total_augmentation_counts.values())
    final_ratio = {k: v / total_augmentations for k, v in total_augmentation_counts.items()}
    print("Final Augmentation Ratios:", final_ratio)
    log_file.write(f'Final Augmentation Ratios: {final_ratio}\n')
    log_file.close()

    with open('logs/log_' + args.DS + '_' + args.aug + '_selector_'+args.d + '_exp_factor_{exp_factor}_start_deterministic_{start_deterministic}', 'a+') as f:
        f.write('Final accuracy: {},{:.2f}\n'.format(args.DS, test_acc*100))
        f.write('Best accuracy: {},{:.2f}\n'.format(args.DS, best_test_acc*100))
