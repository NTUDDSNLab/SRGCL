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
import datetime
import json
from torch_sparse import SparseTensor
from us_aug import TUDataset_aug
from torch_geometric.loader import DataLoader
import sys
import random
from torch import optim
from torch.nn.parameter import Parameter
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from us_losses import *
from us_gin import Encoder
from us_model import *
from us_evaluate_embedding import evaluate_embedding
import torch_geometric.transforms as T
from torch_geometric.transforms import Constant, BaseTransform
import pdb
import json
import argparse
import time
import logging
import shutil
import glob
from torch.autograd import Variable
from torch_geometric import data
from copy import deepcopy
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge, add_random_edge, mask_feature, dropout_node
import math
import random
import collections
from aug_output import get_augmentation




sys.path.append(os.path.abspath(os.path.join('..')))
from datasets import get_dataset
from view_generator import ViewGenerator, GIN_NodeWeightEncoder

from IPython import embed

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
    # parser.add_argument('--decay', dest='lr decay', type=float, default=0, help='Learning rate.')

    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128, help='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=str, default = 'with_sim_loss', help='')
    parser.add_argument('--batch_size', type=int, default = 128, help='')
    parser.add_argument('--epochs', type=int, default = 30, help='')

    parser.add_argument('--d', type=str, default='l2_norm', help='Types of data selector')
    parser.add_argument('--v', type=int, default=50, help='number of views each generation')
    parser.add_argument('--k', type=int, default=2, help='Top k views for contrastive learning')

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class GcnInfomax(nn.Module):
    def __init__(self, args, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
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
    def __init__(self, dataset, hidden_dim, num_gc_layers, prior, alpha=0.5, beta=1., gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers

        self.encoder = Encoder(dataset.num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, data):
        # batch_size = data.num_graphs
        x, edge_index, batch = data.x, data.edge_index, data.batch
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
    
def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss

def calculate_distance(data_batch1, data_batch2, anchor_model, selector): 
    anchor_model.eval()   
    anchor_x = anchor_model(data_batch1)
    aug_x = anchor_model(data_batch2)
    if(selector == 'cosine'):
        cosine_sim = F.cosine_similarity(anchor_x, aug_x)
        distances = 1 - cosine_sim  # Convert similarity to distance
    elif(selector == 'l2_norm'):
        distances = torch.diag(torch.cdist(anchor_x, aug_x, p=2))
    else:
        raise ValueError('Invalid selector')
    return distances.mean().item()

def train_cl_with_sim_loss(view_gen1, view_gen2, view_optimizer, model, anchor_model, optimizer, data_loader, device, selector, generated_views_num=50, topk_views_cl=2):
    loss_all = 0
    model.train()
    total_graphs = 0
    generated_views_num = int(generated_views_num / 2)

    for data in data_loader:

        anchor_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        view_optimizer.zero_grad()

        data = data.to(device)

        for _ in range(generated_views_num):
            distances1 = []
            distances2 = []
            sample1, view1 = view_gen1(data, True)
            sample2, view2 = view_gen2(data, True)

            distance = calculate_distance(data, view1, anchor_model, selector)
            distances1.append((distance, view1, sample1))

            distance = calculate_distance(data, view2, anchor_model, selector)
            distances2.append((distance, view2, sample2))

        distances1.sort(key=lambda x: x[0], reverse=True)
        distances2.sort(key=lambda x: x[0], reverse=True)

        closest_augmentations1 = distances1[:topk_views_cl]
        closest_augmentations2 = distances2[:topk_views_cl]

        closest_data_batch_augmentations1 = [aug for _, aug, _ in closest_augmentations1]
        closest_sample_augmentations1 = [sample for _, _, sample in closest_augmentations1]

        closest_data_batch_augmentations2 = [aug for _, aug, _ in closest_augmentations2]
        closest_sample_augmentations2 = [sample for _, _, sample in closest_augmentations2]

        view1 = closest_data_batch_augmentations1[0]
        view2 = closest_data_batch_augmentations2[0]

        sample1 = closest_sample_augmentations1[0]
        sample2 = closest_sample_augmentations2[0]

        sim_loss = F.mse_loss(sample1, sample2)
        sim_loss = (1 - sim_loss)

        input_list = [data, view1, view2]
        input1, input2 = random.choices(input_list, k=2)

        out1 = model(input1)
        out2 = model(input2)
        
        cl_loss = loss_cl(out1, out2)

        loss = sim_loss + cl_loss

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        loss.backward()        
        optimizer.step()
        view_optimizer.step()
    loss_all /= total_graphs
    return loss_all

def eval_acc(model, data_loader, device):
    model.eval()
    emb, y = model.encoder.get_embeddings(data_loader, device)
    acc, std = evaluate_embedding(emb, y)
    return acc, std

def cl_exp(args):
    set_seed(args.seed)

    joint_log_name = 'joint_log_{}.txt'.format(args.save)
    save_name = args.save
    args.save = '{}-{}-{}-{}'.format(args.dataset, args.seed, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('unsupervised_exp', save_name, args.dataset, args.save)
    create_exp_dir(args.save, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epochs = args.epochs
    log_interval = 10
    batch_size = args.batch_size
    generated_views_num = args.v
    topk_views_cl = args.k
    lr = args.lr
    selector = args.d
    dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root='../../data')
    dataset = dataset.shuffle()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_eval_loader = DataLoader(dataset, batch_size=batch_size)

    model = simclr(dataset, args.hidden_dim, args.num_gc_layers, args.prior).to(device)
    anchor_model = simclr(dataset, args.hidden_dim, args.num_gc_layers, args.prior).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    view_gen1 = ViewGenerator(dataset, args.hidden_dim, GIN_NodeWeightEncoder)
    view_gen2 = ViewGenerator(dataset, args.hidden_dim, GIN_NodeWeightEncoder)
    view_gen1 = view_gen1.to(device)
    view_gen2 = view_gen2.to(device)


    view_optimizer = optim.Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=args.lr
                                , weight_decay=0)

    # logger.info('================')
    # logger.info('lr: {}'.format(lr))
    # logger.info('num_features: {}'.format(dataset.num_features))
    # logger.info('hidden_dim: {}'.format(args.hidden_dim))
    # logger.info('num_gc_layers: {}'.format(args.num_gc_layers))
    # logger.info('================')

    best_test_acc = 0
    best_test_std = 0
    test_accs = []

    for epoch in range(1, epochs+1):
        # train_loss = train_cl(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device)
        train_loss = train_cl_with_sim_loss(view_gen1, view_gen2, view_optimizer, model, anchor_model, optimizer, data_loader, device, selector, generated_views_num, topk_views_cl)
        logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))
        if epoch % log_interval == 0:
            test_acc, test_std = eval_acc(model, data_eval_loader, device)
            test_accs.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std
            logger.info("*" * 50)
            logger.info("Evaluating embedding...")
            logger.info('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epoch, test_acc*100, test_std*100))
    logger.info('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc*100, best_test_std*100))

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time_obj = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time_obj = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time_obj - start_time_obj
    print('Elapsed Time: {}\n'.format(elapsed_time))
    logger.info('Elapsed Time: {}\n'.format(elapsed_time))


    

class simclr_graph_cl(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, prior, alpha=0.5, beta=1., gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

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

    def forward(self, data):
        # batch_size = data.num_graphs
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones(batch.shape[0]).to(edge_index.device)
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

if __name__ == '__main__':
    args = arg_parse()
    cl_exp(args)

