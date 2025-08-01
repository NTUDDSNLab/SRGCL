import argparse

from loader import MoleculeDataset_aug, mask_nodes, permute_edges, drop_nodes
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score
import math
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from torch_geometric.data import Batch

from copy import deepcopy


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
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
        original_embedding = anchor_model.forward_cl(original_batch.x, original_batch.edge_index, original_batch.edge_attr, original_batch.batch)
        aug_embedding = anchor_model.forward_cl(aug_batch.x, aug_batch.edge_index, aug_batch.edge_attr, aug_batch.batch)

    if(selector == 'cosine'):
        cosine_sim = F.cosine_similarity(original_embedding, aug_embedding)
        distance = 1 - cosine_sim  # Convert similarity to distance
    elif(selector == 'l2_norm'):
        distance = torch.dist(original_embedding, aug_embedding, p=2)
    elif(selector == 'l1_norm'):
        distance = torch.dist(original_embedding, aug_embedding, p=1)
    elif(selector == 'kl_divergence'):
        kl_div = nn.KLDivLoss(reduction='batchmean')
        distance = kl_div(F.log_softmax(original_embedding, dim=1), F.softmax(aug_embedding, dim=1))
    elif(selector == 'mahalanobis'):
        diff = original_embedding - aug_embedding
        cov_matrix = torch.cov(diff.T)
        inv_cov_matrix = torch.linalg.inv(cov_matrix + torch.eye(cov_matrix.size(0)).to(device) * 1e-5)
        distance = torch.sqrt(torch.mm(torch.mm(diff, inv_cov_matrix), diff.T).diag())
    elif(selector == 'wasserstein'):
        distance = torch.cdist(original_embedding, aug_embedding, p=1).mean()
    else:
        raise ValueError('Invalid selector')
    # return distances.mean().item()
    return distance


def calculate_temperature(A0, k, current_epoch):
    temperature = A0*math.exp(-k*current_epoch)
    return max(0, min(1, temperature))

def generate_views_with_temperature_topk(device, aug_ratio, selector, exp_factor, data_batch, 
                                        anchor_model, current_epoch=0,
                                        generated_views_num=50, augmentation_type='dnodes', 
                                        total_augmentation_counts=None, topk_views_cl=2):
    grouped_aug_graphs = [[] for _ in range(topk_views_cl)]
    augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0}
    
    temperature = calculate_temperature(A0=1.0, k=exp_factor, current_epoch=current_epoch)
    
    for graph in data_batch.to_data_list():   
        original_graph = graph.clone()
        original_graph_cpu = original_graph.cpu()
        aug_data_list = []
        
        if augmentation_type == 'hybrid':
            hybrid_count = round(generated_views_num / 3)
            aug_types = ['dnodes', 'pedges', 'mask_nodes']
            hybrid_augmentation_list = aug_types * hybrid_count
        else:
            hybrid_augmentation_list = [augmentation_type] * generated_views_num
        
        for aug_type in hybrid_augmentation_list:
            graph_cpu = original_graph_cpu.clone()
            if aug_type == 'dnodes':
                aug_cpu = drop_nodes(graph_cpu, aug_ratio)
            elif aug_type == 'pedges':
                aug_cpu = permute_edges(graph_cpu, aug_ratio)
            elif aug_type == 'mask_nodes':
                aug_cpu = mask_nodes(graph_cpu, aug_ratio)
            aug = aug_cpu.to(device)
            aug_data_list.append((aug, aug_type))
        distances = []
        for aug_graph, aug_type in aug_data_list:
            original_graph = original_graph.to(device)
            distance = calculate_distance(original_graph, aug_graph, anchor_model, selector)
            distances.append((distance, aug_graph, aug_type))
        
        if temperature > 0:
            weights = torch.softmax(-torch.tensor([d[0] for d in distances]) / temperature, dim=0)
            indices = torch.multinomial(weights, topk_views_cl, replacement=False)
            selected_graphs = [distances[i.item()][1] for i in indices]
            selected_graphs_aug_counts = [distances[i.item()][2] for i in indices]
        else:
            distances.sort(key=lambda x: x[0])
            selected_graphs = [distances[i][1] for i in range(min(topk_views_cl, len(distances)))]
            selected_graphs_aug_counts = [distances[i][2] for i in range(min(topk_views_cl, len(distances)))]
        
        for aug_count in selected_graphs_aug_counts:
            augmentation_counts[aug_count] += 1
        
        for i in range(len(selected_graphs)):
            grouped_aug_graphs[i].append(selected_graphs[i])
    
    if total_augmentation_counts is not None:
        for k in augmentation_counts:
            total_augmentation_counts[k] += augmentation_counts[k]
    
    data_batches = []
    for group in grouped_aug_graphs:
        data_batch = Batch.from_data_list(group)
        data_batch = data_batch.to(device)
        data_batches.append(data_batch)
    
    return data_batches



def train(args, model, anchor_model, device, dataset, optimizer, epoch, total_augmentation_counts):
    """
    Single-epoch training with temperature-based top-k view generation.

    Args:
        args: parsed arguments containing augmentation hyperparameters
        model: current GNN model (to be trained)
        anchor_model: frozen copy of the model for distance calculations
        device: torch device
        dataset: PyG dataset (already instantiated)
        optimizer: torch optimizer
        epoch: current epoch (int)
        total_augmentation_counts: dict to accumulate augmentation stats

    Returns:
        avg_loss: average loss over the epoch
    """
    model.train()
    # synchronize and freeze anchor model
    anchor_model.load_state_dict(model.state_dict())
    anchor_model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)
    total_loss = 0.0
    for step, data in enumerate(loader, start=1):
        data = data.to(device)

        # generate top-k augmented views for each graph in the batch
        grouped_batches = generate_views_with_temperature_topk(
            device,                      # torch device
            args.aug_ratio,           # augmentation ratio
            args.d,           # distance selector (cosine, l2_norm, etc.)
            args.exp_factor,             # exponential decay factor for temperature
            data,                        # original batch
            anchor_model,                # frozen model
            current_epoch=epoch,         # epoch index
            generated_views_num=args.v,  # number of candidates per graph
            augmentation_type=args.aug,  # type of augmentation ('dnodes','pedges',...)
            total_augmentation_counts=total_augmentation_counts,
            topk_views_cl=args.k         # number of views to select per graph
        )

        # forward pass: get representations for each selected view
        x_views = [
            model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            for batch in grouped_batches
        ]

        # compute multi-view contrastive loss (pairwise)
        loss = 0.0
        for i in range(len(x_views)):
            for j in range(i + 1, len(x_views)):
                loss = loss + model.loss_cl(x_views[i], x_views[j])

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(dataset)
    return avg_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug', type=str, default = 'hybrid')
    parser.add_argument('--aug_ratio', type=float, default = 0.2)
    parser.add_argument('--v', type=int, default=50, help='number of views each generation')
    parser.add_argument('--k', type=int, default=2, help='Top k views for contrastive learning')
    parser.add_argument('--d', type=str, default='l2_norm', help='Types of data selector (cosine, l2_norm)')
    parser.add_argument('--eta', type=float, default=1.0, help='0.1, 1.0, 10, 100, 1000')
    parser.add_argument('--decay_type', type=str, default='exponential', help='exponential, cosine')
    parser.add_argument('--init_temp', type=float, default=1.0, help='Set initial temperature')
    parser.add_argument('--exp_factor', type=float, default=0.1, help='exponential method factor')

    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    total_augmentation_counts = {'dnodes': 0, 'pedges': 0, 'mask_nodes': 0}
    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    anchor_model = graphcl(gnn)
    model.to(device)
    anchor_model.to(device)
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, anchor_model, device, dataset, optimizer, epoch, total_augmentation_counts)

        print(train_acc)
        print(train_loss)

        if epoch % 20 == 0:
            torch.save(gnn.state_dict(), "./models_srgcl/srgcl_" + str(epoch) + ".pth")
        total_augmentations = sum(total_augmentation_counts.values())
    final_ratio = {k: v / total_augmentations for k, v in total_augmentation_counts.items()}
    print("Final Augmentation Ratios:", final_ratio)
if __name__ == "__main__":
    main()