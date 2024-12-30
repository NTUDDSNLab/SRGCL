import torch

from itertools import repeat
import numpy as np

from copy import deepcopy

# def gen_ran_output(data, model, vice_model, args):
#     for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
#         if name.split('.')[0] == 'proj_head':
#             adv_param.data = param.data
#         else:
#             adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)           
#     z2 = vice_model(data.x, data.edge_index, data.batch, data.num_graphs)
#     return z2

def get_augmentation(data, aug):
    if aug == 'dnodes':
        data_aug = drop_nodes(deepcopy(data))
    elif aug == 'pedges':
        data_aug = permute_edges(deepcopy(data))
    elif aug == 'subgraph':
        data_aug = subgraph(deepcopy(data))
    elif aug == 'mask_nodes':
        data_aug = mask_nodes(deepcopy(data))
    elif aug == 'none':
        data_aug = deepcopy(data)
        data_aug.x = torch.ones((data.edge_index.max()+1, 1))

    elif aug == 'random2':
        n = np.random.randint(2)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False
    elif aug == 'random3':
        n = np.random.randint(3)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = permute_edges(deepcopy(data))
        elif n == 2:
           data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False

    elif aug == 'random4':
        n = np.random.randint(4)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = permute_edges(deepcopy(data))
        elif n == 2:
           data_aug = subgraph(deepcopy(data))
        elif n == 3:
           data_aug = mask_nodes(deepcopy(data))
        else:
            print('sample error')
            assert False
    else:
        print('augmentation error')
        assert False

    # print(data, data_aug)
    # assert False

    return data_aug

def drop_nodes(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.cpu().numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).cpu().numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def subgraph(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.cpu().numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.cpu().numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index



    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


