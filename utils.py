import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, LINKXDataset, WebKB, Actor
from models import *
import torch_geometric.transforms as T

def get_dataset(dataname):
    dataroot = 'datasets/'
    if dataname in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        dataset = LINKXDataset(dataroot, name=dataname,transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataname in ["cora","citeseer","pubmed"]:
        dataset = Planetoid(dataroot, name=dataname, num_train_per_class=20, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataname in ["chameleon","squirrel"]:
        pre_edge_index = WikipediaNetwork(dataroot, name=dataname,geom_gcn_preprocess=False,transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(dataroot, name=dataname,geom_gcn_preprocess=True,transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = pre_edge_index[0].edge_index
    elif dataname in ["cornell","texas","wisconsin"]:
        dataset = WebKB(dataroot, name=dataname,transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataname in ["film"]:
        dataset = Actor(dataroot,transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        raise NotImplementedError
    return dataset,data

def get_geom_mask(dataname,run=0):
    dataname = dataname+'_split_0.6_0.2_'+str(run)+'.npz'
    path = osp.join(osp.dirname(__file__),'split',dataname)
    data_mask = np.load(path)
    train_mask = torch.tensor(data_mask['train_mask']).bool()#
    val_mask = torch.tensor(data_mask['val_mask']).bool()
    test_mask = torch.tensor(data_mask['test_mask']).bool()
    return train_mask,val_mask,test_mask

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def get_few_mask(data, num_class, num_per_class=20, num_val=500):
    indices = []
    for i in range(num_class):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    train_index = torch.cat([i[:num_per_class] for i in indices], dim=0)
    rest_index = torch.cat([i[num_per_class:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    train_mask = index_to_mask(train_index, size=data.num_nodes)
    val_mask = index_to_mask(rest_index[:num_val], size=data.num_nodes)
    test_mask = index_to_mask(rest_index[num_val:], size=data.num_nodes)
    return train_mask, val_mask, test_mask

def get_model(args,dataset,data,device):
    if args.model == 'gprfpa':
        model = FPA(dataset, data, args).to(device)
    else:
        raise ValueError('Invalid method!')
    return model

def norm_adj(data, add_self_loop=True):
    if data.edge_weight == None:
        edge_index = torch.sparse_coo_tensor(data.edge_index, torch.ones(
            data.num_edges, device=data.edge_index.device), (data.num_nodes,
            data.num_nodes))
    else:
        edge_index = torch.sparse_coo_tensor(data.edge_idnex, data.
            edge_weight, (data.num_nodes, data.num_nodes))
    adj = edge_index.to_dense()
    if add_self_loop:
        adj = adj + torch.eye(data.num_nodes, device=data.edge_index.device)
    degree = torch.sum(adj, axis=1)
    degree_sqrt = 1 / torch.sqrt(degree)
    degreeMatrix = torch.diag(degree_sqrt)
    adj = torch.matmul(torch.matmul(degreeMatrix, adj), degreeMatrix)
    return adj

