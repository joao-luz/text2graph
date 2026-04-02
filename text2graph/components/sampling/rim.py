from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from torch_sparse import SparseTensor, spmm
from torch_geometric.utils import remove_self_loops, scatter, add_self_loops
import copy
from src.active.density import get_density
import time


def rim_select(features,labels,edge_index,total_node_number,idx_avilable,total_budget=140,oracle_acc=1,th=0.05,prior = None,batch_size=5):
    seed_everything(42)
    
    reliability_list=torch.ones(total_node_number)
    all_idx = torch.arange(total_node_number)
    
    adj_matrix2 = compute_adj2(edge_index, total_node_number).to_dense()
    norm_aax = compute_norm_aax(features, edge_index, total_node_number) 
    similarity_feature = compute_sim(norm_aax, total_node_number)

    idx_train = []
    idx_available = all_idx[idx_avilable].tolist()
    idx_available_temp = copy.deepcopy(idx_available)
    activated_node = torch.ones(total_node_number)
    count = 0
    train_class = {}

    while True:
        max_ral_node,max_activated_node,max_activated_num = get_max_reliable_info_node_dense(idx_available_temp,activated_node,train_class,labels, oracle_acc, th, adj_matrix2, prior = prior) 
        #print("max_activated_num",max_activated_num)
        idx_train.append(max_ral_node)
        idx_available.remove(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label]=list()
            train_class[node_label].append(max_ral_node)
        count += 1
        # print(count,"  select_node_idx",max_ral_node,"adj2-num",(adj_matrix2[max_ral_node]*activated_node>0.05).sum(),"max_activated_node:",max_activated_num)
        
        if count % batch_size == 0:
            time0=time.time()
            activated_node = update_reliability(idx_train,train_class,labels,total_node_number, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature, prior = prior)
            time1=time.time()
        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)
        if count >= total_budget or max_activated_num <= 0:
            break
    
    train_mask = torch.zeros(total_node_number)
    train_mask[idx_train] = 1
    
    return train_mask.bool()

def get_max_reliable_info_node_dense(high_score_nodes,activated_node,train_class,labels, oracle_acc, th, adj_matrix2, prior = None): 
    def get_activated_node_dense(node,reliable_score,activated_node, adj_matrix2, th): 
        activated_vector=((adj_matrix2[node]*reliable_score)>th)+0
        activated_vector=activated_vector*activated_node
        # num_ones = torch.ones(adj_matrix2.shape[0])
        count = activated_vector.sum()
        return count.item(), activated_vector
    
    max_ral_node = 0
    max_activated_node = 0
    max_activated_num = 0 
    for node in high_score_nodes:
        if prior is not None:
            reliable_score = prior[node]
        else:
            reliable_score = oracle_acc
        activated_num,activated_node_tmp=get_activated_node_dense(node,reliable_score,activated_node, adj_matrix2, th)
        if activated_num > max_activated_num:
            max_activated_num = activated_num
            max_ral_node = node
            max_activated_node = activated_node_tmp        
    return max_ral_node,max_activated_node,max_activated_num


def update_reliability(idx_used,train_class,labels,num_node, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature, prior = None):
    
    def get_reliable_score(similarity, oracle_acc, num_class, prior = None):
        return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))
    
    activated_node = torch.zeros(num_node)
    num_class = labels.max().item()+1
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item()
        if prior is not None:
            prior_acc = prior[node]
        else:
            prior_acc = oracle_acc
        if node_label in train_class:
            total_score = 0.0
            for tmp_node in train_class[node_label]:
                total_score+=reliability_list[tmp_node]
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score(similarity_feature[node][tmp_node], oracle_acc, num_class)
            reliable_score = reliable_score/total_score
        else:
            reliable_score = oracle_acc
        reliability_list[node]=reliable_score
        activated_node+=((adj_matrix2[node]*reliable_score)>th)
    return torch.ones(num_node)-((activated_node>0).to(torch.int))


def compute_rw_norm_edge_index(edge_index, edge_weight = None, num_nodes = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                 device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    # L = I - A_norm.
    edge_index, tmp = add_self_loops(edge_index, edge_weight,
                                        fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp
    return edge_index, edge_weight

def normalize_adj(edge_index, num_nodes, edge_weight = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def compute_adj2(edge_index, num_nodes):
    print("Computing adj2")
    n_edge_index, n_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes = num_nodes)
    adj = SparseTensor(row=n_edge_index[0], col=n_edge_index[1], value=n_edge_weight,
                sparse_sizes=(num_nodes, num_nodes))
    adj2 = adj.matmul(adj)

    return adj2

def compute_norm_aax(x, edge_index, num_nodes):
    print("Start computing aax")
    new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                sparse_sizes=(num_nodes, num_nodes))
    adj_matrix2 = adj.matmul(adj)
    aax = adj_matrix2.matmul(x)
    x = aax.to_dense()

    return x

def compute_sim(norm_aax, num_nodes):
    print("compute sim")

    similarity_feature = torch.mm(norm_aax, norm_aax.t())
    dis_range = torch.max(similarity_feature) - torch.min(similarity_feature)
    similarity_feature = (similarity_feature - torch.min(similarity_feature))/dis_range

    return similarity_feature


class RIMSampler(NodeSampler):
    def __init__():
        