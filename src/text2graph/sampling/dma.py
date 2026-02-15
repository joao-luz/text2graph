from ..llm import LLM
from .sampling import NodeSampler

import copy
import json
import pathlib
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, scatter, add_self_loops
from torch_sparse import SparseTensor
from transformers import AutoModel, AutoTokenizer


# Originaly written in C++, ported to Python https://github.com/PKU-DAIR/Noisy-LLM-Oracle/blob/main/src/lib/cpp_extension/utils.cpp
def get_max_reliable_influ_node(high_score_nodes, activated_node, reliability_list, th, adj_rowptr, adj_col, adj_value, is_first):
    num_node = reliability_list.size(0)
    num_high_score_nodes = high_score_nodes.size(0)

    activated_node_num_list = []

    for i in range(num_high_score_nodes):
        node = high_score_nodes[i].item()

        reliable_score = 1.0 if is_first else reliability_list[node].item()

        node_st = adj_rowptr[node].item()
        node_ed = adj_rowptr[node + 1].item()

        cur_temp_adj = torch.zeros(num_node, dtype=torch.float32)

        neighbors = adj_col[node_st:node_ed]
        values = adj_value[node_st:node_ed]
        cur_temp_adj[neighbors] = values

        mask = (cur_temp_adj * reliable_score > th).float()

        activated_num = int((mask * activated_node).sum().item())

        activated_node_num_list.append(activated_num)

        activated_node_num_tensor = torch.tensor(activated_node_num_list)
        max_idx = torch.argmax(activated_node_num_tensor).item()

        max_ral_node = high_score_nodes[max_idx].item()
        max_activated_num = activated_node_num_list[max_idx]

    reliable_score = 1.0 if is_first else reliability_list[max_ral_node].item()

    max_node_st = adj_rowptr[max_ral_node].item()
    max_node_ed = adj_rowptr[max_ral_node + 1].item()

    temp_max_adj = torch.zeros(num_node, dtype=torch.float32)

    neighbors = adj_col[max_node_st:max_node_ed]
    values = adj_value[max_node_st:max_node_ed]
    temp_max_adj[neighbors] = values

    max_mask = (temp_max_adj * reliable_score > th).float()
    max_activated_node = max_mask * activated_node

    return max_ral_node, max_activated_node, max_activated_num


# Originaly written in C++, ported to Python https://github.com/PKU-DAIR/Noisy-LLM-Oracle/blob/main/src/lib/cpp_extension/utils.cpp
def update_reliability_single_node(reliability_score, num_node, node, adj_vec, labels, idx_used_mask, similarity_feat, labels_sim, sim_label, num_class, visited):
    node_label = labels[node].item()

    relative_influence_vec = torch.zeros(num_node, dtype=torch.float32, device=reliability_score.device)
    influence_sum = 0.0

    for n in range(num_node):
        if adj_vec[n].item() == 0 or n == node:
            continue

        n_label = labels[n].item()

        diag_prob = labels_sim[node_label][node_label]
        sim_feat = similarity_feat[n]

        if idx_used_mask[n].item() == 1:
            off_prob = labels_sim[node_label][node_label]
        else:
            off_prob = sim_label[node_label]

        numerator = diag_prob * sim_feat
        denominator = diag_prob * sim_feat + off_prob * (1 - sim_feat) / (num_class - 1)

        relative_influence = numerator / (denominator + 1e-12)

        relative_influence_vec[n] = relative_influence
        influence_sum += relative_influence.item()

    relative_influence_vec = relative_influence_vec / (influence_sum + 1e-8)

    for n in range(num_node):
        if adj_vec[n].item() == 0 or n == node:
            continue

        if visited[n].item() == 0:
            reliability_score[n] = 0.0

        reliability_score[n] += reliability_score[node] * relative_influence_vec[n]
        visited[n] = 1.0


def compute_rw_norm_edge_index(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                 device=edge_index.device)
    row,_ = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    edge_index, tmp = add_self_loops(edge_index, edge_weight,
                                     fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp
    return edge_index, edge_weight


def normalize_adj(edge_index, num_nodes, edge_weight=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)

    row,_ = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight

    return edge_index, edge_weight


def compute_adj2(edge_index, num_nodes):
    n_edge_index, n_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes=num_nodes)
    adj = SparseTensor(row=n_edge_index[0], col=n_edge_index[1], value=n_edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    return adj


def compute_norm_aax(x, edge_index, num_nodes):
    new_edge_index, new_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes=num_nodes)
    adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    aax = adj.matmul(x)
    aax = adj.matmul(aax)
    x = aax.to_dense()

    return x


def compute_sim(norm_aax, num_nodes):
    similarity_feature = torch.mm(norm_aax, norm_aax.t())
    dis_range = torch.max(similarity_feature) - torch.min(similarity_feature)
    similarity_feature = (similarity_feature -
                          torch.min(similarity_feature))/dis_range

    return similarity_feature


def update_reliability(labels_sim, adj_matrix2, idx_used, labels, num_node, reliability_list, oracle_acc, similarity_feature, th, is_first=None):
    alpha = oracle_acc
    visited = torch.zeros(num_node)
    if is_first:
        reliability_list[idx_used] = alpha

    num_class = len(labels_sim)
    sim_label = []
    sim_label_cor = 0.
    for i in range(num_class):
        sim_label.append(
            (sum(labels_sim[i])-labels_sim[i][i])/(num_class-1))
        sim_label_cor += labels_sim[i][i]
    sim_label_cor /= num_class
    sim_label = torch.tensor(sim_label, dtype=torch.float32)

    cur_similarity_feature = torch.mm(similarity_feature[idx_used], similarity_feature.t())
    dis_range = torch.max(cur_similarity_feature) - torch.min(cur_similarity_feature)
    cur_similarity_feature = (cur_similarity_feature -
                          torch.min(cur_similarity_feature))/dis_range

    for i, node in enumerate(idx_used):
        idx_used_mask = torch.zeros(num_node)
        idx_used_mask[idx_used] = 1

        update_reliability_single_node(reliability_list, num_node, node, adj_matrix2[node].to_dense(), labels, idx_used_mask, 
                                       cur_similarity_feature[i], labels_sim, sim_label, num_class, visited)

    # normalize the realiability list
    dis_range = torch.max(reliability_list) - torch.min(reliability_list)
    reliability_list = (reliability_list -
                            torch.min(reliability_list))/dis_range


def dma(features, labels, edge_index, total_node_number, labels_sim, idx_avilable, total_budget=140, oracle_acc=1, th=0.05, batch_size=5):   
    reliability_list = torch.ones(total_node_number)
    all_idx = torch.arange(total_node_number)
    adj_matrix2 = compute_adj2(edge_index, total_node_number)
    adj_rowptr, adj_col, adj_value = adj_matrix2.csr()
    norm_aax = compute_norm_aax(features, edge_index, total_node_number)
    similarity_feature = norm_aax

    idx_train = []
    idx_available = all_idx[idx_avilable].tolist()
    idx_available_temp = copy.deepcopy(idx_available)
    activated_node = torch.ones(total_node_number)
    count = 0
    iter = 0
    train_class = {}

    while True:
        max_ral_node, max_activated_node, max_activated_num = get_max_reliable_influ_node(
            torch.tensor(idx_available_temp), activated_node, reliability_list, th, adj_rowptr, adj_col, adj_value, is_first=(iter == 0))

        idx_train.append(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label] = list()
            train_class[node_label].append(max_ral_node)
        count += 1

        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)

        if count % batch_size == 0:
            update_reliability(labels_sim, adj_matrix2.to_dense(), idx_train, labels, total_node_number,
                                                reliability_list, oracle_acc, similarity_feature, th, is_first=(iter == 0))
            iter += 1

        if count >= total_budget or max_activated_num <= 0:
            break

    train_mask = torch.zeros(total_node_number)
    train_mask[idx_train] = 1

    return train_mask.bool()


def generate_pseudo_samples(model_path, prompt_template, categories, pseudo_sample_path=None, temperature=0.7):
    if pseudo_sample_path and pathlib.Path(pseudo_sample_path).is_file():
        with open(pseudo_sample_path) as f:
            data = json.load(f)
        pseudo_samples = list(data.values())

        return pseudo_samples

    prompts = [prompt_template.format(category=category) for category in categories]
    
    model = LLM(model_path)
    pseudo_samples = model.query(prompts, temperature)
    del model

    if pseudo_sample_path:
        pathlib.Path(pseudo_sample_path).parent.mkdir(parents=True, exist_ok=True)
        data = {category: sample for category,sample in zip(categories, pseudo_samples)}
        with open(pseudo_sample_path, 'w') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False)

    return pseudo_samples


def generate_similarities(model_path, pseudo_samples, label_embedding_path=None):
    if label_embedding_path and pathlib.Path(label_embedding_path).is_file():
        embeddings = torch.load(label_embedding_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        inputs = tokenizer(pseudo_samples, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_states = outputs['last_hidden_state']
        attention_masks = inputs['attention_mask']
        unsqueezed_attention_masks = torch.unsqueeze(attention_masks, dim=-1)
        masked_hidden_states = last_hidden_states * unsqueezed_attention_masks
        hidden_sum = torch.sum(masked_hidden_states, dim=1)
        attention_sum = torch.sum(unsqueezed_attention_masks, dim=1)
        embeddings = torch.div(hidden_sum, attention_sum)

        if label_embedding_path:
            pathlib.Path(label_embedding_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, label_embedding_path)

    feature_similarity = torch.mm(embeddings, embeddings.t())
    norm = torch.norm(feature_similarity, 2, 1, keepdim=True).add(1e-8)
    feature_similarity = torch.div(feature_similarity, norm)
    feature_similarity = torch.div(feature_similarity, norm.t())
    feature_similarity = F.normalize(feature_similarity, p=1, dim=1)

    return feature_similarity


class DMASampler(NodeSampler):
    def __init__(self, model_path, prompt_template, pseudo_sample_path=None, label_embedding_path=None, n=None):
        super().__init__(n, 'dma_sampler')

        self.pseudo_sample_path = pseudo_sample_path
        self.label_embedding_path = label_embedding_path
        self.model_path = model_path
        self.prompt_template = prompt_template

        if pseudo_sample_path: self.str_parameters['pseudo_sample_path'] = pseudo_sample_path
        if label_embedding_path: self.str_parameters['label_embedding_path'] = label_embedding_path
    
    def sample_nodes(self, data, n=None):
        data = data.clone()

        n = n or self.n

        categories = list(data.label2id.keys())

        # Generate pseudo_samples
        pseudo_samples = generate_pseudo_samples(self.model_path, self.prompt_template, categories, self.pseudo_sample_path)
        # Extract embeddings and compute similarities between representations
        label_similarities = generate_similarities(self.model_path, pseudo_samples, self.label_embedding_path)

        features = data.x
        labels = data.y if hasattr(data, 'y') else torch.full((data.num_nodes,), -1)
        edge_index = data.edge_index
        total_node_number = data.num_nodes
        available_nodes = labels != -1
        sample_mask = dma(features, labels, edge_index, total_node_number, label_similarities, available_nodes, total_budget=n)

        return sample_mask
    
    def __call__(self, data, n=None):
        data = data.clone()

        sample_mask = self.sample_nodes(data, n)

        data.sample_mask = sample_mask

        print(len(sample_mask.nonzero().flatten()))

        return data
