import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from models.models import *
import random
import os
import multiprocessing as mp
from tqdm import tqdm
import time
import sys
from copy import deepcopy
from torch_geometric.data import DataLoader, Data
import torch_geometric.utils as tgu
from debug import *


def check(args):
    if args.dataset == 'foodweb' and not args.directed:
        raise Warning('dataset foodweb is essentially a directed network but currently treated as undirected')
    if args.dataset == 'simulation':
        if args.n is None:
            args.n = [10, 20, 40, 80, 160, 320, 640, 1280]
        if args.max_sp < args.T:
            raise Warning('maximum shortest path distance (max_sp) is less than max number of layers (T), which may deteriorate model capability')


def get_device(args):
    gpu = args.gpu
    return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')


def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 100
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage())) + sys.getsizeof(dataset[i].set_indices.storage())
        gb = storage*total_length/sample_size/1e9
        total_gb += gb
    logger.info('Data roughly takes {:.4f} GB in total'.format(total_gb))
    return total_gb


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_label(dir, task):
    if task == 'node_classification':
        f_path = dir + 'labels.txt'
        fin_labels = open(f_path)
        labels = []
        node_id_mapping = dict()
        for new_id, line in enumerate(fin_labels.readlines()):
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
        fin_labels.close()
    else:
        labels = None
        nodes = []
        with open(dir + 'edges.txt') as ef:
            for line in ef.readlines():
                nodes.extend(line.strip().split()[:2])
        nodes = sorted(list(set(nodes)))
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def read_file(args, logger):
    dataset = args.dataset
    di_flag = args.directed
    if dataset in ['brazil-airports', 'europe-airports', 'usa-airports', 'foodweb', 'karate']:
        task = 'node_classification'
    elif dataset in ['arxiv', 'celegans', 'celegans_small', 'ogbl_collab','facebook', 'ns', 'pb', 'power', 'router', 'usair', 'yeast']:
        task = 'link_prediction'
    elif dataset in ['arxiv_tri', 'celegans_tri', 'celegans_small_tri', 'facebook_tri', 'ns_tri', 'pb_tri', 'power_tri', 'router_tri', 'usair_tri', 'yeast_tri']:
        task = 'triplet_prediction'
    elif dataset in ['simulation']:
        task = 'simulation'
    else:
        raise ValueError('dataset not found')

    directory = './data/' + task + '/' + dataset + '/'
    labels, node_id_mapping = read_label(directory, task=task)
    edges = read_edges(directory, node_id_mapping)
    if not di_flag:
        G = nx.Graph(edges)
    else:
        G = nx.DiGraph(edges)
    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    if args.use_degree:
        attributes += np.expand_dims(np.log(get_degrees(G)+1), 1).astype(np.float32)
    if args.use_attributes:
        # TODO: read in attribute file to concat to axis -1 of attributes, raise error if not found
        raise NotImplementedError
    G.graph['attributes'] = attributes
    logger.info('Read in {} for {} --  number of nodes: {}, number of edges: {}, number of labels: {}. Directed: {}'.format(dataset, task,
                                                                                                                G.number_of_nodes(),
                                                                                                                G.number_of_edges(),
                                                                                                                len(labels) if labels is not None else 0,
                                                                                                               di_flag))
    labels = np.array(labels) if labels is not None else None
    return (G, labels), task


def get_data(G, task, args, labels, logger):
    G = deepcopy(G)  # to make sure original G is unchanged
    if args.debug:
        logger.info(list(G.edges))
    # di_flag = isinstance(G, nx.classes.digraph.DiGraph)
    # deg_flag = args.use_degree
    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature
    # norm_flag = args.adj_norm
    feature_flags = (sp_flag, rw_flag)
    # TODO: adapt the whole branch for simulation
    if task == 'simulation':
        set_indices = np.expand_dims(np.arange(G.number_of_nodes()), 1)
        data_list = extract_subgaphs(G, labels, set_indices, prop_depth=args.prop_depth, layers=args.layers,
                                 feature_flags=feature_flags, task=task,
                                 max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, debug=args.debug)
        loader = DataLoader(data_list, batch_size=args.bs, shuffle=False, num_workers=0)
        return loader
    G, labels, set_indices, (train_mask, val_test_mask) = generate_samples_labels_graph(G, labels, task, args, logger)
    if args.debug:
        logger.info(list(G.edges))
    data_list = extract_subgaphs(G, labels, set_indices, prop_depth=args.prop_depth, layers=args.layers,
                                 feature_flags=feature_flags, task=task,
                                 max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, debug=args.debug)
    train_set, val_set, test_set = split_datalist(data_list, (train_mask, val_test_mask))
    if args.debug:
        print_dataset(train_set, logger)
        print_dataset(val_set, logger)
        print_dataset(test_set, logger)
    train_loader, val_loader, test_loader = load_datasets(train_set, val_set, test_set, bs=args.bs)
    logger.info('Train size :{}, val size: {}, test size: {}, val ratio: {}, test ratio: {}'.format(len(train_set), len(val_set), len(test_set), args.test_ratio, args.test_ratio))
    return (train_loader, val_loader, test_loader), len(np.unique(labels))


def generate_samples_labels_graph(G, labels, task, args, logger):
    if labels is None:
        logger.info('Labels unavailable. Generating training/test instances from dataset ...')
        G, labels, set_indices, (train_mask, val_test_mask) = generate_set_indices_labels(G, task, test_ratio=2*args.test_ratio, data_usage=args.data_usage)
    else:
        # training on nodes or running on synthetic data
        logger.info('Labels provided (node-level task).')
        assert(G.number_of_nodes() == labels.shape[0])
        n_samples = int(round(labels.shape[0] * args.data_usage))
        set_indices = np.random.choice(G.number_of_nodes(), n_samples, replace=False)
        labels = labels[set_indices]
        set_indices = np.expand_dims(set_indices, 1)
        train_mask, val_test_mask = split_dataset(set_indices.shape[0], test_ratio=2*args.test_ratio, stratify=labels)
    logger.info('Generate {} train+val+test instances in total. data_usage: {}.'.format(set_indices.shape[0], args.data_usage))
    return G, labels, set_indices, (train_mask, val_test_mask)


def generate_set_indices_labels(G, task, test_ratio, data_usage=1.0):
    G = G.to_undirected()  # the prediction task completely ignores directions
    pos_edges, neg_edges = sample_pos_neg_sets(G, task, data_usage=data_usage)  # each shape [n_pos_samples, set_size], note hereafter each "edge" may contain more than 2 nodes
    n_pos_edges = pos_edges.shape[0]
    assert(n_pos_edges == neg_edges.shape[0])
    pos_test_size = int(test_ratio * n_pos_edges)

    set_indices = np.concatenate([pos_edges, neg_edges], axis=0)
    test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)  # randomly pick pos edges for test
    test_neg_indices = list(range(n_pos_edges, n_pos_edges + pos_test_size))  # pick first pos_test_size neg edges for test
    test_mask = get_mask(test_pos_indices + test_neg_indices, length=2*n_pos_edges)
    train_mask = np.ones_like(test_mask) - test_mask
    labels = np.concatenate([np.ones((n_pos_edges, )), np.zeros((n_pos_edges, ))]).astype(np.int32)
    G.remove_edges_from([node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index, 2)])

    # permute everything for stable training
    permutation = np.random.permutation(2*n_pos_edges)
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    train_mask, test_mask = train_mask[permutation], test_mask[permutation]

    return G, labels, set_indices, (train_mask, test_mask)


def extract_subgaphs(G, labels, set_indices, prop_depth, layers, feature_flags, task, max_sprw, parallel, logger, debug=False):
    # deal with adj and features
    logger.info('Encode positions ... (Parallel: {})'.format(parallel))
    data_list = []
    hop_num = get_hop_num(prop_depth, layers, max_sprw, feature_flags)
    n_samples = set_indices.shape[0]
    if not parallel:
        for sample_i in tqdm(range(n_samples)):
            data = get_data_sample(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   label=labels[sample_i] if labels is not None else None, debug=debug)
            data_list.append(data)
    else:
        pool = mp.Pool(4)
        results = pool.map_async(parallel_worker,
                                 [(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   labels[sample_i] if labels is not None else None, debug) for sample_i in range(n_samples)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.2)
        data_list = results.get()
        pool.close()
        pbar.close()
    return data_list


def parallel_worker(x):
    return get_data_sample(*x)


def get_data_sample(G, set_index, hop_num, feature_flags, max_sprw, label, debug=False):
    # first, extract subgraph
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if len(set_index) > 1:
        G = G.copy()
        G.remove_edges_from(combinations(set_index, 2))
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1)
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), hop_num, edge_index, num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert(new_G.number_of_nodes() == num_nodes)

    # Construct x from x_list
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    new_set_index = new_set_index.long().unsqueeze(0)
    if not debug:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index)
    else:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index,
                    old_set_indices=torch.tensor(set_index).long().unsqueeze(0), old_subgraph_indices=subgraph_node_old_index)


def get_model(layers, in_features, out_features, prop_depth, args, logger):
    model_name = args.model
    if model_name in ['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hidden_features,
                         out_features=out_features, prop_depth=prop_depth, dropout=args.dropout,
                         model_name=model_name)
    else:
        return NotImplementedError
    logger.info(model.short_summary())
    return model


def get_features_sp_sample(G, node_set, max_sp):
    dim = max_sp + 2
    set_size = len(node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    epsilon = 1e-6
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)
        rw_list.append(rw)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)
    return features_rw


def get_hop_num(prop_depth, layers, max_sprw, feature_flags):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * layers) + 1   # in order to get the correct degree normalization for the subgraph


def shortest_path_length(graph):
    sp_length = np.ones([graph.number_of_nodes(), graph.number_of_nodes()], dtype=np.int32) * -1
    for node1, value in nx.shortest_path_length(graph):
        for node2, length in value.items():
            sp_length[node1][node2] = length

    return sp_length


def split_dataset(n_samples, test_ratio, stratify=None):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio, stratify=stratify)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)


def sample_pos_neg_sets(G, task, data_usage=1.0):
    if task == 'link_prediction':
        pos_edges = np.array(list(G.edges), dtype=np.int32)
        set_size = 2
    elif task == 'triplet_prediction':
        pos_edges = np.array(collect_tri_sets(G))
        set_size = 3
    else:
        raise NotImplementedError

    if data_usage < 1-1e-6:
        pos_edges, sample_i = retain_partial(pos_edges, ratio=data_usage)
    neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=set_size), dtype=np.int32)
    return pos_edges, neg_edges


def sample_neg_sets(G, n_samples, set_size):
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break

    return neg_sets


def collect_tri_sets(G):
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i


def pagerank_inverse(adj, alpha=0.90):
    adj /= (adj.sum(axis=-1, keepdims=True) + 1e-12)
    return np.linalg.inv(np.eye(adj.shape[0]) - alpha * np.transpose(adj, axes=(0,1)))


def split_datalist(data_list, masks):
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert((train_mask.sum()+val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert(train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(0.5*len(val_test_indices)), stratify=val_test_labels)
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def split_indices(num_graphs, test_ratio, stratify=None):
    test_size = int(num_graphs*test_ratio)
    val_size = test_size
    train_val_set, test_set = train_test_split(np.arange(num_graphs), test_size=test_size, shuffle=True, stratify=stratify)
    train_set, val_set = train_test_split(train_val_set, test_size=val_size, shuffle=True, stratify=stratify[train_val_set])
    return train_set, val_set, test_set


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


# ================================== (obsolete) Just for PGNN =================================================
# Adapted from https://github.com/JiaxuanYou/P-GNN
def get_PGNN_anchor_set_distances(layers, G, test_set_indices, c=1):
    G = deepcopy(G)
    num_nodes = G.number_of_nodes()
    device = test_set_indices.device
    if test_set_indices.size(1) > 1:
        edges_to_remove = [[i, j] for set_index in list(test_set_indices.cpu().numpy()) for i, j in combinations(set_index, 2) ]
        G.remove_edges_from(edges_to_remove)
    dists = np.asarray(nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes())).todense().astype(np.float64))  # [n_nodes, n_nodes]
    dists = torch.from_numpy(dists).float().to(device)
    anchorset_id = get_random_anchorset(num_nodes, c=c)
    dists_max_l, dists_argmax_l = [], []
    for layer_i in range(layers):
        dists_max, dists_argmax = get_dist_max(anchorset_id, dists, device=device)
        dists_max_l.append(dists_max)
        dists_argmax_l.append(dists_argmax)
        # TODO: collect the two variables
    dists_max = torch.stack(dists_max_l).float()
    dists_argmax = torch.stack(dists_argmax_l).float()
    compact_distance_scores_and_args = torch.stack([dists_max, dists_argmax], dim=1)
    return compact_distance_scores_and_args  # shape: [layers, 2, N, NAS]


def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = dist_argmax_temp
    return dist_max, dist_argmax


class ObjectView:
    def __init__(self, d):
        self.__dict__ = d

# TODO: 1. check if storage allows, send all data to gpu 5. (optional) add directed graph
# TODO: 6. (optional) enable using original node attributes as initial feature (only need to modify file readin)
# TODO: 7. (optional) rw using sparse matrix for multiplication
