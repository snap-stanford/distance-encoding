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


def print_storage(tensors, names, logger):
    for tensor, name in zip(tensors, names):
        logger.info('{} takes {} MB'.format(name, sys.getsizeof(tensor.storage())/1e6))


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
    if dataset in ['brazil-airports', 'europe-airports', 'usa-airports', 'foodweb']:
        task = 'node_classification'
    elif dataset in ['arxiv', 'celegans', 'celegans_small', 'facebook', 'ns', 'pb', 'power', 'router', 'usair', 'yeast']:
        task = 'link_prediction'
    elif dataset in ['arxiv_tri', 'celegans_tri', 'celegans_small_tri', 'facebook_tri', 'ns_tri', 'pb_tri', 'power_tri', 'router_tri', 'usair_tri', 'yeast_tri']:
        task = 'triplet_prediction'
    elif dataset in ['simulation']:
        task = 'simulation'
    else:
        raise ValueError('dataset not found')

    directory = './data/' + task + '/' + dataset + '/'
    if task == 'simulation':
        labels = None

    else:
        labels, node_id_mapping = read_label(directory, task=task)
        edges = read_edges(directory, node_id_mapping)

        if not di_flag:
            G = nx.Graph(edges)
        else:
            G = nx.DiGraph(edges)
        logger.info('Read in {} for {} --  number of nodes: {}, number of edges: {}, number of labels: {}. Directed: {}'.format(dataset, task,
                                                                                                                    G.number_of_nodes(),
                                                                                                                    G.number_of_edges(),
                                                                                                                    len(labels) if labels is not None else 0,
                                                                                                                   di_flag))
    labels = np.array(labels) if labels is not None else None
    return (G, labels), task


def get_data(G, task, device, args, labels, logger):
    G = deepcopy(G)  # to make sure original G is unchanged
    di_flag = isinstance(G, nx.classes.digraph.DiGraph)
    deg_flag = args.use_degree
    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature
    norm_flag = args.adj_norm
    feature_flags = (di_flag, deg_flag, sp_flag, rw_flag, norm_flag)
    if task == 'simulation':
        set_indices = np.expand_dims(np.arange(G.number_of_nodes()), 1)
        adj_matrix, features = get_adj_features(G, set_indices=set_indices, prop_depth=args.prop_depth,
                                                feature_flags=feature_flags, task=task,
                                                max_sprw=(args.max_sp, args.rw_depth),
                                                parallel=args.parallel, return_adj=args.model != 'PGNN')
        features = torch.from_numpy(features).float().to(device)
        set_indices = torch.from_numpy(set_indices).long().to(device)
        adj_matrix = torch.from_numpy(adj_matrix).float()
        return [features, None, adj_matrix, None, None, set_indices]

    if labels is None:
        logger.info('Labels unavailable. Generating training/test instances from dataset ...')
        G, labels, set_indices, (train_mask, test_mask) = generate_set_indices_labels(G, task, test_ratio=args.test_ratio, data_usage=args.data_usage)
        logger.info('Generate {} pos/neg training & test instances in total.'.format(set_indices.shape[0]))
    else:
        # training on nodes or running on synthetic data
        logger.info('Labels available (node-level task) or no need to train model')
        set_indices = np.expand_dims(np.arange(G.number_of_nodes()), 1)
        if args.data_usage < 1.0 - 1e-6:
            set_indices, sample_i = retain_partial(set_indices, args.data_usage)
            labels = labels[sample_i] if not isinstance(labels, bool) else None
        train_mask, test_mask = split_dataset(set_indices.shape[0], test_ratio=args.test_ratio)
    logger.info('Training size :{}, test size: {}, test ratio: {}'. format(int(train_mask.sum()), int(test_mask.sum()), args.test_ratio))
    # deal with adj and features
    logger.info('Encode positions ... (Parallel: {})'.format(args.parallel))
    adj_matrix, features = get_adj_features(G, set_indices=set_indices, prop_depth=args.prop_depth,
                                            feature_flags=feature_flags, task=task, max_sprw=(args.max_sp, args.rw_depth),
                                            parallel=args.parallel, return_adj=args.model!='PGNN')
    # to device
    features = torch.from_numpy(features).float().to(device)
    train_mask = torch.from_numpy(train_mask).float().to(device)
    test_mask = torch.from_numpy(test_mask).float().to(device)
    set_indices = torch.from_numpy(set_indices).long().to(device)
    if args.model != 'PGNN':
        adj_matrix = torch.from_numpy(adj_matrix).float()  # WARNING: not yet to device here!!!!
    else:
        adj_matrix = get_PGNN_anchor_set_distances(args.layers, G, set_indices[test_mask.bool()])  # adj_batch shape: [layers, 2, N, NAS]
    if not isinstance(labels, bool):
        labels = torch.from_numpy(labels).long().to(device)
        assert(len(labels) == set_indices.size(0))
    return [features, labels, adj_matrix, train_mask, test_mask, set_indices]


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


def get_adj_features(G, set_indices, prop_depth, feature_flags, task, max_sprw, parallel, return_adj=True):
    n_samples = set_indices.shape[0]
    adj_matrix, features = [], []
    if not parallel:
        for sample_i in tqdm(range(n_samples)):
            set_index = set_indices[sample_i]
            adj_sample, features_sample = get_adj_features_sample(G, set_index, prop_depth, feature_flags, max_sprw)
            adj_matrix.append(adj_sample)
            features.append(features_sample)
    else:
        pool = mp.Pool(4)
        results = pool.map_async(parallel_worker, [(G, set_indices[sample_i], prop_depth, feature_flags, max_sprw) for sample_i in range(n_samples)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.5)
        results = results.get()
        pool.close()
        pbar.close()
        for adj_sample, features_sample in results:
            adj_matrix.append(adj_sample)
            features.append(features_sample)
    if return_adj:
        adj_matrix = np.stack(adj_matrix)  # shape [n_samples, prop_depth, N, N]
    else:
        adj_matrix = None
    features = np.stack(features, axis=1)  # shape [N, n_samples, F]
    return adj_matrix, features


def parallel_worker(x):
    return get_adj_features_sample(*x)


def get_adj_features_sample(G, set_index, prop_depth, feature_flags, max_sprw):
    set_index = list(set_index)
    di_flag, deg_flag, sp_flag, rw_flag, norm_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if len(set_index) > 1:
        new_G = deepcopy(G)
        new_G.remove_edges_from(combinations(set_index, 2))
    else:
        new_G = G
    adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=range(new_G.number_of_nodes())).todense().astype(np.float64))  # [n_nodes, n_nodes]
    adj_sample = get_adj_sample(adj, prop_depth, norm=norm_flag)  # shape: [prop_depth, N, N]

    features_sample = []
    if deg_flag:
        degree = np.sum(adj, axis=1)
        features_sample.append(np.expand_dims(np.log(degree+1), axis=1))
    if sp_flag:
        features_sample.append(get_features_sp_sample(new_G, set_index, max_sp=max_sp))
    if rw_flag:
        features_sample.append(get_features_rw_sample(adj, set_index, rw_depth=rw_depth))
    if di_flag:
        adj_t = adj.transpose((0, 1))
        adj_sample_t = get_adj_sample(adj_t, prop_depth=prop_depth, norm=norm_flag)[1:]
        if adj_sample_t.ndim < 3:
            adj_sample_t = np.expand_dims(adj_sample_t, 0)
        adj_sample = np.concatenate([adj_sample, adj_sample_t], axis=0)
        if deg_flag:
            degree_t = np.sum(adj_t, axis=1)
            features_sample.append(np.expand_dims(np.log(degree_t + 1), axis=1))
        if sp_flag:
            features_sample.append(get_features_sp_sample(new_G.reverse(), set_index, max_sp=max_sp))
        if rw_flag:
            features_sample.append(get_features_rw_sample(adj_t, set_index, rw_depth=rw_depth))
    features_sample = np.concatenate(features_sample, axis=1)
    return adj_sample, features_sample


def get_model(layers, in_features, out_features, set_indices, prop_depth, device, args, logger):
    model_name = args.model
    if model_name in ['PE-GCN', 'GIN', 'GCN', 'GraphSAGE', 'PGNN']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hidden_features,
                         out_features=out_features, prop_depth=prop_depth, dropout=args.dropout, set_indices=set_indices,
                         model_name=model_name)
    else:
        return NotImplementedError
    model.to(device)
    logger.info(model.short_summary())
    return model


def get_adj_sample(adj, prop_depth, norm='asym'):
    if prop_depth == -1:
        adj = pagerank_inverse(adj)
        return np.stack([np.eye(adj.shape[0]), adj]).astype(dtype='float32')
    epsilon = 1e-6
    if norm == 'sym':
        adj += np.eye(adj.shape[0])  # renormalization trick
    adj_sample = np.empty((prop_depth+1, adj.shape[0], adj.shape[0]), dtype=np.int16)
    adj_sample[0] = np.identity(adj.shape[0], dtype=np.int16)
    adj_acc = np.identity(adj.shape[0], dtype=np.int16)
    for i in range(0, prop_depth):
        adj_sample[i+1] = np.minimum(np.matmul(adj_sample[i], adj), 1)
        adj_sample[i+1] = np.maximum(adj_sample[i+1] - adj_acc, 0)
        adj_acc += adj_sample[i+1]
    adj_sample = adj_sample.astype(dtype='float32', copy=False)
    if norm == 'asym':
        adj_sample /= (adj_sample.sum(axis=2, keepdims=True) + epsilon)
    elif norm == 'sym':
        adj_sample /= (adj_sample.sum(axis=2, keepdims=True) ** 0.5 + epsilon)
        adj_sample /= (adj_sample.sum(axis=1, keepdims=True) ** 0.5 + epsilon)
    return adj_sample


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
    rw = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])]
    for _ in range(rw_depth):
        rw_list.append(rw)
        rw = np.matmul(rw_list[-1], rw)
    features_rw = np.stack(rw_list, axis=2)[:, node_set, :].sum(axis=1)
    return features_rw


def shortest_path_length(graph):
    sp_length = np.ones([graph.number_of_nodes(), graph.number_of_nodes()], dtype=np.int32) * -1
    for node1, value in nx.shortest_path_length(graph):
        for node2, length in value.items():
            sp_length[node1][node2] = length

    return sp_length


def split_dataset(n_samples, test_ratio=0.2):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)


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

# ================================== Just for PGNN =================================================
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

