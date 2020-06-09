import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
from math import ceil
import torch
import networkx as nx
from utils import *


def simulate(args, logger, device):
    results = {}
    for n in args.n:
        logger.info('n = {}'.format(n))
        G = generate_many_k_regular_graphs(k=args.k, n=n, N=args.N, seed=args.seed)
        data = get_data(G, task='simulation', labels=None, device=device, args=args, logger=logger)  #TODO
        features, labels, adj_matrix, train_mask, test_mask, set_indices = data
        for T in range(1, args.T+1):
            model = GNNModel(layers=T, in_features=features.shape[-1], hidden_features=args.hidden_features,
                             out_features=32, prop_depth=args.prop_depth, dropout=args.dropout, set_indices=set_indices,
                             model_name=args.model)
            model.to(device)
            output = run_simulation(model, data, args=args)  # output shape [G.number_of_nodes(), feat_dim]
            collision_rate = compute_simulation_collisions(output, ratio=True)
            results[(n, T)] = collision_rate
            torch.cuda.empty_cache()
            logger.info('T = {}: {}'.format(T, collision_rate))
        logger.info('#'*30)
    return results


def generate_many_k_regular_graphs(k, n, N, seed=0):
    ngraph = int(ceil(N/n))
    graphs = [generate_k_regular(k, n, s) for s in range(seed, seed+ngraph)]
    index_base = 0
    edge_list = []
    for graph in graphs:
        edge_list.extend([(src+index_base, dst+index_base) for src, dst in list(graph.edges)])
        index_base += n
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G


def generate_k_regular(k, n, seed=0):
    G = nx.random_regular_graph(d=k, n=n, seed=seed)
    return G


def run_simulation(model, data, args):
    features, _, adj_matrix, _, _, _ = data
    device = features.device
    n_samples = features.shape[1]
    bs = args.bs
    model.eval()
    predictions = []
    with torch.no_grad():
        count = 0
        while count < n_samples:
            minibatch = torch.tensor(list(range(count, min(count + bs, n_samples)))).long().to(device)
            count = count + bs
            adj_batch = adj_matrix[minibatch].to(device) if args.model != 'PGNN' else adj_matrix
            prediction = model(features[:, minibatch, :], adj_batch, minibatch)  # shape: [n_nodes, feat_dim]
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
    return predictions


def save_simulation_result(results, pic_format='png'):
    n_l, T_l, r_l = [], [], []
    for (n, T), r in results.items():
        n_l.append(n)
        T_l.append(T)
        r_l.append(r)
    plt.scatter(n_l, T_l, c=r_l, cmap="Greys")
    plt.xscale('log')
    plt.xlabel('number of nodes (n)')
    plt.ylabel('number of network layers (T)')
    plt.savefig('./simulation_results.{}'.format(pic_format), dpi=300)


def compute_simulation_collisions(outputs, ratio=True):
    epsilon = 1e-10
    N = outputs.size(0)
    with torch.no_grad():
        a = outputs.unsqueeze(-1)
        b = outputs.t().unsqueeze(0)
        diff = a-b
        diff = (diff**2).sum(dim=1)
        n_collision = int(((diff < epsilon).sum().item()-N)/2)
        r = n_collision / (N*(N-1)/2)
    if ratio:
        return r
    else:
        return n_collision
