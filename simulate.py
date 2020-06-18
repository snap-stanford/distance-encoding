import matplotlib.pyplot as plt
import logging
from math import ceil
from utils import *
from train import eval_model
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def simulate(args, logger):
    device = get_device(args)
    results = {}
    for n in args.n:
        logger.info('n = {}'.format(n))
        G = generate_many_k_regular_graphs(k=args.k, n=n, N=args.N, seed=args.seed)
        for T in range(1, args.T+1):
            args.layers = T
            loader = get_data(G, task='simulation', args=args, labels=None, logger=logger)
            model = GNNModel(layers=T, in_features=loader.dataset[0].x.shape[-1], hidden_features=args.hidden_features,
                             out_features=32, prop_depth=args.prop_depth, dropout=args.dropout,
                             model_name=args.model)
            model.to(device)
            output = run_simulation(model, loader, device)  # output shape [G.number_of_nodes(), feat_dim]
            if args.debug:
                print('output', output)
            collision_rate = compute_simulation_collisions(output, ratio=True)
            results[(n, T)] = collision_rate
            torch.cuda.empty_cache()
            logger.info('T = {}: {}'.format(T, collision_rate))
            if args.debug:
                print_dataset(loader.dataset, logger)
        logger.info('#'*30)
        if args.debug:
            break
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
    G.graph['attributes'] = np.expand_dims(np.log(get_degrees(G)+1), 1).astype(np.float32)
    return G


def generate_k_regular(k, n, seed=0):
    G = nx.random_regular_graph(d=k, n=n, seed=seed)
    return G


def run_simulation(model, loader, device):
    model.eval()
    with torch.no_grad():
        predictions = eval_model(model, loader, device, return_predictions=True)
    return predictions


def save_simulation_result(results, logger, pic_format='png'):
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
    logger.info('Finished. Results drawn to ./simulation_results.{}'.format(pic_format))


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
