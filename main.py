import argparse
from utils import *
from models.models import *
from log import *
from sklearn.metrics import roc_auc_score
from simulate import *


def check(args):
    if args.model == 'DE-GCN':
        args.model = 'PE-GCN'
    if args.dataset == 'foodweb' and not args.directed:
        raise Warning('dataset foodweb is essentially a directed network but currently treated as undirected')
    if args.dataset == 'simulation':
        if args.n is None:
            args.n = [10, 20, 40, 80, 160, 320, 640, 1280]
        if args.max_sp < args.T:
            raise Warning('maximum shortest path distance (max_sp) is less than max number of layers (T), which may deteriorate model capability')


def split_test_set(test_mask):
    device = test_mask.device
    mask1, mask2 = split_dataset(test_mask.size(0), 0.5)
    mask1, mask2 = torch.from_numpy(mask1).to(device), torch.from_numpy(mask2).to(device)
    with torch.no_grad():
        test_mask1, test_mask2 = test_mask*mask1, test_mask*mask2
    return test_mask1, test_mask2


def train_model(model, data, optimizer, args, logger):
    # prepare for training
    features, labels, adj_matrix, train_mask, test_mask, _ = data
    val_mask, test_mask = split_test_set(test_mask)
    metric = args.metric
    device = features.device
    criterion = torch.nn.functional.cross_entropy
    n_samples = labels.shape[0]
    bs = args.bs
    best_val_metric = 0
    best_val_metric_epoch = -1
    val_metrics, test_metrics = [], []
    best_test_acc_no_val, best_test_auc_no_val = 0, 0  # the four variables here and below are for no validation setting
    best_test_acc_epoch_no_val, best_test_auc_epoch_no_val = -1, -1
    for step in range(args.epoch):
        model.train()
        shuffled_index = np.random.permutation(n_samples)
        count = 0
        while count < n_samples:
            minibatch = torch.from_numpy(shuffled_index[count: min(count + bs, n_samples)]).long().to(device)
            count = count + bs
            train_mask_minibatch = train_mask[minibatch].bool()
            if not train_mask_minibatch.any().item():
                continue
            train_minibatch = minibatch[train_mask_minibatch]
            adj_batch = adj_matrix[train_minibatch].to(device) if args.model != 'PGNN' else adj_matrix
            prediction = model(features[:, train_minibatch, :], adj_batch, train_minibatch)  # shape: [n_nodes, feat_dim]
            loss = criterion(prediction, labels[train_minibatch], reduction='mean')
            loss.backward()
            optimizer.step()

        # evaluate epoch stats
        model.eval()
        predictions = []
        loss_total = 0
        with torch.no_grad():
            count = 0
            while count < n_samples:
                minibatch = torch.tensor(list(range(count, min(count + bs, n_samples)))).long().to(device)
                count = count + bs
                adj_batch = adj_matrix[minibatch].to(device) if args.model != 'PGNN' else adj_matrix
                prediction = model(features[:, minibatch, :], adj_batch, minibatch)  # shape: [n_nodes, feat_dim]
                predictions.append(prediction)
                train_mask_minibatch = train_mask[minibatch].bool()
                if not train_mask_minibatch.any().item():
                    continue
                loss = criterion(prediction[train_mask_minibatch], labels[minibatch][train_mask_minibatch], reduction='sum')
                loss_total += loss.item()
            predictions = torch.cat(predictions, dim=0)

        train_acc, train_auc = compute_acc(predictions, labels, train_mask)
        val_acc, val_auc = compute_acc(predictions, labels, val_mask)
        test_acc, test_auc = compute_acc(predictions, labels, test_mask)

        # update no-val metrics
        if test_acc > best_test_acc_no_val:
            best_test_acc_no_val = test_acc
            best_test_acc_epoch_no_val = step
        if test_auc > best_test_auc_no_val:
            best_test_auc_no_val = test_auc
            best_test_auc_epoch_no_val = step

        # update val metrics
        if metric == 'acc':
            train_metric, val_metric, test_metric = train_acc, val_acc, test_acc
        elif metric == 'auc':
            train_metric, val_metric, test_metric = train_auc, val_auc, test_auc
        else:
            raise NotImplementedError
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)
        logger.info('epoch %d best test %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f test %s: %.4f' %
                    (step, metric, test_metrics[best_val_metric_epoch], loss_total / train_mask.float().sum().item(),
                    metric, train_metric, metric, val_metric, metric, test_metric))
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_val_metric_epoch = step

    best_test_metric = test_metrics[best_val_metric_epoch]
    logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' % (metric, best_test_metric, best_val_metric_epoch, metric, best_val_metric))
    logger.info('(No validation) best test acc: %.4f (epoch: %d)' % (best_test_acc_no_val, best_test_acc_epoch_no_val))
    logger.info('(No validation) best test auc: %.4f (epoch: %d)' % (best_test_auc_no_val, best_test_auc_epoch_no_val))
    if metric == 'auc':
        best_test_metric_no_val = best_test_auc_no_val
    elif metric == 'acc':
        best_test_metric_no_val = best_test_acc_no_val
    else:
        raise NotImplementedError
    return best_test_metric, best_test_metric_no_val


def compute_acc(prediction, labels, mask):
    with torch.no_grad():
        correct_prediction = (torch.argmax(prediction, dim=1) == labels)
        accuracy = (correct_prediction.float() * mask).sum() / mask.sum()
        # compute auc:
        prediction = torch.nn.functional.softmax(prediction[mask.bool()], dim=-1)
        multi_class = 'ovr'
        if prediction.size(1) == 2:
            prediction = prediction[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels[mask.bool()].cpu().numpy(), prediction.cpu().numpy(), multi_class=multi_class)
    return accuracy.item(), auc


def main():
    parser = argparse.ArgumentParser('Interface for DE-GNN framework')

    # general model and training setting
    parser.add_argument('--dataset', type=str, default='celegans', help='dataset name') # currently relying on dataset to determine task
    parser.add_argument('--parallel', default=False, action='store_true', help='whether to use multi cpu cores to prepare data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
    parser.add_argument('--model', type=str, default='DE-GCN', help='model to use', choices=['DE-GCN', 'GIN', 'PGNN', 'GCN', 'GraphSAGE'])
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=32, help='hidden dimension')
    parser.add_argument('--metric', type=str, default='auc', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--adj_norm', type=str, default='asym', help='how to normalize adj', choices=['asym', 'sym', 'None'])
    parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
    parser.add_argument('--directed', type=bool, default=False, help='whether to treat the graph as directed')

    # positional encoding
    parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
    parser.add_argument('--use_degree', type=bool, default=True, help='whether to use node degree as the initial feature')
    parser.add_argument('--feature', type=str, default='sp', help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
    parser.add_argument('--max_sp', type=int, default=3, help='maximum distance to be encoded for shortest path feature')

    # model training
    parser.add_argument('--epoch', type=int, default=3000, help='number of epochs to train')
    parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # simulation (valid only when dataset == 'simulation')
    parser.add_argument('--k', type=int, default=3, help='node degree (k) or synthetic k-regular graph')
    parser.add_argument('--n', nargs='*', help='a list of number of nodes in each connected k-regular subgraph')
    parser.add_argument('--N', type=int, default=1000, help='total number of nodes in simultation')
    parser.add_argument('--T', type=int, default=6, help='largest number of layers to be tested')

    # logging
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')  # sp (shortest path) or rw (random walk)

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)
    device = get_device(args)
    if args.dataset == 'simulation':  # a special side branch for simulation
        results = simulate(args, logger, device)
        save_simulation_result(results)
        return
    (G, labels), task = read_file(args, logger)
    data = get_data(G, task=task, labels=labels, device=device, args=args, logger=logger)
    features, labels, adj_matrix, train_mask, test_mask, set_indices = data
    prop_depth = adj_matrix.shape[1] if args.model != 'PGNN' else adj_matrix.shape[-1]
    print_storage([features, adj_matrix], ['features', 'adj_matrix'], logger)
    model = get_model(layers=args.layers, in_features=features.shape[-1], out_features=len(torch.unique(labels)),
                      prop_depth=prop_depth, set_indices=set_indices, device=device, args=args, logger=logger)
    optimizer = get_optimizer(model, args)
    results = train_model(model, data, optimizer, args, logger)
    save_performance_result(args, logger, results)


if __name__ == '__main__':
    main()
