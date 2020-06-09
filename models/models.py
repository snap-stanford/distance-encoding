from models.layers import *
from itertools import combinations


class GNNModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, set_indices, dropout=0.0, model_name='PE-GNN'):
        super(GNNModel, self).__init__()
        self.layers, self.in_features, self.hidden_features, self.out_features, self.model_name = layers, in_features, hidden_features, out_features, model_name
        Layer = self.get_layer_class()
        self.set_indices = set_indices
        self.act = lambda x: x  # was torch.relu
        self.dropout = dropout
        self.layer_list = nn.ModuleList()
        self.layer_list.append(
            Layer(input_dim=in_features, output_dim=hidden_features,
                       prop_depth=prop_depth, act=self.act, dropout=dropout, layer_i=0))
        self.layer_list[-1].last_layer_flag = True
        if layers > 1:
            for i in range(layers - 1):
                self.layer_list.append(Layer(input_dim=hidden_features, output_dim=hidden_features,
                                                  prop_depth=prop_depth, act=self.act, dropout=dropout, layer_i=i+1))
        self.merger = nn.Linear(3*hidden_features, hidden_features)
        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

    def forward(self, x, adj_batch, minibatch):
        # x shape: [N, B, F], adj_batch shape [B, prop_depth, N, N] / [1, prop_depth, N, N]
        for i, layer in enumerate(self.layer_list):
            x = layer(x, adj_batch)
        x = self.get_minibatch_embeddings(x, minibatch)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, minibatch):
        minibatch_set_indices = self.set_indices[minibatch]
        dim1_indices = torch.arange(len(minibatch), device=x.device).unsqueeze(1).repeat(1, minibatch_set_indices.size(1))
        x = x[minibatch_set_indices, dim1_indices, :]  # shape [batch_size, set_size, feat_dim]
        x = self.pool(x)
        return x

    def pool(self, x):
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        if self.model_name == 'PGNN': # PGNN uses dot product of node embeddings to score the existence of the link
            return torch.prod(x, dim=1)
        # use mean/diff/max to pool each set's representations
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :]-x[:, j, :])
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x

    def get_layer_class(self):
        layer_dict = {'PE-GCN': PEGCNLayer, 'GIN': GINLayer, 'GCN': GCNLayer, 'GraphSAGE': GraphSAGELayer, 'PGNN': PGNNLayer}
        model_name = self.model_name
        Layer = layer_dict.get(model_name)
        if Layer is None:
            raise NotImplementedError('Unknown model name: {}'.format(model_name))
        return Layer

    def short_summary(self):
        return 'Model: {}, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(self.model_name, self.layers, self.in_features, self.hidden_features, self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout, nn.Linear(in_features, in_features))
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x


# TODO: add if necessary
class GeneralizedPageRankModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, set_indices, dropout=0.0):
        super(GeneralizedPageRankModel, self).__init__()
        self.dropout = dropout
        self.layers, self.in_features, self.hidden_features, self.out_features = layers, in_features, hidden_features, out_features
        self.act = lambda x: x  # was torch.relu
        self.set_indices = set_indices

    def forward(self, x, adj_batch, minibatch):
        pass