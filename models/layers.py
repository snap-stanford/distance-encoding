import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.mlp import MLP


class PEGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth, act=torch.relu, dropout=0.0, layer_i=0):
        super(PEGCNLayer, self).__init__()
        self.prop_depth = prop_depth
        self.act = act
        self.weight = nn.Parameter(torch.empty(1, prop_depth, input_dim, output_dim, dtype=torch.float), requires_grad=True)
        nn.init.uniform_(self.weight.data)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: [B, prop_depth, N, N]
        if adj_batch.dim() < 4:
            adj_batch = adj_batch.unsqueeze(0)
        x = x.transpose(0, 1).unsqueeze(dim=1).repeat(1, self.prop_depth, 1, 1)  # shape [B, prop_depth, N, F]
        x = torch.matmul(x, self.weight)  # shape [B, prop_depth, N, F]
        x = torch.matmul(adj_batch, x)  # shape [B, prop_depth, N, F]
        x = x.sum(dim=1)  # shape [B, N, F]
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        x = self.act(x)
        x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        x = self.layer_norm(x)
        return x


class GINLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth=1, act=torch.relu, dropout=0.0, layer_i=0):
        super(GINLayer, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(p=dropout)
        self.prop_depth = 1
        self.epsilon = 1/math.e**2  # simulate "irrational number"
        self.mlp = MLP(num_layers=2, input_dim=input_dim, hidden_dim=output_dim, output_dim=output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: [B, prop_depth, N, N]
        x = x.transpose(0, 1)  # shape: [B, N, F]
        adj_batch = adj_batch[:, 1, :, :]  # shape: [B, N, N]
        x = torch.matmul(adj_batch, x) + (1 + self.epsilon) * x  # shape: [B, N, F]
        B, N, F = x.shape
        x = self.mlp(x.view(-1, x.size(-1))).view(B, N, -1)
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        x = self.act(x)
        x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth=1, act=torch.relu, dropout=0.0, layer_i=0):
        super(GCNLayer, self).__init__()
        self.prop_depth = 1
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim, dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.weight.data)
        self.act = act
        self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = nn.LayerNorm(output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: [B, prop_depth, N, N]
        x = x.transpose(0, 1)  # shape: [B, N, F]
        adj_batch = adj_batch[:, 1, :, :]  # shape: [B, N, N]
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj_batch, x)
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        x = self.act(x)
        x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        return x


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth=1, act=torch.relu, dropout=0.0, layer_i=0):
        super(GraphSAGELayer, self).__init__()
        self.prop_depth = 1
        self.act = act
        self.weight = nn.Parameter(torch.empty(2*input_dim, output_dim, dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.weight.data)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: [B, prop_depth, N, N]
        x = x.transpose(0, 1)  # shape: [B, N, F]
        adj_batch = adj_batch[:, 1, :, :]  # shape: [B, N, N]
        x = torch.cat([torch.matmul(adj_batch, x), x], dim=2)
        x = torch.matmul(x, self.weight)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        x = self.act(x)
        x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        return x


class PGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth, act=torch.relu, dropout=0.0, layer_i=0):
        super(PGNNLayer, self).__init__()
        self.num_anchor_sets = prop_depth  # abuse the prop_depth variable
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)
        self.post_projection = nn.Linear(self.num_anchor_sets, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)
        self.act = act
        self.dropout = nn.Dropout(p=dropout)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: []
        dists_max, dists_argmax = self.extract_distances_and_indices(adj_batch)  # both shape: [N, NAS]
        N, B, F_ = list(x.size())
        NAS = dists_argmax.size(1)
        x = x.transpose(0, 1)  # shape: [B, N, F]
        batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).repeat(1, N, NAS).to(x.device)  # shape: [B, N, NAS]
        node_indices = dists_argmax.repeat(x.size(0), 1, 1)  # shape: [B, N, NAS]
        inv_distance_score = dists_max.unsqueeze(0).unsqueeze(-1)  # shape: [1, N, NAS, 1]
        anchor_set_features = x[batch_indices, node_indices, :] * inv_distance_score # shape: [B, N, NAS, F]
        source_node_features = x.unsqueeze(2).repeat(1, 1, NAS, 1)  # shape: [B, N, NAS, F]
        messages = torch.cat([anchor_set_features, source_node_features], dim=3)   # shape: [B, N, NAS, 2F]
        messages = self.linear_hidden(messages)
        if not self.last_layer_flag:
            x = messages.mean(dim=2)
        else:
            x = self.linear_out_position(messages).squeeze(3)
            x = self.post_projection(x)
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        # x shape: [N, B, F], adj_batch shape: []
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        # x = self.act(x)  # PGNN repo makes it optional
        if not self.last_layer_flag:
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        else:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def extract_distances_and_indices(self, adj_batch):
        # adj_batch shape: [layers, 2,  N, NAS], but the first dim is duplicated
        assert(adj_batch.dim() == 4)
        adj_batch = adj_batch[self.layer_i]
        dists_max = adj_batch[0]
        dists_argmax = adj_batch[1].long()
        return dists_max, dists_argmax


#  TODO: still under development
class PageRankLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prop_depth=1, act=torch.relu, dropout=0.0, layer_i=0):
        super(PageRankLayer, self).__init__()
        self.prop_depth = 1
        self.epsilon = 1/math.e  # simulate "irrational number"
        self.mlp = MLP(num_layers=2, input_dim=input_dim, hidden_dim=output_dim, output_dim=output_dim)
        self.layer_i = layer_i
        self.last_layer_flag = False

    def layer(self, x, adj_batch):
        # inputs shape: [N, B, F]
        # adj_batch shape: [B, prop_depth, N, N]
        x = x.transpose(0, 1)  # shape: [B, N, F]
        adj_batch = adj_batch[:, 1, :, :]  # shape: [B, N, N]
        x = torch.matmul(adj_batch, x) + (1 + self.epsilon) * x  # shape: [B, N, F]
        x = self.mlp(x)
        x = x.transpose(0, 1)  # shape [N, B, F]
        return x

    def forward(self, x, adj_batch):
        # NOTE that inputs cannot be modified
        x = self.layer(x, adj_batch)  # [[n_nodes, mini_batch, output_dim] * len(adj_list)]
        x = self.act(x)
        x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
        return x

