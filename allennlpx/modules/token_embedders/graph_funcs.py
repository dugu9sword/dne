import torch
from torch_geometric.nn import MessagePassing


class MeanAggregator(MessagePassing):
    def __init__(self, dim):
        super(MeanAggregator, self).__init__(aggr='mean', flow='target_to_source')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Tanh(),
        )

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        # return self.mlp(tmp)
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return self.mlp(aggr_out)


class PoolingAggregator(MessagePassing):
    def __init__(self, dim):
        super(PoolingAggregator, self).__init__(aggr='max', flow='target_to_source')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Tanh(),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return self.mlp(x_j)

    def update(self, aggr_out):
        return aggr_out
