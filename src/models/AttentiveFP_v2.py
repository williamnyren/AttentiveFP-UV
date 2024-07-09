from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter, Module
from torch_geometric.nn import GATv2Conv, GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense import DenseGATConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, to_dense_batch, to_dense_adj
from torch_geometric.nn import MLP
import numpy as np

class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)

class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning with various attention mechanisms.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        attention_mode (str, optional): Type of attention mechanism to use. (default: :obj:`GATv2`)
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
        attention_mode: str = 'GATv2',
        heads: int = 1
    ):
        super().__init__()
        assert attention_mode in ['MoGATv2', 'MoGAT', 'GATv2', 'GAT', 'DenseGAT']

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.masked_attention = attention_mode != 'DenseGAT'
        self.attention_mode = attention_mode

        self.lin1 = Linear(in_channels, hidden_channels)
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        self.mol_convs = torch.nn.ModuleList()
        self.mol_grus = torch.nn.ModuleList()

        if self.masked_attention:
            if attention_mode in ['GATv2', 'MoGATv2']:
                Attention_mechanism_atom = GATv2Conv
                Attention_mechanism_mol = GATv2Conv
            elif attention_mode in ['GAT', 'MoGAT']:
                Attention_mechanism_atom = GATConv
                Attention_mechanism_mol = GATConv
            else:
                raise NotImplementedError
        else:
            Attention_mechanism_atom = DenseGATConv
            Attention_mechanism_mol = GATv2Conv

        for ix in range(num_layers - 1):
            if self.masked_attention:
                conv = Attention_mechanism_atom(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=False, negative_slope=0.01)
                conv_mol = Attention_mechanism_mol(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=False, negative_slope=0.01)
                self.atom_convs.append(conv)
                self.atom_grus.append(GRUCell(hidden_channels * heads, hidden_channels))
                if attention_mode in ['MoGATv2', 'MoGAT']:
                    conv_mol.explain = False
                    self.mol_convs.append(conv_mol)
                    self.mol_grus.append(GRUCell(hidden_channels * heads, hidden_channels))
            elif attention_mode == 'DenseGAT':
                conv = Attention_mechanism_atom(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout, negative_slope=0.01)
                self.atom_convs.append(conv)
                self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))
            else:
                raise NotImplementedError
        if attention_mode in ['DenseGAT', 'GATv2', 'GAT']:
            self.mol_conv = Attention_mechanism_mol(hidden_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False, dropout=dropout, negative_slope=0.01)
            self.mol_conv.explain = False
            self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        if attention_mode in ['MoGATv2', 'MoGAT']:
            self.lin2 = MLP([hidden_channels * (num_layers - 1), (hidden_channels * (num_layers - 1)) // 2, (hidden_channels * (num_layers - 1)) // max(num_layers - 2, 2), out_channels])
        elif attention_mode in ['GATv2', 'GAT', 'DenseGAT']:
            self.lin2 = MLP([hidden_channels, hidden_channels // 2, hidden_channels // 4, out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        if self.attention_mode in ['MoGATv2', 'MoGAT']:
            for conv, gru, conv_mol, gru_mol in zip(self.atom_convs, self.atom_grus, self.mol_convs, self.mol_grus):
                conv.reset_parameters()
                gru.reset_parameters()
                conv_mol.reset_parameters()
                gru_mol.reset_parameters()
        elif self.attention_mode in ['GATv2', 'GAT', 'DenseGAT']:
            for conv, gru in zip(self.atom_convs, self.atom_grus):
                conv.reset_parameters()
                gru.reset_parameters()
            self.mol_conv.reset_parameters()
            self.mol_gru.reset_parameters()
        else:
            raise NotImplementedError
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        middle_states = []
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index_global = torch.stack([row, batch], dim=0)

        if self.attention_mode in ['MoGATv2', 'MoGAT']:
            for conv, gru, conv_mol, gru_mol in zip(self.atom_convs, self.atom_grus, self.mol_convs, self.mol_grus):
                h = conv(x, edge_index)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = gru(h, x).relu()
                out = global_add_pool(x, batch).relu_()
                for t in range(self.num_timesteps):
                    h = F.elu_(conv_mol((x, out), edge_index_global))
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    out = gru_mol(h, out).relu_()
                middle_states.append(gru_mol(h, out).relu_())
            out = torch.cat(middle_states, dim=1)
            out = F.dropout(out, p=self.dropout, training=self.training)
        elif self.attention_mode in ['GATv2', 'GAT']:
            for conv, gru in zip(self.atom_convs, self.atom_grus):
                h = conv(x, edge_index)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = gru(h, x).relu()
            out = global_add_pool(x, batch).relu_()
            for t in range(self.num_timesteps):
                h = F.elu_(self.mol_conv((x, out), edge_index_global))
                h = F.dropout(h, p=self.dropout, training=self.training)
                out = self.mol_gru(h, out).relu_()
            out = F.dropout(out, p=self.dropout, training=self.training)
        else:
            adj_dense = torch.ones_like(to_dense_adj(edge_index, batch))
            adj_dense = adj_dense - torch.eye(adj_dense.size(1), device=edge_index.device)

            for conv, gru in zip(self.atom_convs, self.atom_grus):
                x, mask = to_dense_batch(x, batch)
                h = conv(x, adj_dense, mask=mask, add_loop=False)
                h = h[mask].view(-1, h.size(-1))
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = x[mask].view(-1, x.size(-1))
                x = gru(h, x).relu()

            out = global_add_pool(x, batch).relu_()
            for t in range(self.num_timesteps):
                h = F.elu_(self.mol_conv((x, out), edge_index_global))
                h = F.dropout(h, p=self.dropout, training=self.training)
                out = self.mol_gru(h, out).relu_()
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')