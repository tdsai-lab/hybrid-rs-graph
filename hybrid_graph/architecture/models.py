import math

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .layers import GraphConvolution
except ImportError:  # script-style execution from hybrid_graph/
    from architecture.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("GCN requires num_layers >= 2.")

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.hidden_layers = nn.ModuleList(
            GraphConvolution(nhid, nhid) for _ in range(max(0, self.num_layers - 2))
        )
        self.gc2 = GraphConvolution(nhid, nclass)
        self.graph_layers = nn.ModuleList([self.gc1, *self.hidden_layers, self.gc2])
        self.dropout = dropout

    def forward(self, x, adj):
        for layer_idx, layer in enumerate(self.graph_layers):
            x = layer(x, adj)
            if layer_idx != len(self.graph_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


def _bernstein_theta(order):
    """Return the BWGNN Bernstein polynomial coefficients in ascending powers."""
    order = int(order)
    if order < 0:
        raise ValueError("BWGNN order must be >= 0.")

    inv_beta = []
    for i in range(order + 1):
        # 1 / beta(i + 1, order + 1 - i) for integer arguments.
        inv_beta.append(math.exp(math.lgamma(order + 2) - math.lgamma(i + 1) - math.lgamma(order - i + 1)))

    thetas = []
    for i in range(order + 1):
        coeffs = [0.0 for _ in range(order + 1)]
        for k in range(i, order + 1):
            j = k - i
            coeffs[k] = (
                math.comb(order - i, j)
                * ((-1.0) ** j)
                * inv_beta[i]
                / (2.0 ** k)
            )
        thetas.append(coeffs)
    return thetas


def _adj_mm(adj, feat):
    if getattr(adj, "is_sparse", False):
        return torch.sparse.mm(adj, feat)
    return torch.matmul(adj, feat)


class BWGNN(nn.Module):
    """
    BWGNN-style Bernstein polynomial graph filter with the CertifyGNN model API.

    The upstream BWGNN implementation stores a DGL graph in the module and uses
    ``forward(features)``. Certification needs to pass a freshly sampled
    adjacency for every Monte Carlo draw, so this variant uses
    ``forward(features, adj_norm)`` and sparse PyTorch matrix multiplies.
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.0, order=2):
        super().__init__()
        self.order = int(order)
        self.dropout = float(dropout)
        theta = torch.tensor(_bernstein_theta(self.order), dtype=torch.float32)
        self.register_buffer("theta", theta)

        n_filters = self.order + 1
        self.linear = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nhid)
        self.linear3 = nn.Linear(nhid * n_filters, nhid)
        self.linear4 = nn.Linear(nhid, nclass)

    def _poly_filter(self, feat, adj_norm, theta):
        lap_feat = feat
        out = theta[0] * lap_feat
        for k in range(1, theta.numel()):
            lap_feat = lap_feat - _adj_mm(adj_norm, lap_feat)
            out = out + theta[k] * lap_feat
        return out

    def forward(self, x, adj):
        h = F.relu(self.linear(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.linear2(h))
        h = F.dropout(h, self.dropout, training=self.training)

        filters = [self._poly_filter(h, adj, theta_i) for theta_i in self.theta]
        h = torch.cat(filters, dim=-1)
        h = F.relu(self.linear3(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear4(h)
        return F.log_softmax(h, dim=1)
