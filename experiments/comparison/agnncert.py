"""AGNNCert's actual partition and vote code, adapted to our row-normalized GNN."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from experiments.comparison.common import new_model, normalized, seed_all
from experiments.comparison.upstream import load_module


class RowModel(torch.nn.Module):
    """Convert PyG source->target edge indices to the paper's A_norm @ X."""
    def __init__(self, model, n, device):
        super().__init__()
        self.base = model
        self.n, self.device, self.adjacencies = n, device, {}

    def forward(self, features, edges):
        key = edges
        if key not in self.adjacencies:
            coo = edges.detach().cpu().numpy()
            # PyG [source, destination]; matrix multiplication uses [destination, source].
            adj = sp.csr_matrix((np.ones(coo.shape[1], dtype=np.float32),
                                 (coo[1], coo[0])), shape=(self.n, self.n))
            self.adjacencies[key] = normalized(adj, self.device)
        return self.base(features, self.adjacencies[key])


def train_vote(data, nodes, division, partitions, epochs, seed, checkpoint):
    native, record = load_module("AGNNCert", f"{division}_hash.py")
    seed_all(seed)
    coo = data["adj_sp"].tocoo()
    edges = torch.from_numpy(np.stack((coo.row, coo.col))).long()
    x, y = data["features"], data["labels"]
    hasher = native.HashAgent(T=partitions)
    subgraphs = hasher.generate_node_subgraphs(edges, x, y)
    if not subgraphs:
        raise ValueError("AGNNCert generated no nonempty graph partitions.")
    # Cache only graph division, keeping the authors' hashing and vote certificate.
    hasher.generate_node_subgraphs = lambda *_: subgraphs
    wrapped = RowModel(new_model(data, dropout=.5), len(y), data["device"])
    native_classifier = native.RobustNodeClassifier(wrapped, hasher, edges, x, y,
        data["idx_train"], data["idx_val"], torch.as_tensor(nodes, device=data["device"]), data["n_classes"])
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=.01, weight_decay=.0005)
    best, state = -1., None
    train, val = data["idx_train"], data["idx_val"]
    for epoch in range(epochs):
        wrapped.train()
        optimizer.zero_grad()
        # Sum full-graph and all subgraph losses, as in the upstream training loop.
        for graph_edges in [edges] + [g.edge_index for g in subgraphs]:
            F.nll_loss(wrapped(x, graph_edges)[train], y[train]).backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=2.)
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                val_votes, _ = native_classifier.vote(val)
            score = float((val_votes.argmax(1).numpy() == y[val].cpu().numpy()).mean())
            if score > best:
                best, state = score, {k: v.detach().clone() for k, v in wrapped.state_dict().items()}
    wrapped.load_state_dict(state)
    wrapped.eval()
    torch.save(dict(state_dict=state, division=division, partitions=partitions,
                    upstream=record, seed=seed, epochs=epochs), checkpoint)
    with torch.no_grad():
        votes, budget = native_classifier.vote(torch.as_tensor(nodes, device=data["device"]))
    record.update(nonempty_partitions=len(subgraphs),
                  adaptation="row-normalized GCN/BWGNN; shared data/split; cached division; incremental summed gradients")
    return wrapped, native_classifier, votes.numpy().astype(np.int64), budget.numpy().astype(int), record
