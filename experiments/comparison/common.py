"""Shared populations, base models and result tables for SOTA comparisons."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from hybrid_graph.architecture import BWGNN, GCN
from hybrid_graph.data import load_data_new, normalize, sparse_mx_to_torch_sparse_tensor
from hybrid_graph.fraud_utils import load_fraud_graph_data
from experiments.mgtab_data import (_load_mgtab_tensor_dataset, _build_mgtab_splits,
    _make_balanced_training_indices, _standardize_features_from_train)
from experiments.run_mgtab import sample_nodes


def multiclass_bounds(counts, selected, alpha):
    """Bound the selected class against every other class, with Bonferroni."""
    from hybrid_smoothing.utils_hybrid import clopper_pearson_lcb, clopper_pearson_ucb
    n, classes = int(np.sum(counts)), len(counts)
    lower = clopper_pearson_lcb(n, int(counts[selected]), alpha / classes)
    upper = max(clopper_pearson_ucb(n, int(counts[c]), alpha / classes)
                for c in range(classes) if c != selected)
    return lower, upper


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sparse_tensor(adj, device):
    return sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj)).coalesce().to(device)


def normalized(adj, device):
    return sparse_tensor(normalize(adj + sp.eye(adj.shape[0], dtype=np.float32)), device)


def load_data(dataset, data_root, device, seed, n_per_class=None, binary=False):
    os.environ["PLANETOID_ROOT"] = str(data_root / "planetoid")
    fraud = dataset in ("amazon_fraud", "yelpchi")
    if fraud:
        data = load_fraud_graph_data(dataset=dataset,
            caregnn_root=str(data_root / "CARE-GNN/processed"),
            n_per_class=n_per_class or (100 if dataset == "amazon_fraud" else 2000),
            split_seed=seed, cuda=device.type == "cuda",
            fraud_graph_source="care_static", care_static_topk=50, adj_norm_type="row")
    else:
        if dataset == "mgtab":
            x, y, adj, _ = _load_mgtab_tensor_dataset(str(data_root / "MGTAB"), 0, True)
            full, val, test = _build_mgtab_splits(y, .8, .1, seed)
            train = _make_balanced_training_indices(y, full, seed)
            x = _standardize_features_from_train(x, full)
            features, labels = torch.from_numpy(x), torch.from_numpy(y)
            idx_train, idx_val, idx_test = map(torch.from_numpy, (train, val, test))
        else:
            adj, features, labels, idx_train, idx_val, idx_test = load_data_new(
                dataset=dataset)
        data = dict(adj_sp=adj.tocsr(), features=features.to(device), labels=labels.to(device),
                    idx_train=idx_train.to(device), idx_val=idx_val.to(device), idx_test=idx_test.to(device))
    data["adj_sp"] = data["adj_sp"].astype(np.float32).tocsr()
    data["adj_sp"].setdiag(0)
    data["adj_sp"].eliminate_zeros()
    if binary:
        data["features"] = (data["features"] > 0).float()
    data.update(dataset=dataset, device=device, binary=binary, fraud=fraud,
                n_classes=int(data["labels"].max()) + 1,
                adj_norm_t=normalized(data["adj_sp"], device),
                adj_bin_t=sparse_tensor(data["adj_sp"], device))
    return data


def select_nodes(data, count, seed):
    return np.asarray(sample_nodes(data["labels"].cpu().numpy(),
        data["idx_test"].cpu().numpy(), count, seed), dtype=np.int64)


def new_model(data, dropout=None):
    if data["fraud"]:
        return BWGNN(nfeat=data["features"].shape[1], nhid=64, nclass=data["n_classes"],
                     dropout=0 if dropout is None else dropout, order=2).to(data["device"])
    return GCN(nfeat=data["features"].shape[1], nhid=16, nclass=data["n_classes"],
               dropout=.5 if dropout is None else dropout).to(data["device"])


def train_base(data, epochs, seed, sigma_train, checkpoint, sample=None):
    """Shared base family; fraud weighted NLL and validation balanced accuracy."""
    seed_all(seed)
    model = new_model(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005 if data["fraud"] else .01,
                                 weight_decay=.0001 if data["fraud"] else .0005)
    labels, train, val = data["labels"], data["idx_train"], data["idx_val"]
    weights = None
    if data["fraud"]:
        counts = torch.bincount(labels[train], minlength=data["n_classes"]).float().clamp_min(1)
        weights = counts.sum() / (data["n_classes"] * counts)
    best, state = -1., None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if sample:
            x, adj = sample()
        else:
            x = data["features"] + sigma_train * torch.randn_like(data["features"])
            adj = data["adj_norm_t"]
        F.nll_loss(model(x, adj)[train], labels[train], weight=weights).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if sample:
                x, adj = sample()
            else:
                x, adj = data["features"], data["adj_norm_t"]
            pred = model(x, adj)[val].argmax(1).cpu().numpy()
            truth = labels[val].cpu().numpy()
            score = balanced_accuracy_score(truth, pred) if data["fraud"] else float((truth == pred).mean())
        if score > best:
            best, state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Citation models in the supplementary harness select validation accuracy.
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, sigma_train=sigma_train, seed=seed,
                    epochs=epochs, selection="val_bacc" if data["fraud"] else "val_accuracy"), checkpoint)
    return model.eval()


def summarize(method, correct, certified, **fields):
    correct, certified = np.asarray(correct), np.asarray(certified)
    hit, n, n_correct = int((correct & certified).sum()), len(correct), int(correct.sum())
    return dict(method=method, n_all=n, n_correct=n_correct, n_certified_correct=hit,
                accuracy=n_correct / n, cert_acc_all=hit / n,
                cert_acc_correct=hit / n_correct if n_correct else None, **fields)


def write_results(out, rows, meta):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    metadata = dict(meta, torch=torch.__version__, numpy=np.__version__, completed_unix=time.time())
    metadata["output_sha256"] = hashlib.sha256((out / "comparison.csv").read_bytes()).hexdigest()
    (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # A compact table that keeps both denominators and budgets visible.
    lines = ["Native supports and predictors differ; see comparison.csv and meta.json for each protocol.", "",
             "| Method | Dataset | d | feature budget | Targets | MC draws | CA all | CA correct |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        ca = "n/a" if row["cert_acc_correct"] is None else f"{100*row['cert_acc_correct']:.1f}%"
        feature = row.get("epsilon", row.get("d_features", 0))
        lines.append(f"| {row['method']} | {row.get('dataset', '')} | {row.get('d', 0)} | "
                     f"{feature} | {row['n_all']} | {row.get('n_mc', '-')} | {100*row['cert_acc_all']:.1f}% | {ca} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
