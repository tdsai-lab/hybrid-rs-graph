"""MGTAB dataset loading, feature standardization and GCN training."""
from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split as _sk_train_test_split

def _load_tensor_cpu(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")

def _load_mgtab_tensor_dataset(
    dataset_dir: str,
    edge_type_to_keep: int,
    make_undirected: bool,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix, Dict[str, Any]]:
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    required = {
        "features": os.path.join(dataset_dir, "features.pt"),
        "edge_index": os.path.join(dataset_dir, "edge_index.pt"),
        "edge_type": os.path.join(dataset_dir, "edge_type.pt"),
        "labels_bot": os.path.join(dataset_dir, "labels_bot.pt"),
    }
    missing = [path for path in required.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing MGTAB tensor files: {missing}")

    features_t = _load_tensor_cpu(required["features"])
    labels_t = _load_tensor_cpu(required["labels_bot"])
    edge_index_t = _load_tensor_cpu(required["edge_index"]).long()
    edge_type_t = _load_tensor_cpu(required["edge_type"]).long().view(-1)

    if getattr(features_t, "is_sparse", False):
        features_t = features_t.to_dense()
    features_t = features_t.float()
    labels_t = labels_t.long().view(-1)

    if features_t.dim() != 2:
        raise ValueError(f"features.pt must be 2-D, got shape {tuple(features_t.shape)}")
    if labels_t.dim() != 1 or labels_t.numel() != features_t.size(0):
        raise ValueError(
            f"labels_bot.pt must be [num_nodes], got shape {tuple(labels_t.shape)} "
            f"for {features_t.size(0)} nodes"
        )
    if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
        raise ValueError(f"edge_index.pt must be [2, num_edges], got shape {tuple(edge_index_t.shape)}")
    if edge_type_t.dim() != 1 or edge_type_t.numel() != edge_index_t.size(1):
        raise ValueError(
            f"edge_type.pt must align with edge_index columns, got {edge_type_t.numel()} types "
            f"for {edge_index_t.size(1)} edges"
        )

    original_edges = int(edge_index_t.size(1))
    keep_mask = edge_type_t == int(edge_type_to_keep)
    edge_index_t = edge_index_t[:, keep_mask]
    filtered_edges = int(edge_index_t.size(1))
    if filtered_edges > 0:
        edge_index_t = torch.unique(edge_index_t, dim=1)
    else:
        edge_index_t = edge_index_t.new_empty((2, 0))

    edge_index_np = edge_index_t.numpy().astype(np.int64, copy=False)
    n_nodes = int(features_t.size(0))
    adj = sp.csr_matrix(
        (
            np.ones(edge_index_np.shape[1], dtype=np.float32),
            (edge_index_np[0], edge_index_np[1]),
        ),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    binary_edges_before_undirected = int(adj.nnz)
    if make_undirected:
        adj = adj.maximum(adj.T).tocsr()
        if adj.nnz > 0:
            adj.data[:] = 1.0

    metadata = {
        "loader": "mgtab_tensor",
        "dataset_dir": dataset_dir,
        "nodes": n_nodes,
        "feature_dim": int(features_t.size(1)),
        "original_edges": original_edges,
        "kept_edge_type": int(edge_type_to_keep),
        "filtered_edges_before_binary_unique": filtered_edges,
        "binary_edges_before_undirected": binary_edges_before_undirected,
        "binary_edges": int(adj.nnz),
        "make_undirected": bool(make_undirected),
    }
    return (
        features_t.numpy().astype(np.float32, copy=False),
        labels_t.numpy().astype(np.int64, copy=False),
        adj,
        metadata,
    )

def _manual_stratified_train_val_test_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train_parts, val_parts, test_parts = [], [], []
    for class_value in np.unique(labels):
        class_indices = np.where(labels == class_value)[0]
        perm = rng.permutation(class_indices)
        n_class = int(perm.size)
        n_train = int(round(n_class * float(train_ratio)))
        n_val = int(round(n_class * float(val_ratio)))
        if n_class >= 3:
            n_train = min(max(1, n_train), n_class - 2)
            n_val = min(max(1, n_val), n_class - n_train - 1)
        else:
            n_train = max(1, min(n_train, n_class))
            n_val = max(0, min(n_val, n_class - n_train))
        train_parts.append(perm[:n_train])
        val_parts.append(perm[n_train:n_train + n_val])
        test_parts.append(perm[n_train + n_val:])
    return (
        np.sort(rng.permutation(np.concatenate(train_parts)).astype(np.int64, copy=False)),
        np.sort(rng.permutation(np.concatenate(val_parts)).astype(np.int64, copy=False)),
        np.sort(rng.permutation(np.concatenate(test_parts)).astype(np.int64, copy=False)),
    )

def _build_mgtab_splits(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError(
            "--mgtab_train_ratio and --mgtab_val_ratio must leave non-empty train, "
            "validation, and test fractions."
        )

    indices = np.arange(labels.shape[0], dtype=np.int64)
    if _sk_train_test_split is not None:
        try:
            stratify = labels if np.unique(labels).size > 1 else None
            train_idx, holdout_idx = _sk_train_test_split(
                indices,
                train_size=train_ratio,
                random_state=int(seed),
                stratify=stratify,
            )
            holdout_labels = labels[holdout_idx]
            holdout_stratify = holdout_labels if np.unique(holdout_labels).size > 1 else None
            val_fraction = val_ratio / (1.0 - train_ratio)
            val_idx, test_idx = _sk_train_test_split(
                holdout_idx,
                train_size=val_fraction,
                random_state=int(seed),
                stratify=holdout_stratify,
            )
            return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
        except ValueError:
            pass

    return _manual_stratified_train_val_test_split(
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

def _make_balanced_training_indices(labels: np.ndarray, train_idx: np.ndarray, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train_labels = labels[train_idx]
    class_values, class_counts = np.unique(train_labels, return_counts=True)
    if class_values.size < 2:
        raise ValueError("Balanced training requires at least two classes in the training split.")
    target_count = int(class_counts.min())
    parts = []
    for class_value in class_values:
        class_indices = train_idx[train_labels == class_value]
        parts.append(rng.choice(class_indices, size=target_count, replace=False))
    balanced = np.concatenate(parts).astype(np.int64, copy=False)
    rng.shuffle(balanced)
    return balanced

def _standardize_features_from_train(features: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return ((features - mean) / std).astype(np.float32, copy=False)

def train_model(model, optimizer, features, adj_norm, labels, idx_train, idx_val, epochs, feature_noise, sigma_train):
    best_state = None
    best_val = -1.0
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        x_train = features
        if feature_noise == "gaussian" and sigma_train > 0:
            x_train = features + torch.randn_like(features) * float(sigma_train)
        out = model(x_train, adj_norm)
        loss = F.nll_loss(out[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(features, adj_norm)
            val_acc = float((out_val[idx_val].argmax(1) == labels[idx_val]).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
