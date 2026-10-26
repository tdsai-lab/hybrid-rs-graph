from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


try:
    from hybrid_graph.data import load_data_new, normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor
except ImportError:
    from data import load_data_new, normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor


FRAUD_DATASETS = ("amazon_fraud", "yelpchi")


def require_fraud_dataset(dataset: str) -> str:
    dataset_key = str(dataset).lower()
    if dataset_key not in FRAUD_DATASETS:
        raise ValueError(
            f"Unsupported fraud dataset '{dataset}'. Expected one of {list(FRAUD_DATASETS)}."
        )
    return dataset_key


def set_random_seeds(seed: int, cuda: bool) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def load_fraud_graph_data(
    dataset: str,
    *,
    caregnn_root: str | None,
    n_per_class: int,
    split_seed: int,
    fraud_graph_source: str,
    care_static_topk: int,
    cuda: bool,
    move_adj_bin_to_cuda: bool = False,
    adj_norm_type: str = "row",
) -> Dict[str, Any]:
    dataset_key = require_fraud_dataset(dataset)
    if caregnn_root:
        os.environ["CAREGNN_ROOT"] = os.path.abspath(os.path.expanduser(caregnn_root))

    adj_sp, features, labels, idx_train, idx_val, idx_test = load_data_new(
        dataset=dataset_key,
        n_per_class=n_per_class,
        split_seed=split_seed,
        caregnn_root=caregnn_root,
        fraud_graph_source=fraud_graph_source,
        care_static_topk=care_static_topk,
    )
    adj_sp = adj_sp.tocsr()
    adj_plus_i = adj_sp + sp.eye(adj_sp.shape[0], dtype=np.float32, format="csr")
    if str(adj_norm_type).strip().lower() == "sym":
        adj_norm_sp = normalize_adj(adj_plus_i)
    else:
        adj_norm_sp = normalize(adj_plus_i)
    adj_norm_t = sparse_mx_to_torch_sparse_tensor(adj_norm_sp)
    adj_bin_t = sparse_mx_to_torch_sparse_tensor(adj_sp).coalesce()

    if cuda:
        features = features.cuda()
        adj_norm_t = adj_norm_t.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        if move_adj_bin_to_cuda:
            adj_bin_t = adj_bin_t.cuda()

    return {
        "dataset": dataset_key,
        "fraud_graph_source": str(fraud_graph_source),
        "care_static_topk": int(care_static_topk),
        "adj_norm_type": str(adj_norm_type).strip().lower(),
        "adj_sp": adj_sp,
        "adj_norm_t": adj_norm_t,
        "adj_bin_t": adj_bin_t,
        "features": features,
        "labels": labels,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "n_nodes": int(adj_sp.shape[0]),
        "n_features": int(features.shape[1]),
        "n_classes": int(labels.max().item() + 1),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_binary_split_metrics(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    split_idx: torch.Tensor,
) -> Dict[str, float]:
    split_out = log_probs[split_idx]
    split_labels = labels[split_idx]

    if split_labels.numel() == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "auc": float("nan"),
            "ap": float("nan"),
            "n": 0,
        }

    loss = F.nll_loss(split_out, split_labels).item()
    probs = split_out.exp()
    y_true = split_labels.detach().cpu().numpy()
    y_pred_t = split_out.argmax(1)
    y_pred = y_pred_t.detach().cpu().numpy()
    y_score = probs[:, 1].detach().cpu().numpy()

    topk_vals = probs.topk(k=min(2, probs.shape[1]), dim=1).values
    pA = topk_vals[:, 0]
    if probs.shape[1] >= 2:
        pB = topk_vals[:, 1]
    else:
        pB = torch.zeros_like(pA)
    margin = pA - pB
    correct_mask = (y_pred_t == split_labels)
    if bool(correct_mask.any().item()):
        margin_correct_mean = float(margin[correct_mask].mean().item())
    else:
        margin_correct_mean = float("nan")

    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": _safe_auc(y_true, y_score),
        "ap": _safe_ap(y_true, y_score),
        "pA_mean": float(pA.mean().item()),
        "pB_mean": float(pB.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "margin_correct_mean": margin_correct_mean,
        "n": int(split_labels.numel()),
    }


def build_inverse_frequency_class_weights(
    labels: torch.Tensor,
    split_idx: torch.Tensor,
) -> torch.Tensor:
    split_labels = labels[split_idx].detach().cpu().numpy().astype(np.int64, copy=False)
    counts = np.bincount(split_labels, minlength=int(labels.max().item()) + 1).astype(np.float32)
    counts[counts <= 0.0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=labels.device)


def build_fraud_model(
    *,
    model_name: str,
    n_features: int,
    n_hidden: int,
    n_classes: int,
    dropout: float,
    bwgnn_order: int = 2,
) -> torch.nn.Module:
    model_key = str(model_name).strip().lower()
    try:
        from hybrid_graph.architecture import BWGNN
    except ImportError:
        from architecture import BWGNN

    if model_key == "bwgnn":
        return BWGNN(
            nfeat=int(n_features),
            nhid=int(n_hidden),
            nclass=int(n_classes),
            dropout=float(dropout),
            order=int(bwgnn_order),
        )
    raise ValueError("Unsupported fraud model '{}'. Expected 'bwgnn'.".format(model_name))


def evaluate_binary_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    adj_norm_t: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        log_probs = model(features, adj_norm_t)

    return {
        "train": compute_binary_split_metrics(log_probs, labels, idx_train),
        "val": compute_binary_split_metrics(log_probs, labels, idx_val),
        "test": compute_binary_split_metrics(log_probs, labels, idx_test),
    }


def _safe_metric_key(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("-inf")


def _score_key(
    metrics_by_split: Dict[str, Dict[str, float]],
    selection_metric: str = "val_auc",
) -> tuple[float, float, float]:
    val_metrics = metrics_by_split["val"]
    auc = _safe_metric_key(val_metrics["auc"])
    ap = _safe_metric_key(val_metrics["ap"])
    acc = _safe_metric_key(val_metrics["accuracy"])
    loss = -float(val_metrics["loss"]) if np.isfinite(float(val_metrics["loss"])) else float("-inf")

    if selection_metric == "val_auc":
        return (auc, ap, acc)
    if selection_metric == "val_ap":
        return (ap, auc, acc)
    if selection_metric == "val_bacc":
        bacc = _safe_metric_key(val_metrics["balanced_accuracy"])
        return (bacc, ap, auc)
    if selection_metric == "val_accuracy":
        return (acc, auc, ap)
    if selection_metric == "val_loss":
        return (loss, acc, auc)
    raise ValueError(
        f"Unsupported selection_metric '{selection_metric}'. "
        "Expected one of ['val_auc', 'val_ap', 'val_bacc', 'val_accuracy', 'val_loss']."
    )


def is_better_checkpoint(
    metrics_by_split: Dict[str, Dict[str, float]],
    best_metrics_by_split: Dict[str, Dict[str, float]] | None,
    *,
    selection_metric: str = "val_auc",
) -> bool:
    if best_metrics_by_split is None:
        return True
    return _score_key(metrics_by_split, selection_metric) > _score_key(best_metrics_by_split, selection_metric)


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args_dict: Dict[str, Any],
    metrics_by_split: Dict[str, Dict[str, float]],
    extra: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": dict(args_dict),
        "metrics": metrics_by_split,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint["state_dict"])
    graph_layers = getattr(model, "graph_layers", None)
    if graph_layers is not None and len(graph_layers) >= 2:
        remap_pairs = {
            "gc1.weight": "graph_layers.0.weight",
            "gc1.bias": "graph_layers.0.bias",
            "gc2.weight": f"graph_layers.{len(graph_layers) - 1}.weight",
            "gc2.bias": f"graph_layers.{len(graph_layers) - 1}.bias",
        }
        for src_key, dst_key in remap_pairs.items():
            if src_key in state_dict and dst_key not in state_dict:
                state_dict[dst_key] = state_dict[src_key]
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys while loading {path}: {unexpected}")
    bad_missing = [key for key in missing if not str(key).startswith("graph_layers.")]
    if bad_missing:
        raise RuntimeError(f"Missing checkpoint keys while loading {path}: {bad_missing}")
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def flatten_metrics(metrics_by_split: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for split_name, split_metrics in metrics_by_split.items():
        for metric_name, metric_value in split_metrics.items():
            row[f"{split_name}_{metric_name}"] = metric_value
    return row


def write_metrics_history_csv(path: str, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def fmt_metric(value: float) -> str:
    if value is None:
        return "nan"
    try:
        if not np.isfinite(value):
            return "nan"
    except TypeError:
        return str(value)
    return f"{float(value):.4f}"
