#!/usr/bin/env python3
"""Train MGTAB's GCN or sample and certify it under the Figure 2 protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch
from hybrid_graph.architecture import GCN
from hybrid_graph.certification.core_hybrid import SmoothHybrid
from hybrid_graph.data import normalize, sparse_mx_to_torch_sparse_tensor
from experiments.mgtab_data import (
    _load_mgtab_tensor_dataset, _build_mgtab_splits, _make_balanced_training_indices,
    _standardize_features_from_train, train_model,
)
from experiments.offline_certificates import confidence_bounds, node_grids


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("No test nodes were sampled.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_nodes(labels, test, count, seed):
    if count >= len(test):
        return test.tolist()
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[test])
    per_class = max(1, count // len(classes))
    picks = []
    for c in classes:
        pool = test[labels[test] == c]
        take = min(per_class, len(pool), count - len(picks))
        if take:
            picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    remaining = count - len(picks)
    if remaining:
        rest = np.setdiff1d(test, picks)
        picks.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())
    return sorted(picks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("train", "certify"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--nodes", type=int, default=400)
    p.add_argument("--mc-samples", type=int, default=10000)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    if min(args.epochs, args.nodes, args.mc_samples, args.batch) < 1:
        p.error("Counts must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable.")
    if args.stage == "train" and args.checkpoint.exists():
        p.error("Checkpoint already exists; choose a fresh output location.")
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    features_np, labels_np, adj, loader_meta = _load_mgtab_tensor_dataset(str(args.data), 0, True)
    train_full, val, test = _build_mgtab_splits(labels_np, 0.8, 0.1, args.seed)
    train = _make_balanced_training_indices(labels_np, train_full, args.seed)
    features_np = _standardize_features_from_train(features_np, train_full)
    features = torch.from_numpy(features_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    adj_norm = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0]))).coalesce().to(device)
    adj_bin = sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device)
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5).to(device)
    cfg = dict(q=0.4, sigma=0.4, sigma_train=0.4, epochs=args.epochs, seed=args.seed,
               hidden=16, dropout=0.5, lr=0.01, weight_decay=0.0005,
               edge_type=0, undirected=True, train_ratio=0.8, val_ratio=0.1,
               balanced_training=True, standardize_from_train=True,
               checkpoint_metric="validation accuracy", feature_noise_scope="all")
    inputs = {name: digest(args.data / name) for name in
              ("features.pt", "labels_bot.pt", "edge_index.pt", "edge_type.pt")}
    start = time.time()
    if args.stage == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        train_model(model, optimizer, features, adj_norm, labels,
                    torch.from_numpy(train).to(device), torch.from_numpy(val).to(device),
                    args.epochs, "gaussian", 0.4)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), config=cfg, input_sha256=inputs), args.checkpoint)
        (args.out / "training_meta.json").write_text(json.dumps(
            dict(config=cfg, input_sha256=inputs, loader=loader_meta,
                 checkpoint_sha256=digest(args.checkpoint), seconds=time.time() - start), indent=2) + "\n")
        print(f"Saved {args.checkpoint}")
        return
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("input_sha256", inputs) != inputs:
        raise ValueError("Checkpoint was trained on different input tensors.")
    checkpoint_cfg = checkpoint.get("config", {})
    for key in ("seed", "sigma_train", "edge_type", "train_ratio", "val_ratio",
                "standardize_from_train", "balanced_training"):
        if key in checkpoint_cfg and checkpoint_cfg[key] != cfg[key]:
            raise ValueError(f"Checkpoint configuration differs: {key}")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    with torch.no_grad():
        clean_preds = model(features, adj_norm).argmax(1)
    nodes = sample_nodes(labels_np, test, args.nodes, args.seed)
    if not nodes:
        raise ValueError("Empty test split.")
    smoother = SmoothHybrid(base_classifier=model, num_classes=int(labels.max()) + 1,
                            prob=0.4, adj=adj_bin, features=features, sigma=0.4,
                            cuda=args.device == "cuda", feature_noise_scope="all", adj_norm_type="row")
    from hybrid_graph.certification.batched import enable_batched_sampling
    enable_batched_sampling(smoother)
    # The completed reproduction resets the MC seed after training/loading.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rows = []
    for i, node in enumerate(nodes, 1):
        before = time.time()
        pre = smoother.sample_votes_target(node_idx=node, n_samples=100, batch_size=args.batch)
        best = int(np.argmax(pre))
        est = smoother.sample_votes_target(node_idx=node, n_samples=args.mc_samples, batch_size=args.batch)
        second = int(np.where(np.arange(len(est)) != best, est, -1).argmax())
        cache = smoother._get_target_cache(node)
        rows.append(dict(dataset="mgtab", seed=args.seed, node=node, label=int(labels[node]),
                         clean_base_pred=int(clean_preds[node]), best_class=best, second_class=second,
                         n_best=int(est[best]), n_second=int(est[second]), n0=100, n=args.mc_samples,
                         m1=int(cache["m1"][0]), m0=int(cache["m0"][0]),
                         num_classes=int(labels.max()) + 1, prob=0.4, sigma=0.4,
                         mc_seconds=time.time() - before))
        write_csv(args.out / "per_node_stats.csv", rows)
        print(f"Sampled {i}/{len(nodes)} test nodes", flush=True)
    eps = np.linspace(0, 2.0, 200)
    budgets = list(range(9))
    hybrid = np.zeros((9, len(eps)))
    sequential = np.zeros_like(hybrid)
    fraction = np.zeros_like(hybrid)
    for row in rows:
        pa, pb = confidence_bounds(row["n_best"], row["n_second"], row["n"], row["num_classes"])
        grids = node_grids(pA_lower=pa, pB_upper=pb, m1=row["m1"], m0=row["m0"],
                           sigma=0.4, p_minus=0.3, p_plus=0.3, d_list=budgets,
                           eps_values=eps, variants=("hybrid", "seq_topo_feat"))
        hm = grids["hybrid"]["lower"] > grids["hybrid"]["upper"]
        sm = grids["seq_topo_feat"]["lower"] > grids["seq_topo_feat"]["upper"]
        correct = row["best_class"] == row["label"]
        hybrid += hm * correct
        sequential += sm * correct
        fraction += hm
    summary = dict(dataset="mgtab", cert_mode="tight", d_list=budgets, eps_values=eps.tolist(),
                   cert_acc_mode=(hybrid / len(rows)).tolist(),
                   cert_acc_seq_mode=(sequential / len(rows)).tolist(),
                   frac_cert_mode=(fraction / len(rows)).tolist(),
                   N=args.mc_samples, N0=100, alpha=0.001,
                   n_targets=len(rows), q=0.4, sigma=0.4, feature_noise_scope="all")
    (args.out / "hybrid_curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    curve_rows = [dict(d=d, epsilon=float(e), hybrid=float(hybrid[d, j] / len(rows)),
                       sequential=float(sequential[d, j] / len(rows)))
                  for d in budgets for j, e in enumerate(eps)]
    write_csv(args.out / "hybrid_curve.csv", curve_rows)
    (args.out / "meta.json").write_text(json.dumps(
        dict(config=cfg, trained_config=checkpoint_cfg, input_sha256=inputs,
             checkpoint_sha256=digest(args.checkpoint), selected_nodes=nodes,
             N=args.mc_samples, N0=100, alpha=0.001,
             device=str(device), torch=torch.__version__, numpy=np.__version__,
             seconds=time.time() - start), indent=2) + "\n")
    print(f"Wrote sampling statistics and curves to {args.out}")


if __name__ == "__main__":
    main()
