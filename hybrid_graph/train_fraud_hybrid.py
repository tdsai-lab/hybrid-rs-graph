"""Train a fraud BWGNN and export Hybrid/Sequential NP certificate curves."""

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_graph.certification import SmoothHybrid
from hybrid_graph.certification.batched import enable_batched_sampling
from hybrid_graph.fraud_results import certificate_row, certificate_curves, write_curve_csv
from hybrid_graph.fraud_utils import (
    build_inverse_frequency_class_weights,
    build_fraud_model,
    evaluate_binary_model,
    flatten_metrics,
    fmt_metric,
    is_better_checkpoint,
    load_checkpoint,
    load_fraud_graph_data,
    require_fraud_dataset,
    save_checkpoint,
    set_random_seeds,
    write_json,
    write_metrics_history_csv,
)


def _parse_d_list(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--d_list must contain at least one integer.")
    values = sorted(set(values))
    if values[0] < 0:
        raise ValueError("--d_list cannot contain negative values.")
    return values


def _make_eps_values(eps_max, n_eps):
    n_eps = int(n_eps)
    eps_max = float(eps_max)
    if n_eps <= 1:
        return np.array([max(0.0, eps_max)], dtype=np.float64)
    if eps_max < 0.0:
        raise ValueError("--eps_max must be >= 0.")
    return np.linspace(0.0, eps_max, n_eps, dtype=np.float64)


def _select_certify_nodes(idx_test, labels, strategy, skip, max_nodes, seed):
    test_nodes = [int(x) for x in idx_test.detach().cpu().tolist()]
    skip = max(1, int(skip))
    candidate_nodes = test_nodes[::skip]
    if max_nodes < 0:
        target_count = len(candidate_nodes)
    else:
        target_count = min(int(max_nodes), len(candidate_nodes))

    if target_count <= 0:
        return []

    strategy = str(strategy).strip().lower()
    rng = np.random.RandomState(int(seed))

    if strategy == "stride":
        return candidate_nodes[:target_count]

    if strategy == "random":
        shuffled = list(candidate_nodes)
        rng.shuffle(shuffled)
        return shuffled[:target_count]

    if strategy == "random_stratified":
        labels_cpu = labels.detach().cpu().numpy()
        by_label = {}
        for node_idx in candidate_nodes:
            label_i = int(labels_cpu[node_idx])
            by_label.setdefault(label_i, []).append(int(node_idx))

        label_keys = sorted(by_label.keys())
        for key in label_keys:
            rng.shuffle(by_label[key])

        selected = []
        # Round-robin over labels to keep the sample balanced when possible.
        while len(selected) < target_count:
            progressed = False
            for key in label_keys:
                bucket = by_label[key]
                if bucket:
                    selected.append(bucket.pop())
                    progressed = True
                    if len(selected) >= target_count:
                        break
            if not progressed:
                break
        return selected

    raise ValueError(f"Unsupported certify node selection strategy '{strategy}'.")


def _capture_model_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _print_metrics(prefix, metrics):
    fields = flatten_metrics(metrics)
    print(prefix, " ".join(f"{key}={fmt_metric(value)}" for key, value in fields.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str, help="folder to save model and logs")
    parser.add_argument(
        "--dataset", type=str, default="amazon_fraud", help="fraud dataset (amazon_fraud|yelpchi)"
    )
    parser.add_argument(
        "--caregnn_root",
        type=str,
        default=None,
        help="root containing prepared CARE-GNN fraud .npz files",
    )
    parser.add_argument("--n_per_class", type=int, default=100, help="train/val nodes per class")
    parser.add_argument(
        "--fraud_graph_source",
        type=str,
        default="homo",
        choices=["homo", "care_static"],
        help="fixed fraud graph used by the certifiable GCN",
    )
    parser.add_argument(
        "--care_static_topk",
        type=int,
        default=50,
        help="top-k used when fraud_graph_source=care_static",
    )
    parser.add_argument(
        "--adj_norm_type",
        type=str,
        default="row",
        choices=["row", "sym"],
        help="adjacency normalization used by the base model",
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--batch", type=int, default=32, help="MC batch size for certification")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--hidden", type=int, default=64, help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--model", type=str, default="bwgnn", choices=["bwgnn"], help="base fraud model"
    )
    parser.add_argument(
        "--bwgnn_order", type=int, default=2, help="Bernstein polynomial order when --model=bwgnn"
    )
    parser.add_argument(
        "--prob", type=float, default=0.95, help="status-keep probability for Bernoulli overwrite"
    )
    parser.add_argument(
        "--train_loss",
        type=str,
        default="weighted_nll",
        choices=["weighted_nll"],
        help="training loss for the fraud GCN",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="last",
        choices=["last", "val_accuracy", "val_auc", "val_ap", "val_bacc", "val_loss"],
        help="checkpoint selection rule; 'last' matches the original CertifyGNN training behavior",
    )
    parser.add_argument(
        "--feature_noise", type=str, default="gaussian", choices=["none", "gaussian"]
    )
    parser.add_argument(
        "--sigma_train", type=float, default=0.01, help="gaussian std used only during training"
    )
    parser.add_argument(
        "--feature_noise_scope",
        type=str,
        default="target",
        choices=["all", "target"],
    )
    parser.add_argument(
        "--sigma", type=float, default=0.01, help="gaussian smoothing std for hybrid certification"
    )
    parser.add_argument(
        "--d_list", type=str, default="0,1,2,3,4,5", help="comma-separated discrete budgets"
    )
    parser.add_argument(
        "--eps_max",
        type=float,
        default=0.03,
        help="max epsilon for curve (clamped to at least 3*sigma)",
    )
    parser.add_argument("--n_eps", type=int, default=200, help="number of epsilon points")
    parser.add_argument("--cert_mode", type=str, default="tight", choices=["tight", "half"])
    parser.add_argument("--certifyfile", type=str, default=None, help="per-node tsv output")
    parser.add_argument("--curve_csv", type=str, default=None, help="aggregate curve csv output")
    parser.add_argument("--curve_json", type=str, default=None, help="aggregate curve json output")
    parser.add_argument(
        "--metrics_csv", type=str, default=None, help="training metrics history csv"
    )
    parser.add_argument(
        "--metrics_json", type=str, default=None, help="training + evaluation summary json"
    )
    parser.add_argument(
        "--checkpoint_out", type=str, default=None, help="best checkpoint output path"
    )
    parser.add_argument(
        "--checkpoint_in",
        type=str,
        default=None,
        help="checkpoint to load before evaluation/certification",
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        default=False,
        help="persist the best checkpoint to disk",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="skip training and only run evaluation/certification from a checkpoint",
    )
    parser.add_argument(
        "--skip_certify",
        action="store_true",
        default=False,
        help="skip hybrid certification and only run training/evaluation",
    )
    parser.add_argument("--skip", type=int, default=1, help="how many test examples to skip")
    parser.add_argument("--max", type=int, default=50, help="stop after this many examples")
    parser.add_argument(
        "--certify_node_selection",
        type=str,
        default="stride",
        choices=["stride", "random", "random_stratified"],
        help="how to select the certified test nodes from idx_test",
    )
    parser.add_argument(
        "--certify_sample_seed",
        type=int,
        default=None,
        help="seed used for random/random_stratified certified-node sampling; defaults to --seed",
    )
    parser.add_argument("--N0", type=int, default=100, help="MC samples for class selection")
    parser.add_argument("--N", type=int, default=1000, help="MC samples for probability estimation")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    args = parser.parse_args()
    args.dataset = require_fraud_dataset(args.dataset)
    args.cuda = bool(args.cuda and torch.cuda.is_available())
    if args.prob <= 0.0 or args.prob >= 1.0:
        raise ValueError("--prob must be in (0, 1).")
    if args.sigma <= 0.0:
        raise ValueError("--sigma must be > 0 for hybrid certification.")
    if args.sigma_train < 0.0:
        raise ValueError("--sigma_train must be >= 0.")
    min_eps_max = 3.0 * float(args.sigma)
    if float(args.eps_max) < min_eps_max:
        print(
            f"[fraud-hybrid] increasing eps_max from {args.eps_max} to {min_eps_max} "
            f"to satisfy eps_max >= 3*sigma."
        )
        args.eps_max = min_eps_max
    os.makedirs(args.outdir, exist_ok=True)
    output_names = dict(
        certifyfile=f"certify_hybrid.prob.{args.prob}.sigma.{args.sigma}.tsv",
        curve_csv="hybrid_curve.csv",
        curve_json="hybrid_curve_summary.json",
        metrics_csv="metrics_history.csv",
        metrics_json="metrics_summary.json",
        checkpoint_out="checkpoint.best.pt",
    )
    for field, filename in output_names.items():
        if getattr(args, field) is None:
            setattr(args, field, os.path.join(args.outdir, filename))
    return args


class FraudExperiment:
    """Own the data, model and outputs for one training/certification run."""

    def __init__(self, args):
        self.args = args
        self.d_list = _parse_d_list(args.d_list)
        self.eps_values = _make_eps_values(args.eps_max, args.n_eps)
        print("[fraud-hybrid] config:", vars(args))
        set_random_seeds(args.seed, args.cuda)
        self.data = load_fraud_graph_data(
            dataset=args.dataset,
            caregnn_root=args.caregnn_root,
            n_per_class=args.n_per_class,
            split_seed=args.seed,
            fraud_graph_source=args.fraud_graph_source,
            care_static_topk=args.care_static_topk,
            adj_norm_type=args.adj_norm_type,
            cuda=args.cuda,
            move_adj_bin_to_cuda=args.cuda,
        )
        self.adj_norm_t = self.data["adj_norm_t"]
        self.adj_bin_t = self.data["adj_bin_t"]
        self.features = self.data["features"]
        self.labels = self.data["labels"]
        self.idx_train = self.data["idx_train"]
        self.idx_val = self.data["idx_val"]
        self.idx_test = self.data["idx_test"]
        self.n_nodes = self.data["n_nodes"]
        self.n_features = self.data["n_features"]
        self.n_classes = self.data["n_classes"]
        self.model = build_fraud_model(
            model_name=args.model,
            n_features=self.n_features,
            n_hidden=args.hidden,
            n_classes=self.n_classes,
            dropout=args.dropout,
            bwgnn_order=args.bwgnn_order,
        )
        print(
            f"[fraud-hybrid] model={args.model} hidden={args.hidden} dropout={args.dropout} bwgnn_order={args.bwgnn_order}"
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if args.cuda:
            self.model.cuda()
        self.train_class_weights = None
        if args.train_loss == "weighted_nll":
            self.train_class_weights = build_inverse_frequency_class_weights(
                self.labels, self.idx_train
            )
            print(
                f"[fraud-hybrid] class weights for weighted_nll: {self.train_class_weights.detach().cpu().tolist()}"
            )

    def evaluate(self):
        return evaluate_binary_model(
            model=self.model,
            features=self.features,
            adj_norm_t=self.adj_norm_t,
            labels=self.labels,
            idx_train=self.idx_train,
            idx_val=self.idx_val,
            idx_test=self.idx_test,
        )

    def train_one_epoch(self, epoch):
        start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        x_train = self.features
        if self.args.feature_noise == "gaussian" and self.args.sigma_train > 0.0:
            x_train = self.features + torch.randn_like(self.features) * self.args.sigma_train
        output = self.model(x_train, self.adj_norm_t)
        loss_train = F.nll_loss(
            output[self.idx_train], self.labels[self.idx_train], weight=self.train_class_weights
        )
        acc_train_noisy = float(
            (output[self.idx_train].argmax(1) == self.labels[self.idx_train]).float().mean().item()
        )
        loss_train.backward()
        self.optimizer.step()
        metrics_by_split = self.evaluate()
        elapsed = time.time() - start
        row = {
            "epoch": int(epoch + 1),
            "time_seconds": float(elapsed),
            "train_loss": float(loss_train.item()),
            "train_accuracy_noisy": float(acc_train_noisy),
            "prob": float(self.args.prob),
            "sigma": float(self.args.sigma),
            "sigma_train": float(self.args.sigma_train),
            "model": self.args.model,
            "bwgnn_order": int(self.args.bwgnn_order),
            "adj_norm_type": self.args.adj_norm_type,
            "train_loss_name": self.args.train_loss,
        }
        row.update(flatten_metrics(metrics_by_split))
        prefix = (
            f"Epoch: {epoch + 1:04d} loss_train={fmt_metric(row['train_loss'])} "
            f"acc_train_noisy={fmt_metric(acc_train_noisy)} time={elapsed:.4f}s"
        )
        _print_metrics(prefix, metrics_by_split)
        return row, metrics_by_split

    def certify_hybrid(self):
        args = self.args
        p_flip = (1.0 - float(args.prob)) * 0.5
        smoother = SmoothHybrid(
            self.model,
            self.n_classes,
            args.prob,
            self.adj_bin_t,
            self.features,
            args.sigma,
            args.cuda,
            args.feature_noise_scope,
            args.adj_norm_type,
            self.adj_norm_t,
        )
        enable_batched_sampling(smoother)
        sample_seed = int(
            args.seed if args.certify_sample_seed is None else args.certify_sample_seed
        )
        selected = _select_certify_nodes(
            self.idx_test,
            self.labels,
            args.certify_node_selection,
            args.skip,
            args.max,
            sample_seed,
        )
        if not selected:
            raise ValueError("No test nodes selected; adjust --skip/--max.")
        print(f"[certify-hybrid] selected {len(selected)} test nodes; N0={args.N0}, N={args.N}")
        results, rows = [], []
        with open(args.certifyfile, "w", newline="", encoding="utf-8") as handle:
            writer = None
            for node in tqdm(selected, desc="certify-hybrid-test-nodes"):
                started = time.time()
                result = smoother.certify_node_hybrid(
                    node,
                    args.N0,
                    args.N,
                    args.alpha,
                    args.batch,
                    self.d_list,
                    self.eps_values,
                    p_flip,
                    p_flip,
                )
                row = certificate_row(
                    node, int(self.labels[node]), result, self.d_list, self.eps_values
                )
                row["time"] = str(datetime.timedelta(seconds=time.time() - started))
                if writer is None:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(row), delimiter="\t", lineterminator="\n"
                    )
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                results.append(result)
                rows.append(row)

        correct = np.array([row["correct"] for row in rows], dtype=bool)
        curves = certificate_curves(results, correct, args.cert_mode)
        write_curve_csv(args.curve_csv, self.d_list, self.eps_values, curves)
        metadata_fields = (
            "dataset seed model bwgnn_order adj_norm_type prob sigma feature_noise_scope "
            "sigma_train N N0 certify_node_selection cert_mode"
        ).split()
        summary = {name: getattr(args, name) for name in metadata_fields}
        labels, counts = np.unique([row["label"] for row in rows], return_counts=True)
        accuracy = float(correct.mean())
        accuracy_fields = dict(
            smoothed_accuracy=accuracy,
            smoothed_accuracy_selected_nodes=accuracy,
            selected_clean_majority_acc=accuracy,
        )
        summary.update(
            pipeline="fraud_hybrid",
            p_plus_adj=p_flip,
            p_minus_adj=p_flip,
            feature_noise_train=args.feature_noise,
            n_selected_nodes=len(selected),
            certify_sample_seed=sample_seed,
            selected_label_counts={int(label): int(count) for label, count in zip(labels, counts)},
            d_list=self.d_list,
            eps_values=self.eps_values.tolist(),
            certifier_methods=["hybrid_np", "sequential_np"],
            **accuracy_fields,
        )
        summary.update({name: values.tolist() for name, values in curves.items()})
        for name in ("pA_lower", "pB_upper", "m1", "m0"):
            summary[f"mean_{name}"] = float(np.mean([row[name] for row in rows]))
        write_json(args.curve_json, summary)
        print(f"[stats] saved certificate curves: {args.curve_json}")
        return dict(
            n_selected_nodes=len(selected),
            **accuracy_fields,
            certifyfile=args.certifyfile,
            curve_csv=args.curve_csv,
            curve_json=args.curve_json,
        )

    def run(self):
        history_rows = []
        best_metrics = None
        best_epoch = None
        best_state_dict = None
        if self.args.skip_train:
            checkpoint_to_load = self.args.checkpoint_in or self.args.checkpoint_out
            load_checkpoint(checkpoint_to_load, model=self.model)
            print(
                "[fraud-hybrid] skip_train=1 -> training skipped, running evaluation/certification only."
            )
        else:
            t_total = time.time()
            for epoch in range(self.args.epochs):
                row, metrics_by_split = self.train_one_epoch(epoch)
                history_rows.append(row)
                should_save = False
                if self.args.selection_metric == "last":
                    should_save = True
                elif is_better_checkpoint(
                    metrics_by_split, best_metrics, selection_metric=self.args.selection_metric
                ):
                    should_save = True
                if should_save:
                    best_metrics = metrics_by_split
                    best_epoch = int(epoch + 1)
                    best_state_dict = _capture_model_state_dict(self.model)
                    if self.args.save_checkpoint:
                        save_checkpoint(
                            self.args.checkpoint_out,
                            model=self.model,
                            optimizer=self.optimizer,
                            epoch=epoch + 1,
                            args_dict=vars(self.args),
                            metrics_by_split=metrics_by_split,
                            extra={"selection_metric": self.args.selection_metric},
                        )
                        print(
                            f"[fraud-hybrid] updated best checkpoint at epoch {best_epoch}: {self.args.checkpoint_out}"
                        )
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            write_metrics_history_csv(self.args.metrics_csv, history_rows)
            print(f"[fraud-hybrid] saved metrics history: {self.args.metrics_csv}")
            if best_state_dict is None:
                best_state_dict = _capture_model_state_dict(self.model)
            if self.args.save_checkpoint:
                load_checkpoint(self.args.checkpoint_out, model=self.model)
            else:
                self.model.load_state_dict(best_state_dict)
                print(
                    "[fraud-hybrid] restored best model from in-memory state (checkpoint saving disabled)."
                )
        final_metrics = self.evaluate()
        _print_metrics("[fraud-hybrid] best metrics:", final_metrics)
        if self.args.skip_certify:
            print("[fraud-hybrid] skip_certify=1 -> certification skipped.")
            cert_summary = {
                "skipped": True,
                "reason": "skip_certify",
                "certifyfile": None,
                "curve_csv": None,
                "curve_json": None,
            }
        else:
            cert_summary = self.certify_hybrid()
        metadata_fields = (
            "dataset seed n_per_class model bwgnn_order fraud_graph_source care_static_topk "
            "adj_norm_type prob sigma sigma_train feature_noise feature_noise_scope "
            "train_loss selection_metric epochs N N0 save_checkpoint"
        ).split()
        summary = {name: getattr(self.args, name) for name in metadata_fields}
        summary.update(
            pipeline="fraud_hybrid",
            best_epoch=best_epoch,
            checkpoint_file=self.args.checkpoint_out if self.args.save_checkpoint else None,
            metrics_history_csv=self.args.metrics_csv,
            metrics=final_metrics,
            smoothed_accuracy=cert_summary.get("smoothed_accuracy"),
            certification=cert_summary,
        )
        write_json(self.args.metrics_json, summary)
        print(f"[fraud-hybrid] saved metrics summary: {self.args.metrics_json}")


def main():
    FraudExperiment(parse_args()).run()


if __name__ == "__main__":
    main()
