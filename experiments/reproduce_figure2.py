#!/usr/bin/env python3
"""Train and certify Figure 2 datasets, with all parameters visible in the plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("amazon_fraud", "yelpchi", "pubmed", "mgtab")
CONFIG = {
    "amazon_fraud": dict(q=0.5, q_train=0.5, sigma=0.1, sigma_train=0.5,
                         nodes=500, n0=50, d="0,1,2,3,4,5", eps=0.5, per_class=100),
    "yelpchi": dict(q=0.6, q_train=0.7, sigma=0.005, sigma_train=0.005,
                    nodes=500, n0=50, d="0,1,2,3,4,5", eps=0.03, per_class=2000),
    "pubmed": dict(q=0.7, q_train=0.7, sigma=0.01, sigma_train=0.04,
                   nodes=1000, n0=10, d="0,1,2,3", eps=0.1),
    "mgtab": dict(q=0.4, q_train=0.4, sigma=0.4, sigma_train=0.4,
                  nodes=400, n0=100, d="0,1,2,3,4,5,6,7,8", eps=2.0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    p.add_argument("--stage", choices=("all", "train", "certify"), default="all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "results/figure2")
    p.add_argument("--checkpoint", type=Path,
                   help="Single-dataset certification from an existing checkpoint.")
    p.add_argument("--mc-samples", type=int, default=10000)
    p.add_argument("--nodes", type=int, help="Override the recorded number of test nodes.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="Print commands; do not run them.")
    args = p.parse_args(argv)
    if min(args.mc_samples, args.epochs, args.batch) < 1 or (args.nodes is not None and args.nodes < 1):
        p.error("Sample, epoch, batch and node counts must be positive.")
    if args.checkpoint and (args.dataset == "all" or args.stage != "certify"):
        p.error("--checkpoint requires a single dataset and --stage certify.")
    args.data_root = args.data_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    for protected in (ROOT / "hybrid_graph", ROOT / "hybrid_smoothing", ROOT / "graph_utils",
                      ROOT / "experiments", ROOT / "external", ROOT / "scripts", args.data_root):
        if args.out == ROOT or args.out.is_relative_to(protected):
            p.error("Choose an output directory outside source code and raw data.")
    return args


def build_plan(args):
    """Return explicit commands and their logs, without importing training libraries."""
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    plan = []
    for dataset in selected:
        c = CONFIG[dataset]
        run = args.out / dataset
        train, cert = run / "train", run / "certify"
        nodes = args.nodes if args.nodes is not None else c["nodes"]
        cuda = ["--cuda"] if args.device == "cuda" else []
        if dataset == "mgtab":
            for stage in ("train", "certify"):
                if args.stage not in ("all", stage):
                    continue
                checkpoint = args.checkpoint or train / "checkpoint.pt"
                command = [sys.executable, str(ROOT / "experiments/run_mgtab.py"),
                           "--stage", stage, "--data", str(args.data_root / "MGTAB"),
                           "--out", str(train if stage == "train" else cert),
                           "--checkpoint", str(checkpoint), "--device", args.device,
                           "--seed", str(args.seed), "--epochs", str(args.epochs),
                           "--nodes", str(nodes), "--mc-samples", str(args.mc_samples),
                           "--batch", str(args.batch)]
                plan.append((dataset, stage, command, (train if stage == "train" else cert) / "run.log"))
            continue

        common = ["--dataset", dataset, "--seed", str(args.seed),
                  "--epochs", str(args.epochs),
                  "--feature_noise", "gaussian", "--sigma_train", str(c["sigma_train"]),
                  "--feature_noise_scope", "all"] + cuda
        if dataset in ("amazon_fraud", "yelpchi"):
            driver = ROOT / "hybrid_graph/train_fraud_hybrid.py"
            checkpoint = args.checkpoint or train / "checkpoint.best.pt"
            common += ["--caregnn_root", str(args.data_root / "CARE-GNN/processed"),
                       "--n_per_class", str(c["per_class"]), "--fraud_graph_source", "care_static",
                       "--care_static_topk", "50",
                       "--adj_norm_type", "row", "--model", "bwgnn", "--bwgnn_order", "2",
                       "--hidden", "64", "--dropout", "0", "--lr", "0.005",
                       "--weight_decay", "0.0001", "--train_loss", "weighted_nll",
                       "--selection_metric", "val_bacc"]
            # Match the saved training invocation; certification sigma is separate.
            train_sigma = 0.01 if dataset == "amazon_fraud" else c["sigma"]
            save = ["--save_checkpoint", "--checkpoint_out", str(checkpoint)]
        else:
            driver = ROOT / "experiments/run_pubmed.py"
            checkpoint = args.checkpoint or train / "checkpoint.hybrid.prob.0.7.sigma.0.01.pt"
            common += ["--hidden", "16", "--num_layers", "2", "--dropout", "0.5",
                       "--model_arch", "gcn", "--lr", "0.01", "--weight_decay", "0.0005"]
            train_sigma, save = c["sigma"], []
        if args.stage in ("all", "train"):
            command = [sys.executable, str(driver), str(train)] + common + [
                "--prob", str(c["q_train"]), "--sigma", str(train_sigma), "--skip_certify"] + save
            plan.append((dataset, "train", command, train / "run.log"))
        if args.stage in ("all", "certify"):
            command = [sys.executable, str(driver), str(cert)] + common + [
                "--prob", str(c["q"]), "--sigma", str(c["sigma"]),
                "--skip_train", "--checkpoint_in", str(checkpoint),
                "--N0", str(c["n0"]), "--N", str(args.mc_samples),
                "--batch", str(args.batch), "--alpha", "0.001", "--skip", "1",
                "--max", str(nodes), "--certify_node_selection", "random_stratified",
                "--certify_sample_seed", str(args.seed), "--d_list", c["d"],
                "--eps_max", str(c["eps"]), "--n_eps", "200", "--cert_mode", "tight"]
            plan.append((dataset, "certify", command, cert / "run.log"))
    return plan


def main():
    args = parse_args()
    plan = build_plan(args)
    for dataset, stage, command, log in plan:
        print(f"[{dataset}/{stage}] {shlex.join(command)}", flush=True)
        if not args.dry_run and log.parent.exists():
            raise SystemExit(f"Output already exists: {log.parent}. Choose a fresh --out.")
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PLANETOID_ROOT"] = str(args.data_root / "planetoid")
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(args.out / ".mplconfig")
    env["XDG_CACHE_HOME"] = str(args.out / ".cache")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "2")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        subprocess.run([sys.executable, "-c",
                        "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"],
                       check=True, env=env, cwd=ROOT)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        if dataset in ("amazon_fraud", "yelpchi"):
            path = args.data_root / "CARE-GNN/processed" / f"{dataset}_care_static_topk50.npz"
            if not path.is_file():
                raise SystemExit(f"Missing {path}. Run experiments/prepare_fraud.py first.")
        if dataset == "mgtab":
            for name in ("features.pt", "labels_bot.pt", "edge_index.pt", "edge_type.pt"):
                path = args.data_root / "MGTAB" / name
                if not path.is_file():
                    raise SystemExit(f"Missing {path}. See the data section in README.md.")
    if args.stage == "certify":
        for _, _, command, _ in plan:
            flag = "--checkpoint" if "--checkpoint" in command else "--checkpoint_in"
            path = Path(command[command.index(flag) + 1])
            if not path.is_file():
                raise SystemExit(f"Missing checkpoint: {path}. Run --stage train first.")
    args.out.mkdir(parents=True, exist_ok=True)
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  commands=[dict(dataset=d, stage=s, argv=c, log=str(l)) for d, s, c, l in plan],
                  confidence_budget="N independent estimation draws", status="running")
    record_path = args.out / f"{args.dataset}_{args.stage}_run.json"
    if record_path.exists():
        raise SystemExit(f"Run record already exists: {record_path}")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for dataset, stage, command, log in plan:
            log.parent.mkdir(parents=True, exist_ok=False)
            print(f"Running {dataset}/{stage}; log: {log}", flush=True)
            with log.open("w") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            if stage == "certify":
                path = log.parent / "hybrid_curve_summary.json"
                summary = json.loads(path.read_text())
                summary.update(N=args.mc_samples, N0=CONFIG[dataset]["n0"],
                               feature_noise_scope="all")
                path.write_text(json.dumps(summary, indent=2) + "\n")
        record["status"] = "complete"
    except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
