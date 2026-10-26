#!/usr/bin/env python3
"""Train external SOTA methods and Hybrid on shared datasets; export comparisons."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True, choices=("structure", "binary", "features"))
    p.add_argument("--dataset", choices=("cora", "citeseer", "pubmed", "amazon_fraud", "yelpchi", "mgtab"))
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--nodes", type=int)
    p.add_argument("--mc-samples", type=int)
    p.add_argument("--n0", type=int)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--node-seed", type=int, default=0)
    p.add_argument("--partitions", type=int)
    p.add_argument("--d-max", type=int, default=3)
    p.add_argument("--n-per-class", type=int, help="Override fraud class train/val sizes.")
    p.add_argument("--sigmas", default="0.5,1", help="Binary-suite Gaussian sigmas.")
    p.add_argument("--epsilon-ratios", default="0.5,1,2.5", help="Feature-suite epsilon/sigma values.")
    p.add_argument("--hybrid-mc-samples", type=int, default=1000, help="Binary-suite Hybrid draws.")
    p.add_argument("--hybrid-nodes", type=int, default=200, help="Binary-suite Hybrid population.")
    p.add_argument("--hybrid-n0", type=int, default=100, help="Binary-suite Hybrid selection draws.")
    p.add_argument("--dry-run", action="store_true", help="Print the comparison protocol without loading datasets.")
    args = p.parse_args(argv)
    args.dataset = args.dataset or ("cora" if args.suite == "binary" else "pubmed")
    if args.suite == "binary" and args.dataset not in ("cora", "citeseer"):
        p.error("The binary comparison requires binary citation attributes (cora or citeseer).")
    args.nodes_overridden = args.nodes is not None
    args.nodes = args.nodes if args.nodes is not None else (1000 if args.suite == "binary" else 200 if args.suite == "features" else 300)
    args.mc_samples = args.mc_samples if args.mc_samples is not None else (1000 if args.suite == "structure" else 10000)
    args.n0 = args.n0 if args.n0 is not None else (1000 if args.suite in ("features", "binary") else 100)
    args.partitions = args.partitions if args.partitions is not None else (300 if args.suite == "features" else 10)
    for key in ("nodes", "mc_samples", "n0", "partitions", "epochs", "batch", "hybrid_mc_samples", "hybrid_nodes", "hybrid_n0"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive.")
    if args.d_max < 0 or (args.n_per_class is not None and args.n_per_class < 1):
        p.error("Budgets must be nonnegative and class sizes positive.")
    try:
        for values in (args.sigmas, args.epsilon_ratios):
            parsed = [float(x) for x in values.split(",")]
            if not parsed or any(not math.isfinite(x) or x <= 0 for x in parsed):
                raise ValueError
    except ValueError:
        p.error("Sigmas and epsilon ratios must contain positive finite values.")
    args.data_root, args.out = args.data_root.expanduser().resolve(), args.out.expanduser().resolve()
    for path in (ROOT / "external", ROOT / "experiments", ROOT / "hybrid_graph",
                 ROOT / "hybrid_smoothing", ROOT / "graph_utils", ROOT / "scripts", args.data_root):
        if args.out == ROOT or args.out.is_relative_to(path):
            p.error("Choose results outside code and dataset directories.")
    return args


def sample_hybrid(data, model, nodes, q, sigma, args, *, scope="target", p_minus=None, p_plus=None):
    import numpy as np
    from hybrid_graph.certification.core_hybrid import SmoothHybrid
    from experiments.comparison.common import seed_all
    from experiments.comparison.common import multiclass_bounds
    smoother = SmoothHybrid(model, data["n_classes"], q, data["adj_bin_t"], data["features"], sigma,
        cuda=data["device"].type == "cuda", feature_noise_scope=scope, adj_norm_type="row")
    if p_minus is not None or p_plus is not None:
        from experiments.comparison.asymmetric import enable_asymmetric_kernel
        enable_asymmetric_kernel(smoother, p_minus=p_minus, p_plus=p_plus)
    seed_all(args.seed)
    rows = []
    for i, node in enumerate(nodes):
        pre = smoother.sample_votes_target(int(node), args.n0, args.batch)
        best = int(np.argmax(pre))
        counts = smoother.sample_votes_target(int(node), args.mc_samples, args.batch)
        rivals = counts.copy()
        rivals[best] = -1
        second = int(np.argmax(rivals))
        cache = smoother._get_target_cache(int(node))
        pa, pb = multiclass_bounds(counts, best, .001)
        rows.append(dict(node=int(node), label=int(data["labels"][node]), prediction=best,
                         n_best=int(counts[best]), n_second=int(counts[second]),
                         m1=int(cache["m1"][0]), m0=int(cache["m0"][0]), p_lower=pa, p_upper=pb))
        print(f"[Hybrid] {i+1}/{len(nodes)} nodes", flush=True)
    return rows


def structure(data, nodes, args):
    import numpy as np
    import torch
    from hybrid_graph.certification.core_hybrid import SmoothHybrid
    from experiments.comparison.agnncert import train_vote
    from experiments.comparison.common import seed_all, summarize, train_base
    from experiments.comparison.upstream import load_module, topology_thresholds
    from experiments.offline_certificates import node_grids
    truth = data["labels"].cpu().numpy()[nodes]
    rows, sources = [], []
    d_list = list(range(args.d_max + 1))
    for division in ("edge", "node"):
        _, _, votes, budget, source = train_vote(data, nodes, division, args.partitions, args.epochs,
            args.seed, args.out / f"agnncert_{division}.pt")
        np.savez(args.out / f"agnncert_{division}_votes.npz", nodes=nodes, votes=votes, M=budget)
        sources.append(source)
        for d in d_list:
            rows.append(summarize(f"AGNNCert-{division}", votes.argmax(1) == truth, budget >= d,
                dataset=args.dataset, d=d, epsilon=0, partitions=args.partitions,
                guarantee="deterministic", support="native partition edge convention",
                mean_vote_budget=float(budget.mean())))
    native, source = load_module("CertifyGNN", "core_Ber.py")
    thresholds, table_source = topology_thresholds(.7)
    sources.extend((source, table_source))
    topo_model = train_base(data, args.epochs, args.seed, 0, args.out / "topology_model.pt")
    # Native P_Base is a flip-retention table. Its equivalent overwrite q is 2p-1.
    efficient = SmoothHybrid(topo_model, data["n_classes"], .4, data["adj_bin_t"], data["features"], 0,
        cuda=data["device"].type == "cuda", feature_noise_scope="target", adj_norm_type="row")
    class EfficientNative(native.Smooth_Ber):
        def _sample_noise_ber(self, idx, num, batch_size):
            counts = efficient.sample_votes_target(idx, num, batch_size)
            self.sampled_counts.append(counts)
            return counts
    topo = EfficientNative(topo_model, data["n_classes"], len(data["labels"]), .4,
                          data["adj_bin_t"], data["features"], args.device == "cuda")
    seed_all(args.seed)
    predictions, selected_predictions, p_lower = [], [], []
    for node in nodes:
        topo.sampled_counts = []
        pred, bound = topo.certify_Ber(int(node), args.n0, args.mc_samples, .001, args.batch)
        predictions.append(pred)
        selected_predictions.append(int(topo.sampled_counts[0].argmax()))
        p_lower.append(bound)
    p_lower = np.asarray(p_lower)
    np.savez(args.out / "topology_bounds.npz", nodes=nodes, prediction=predictions,
             selected_prediction=selected_predictions, p_lower=p_lower)
    for d in d_list:
        if d > len(thresholds):
            raise ValueError("Requested topology budget exceeds the authors' table.")
        rows.append(summarize("TopoRS", np.asarray(selected_predictions) == truth,
            p_lower > (thresholds[d - 1] if d else .5), dataset=args.dataset, d=d, epsilon=0,
            guarantee="probabilistic", support="target-incident edges", sampler_q=.4, table_keep=.7,
            n_mc=args.mc_samples, alpha=.001))
    # Structure-axis supplementary citation settings; independent of Figure 2 presets.
    sigma = .01
    model = train_base(data, args.epochs, args.seed, .04 if args.dataset == "pubmed" else sigma,
                       args.out / "hybrid_model.pt")
    hybrid_nodes = nodes
    if args.dataset == "pubmed" and not args.nodes_overridden:
        from experiments.comparison.common import select_nodes
        hybrid_nodes = select_nodes(data, 1000, args.node_seed)
    stats = sample_hybrid(data, model, hybrid_nodes, .7, sigma, args)
    masks = []
    for row in stats:
        grid = node_grids(pA_lower=row["p_lower"], pB_upper=row["p_upper"], m1=row["m1"], m0=row["m0"],
            sigma=sigma, p_minus=.15, p_plus=.15, d_list=d_list, eps_values=[0.], variants=("hybrid",))["hybrid"]
        masks.append((grid["lower"] > grid["upper"])[:, 0])
    correct = np.array([r["prediction"] == r["label"] for r in stats])
    for i, d in enumerate(d_list):
        rows.append(summarize("Hybrid-NP", correct, np.asarray(masks)[:, i], dataset=args.dataset,
            d=d, epsilon=0, guarantee="probabilistic", support="target-incident edges; target Gaussian noise",
            sampler_q=.7, sigma=sigma, n_mc=args.mc_samples, alpha=.001))
    (args.out / "hybrid_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return rows, sources


def features(data, nodes, args):
    import numpy as np
    from scipy.stats import norm
    from experiments.comparison.agnncert import train_vote
    from experiments.comparison.common import summarize, train_base, seed_all, multiclass_bounds
    sigma = {"amazon_fraud": .02, "yelpchi": .005, "mgtab": .4, "cora": .5}.get(args.dataset, .01)
    _, _, votes, budget, source = train_vote(data, nodes, "node", args.partitions, args.epochs,
        args.seed, args.out / "agnncert_node.pt")
    model = train_base(data, args.epochs, args.seed, sigma, args.out / "hybrid_model.pt")
    import torch
    seed_all(args.seed)
    # One global Gaussian graph draw predicts every selected target. Sharing the
    # draw keeps each target's marginal unchanged and avoids repeating N passes
    # for each node. Confidence statements are per target, as in the paper.
    targets = torch.as_tensor(nodes, device=data["device"])
    def global_votes(count):
        result = torch.zeros((len(nodes), data["n_classes"]), dtype=torch.long, device=data["device"])
        with torch.no_grad():
            for draw in range(count):
                noisy = data["features"] + sigma * torch.randn_like(data["features"])
                predictions = model(noisy, data["adj_norm_t"])[targets].argmax(1)
                result.scatter_add_(1, predictions[:, None], torch.ones_like(predictions[:, None]))
                if (draw + 1) % max(1, count // 10) == 0:
                    print(f"[Global Gaussian] {draw+1}/{count} draws for {len(nodes)} targets", flush=True)
        return result.cpu().numpy()
    pre, est = global_votes(args.n0), global_votes(args.mc_samples)
    np.savez(args.out / "global_gaussian_votes.npz", nodes=nodes, pre_votes=pre, votes=est)
    stats = []
    for i, node in enumerate(nodes):
        best = int(pre[i].argmax())
        rivals = est[i].copy()
        rivals[best] = -1
        second = int(rivals.argmax())
        pa_i, pb_i = multiclass_bounds(est[i], best, .001)
        stats.append(dict(node=int(node), label=int(data["labels"][node]), prediction=best,
            n_best=int(est[i, best]), n_second=int(est[i, second]), p_lower=pa_i, p_upper=pb_i,
            feature_noise_scope="all", n=args.mc_samples))
    pa, pb = np.array([r["p_lower"] for r in stats]), np.array([r["p_upper"] for r in stats])
    radii = np.maximum(0, sigma / 2 * (norm.ppf(pa) - norm.ppf(pb)))
    radii[pa <= pb] = 0
    np.savez(args.out / "per_node_feature_budgets.npz", nodes=nodes, agnn_votes=votes,
             M=budget, hybrid_radius=radii, hybrid_p_lower=pa, hybrid_p_upper=pb)
    correct_a = votes.argmax(1) == data["labels"].cpu().numpy()[nodes]
    correct_h = np.array([r["prediction"] == r["label"] for r in stats])
    rows = []
    for rule, k in (("half_M", budget // 2), ("M", budget), ("M_plus_1", budget + 1), ("twice_M", 2 * budget)):
        for ratio in map(float, args.epsilon_ratios.split(",")):
            if ratio <= 0:
                raise ValueError("Epsilon ratios must be positive.")
            epsilon = sigma * ratio
            fields = dict(dataset=args.dataset, d=0, epsilon=epsilon, epsilon_sigma_ratio=ratio,
                          k_rule=rule, min_k=int(k.min()), max_k=int(k.max()),
                          support="fixed topology; non-target feature rows; total Frobenius norm")
            rows.append(summarize("AGNNCert-N", correct_a, (k >= 0) & (k <= budget),
                                  guarantee="deterministic count-only sufficient bound", **fields))
            rows.append(summarize("Hybrid-global-Gaussian", correct_h, (pa > pb) & (radii > epsilon),
                                  guarantee="probabilistic global norm bound", n_mc=args.mc_samples,
                                  sigma=sigma, alpha=.001, **fields))
    (args.out / "hybrid_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return rows, [source]


def binary(data, nodes, args):
    import numpy as np
    import torch
    from experiments.comparison.common import normalized, seed_all, summarize, train_base
    from experiments.comparison.upstream import load_module
    from experiments.offline_certificates import restricted_grids
    import scipy.sparse as sp
    native, source = load_module("sparse_smoothing", "sparse_smoothing/cert.py")
    coo = sp.triu(data["adj_sp"], k=1).tocoo()
    pairs = torch.tensor(np.stack((coo.row, coo.col)), device=data["device"], dtype=torch.long)
    def sample():
        keep = torch.rand(pairs.shape[1], device=data["device"]) >= .4
        edges = pairs[:, keep].cpu().numpy()
        adj = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
                            shape=data["adj_sp"].shape)
        x = data["features"]
        noisy = torch.where(x > .5, (torch.rand_like(x) >= .6).float(), (torch.rand_like(x) < .01).float())
        return noisy, normalized(adj.maximum(adj.T), data["device"])
    model = train_base(data, args.epochs, args.seed, 0, args.out / "sparse_model.pt", sample=sample)
    def votes(count):
        result = np.zeros((len(nodes), data["n_classes"]), dtype=np.int64)
        with torch.no_grad():
            for _ in range(count):
                x, adj = sample()
                pred = model(x, adj)[nodes].argmax(1).cpu().numpy()
                result[np.arange(len(nodes)), pred] += 1
        return result
    seed_all(args.seed)
    pre, est = votes(args.n0), votes(args.mc_samples)
    pa, pb = native.p_lower_upper_from_votes(est, pre, .01, args.mc_samples)
    np.savez(args.out / "sparse_votes.npz", nodes=nodes, pre=pre, votes=est, p_lower=pa, p_upper=pb)
    truth = data["labels"].cpu().numpy()[nodes]
    # Use the pre-selected class for both correctness and its probability bounds.
    selected = pre.argsort(axis=1)[:, -1]
    correct = selected == truth
    feature_budgets = [0, 1, 2, 4, 9]
    rows = []
    for d in range(args.d_max + 1):
        for dx in feature_budgets:
            certified = np.ones(len(nodes), dtype=bool)
            for additions in range(dx + 1):
                if not (d or dx):
                    lower, upper = pa, pb
                else:
                    _, _, regions = native.joint_regions(0, d, additions, dx - additions, 0., .4, .01, .6)
                    lower = native.compute_rho_for_many(regions, pa, is_sorted=False)
                    upper = native.compute_rho_for_many(regions, pb, is_sorted=False, reverse=True)
                certified &= np.asarray(lower) > np.asarray(upper)
            rows.append(summarize("sparsity-aware-RS", correct, certified, dataset=args.dataset,
                d=d, d_features=dx, epsilon=math.sqrt(dx), n_mc=args.mc_samples, alpha=.01,
                support="global binary attributes; global edge deletions", guarantee="probabilistic"))
    from experiments.comparison.common import select_nodes
    hybrid_nodes = select_nodes(data, min(len(nodes), args.hybrid_nodes), args.node_seed)
    original_n = args.mc_samples
    original_n0 = args.n0
    args.mc_samples = args.hybrid_mc_samples
    args.n0 = args.hybrid_n0
    try:
        for sigma in map(float, args.sigmas.split(",")):
            if sigma <= 0:
                raise ValueError("Sigmas must be positive.")
            model = train_base(data, args.epochs, args.seed, sigma, args.out / f"hybrid_sigma_{sigma}.pt")
            stats = sample_hybrid(data, model, hybrid_nodes, .2, sigma, args, p_minus=.4, p_plus=1e-6)
            correct_h = np.asarray([r["prediction"] == r["label"] for r in stats])
            masks = []
            eps = [math.sqrt(dx) for dx in feature_budgets]
            for row in stats:
                grid = restricted_grids(pA_lower=row["p_lower"], pB_upper=row["p_upper"],
                    m1=row["m1"], m0=row["m0"], sigma=sigma, p_minus=.4, p_plus=1e-6,
                    d_list=list(range(args.d_max + 1)), eps_values=eps, max_d_plus=0)
                masks.append(grid["lower"] > grid["upper"])
            for d in range(args.d_max + 1):
                for j, dx in enumerate(feature_budgets):
                    rows.append(summarize(f"Hybrid-sigma-{sigma}", correct_h, np.asarray(masks)[:, d, j],
                        dataset=args.dataset, d=d, d_features=dx, epsilon=eps[j],
                        n_mc=args.mc_samples, alpha=.001, support="target features; target-incident edge deletions",
                        guarantee="probabilistic"))
            (args.out / f"hybrid_stats_sigma_{sigma}.json").write_text(json.dumps(stats, indent=2) + "\n")
    finally:
        args.mc_samples = original_n
        args.n0 = original_n0
    return rows, [source]


def main():
    args = parse_args()
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if args.out.exists():
        raise SystemExit(f"Output already exists: {args.out}; choose a fresh --out.")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "2")
    import torch
    from experiments.comparison.common import load_data, select_nodes, write_results
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    args.out.mkdir(parents=True)
    start = time.time()
    try:
        data = load_data(args.dataset, args.data_root, torch.device(args.device), args.seed,
                         args.n_per_class, binary=args.suite == "binary" or (args.suite == "features" and args.dataset == "cora"))
        nodes = select_nodes(data, args.nodes, args.node_seed)
        (args.out / "selected_nodes.json").write_text(json.dumps(nodes.tolist()) + "\n")
        rows, sources = {"structure": structure, "features": features, "binary": binary}[args.suite](data, nodes, args)
        import hashlib
        def digest_array(array):
            return hashlib.sha256(array.tobytes()).hexdigest()
        inputs = dict(features=digest_array(data["features"].cpu().numpy()),
            labels=digest_array(data["labels"].cpu().numpy()),
            adjacency_indptr=digest_array(data["adj_sp"].indptr),
            adjacency_indices=digest_array(data["adj_sp"].indices),
            train_nodes=digest_array(data["idx_train"].cpu().numpy()))
        write_results(args.out, rows, dict(protocol=protocol, upstream_executed=sources,
            input_array_sha256=inputs, hybrid_confidence="simultaneous class-wise Clopper-Pearson; maximum rival UCB",
            seconds=time.time() - start, comparison="native threat models and distinct predictors; supports and population sizes stated per row"))
        (args.out / "status.json").write_text(json.dumps(dict(status="complete", rows=len(rows))) + "\n")
        print(f"Saved {len(rows)} comparison rows: {args.out / 'comparison.csv'}")
    except Exception as exc:
        (args.out / "status.json").write_text(json.dumps(dict(status="failed", error=str(exc))) + "\n")
        raise


if __name__ == "__main__":
    main()
