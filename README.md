# Joint Certification for Attributed Graphs

This repository contains the code for the paper
[**Joint Certification for Attributed Graphs: Beyond Topology-Only Robustness**](https://openreview.net/forum?id=1qibUxCQxG).


## How to cite

Blaise Delattre, Hengyu Wu, Wei Yang Bryan Lim, and Yang Cao.
**Joint Certification for Attributed Graphs: Beyond Topology-Only Robustness.**
*Advances in Neural Information Processing Systems (NeurIPS)*, 2026.

```bibtex
@inproceedings{delattre2026joint,
  title = {Joint Certification for Attributed Graphs: Beyond Topology-Only Robustness},
  author = {Blaise Delattre and Hengyu Wu and Wei Yang Bryan Lim and Yang Cao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026}
}
```

## Installation and data

Install PyTorch with CUDA support for GPU runs, then:

```sh
pip install -r requirements.txt
python experiments/prepare_fraud.py --root data/CARE-GNN \
  --datasets amazon_fraud yelpchi --care_static_topk 50
```

Citation datasets download automatically through PyTorch Geometric. Download
MGTAB from the [dataset authors](https://github.com/GraphDetec/MGTAB) and place
`features.pt`, `labels_bot.pt`, `edge_index.pt`, and `edge_type.pt` in `data/MGTAB/`.

## Included paper experiments

This package runs Figure 2 and the certificate comparisons in Appendix I.2,
Tables 6–9. For Table 9, it computes the certificate-coverage columns.


| Code | Paper experiment |
| --- | --- |
| `experiments/reproduce_figure2.py`, `run_pubmed.py`, `run_mgtab.py` | Figure 2 training, sampling and certification |
| `experiments/compare_sota.py`, `comparison/agnncert.py` | AGNNCert and TopoRS comparisons, Tables 6, 8 and 9 |
| `comparison/asymmetric.py` | Binary-feature comparison, Table 7 |
| `comparison/common.py`, `comparison/upstream.py`, `scripts/fetch_external.py`, `external/repos.json` | Shared data/models and pinned baseline implementations |
| `hybrid_graph/`, `hybrid_smoothing/`, `experiments/offline_certificates.py` | Models, training, topology/feature sampling, Hybrid NP and Sequential NP bounds |
| `experiments/prepare_fraud.py`, `mgtab_data.py`, `graph_utils/` | Data preparation and graph storage |

Paths under `comparison/` refer to `experiments/comparison/`.

## Code layout

`hybrid_graph/certification/core_hybrid.py` contains the `SmoothHybrid` entry point:
the smoothing channels, independent pilot/estimation draws, confidence bounds,
and certificate assembly. Its CPU sampling methods are grouped in
`local_sampling.py` (dispatch and target caches), `gcn_sampling.py` and
`gcn_global.py` (GCN), and `bwgnn_sampling.py` (BWGNN). These private mixins share
the sampler state; CUDA batch evaluation is in `batched.py`.

`hybrid_smoothing/certif_hybrid.py` computes the Hybrid NP, topology-only NP,
and Sequential NP grids, using the numerical routines in `utils_hybrid.py`.
`hybrid_graph/train_fraud_hybrid.py` owns one fraud experiment's data, training,
checkpoint selection and certification; `fraud_results.py` aggregates its
per-target certificates and writes the curve CSVs.

## Main experiments: Figure 2

```sh
python experiments/reproduce_figure2.py --dataset all --dry-run
CUDA_VISIBLE_DEVICES=0 python experiments/reproduce_figure2.py \
  --dataset all --device cuda --out results/figure2
```

Choose `--dataset amazon_fraud`, `yelpchi`, `pubmed`, or `mgtab` for one dataset.
`--stage train` and `--stage certify` run separately; a single dataset can use
`--stage certify --checkpoint PATH`. Defaults are seed 42, 200 epochs,
N=10,000 independent estimation draws and alpha=0.001 per target.

| Dataset | Model | q | sigma | sigma train | Targets | N0 |
| --- | --- | --- | --- | --- | --- | --- |
| Amazon Fraud | BWGNN, order 2, hidden 64 | 0.5 | 0.1 | 0.5 | 500 | 50 |
| YelpChi | BWGNN, order 2, hidden 64 | 0.6 | 0.005 | 0.005 | 500 | 50 |
| PubMed | GCN, hidden 16 | 0.7 | 0.01 | 0.04 | 1000 | 10 |
| MGTAB | GCN, hidden 16 | 0.4 | 0.4 | 0.4 | 400 | 100 |

Figure 2 uses global Gaussian feature noise, with one total Frobenius
budget epsilon. Topology smoothing overwrites target-incident edge slots with
keep probability q; effective retention is (1+q)/2.

## Baseline comparisons

```sh
pip install -r requirements-comparisons.txt
python scripts/fetch_external.py AGNNCert CertifyGNN sparse_smoothing
python experiments/compare_sota.py --suite structure --dataset pubmed \
  --device cuda --out results/structure
python experiments/compare_sota.py --suite binary --dataset cora \
  --device cuda --out results/binary
python experiments/compare_sota.py --suite features --dataset amazon_fraud \
  --device cuda --out results/features
```

[Comparison protocols](experiments/README.md) describe the
method-specific settings. `external/repos.json` pins the external code.

`--data-root`, `--seed`, `--epochs`, `--nodes`, `--mc-samples` and `--batch`
control run sizes. Smaller counts change the experiment. Commands generate
checkpoints, sampling statistics, curves and metrics under `results/`;
use a fresh output directory for each run.


## Codebase and attribution

We use the original [CertifyGNN code](https://github.com/binghuiwang/CertifyGNN)
and extend its topology-only randomized smoothing to hybrid randomized
smoothing for joint edge flips and continuous node-feature perturbations.
Our modified implementation lives in `hybrid_graph/`, adapting CertifyGNN's
GCN layers, dataset utilities and graph-certification pipeline. Reused CertifyGNN code retains its
[MIT license and copyright notice](LICENSES/CertifyGNN-MIT.txt).

For hybrid randomized smoothing, we use and adapt the
[hybrid-randomized-smoothing codebase](https://github.com/tdsai-lab/hybrid-randomized-smoothing).
Its hybrid discrete–continuous certification routines are specialized here
to graph topology and node features in `hybrid_smoothing/`. This repository
integrates these two codebases and adds the graph-specific Hybrid NP and
Sequential NP certificates.

The graph storage helpers in `graph_utils/` are
adapted from [sparse_smoothing](https://github.com/abojchevski/sparse_smoothing).
Fraud dataset preparation uses the data released with
[CARE-GNN](https://github.com/YingtongDou/CARE-GNN).

Baseline comparisons execute the authors' implementations of
[AGNNCert](https://github.com/JetRichardLee/AGNNCert),
[CertifyGNN](https://github.com/binghuiwang/CertifyGNN),
[sparse_smoothing](https://github.com/abojchevski/sparse_smoothing).
