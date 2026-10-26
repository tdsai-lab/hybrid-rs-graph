# Baseline comparison protocols

```sh
pip install -r requirements-comparisons.txt
python scripts/fetch_external.py AGNNCert CertifyGNN sparse_smoothing
```

The adapters execute the pinned upstream implementations in
`external/repos.json`. Each run exports comparison tables, checkpoints and
sampling statistics into its chosen `--out` directory. Use `--dry-run`
to inspect settings. Existing output directories are protected.

## Structure: AGNNCert, TopoRS and Hybrid

```sh
python experiments/compare_sota.py --suite structure --dataset pubmed \
  --device cuda --out results/structure_pubmed
```

Also supports Cora and Citeseer. Defaults: seed 42, node seed 0, 200 epochs,
T=10, N0=100, N=1000, alpha=0.001, d=0..3, epsilon=0. AGNNCert/TopoRS evaluate
300 targets. Hybrid evaluates 1000 on PubMed and 300 otherwise.
`--nodes` explicitly overrides the population. `--partitions 200` or `300`
produces the paper's partition-budget diagnostics.

AGNNCert uses the authors' hash partitioning and native vote certificate,
with the shared row-normalized GCN/BWGNN and data split. TopoRS uses upstream
`Smooth_Ber.certify_Ber` and P_Base thresholds, with an efficient target-row
sampler and fixed self-loops. Its flip retention 0.7 corresponds to overwrite
q=0.4. Hybrid uses q=0.7 and sigma=0.01. These native predictors differ.

## Binary attributes: sparsity-aware smoothing and Hybrid

```sh
python experiments/compare_sota.py --suite binary --dataset cora \
  --device cuda --out results/binary_cora
```

Also supports Citeseer. The upstream joint binary NP certificate uses global
binary noise: edge deletion/insertion (0.4, 0), attribute deletion/insertion
(0.6, 0.01). It evaluates 1000 targets, N0=1000, N=10000, alpha=0.01.
Hybrid evaluates 200 stratified targets, N0=100, N=1000, alpha=0.001,
sigma=0.5/1, topology probabilities (0.4, 1e-6), and target-only Gaussian features.
Both certify structural deletions; their feature supports differ.
The table covers d_X=0,1,2,4,9 and epsilon=sqrt(d_X).
Controls include `--d-max`, `--sigmas`, `--hybrid-nodes`, `--hybrid-n0`,
`--hybrid-mc-samples`.

## Multiple feature rows: AGNNCert and Hybrid

```sh
python experiments/compare_sota.py --suite features --dataset amazon_fraud \
  --device cuda --out results/features_amazon
```

Also supports Cora, Citeseer, PubMed, YelpChi and MGTAB. Both predictors evaluate
200 stratified targets. AGNNCert-N uses T=300. Hybrid uses fixed topology,
global Gaussian features, N0=1000, N=10000, alpha=0.001, with sigma=0.02 for
Amazon, 0.5 for Cora, 0.01 for Citeseer/PubMed, 0.005 for YelpChi and 0.4 for MGTAB.
MGTAB features use train-split standardization. Gaussian draws are shared across
targets; confidence statements are per target.

Outputs compare k=floor(M/2), M, M+1, 2M at epsilon/sigma=0.5,1,2.5.
M is each target's native AGNNCert vote budget. Its count guarantee excludes
the target feature row. The global Gaussian certificate bounds one total
Frobenius norm, including the target. This suite computes certificate coverage;
it does not run the paper's separate PGD diagnostics.
