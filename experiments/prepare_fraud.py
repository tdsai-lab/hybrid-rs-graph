#!/usr/bin/env python3
"""Download and preprocess CARE-GNN fraud datasets for static node classification."""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_utils.sparsegraph import SparseGraph


DATASET_SPECS = {
    'amazon_fraud': {
        'archive_name': 'Amazon.zip',
        'archive_url': 'https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip',
        'mat_name': 'Amazon.mat',
        'output_name': 'amazon_fraud_homo.npz',
        'summary_name': 'amazon_fraud_summary.json',
        'label_semantics': {
            '0': 'benign / non-fraudulent user',
            '1': 'fraudulent user',
        },
        'relation_names': ['net_upu', 'net_usu', 'net_uvu'],
        'relation_description': {
            'net_upu': 'user-product-user',
            'net_usu': 'user-star-user',
            'net_uvu': 'user-review-vector-user',
        },
    },
    'yelpchi': {
        'archive_name': 'YelpChi.zip',
        'archive_url': 'https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip',
        'mat_name': 'YelpChi.mat',
        'output_name': 'yelpchi_homo.npz',
        'summary_name': 'yelpchi_summary.json',
        'label_semantics': {
            '0': 'benign / non-fraudulent review node',
            '1': 'fraudulent / spam review node',
        },
        'relation_names': ['net_rur', 'net_rtr', 'net_rsr'],
        'relation_description': {
            'net_rur': 'review-user-review',
            'net_rtr': 'review-time-review',
            'net_rsr': 'review-star-review',
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and preprocess CARE-GNN fraud datasets under ./data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--root',
        type=str,
        default='data/CARE-GNN',
        help='Root directory for raw and processed CARE-GNN fraud data.',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['amazon_fraud', 'yelpchi'],
        choices=sorted(DATASET_SPECS),
        help='Datasets to prepare.',
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip archive downloads and only preprocess existing local files.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Redownload and regenerate processed outputs even if they already exist.',
    )
    parser.add_argument(
        '--care_static_topk',
        type=str,
        default='10,25,50,100',
        help='Comma-separated top-k values for CARE-inspired static graph variants.',
    )
    return parser.parse_args()


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(archive_path: Path, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as zf:
        for member in zf.namelist():
            target = output_dir / member
            if target.exists() and not force:
                continue
            zf.extract(member, path=output_dir)


def _sparse_float_matrix(obj) -> sp.csr_matrix:
    if sp.isspmatrix(obj):
        return obj.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(obj, dtype=np.float32))


def _binary_undirected_adjacency(obj) -> sp.csr_matrix:
    adj = _sparse_float_matrix(obj)
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    if adj.nnz:
        adj.data[:] = 1.0
    return adj


def _feature_stats(features: sp.csr_matrix) -> dict:
    if features.nnz == 0:
        return {
            'is_sparse': True,
            'is_continuous': False,
            'is_binary': True,
            'min': 0.0,
            'max': 0.0,
            'nnz': 0,
        }
    data = features.data
    is_binary = bool(np.all((data == 0.0) | (data == 1.0)))
    return {
        'is_sparse': True,
        'is_continuous': not is_binary,
        'is_binary': is_binary,
        'min': float(data.min()),
        'max': float(data.max()),
        'nnz': int(features.nnz),
    }


def _relation_summary(mat_data: dict, relation_names) -> list[dict]:
    summaries = []
    union = None
    for rel_name in relation_names:
        rel = _binary_undirected_adjacency(mat_data[rel_name])
        union = rel if union is None else union.maximum(rel)
        summaries.append({
            'name': rel_name,
            'nnz': int(rel.nnz),
            'undirected_edges': int(rel.nnz // 2),
            'symmetric': bool((rel != rel.T).nnz == 0),
        })
    return summaries, union.tocsr()


def _parse_topk_values(text: str) -> list[int]:
    values = []
    for token in str(text).split(','):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("--care_static_topk must contain positive integers only.")
        values.append(value)
    if not values:
        raise ValueError("--care_static_topk must contain at least one value.")
    return sorted(set(values))


def _row_normalize_csr(features: sp.csr_matrix) -> sp.csr_matrix:
    x_unit = features.tocsr(copy=True).astype(np.float32)
    row_norms = np.sqrt(x_unit.multiply(x_unit).sum(axis=1)).A1
    row_norms[row_norms == 0.0] = 1.0
    inv_row_norms = 1.0 / row_norms
    row_nnz = np.diff(x_unit.indptr)
    x_unit.data *= np.repeat(inv_row_norms, row_nnz)
    return x_unit


def _build_care_static_adj(
    relation_adjs: list[sp.csr_matrix],
    raw_features: sp.csr_matrix,
    *,
    topk: int,
) -> tuple[sp.csr_matrix, dict]:
    if topk <= 0:
        raise ValueError("topk must be > 0 for care_static graph construction.")
    if not relation_adjs:
        raise ValueError("Need at least one relation adjacency to build care_static.")

    n_rel = len(relation_adjs)
    rel_count = relation_adjs[0].copy().astype(np.float32)
    for rel in relation_adjs[1:]:
        rel_count = rel_count + rel.astype(np.float32)
    rel_count = rel_count.tocsr()
    rel_count.setdiag(0)
    rel_count.eliminate_zeros()

    x_unit = _row_normalize_csr(raw_features)
    indptr = rel_count.indptr
    indices = rel_count.indices
    rel_data = rel_count.data.astype(np.float32, copy=False)

    rows = []
    cols = []
    scores_out = []
    kept_degrees = np.zeros(rel_count.shape[0], dtype=np.int32)

    for u in range(rel_count.shape[0]):
        start = indptr[u]
        end = indptr[u + 1]
        neigh = indices[start:end]
        if neigh.size == 0:
            continue

        rel_score = rel_data[start:end] / float(n_rel)
        sims = np.asarray(x_unit[neigh].multiply(x_unit.getrow(u)).sum(axis=1)).ravel()
        fused = rel_score + sims

        if neigh.size > topk:
            top_idx = np.argpartition(-fused, topk - 1)[:topk]
            top_idx = top_idx[np.argsort(-fused[top_idx], kind='mergesort')]
            keep = neigh[top_idx]
            keep_scores = fused[top_idx]
        else:
            order = np.argsort(-fused, kind='mergesort')
            keep = neigh[order]
            keep_scores = fused[order]

        kept_degrees[u] = keep.shape[0]
        rows.append(np.full(keep.shape[0], u, dtype=np.int32))
        cols.append(keep.astype(np.int32, copy=False))
        scores_out.append(keep_scores.astype(np.float32, copy=False))

    if not rows:
        return sp.csr_matrix(rel_count.shape, dtype=np.float32), {
            'topk': int(topk),
            'candidate_directed_nnz': int(rel_count.nnz),
            'candidate_undirected_edges': int(rel_count.nnz // 2),
            'output_directed_nnz': 0,
            'output_undirected_edges': 0,
            'avg_kept_degree_before_sym': 0.0,
            'max_kept_degree_before_sym': 0,
        }

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    score_vals = np.concatenate(scores_out)
    directed = sp.csr_matrix((score_vals, (row_idx, col_idx)), shape=rel_count.shape, dtype=np.float32)
    fused_adj = directed.maximum(directed.T).tocsr()
    fused_adj.setdiag(0)
    fused_adj.eliminate_zeros()
    if fused_adj.nnz:
        fused_adj.data[:] = 1.0

    stats = {
        'topk': int(topk),
        'candidate_directed_nnz': int(rel_count.nnz),
        'candidate_undirected_edges': int(rel_count.nnz // 2),
        'output_directed_nnz': int(fused_adj.nnz),
        'output_undirected_edges': int(fused_adj.nnz // 2),
        'avg_kept_degree_before_sym': float(kept_degrees.mean()),
        'max_kept_degree_before_sym': int(kept_degrees.max()) if kept_degrees.size else 0,
    }
    return fused_adj, stats


def preprocess_dataset(
    dataset_key: str,
    root: Path,
    *,
    force: bool = False,
    care_static_topk_values: list[int] | None = None,
) -> None:
    spec = DATASET_SPECS[dataset_key]
    archive_path = root / spec['archive_name']
    raw_dir = root / f'{dataset_key}_raw'
    mat_path = raw_dir / spec['mat_name']
    processed_dir = root / 'processed'
    processed_path = processed_dir / spec['output_name']
    summary_path = processed_dir / spec['summary_name']

    if care_static_topk_values is None:
        care_static_topk_values = [10, 25, 50, 100]

    care_static_paths = [
        processed_dir / f"{dataset_key}_care_static_topk{k}.npz"
        for k in care_static_topk_values
    ]
    required_outputs_exist = (
        processed_path.exists()
        and summary_path.exists()
        and all(path.exists() for path in care_static_paths)
    )
    if required_outputs_exist and not force:
        return

    extract_zip(archive_path, raw_dir, force=force)
    mat_data = loadmat(mat_path)

    adj = _binary_undirected_adjacency(mat_data['homo'])
    features = _sparse_float_matrix(mat_data['features'])
    labels = np.asarray(mat_data['label']).reshape(-1).astype(np.int64)

    graph = SparseGraph(
        adj_matrix=adj,
        attr_matrix=features,
        labels=labels,
        class_names=np.array(['benign', 'fraud'], dtype=object),
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_path, **graph.to_flat_dict())

    relation_summaries, relation_union = _relation_summary(mat_data, spec['relation_names'])
    relation_adjs = [
        _binary_undirected_adjacency(mat_data[rel_name])
        for rel_name in spec['relation_names']
    ]
    care_static_outputs = {}
    for topk, care_static_path in zip(care_static_topk_values, care_static_paths):
        care_static_adj, care_static_stats = _build_care_static_adj(
            relation_adjs,
            features,
            topk=topk,
        )
        care_static_graph = SparseGraph(
            adj_matrix=care_static_adj,
            attr_matrix=features,
            labels=labels,
            class_names=np.array(['benign', 'fraud'], dtype=object),
        )
        np.savez_compressed(care_static_path, **care_static_graph.to_flat_dict())
        care_static_outputs[str(topk)] = {
            'processed_graph': str(care_static_path),
            **care_static_stats,
        }
    feature_stats = _feature_stats(features)
    label_values, label_counts = np.unique(labels, return_counts=True)

    summary = {
        'dataset': dataset_key,
        'source_repo': 'https://github.com/YingtongDou/CARE-GNN',
        'archive_url': spec['archive_url'],
        'processed_graph': str(processed_path),
        'adjacency_source': 'official homo matrix provided by CARE-GNN',
        'raw_graph_type': 'multi-relational',
        'integrated_graph_type': 'homogeneous',
        'num_nodes': int(adj.shape[0]),
        'num_edges_directed_nnz': int(adj.nnz),
        'num_edges_undirected': int(adj.nnz // 2),
        'feature_dim': int(features.shape[1]),
        'label_semantics': spec['label_semantics'],
        'label_counts': {str(int(v)): int(c) for v, c in zip(label_values, label_counts)},
        'features_continuous': bool(feature_stats['is_continuous']),
        'features_binary': bool(feature_stats['is_binary']),
        'features_sparse': bool(feature_stats['is_sparse']),
        'feature_value_min': feature_stats['min'],
        'feature_value_max': feature_stats['max'],
        'feature_nnz': feature_stats['nnz'],
        'raw_relations': relation_summaries,
        'relation_descriptions': spec['relation_description'],
        'homo_equals_union_of_relations': bool((adj != relation_union).nnz == 0),
        'care_static_variants': care_static_outputs,
    }
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"Prepared {dataset_key}: nodes={summary['num_nodes']} "
        f"edges={summary['num_edges_undirected']} feat_dim={summary['feature_dim']} "
        f"continuous_features={summary['features_continuous']} -> {processed_path}"
    )


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        archive_path = root / spec['archive_name']
        if not args.skip_download:
            download_file(spec['archive_url'], archive_path, force=args.force)
        elif not archive_path.exists():
            raise FileNotFoundError(
                f"Missing archive {archive_path}. Remove --skip_download or place the file there first."
            )
        preprocess_dataset(
            dataset_key,
            root=root,
            force=args.force,
            care_static_topk_values=_parse_topk_values(args.care_static_topk),
        )


if __name__ == '__main__':
    main()
