import numpy as np
import scipy.sparse as sp
import torch
import sys
import os

import pickle as pkl
import networkx as nx


CARE_GNN_FRAUD_DATASETS = (
    'amazon_fraud',
    'yelpchi',
)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _default_caregnn_root():
    return os.environ.get('CAREGNN_ROOT', os.path.join(_project_root(), 'data', 'CARE-GNN', 'processed'))


def _default_split_seed():
    return int(os.environ.get('CERTIFYGNN_SPLIT_SEED', '0'))


def _default_n_per_class():
    return int(os.environ.get('CERTIFYGNN_N_PER_CLASS', '20'))


def _caregnn_graph_filename(dataset_key: str, fraud_graph_source: str = 'homo', care_static_topk: int = 50) -> str:
    dataset_key = str(dataset_key).lower()
    if dataset_key not in CARE_GNN_FRAUD_DATASETS:
        raise ValueError(
            f"Unsupported CARE-GNN fraud dataset '{dataset_key}'. "
            f"Expected one of {sorted(CARE_GNN_FRAUD_DATASETS)}."
        )

    graph_source = str(fraud_graph_source).lower()
    if graph_source == 'homo':
        return f'{dataset_key}_homo.npz'
    if graph_source == 'care_static':
        topk = int(care_static_topk)
        if topk <= 0:
            raise ValueError("--care_static_topk must be > 0 when fraud_graph_source=care_static.")
        return f'{dataset_key}_care_static_topk{topk}.npz'

    raise ValueError(
        f"Unsupported fraud_graph_source '{fraud_graph_source}'. "
        "Expected one of ['homo', 'care_static']."
    )


def _load_csr_from_flat_dict(loader, prefix):
    return sp.csr_matrix(
        (
            loader[f'{prefix}.data'],
            loader[f'{prefix}.indices'],
            loader[f'{prefix}.indptr'],
        ),
        shape=tuple(loader[f'{prefix}.shape']),
    )




def _stratified_split(labels, n_per_class=20, seed=0):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)

    idx_train, idx_val = [], []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.shape[0] < 2 * n_per_class:
            raise ValueError(
                f"Class {cls} only has {cls_idx.shape[0]} nodes, "
                f"but need at least {2 * n_per_class} for train/val splits."
            )
        perm = rng.permutation(cls_idx)
        idx_train.append(perm[:n_per_class])
        idx_val.append(perm[n_per_class:2 * n_per_class])

    idx_train = rng.permutation(np.concatenate(idx_train))
    idx_val = rng.permutation(np.concatenate(idx_val))
    idx_test = np.setdiff1d(np.arange(labels.shape[0]), np.concatenate((idx_train, idx_val)))
    return idx_train, idx_val, idx_test


def _load_caregnn_processed(
    dataset,
    caregnn_root=None,
    n_per_class=None,
    split_seed=None,
    fraud_graph_source='homo',
    care_static_topk=50,
):
    dataset_key = str(dataset).lower()
    if dataset_key not in CARE_GNN_FRAUD_DATASETS:
        raise ValueError(
            f"Unsupported CARE-GNN fraud dataset '{dataset}'. "
            f"Expected one of {sorted(CARE_GNN_FRAUD_DATASETS)}."
        )

    if caregnn_root is None:
        caregnn_root = _default_caregnn_root()
    if n_per_class is None:
        n_per_class = _default_n_per_class()
    if split_seed is None:
        split_seed = _default_split_seed()

    graph_filename = _caregnn_graph_filename(
        dataset_key,
        fraud_graph_source=fraud_graph_source,
        care_static_topk=care_static_topk,
    )
    file_path = os.path.join(os.path.abspath(os.path.expanduser(caregnn_root)), graph_filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Prepared CARE-GNN dataset not found: {file_path}. "
            "Run `python experiments/prepare_fraud.py` first."
        )

    with np.load(file_path, allow_pickle=True) as loader_npz:
        loader = dict(loader_npz)

    adj = _load_csr_from_flat_dict(loader, 'adj_matrix')
    features = _load_csr_from_flat_dict(loader, 'attr_matrix')
    labels = np.asarray(loader['labels']).reshape(-1).astype(np.int64)
    idx_train, idx_val, idx_test = _stratified_split(labels, n_per_class=n_per_class, seed=split_seed)
    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_new(
    dataset,
    n_per_class=None,
    split_seed=None,
    caregnn_root=None,
    fraud_graph_source='homo',
    care_static_topk=50,
): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    """
    Loads input data from gcn/data directory

    ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset.ally => the labels for instances in ind.dataset.allx as numpy.ndarray object;
    ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    dataset_key = str(dataset).lower()
    # Accept the standard PyG raw cache as well as legacy source-local files.
    configured_root = os.environ.get('PLANETOID_ROOT')
    if configured_root:
        raw_root = os.path.join(configured_root, dataset_key, 'raw')
        if os.path.isfile(os.path.join(raw_root, f'ind.{dataset_key}.test.index')):
            data_root = raw_root
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    required = [os.path.join(data_root, f'ind.{dataset}.{name}') for name in names]
    test_index_path = os.path.join(data_root, f'ind.{dataset}.test.index')
    has_legacy_files = all(os.path.isfile(p) for p in required) and os.path.isfile(test_index_path)

    if has_legacy_files:
        objects = []
        for path in required:
            with open(path, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        print(allx.shape, tx.shape)
        test_idx_reorder = parse_index_file(test_index_path)
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)
    elif dataset_key in CARE_GNN_FRAUD_DATASETS:
        adj, features, labels, idx_train, idx_val, idx_test = _load_caregnn_processed(
            dataset=dataset_key,
            caregnn_root=caregnn_root,
            n_per_class=n_per_class,
            split_seed=split_seed,
            fraud_graph_source=fraud_graph_source,
            care_static_topk=care_static_topk,
        )
    else:
        from torch_geometric.datasets import Planetoid

        dataset_map = {
            'cora': 'Cora',
            'citeseer': 'CiteSeer',
            'pubmed': 'PubMed',
        }
        if dataset_key not in dataset_map:
            raise ValueError(
                f"Unsupported dataset '{dataset}' for load_data_new fallback. "
                f"Expected one of {sorted(dataset_map) + sorted(CARE_GNN_FRAUD_DATASETS)}."
            )

        pyg_root = os.environ.get('PLANETOID_ROOT', os.path.join(data_root, 'planetoid'))
        pyg_dataset = Planetoid(root=pyg_root, name=dataset_map[dataset_key], split='public')
        data = pyg_dataset[0]

        num_nodes = int(data.num_nodes)
        edge_index = data.edge_index.cpu().numpy()
        adj = sp.csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        )
        # build symmetric adjacency matrix
        adj = adj.maximum(adj.T).tocsr()

        x_t = data.x
        if x_t.is_sparse:
            x_t = x_t.to_dense()
        features = sp.csr_matrix(x_t.cpu().numpy())

        labels = data.y.cpu().numpy()

        def _mask_to_idx(mask):
            mask_np = mask.cpu().numpy()
            if mask_np.ndim == 2:
                mask_np = mask_np[:, 0]
            return np.where(mask_np)[0]

        idx_train = _mask_to_idx(data.train_mask)
        idx_val = _mask_to_idx(data.val_mask)
        idx_test = _mask_to_idx(data.test_mask)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    if labels.ndim > 1:
        labels = torch.LongTensor(np.argmax(labels, 1))
    else:
        labels = torch.LongTensor(labels.astype(np.int64, copy=False))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    #r_mat_inv = sp.diags(r_inv)
    r_mat_inv = sp.diags(r_inv,0)
    mx = r_mat_inv.dot(mx)
    return mx




def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
