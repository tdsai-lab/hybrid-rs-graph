from typing import Dict, Union, Any
import numpy as np
import scipy.sparse as sp


__all__ = ['SparseGraph']

sparse_graph_properties = [
        'adj_matrix', 'attr_matrix', 'edge_attr_matrix', 'labels',
        'node_names', 'attr_names', 'edge_attr_names', 'class_names',
        'metadata']


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions (e.g. of edge_attr_matrix).
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    edge_attr_matrix
        Edge attribute matrix in CSR or numpy format. Shape [num_edges, num_edge_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    edge_attr_names
        Names of the edge attributes (as strings). Shape [num_edge_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """
    def __init__(
            self, adj_matrix: sp.spmatrix,
            attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            edge_attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            labels: Union[np.ndarray, sp.spmatrix] = None,
            node_names: np.ndarray = None,
            attr_names: np.ndarray = None,
            edge_attr_names: np.ndarray = None,
            class_names: np.ndarray = None,
            metadata: Any = None):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)."
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree.")

        if edge_attr_matrix is not None:
            if sp.isspmatrix(edge_attr_matrix):
                edge_attr_matrix = edge_attr_matrix.tocsr().astype(np.float32)
            elif isinstance(edge_attr_matrix, np.ndarray):
                edge_attr_matrix = edge_attr_matrix.astype(np.float32)
            else:
                raise ValueError("Edge attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(edge_attr_matrix)))

            if edge_attr_matrix.shape[0] != adj_matrix.count_nonzero():
                raise ValueError("Number of edges and dimension of the edge attribute matrix don't agree.")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree.")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree.")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree.")

        if edge_attr_names is not None:
            if len(edge_attr_names) != edge_attr_matrix.shape[1]:
                raise ValueError("Dimensions of the edge attribute matrix and the edge attribute names don't agree.")

        # TODO: check that class names matches the labels vector (including the multi-label case)

        self._adj_matrix = adj_matrix
        self._attr_matrix = attr_matrix
        self._edge_attr_matrix = edge_attr_matrix
        self._labels = labels
        self._node_names = node_names
        self._attr_names = attr_names
        self._edge_attr_names = edge_attr_names
        self._class_names = class_names
        self._metadata = metadata

        self._flag_writeable(False)


    # Quality of life (shortcuts)


    def _flag_writeable(self, writeable: bool):
        """Flag all Numpy arrays and sparse matrices as non-writeable.
        """
        flag_writeable(self._adj_matrix, writeable)
        flag_writeable(self._attr_matrix, writeable)
        flag_writeable(self._edge_attr_matrix, writeable)
        flag_writeable(self._labels, writeable)
        flag_writeable(self._node_names, writeable)
        flag_writeable(self._attr_names, writeable)
        flag_writeable(self._edge_attr_names, writeable)
        flag_writeable(self._class_names, writeable)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return flat dictionary containing all SparseGraph properties.
        """
        data_dict = {}
        for key in sparse_graph_properties:
            val = getattr(self, key)
            if sp.isspmatrix(val):
                data_dict['{}.data'.format(key)] = val.data
                data_dict['{}.indices'.format(key)] = val.indices
                data_dict['{}.indptr'.format(key)] = val.indptr
                data_dict['{}.shape'.format(key)] = val.shape
            else:
                data_dict[key] = val
        return data_dict


    @property
    def adj_matrix(self) -> sp.csr_matrix:
        return self._adj_matrix

    @property
    def attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._attr_matrix

    @property
    def edge_attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._edge_attr_matrix

    @property
    def labels(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._labels

    @property
    def node_names(self) -> np.ndarray:
        return self._node_names

    @property
    def attr_names(self) -> np.ndarray:
        return self._attr_names

    @property
    def edge_attr_names(self) -> np.ndarray:
        return self._edge_attr_names

    @property
    def class_names(self) -> np.ndarray:
        return self._class_names

    @property
    def metadata(self) -> Any:
        return self._metadata


def flag_writeable(matrix: Union[np.ndarray, sp.csr_matrix], writeable: bool):
    if matrix is not None:
        if sp.isspmatrix(matrix):
            matrix.data.flags.writeable = writeable
            matrix.indices.flags.writeable = writeable
            matrix.indptr.flags.writeable = writeable
        elif isinstance(matrix, np.ndarray):
            matrix.flags.writeable = writeable
        else:
            raise ValueError(
                    "Matrix must be an sp.spmatrix or an np.ndarray"
                    " (got {0} instead).".format(type(matrix)))
