import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import distances


class KNNRegressor:

    def __init__(self, k, strategy='my_own', metric='euclidean', mode='uniform'):
        if not isinstance(k, int) or k < 1:
            raise AttributeError('Incorrect "k" parameter')
        if not isinstance(mode, str) or mode != 'uniform' and mode != 'distance':
            raise AttributeError('Mode parameter can be uniform or distance only')

        self.mode = mode
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.training_labels = None
        if strategy == 'my_own':
            self.training_data = None
        else:
            self.nn = NearestNeighbors(n_neighbors=k, algorithm=strategy, leaf_size=30, metric=metric)

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise AttributeError('Mismatch between training set and its labels')

        self.training_labels = y
        if self.strategy == 'my_own':
            self.training_data = x
        else:
            self.nn.fit(x)

    def find_kneighbors(self, x, return_distance=True):
        if not isinstance(return_distance, bool):
            raise AttributeError('Incorrect "return_distance" parameter')

        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                dist_matrix = distances.euclidean_distance(x, self.training_data)
            elif self.metric == 'cosine':
                dist_matrix = distances.cosine_distance(x, self.training_data)
            else:
                dist_matrix = self.metric(self.training_data, x).astype(np.float64).T
            if not return_distance:
                res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                np.argmin(dist_matrix, axis=1, out=res_index)
                dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                res_index = res_index.reshape((-1, 1))
                for i in range(self.k - 1):
                    np.argmin(dist_matrix, axis=1, out=tmp_index)
                    dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                    res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                return res_index
            else:
                res_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                tmp_index = np.empty(dist_matrix.shape[0], dtype=np.int64)
                np.argmin(dist_matrix, axis=1, out=res_index)
                res_dist = dist_matrix[np.arange(dist_matrix.shape[0]), res_index]
                dist_matrix[np.arange(dist_matrix.shape[0]), res_index] = np.inf
                res_index = res_index.reshape((-1, 1))
                res_dist = res_dist.reshape((-1, 1))
                for i in range(self.k - 1):
                    np.argmin(dist_matrix, axis=1, out=tmp_index)
                    res_dist = np.hstack((res_dist,
                                          dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index][:, np.newaxis]))
                    dist_matrix[np.arange(dist_matrix.shape[0]), tmp_index] = np.inf
                    res_index = np.hstack((res_index, tmp_index[:, np.newaxis]))
                return res_dist, res_index
        else:
            return self.nn.kneighbors(x, return_distance=return_distance)

    def predict(self, x, k=None):
        if k is not None:
            if not isinstance(k, int) or k < 1:
                raise AttributeError('Incorrect "k" parameter')
            else:
                if self.strategy == 'my_own':
                    self.k = k
                else:
                    params = self.nn.get_params()
                    params['n_neighbors'] = k
                    self.nn = self.nn.set_params(**params)

        if self.mode == 'uniform':
            nn_index = self.training_labels[self.find_kneighbors(x, return_distance=False)]
            return np.mean(nn_index, axis=1)
        else:
            vec_weight = np.vectorize(lambda z: 1 / (z + 0.00001))
            nn_dist, nn_index = self.find_kneighbors(x)
            nn_index = self.training_labels[nn_index]
            nn_dist = vec_weight(nn_dist)
            return np.sum(nn_index * nn_dist, axis=1) / np.sum(nn_dist, axis=1)
