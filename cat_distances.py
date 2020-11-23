import numpy as np


def overlap(x, z):
    return np.apply_along_axis(lambda y: np.sum(y != z, axis=1), axis=1, arr=x)


def flattened_overlap(x, z):

    def vec_fun(y):
        quantities = np.sum(y == x, axis=0)
        res_quantities = np.array([])
        for num, col in enumerate(x.T):
            counts = np.unique(col, return_counts=True)[1]
            res_quantities = np.append(res_quantities, np.sum((counts[counts <= quantities[num]] * (counts[counts <= quantities[num]] - 1)) / (x.shape[0] * (x.shape[0] - 1))))

        return np.sum((y != z) * 1 + (y == z) * res_quantities, axis=1)

    return np.apply_along_axis(vec_fun, axis=1, arr=x)


def log_overlap(x, z):

    def vec_fun(y):
        log_quantities = np.log(np.sum(x == y, axis=0) + 1)
        return np.sum((y != z) * log_quantities * z_counts, axis=1)

    def vec_fun2(y):
        return np.log(np.sum(x == y, axis=0) + 1)

    z_counts = np.apply_along_axis(vec_fun2, axis=1, arr=z)

    return np.apply_along_axis(vec_fun, axis=1, arr=x)
