import numpy as np


def kfold(n, n_folds):
    if not isinstance(n, int) or not isinstance(n_folds, int) or n < 1 or n_folds < 2:
        raise AttributeError('Incorrect parameters')

    index_list = []
    d = n // n_folds
    for i in range(n_folds - 1):
        arr_val = np.arange(i * d, (i + 1) * d)
        arr_train = np.hstack((np.arange(0, i * d), np.arange((i + 1) * d, n)))
        index_list.append((arr_train, arr_val))
    index_list.append((np.arange((n_folds - 1) * d), np.arange((n_folds - 1) * d, n)))
    return index_list


def counters(x_train, y_train, x_test, cat_features, num_folds=None):
    if num_folds is None:
        for feature in cat_features:
            x_train[feature], x_test[feature] = counters_maker(x_train[feature].to_numpy().flatten(),
                                                               x_test[feature].to_numpy().flatten(),
                                                               y_train.to_numpy().flatten())
    else:
        for feature in cat_features:
            x_train[feature], x_test[feature] = counters_maker_folding(x_train[feature].to_numpy().flatten(),
                                                                       x_test[feature].to_numpy().flatten(),
                                                                       y_train.to_numpy().flatten(), num_folds)


def counters_maker(x_train_feature, x_test_feature, target, a=1, b=2):
    categories, counts = np.unique(x_train_feature, return_counts=True)
    categories_cond, counts_cond = np.unique(x_train_feature[target == 1], return_counts=True)
    categories_test = np.unique(x_test_feature)
    x_train_res, x_test_res = np.empty(x_train_feature.shape[0], dtype=np.float64), \
                              np.full(x_test_feature.shape[0], -1, dtype=np.float64)

    for cat in categories:
        if np.any(categories_cond == cat):
            res = (counts_cond[categories_cond == cat][0] + a) / (counts[categories == cat][0] + b)
            x_train_res[x_train_feature == cat] = res
        else:
            res = a / (counts[categories == cat][0] + b)
            x_train_res[x_train_feature == cat] = res

        if np.any(categories_test == cat):
            x_test_res[x_test_feature == cat] = res

    x_test_res[x_test_res == -1.0] = a / b
    return x_train_res, x_test_res


def counters_maker_folding(x_train_feature, x_test_feature, target, num_folds, a=1, b=2):
    index_list = kfold(x_train_feature.shape[0], num_folds)
    x_test_res = np.full(x_test_feature.shape[0], -1, dtype=np.float64)
    x_train_res = np.array([], dtype=np.float64)
    for index in index_list:
        categories, counts = np.unique(x_train_feature[index[0]], return_counts=True)
        categories_cond, counts_cond = np.unique(x_train_feature[index[0]][target[index[0]] == 1], return_counts=True)
        categories_fold = np.unique(x_train_feature[index[1]])
        x_fold_tmp = np.full(index[1].shape[0], -1, dtype=np.float64)
        for cat in categories:
            if np.any(categories_cond == cat):
                res = (counts_cond[categories_cond == cat][0] + a) / (counts[categories == cat][0] + b)
            else:
                res = a / (counts[categories == cat][0] + b)

            if np.any(categories_fold == cat):
                x_fold_tmp[x_train_feature[index[1]] == cat] = res

        x_fold_tmp[x_fold_tmp == -1.0] = a / b
        x_train_res = np.append(x_train_res, x_fold_tmp)

    categories, counts = np.unique(x_train_feature, return_counts=True)
    categories_cond, counts_cond = np.unique(x_train_feature[target == 1], return_counts=True)
    categories_test = np.unique(x_test_feature)
    for cat in categories:
        if np.any(categories_cond == cat):
            res = (counts_cond[categories_cond == cat][0] + a) / (counts[categories == cat][0] + b)
        else:
            res = a / (counts[categories == cat][0] + b)

        if np.any(categories_test == cat):
            x_test_res[x_test_feature == cat] = res

    x_test_res[x_test_res == -1.0] = a / b
    return x_train_res, x_test_res
