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


def counters(x_train, y_train, x_test, cat_features, counter_type='mean', num_folds=None):
    if num_folds is None:
        for feature in cat_features:
            x_train[feature], x_test[feature] = counters_maker(x_train[feature].to_numpy().flatten(),
                                                               x_test[feature].to_numpy().flatten(),
                                                               y_train.to_numpy().flatten(), counter_type)
    else:
        for feature in cat_features:
            x_train[feature], x_test[feature] = counters_maker_folding(x_train[feature].to_numpy().flatten(),
                                                                       x_test[feature].to_numpy().flatten(),
                                                                       y_train.to_numpy().flatten(), num_folds,
                                                                       counter_type)


def counters_maker(x_train_feature, x_test_feature, target, counter_type):
    categories = np.unique(x_train_feature)
    categories_test = np.unique(x_test_feature)
    x_train_res, x_test_res = np.empty(x_train_feature.shape[0], dtype=np.float64), \
                              np.full(x_test_feature.shape[0], np.nan, dtype=np.float64)

    for cat in categories:
        if counter_type == 'mean':
            res = np.mean(target[x_train_feature == cat])
        else:
            res = np.std(target[x_train_feature == cat])
        x_train_res[x_train_feature == cat] = res
        if np.any(categories_test == cat):
            x_test_res[x_test_feature == cat] = res

    if counter_type == 'mean':
        x_test_res[np.isnan(x_test_res)] = np.mean(target)
    else:
        x_test_res[np.isnan(x_test_res)] = np.std(target)
    return x_train_res, x_test_res


def counters_maker_folding(x_train_feature, x_test_feature, target, num_folds, counter_type):
    index_list = kfold(x_train_feature.shape[0], num_folds)
    x_test_res = np.full(x_test_feature.shape[0], np.nan, dtype=np.float64)
    x_train_res = np.array([], dtype=np.float64)
    for index in index_list:
        categories = np.unique(x_train_feature[index[0]])
        categories_fold = np.unique(x_train_feature[index[1]])
        x_fold_tmp = np.full(index[1].shape[0], np.nan, dtype=np.float64)
        for cat in categories:
            if counter_type == 'mean':
                res = np.mean(target[index[0]][x_train_feature[index[0]] == cat])
            else:
                res = np.std(target[index[0]][x_train_feature[index[0]] == cat])

            if np.any(categories_fold == cat):
                x_fold_tmp[x_train_feature[index[1]] == cat] = res

        if counter_type == 'mean':
            x_fold_tmp[np.isnan(x_fold_tmp)] = np.mean(target[index[0]])
        else:
            x_fold_tmp[np.isnan(x_fold_tmp)] = np.std(target[index[0]])
        x_train_res = np.append(x_train_res, x_fold_tmp)

    categories = np.unique(x_train_feature)
    categories_test = np.unique(x_test_feature)
    for cat in categories:
        if counter_type == 'mean':
            res = np.mean(target[x_train_feature == cat])
        else:
            res = np.std(target[x_train_feature == cat])

        if np.any(categories_test == cat):
            x_test_res[x_test_feature == cat] = res

    if counter_type == 'mean':
        x_test_res[np.isnan(x_test_res)] = np.mean(target)
    else:
        x_test_res[np.isnan(x_test_res)] = np.std(target)
    return x_train_res, x_test_res
