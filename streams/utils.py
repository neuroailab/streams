from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats


def splithalf(data, rng=None):
    data = np.array(data)
    if rng is None:
        rng = np.random.RandomState(None)
    inds = range(data.shape[0])
    rng.shuffle(inds)
    half = len(inds) // 2
    split1 = data[inds[:half]].mean(axis=0)
    split2 = data[inds[half:2*half]].mean(axis=0)
    return split1, split2


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)


def spearman_brown_correct(pearsonr, n=2):
    pearsonr = np.array(pearsonr)
    return n * pearsonr / (1 + (n-1) * pearsonr)


def resample(data, rng=None):
    data = np.array(data)
    if rng is None:
        rng = np.random.RandomState(None)
    inds = rng.choice(range(data.shape[0]), size=data.shape[0], replace=True)
    return data[inds]


def bootstrap_resample(data, func=np.mean, niter=100, ci=95, rng=None):
    df = [func(resample(data, rng=rng)) for i in range(niter)]
    if ci is not None:
        return np.percentile(df, 50-ci/2.), np.percentile(df, 50+ci/2.)
    else:
        return df