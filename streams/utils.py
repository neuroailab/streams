from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats
import pandas
import seaborn as sns


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


def _timeplot_bootstrap(x, estimator=np.mean, ci=95, n_boot=100):
    ci = bootstrap_resample(x, func=estimator, ci=ci, niter=n_boot)
    return pandas.Series({'emin': ci[0], 'emax': ci[1]})


def timeplot(data=None, x=None, y=None, hue=None,
             estimator=np.mean, ci=95, n_boot=100, **kwargs):
    if hue is None:
        hues = ['']
    else:
        hues = data[hue].unique()

    for h, color in zip(hues, sns.color_palette()):
        if hue is None:
            d = data
        else:
            d = data[data[hue] == h]

        mn = d.groupby(x)[y].apply(estimator)
        ebars = d.groupby(x)[y].apply(lambda x: _timeplot_bootstrap(x, estimator, ci, n_boot)).unstack()

        sns.plt.fill_between(mn.index, ebars.emin, ebars.emax, alpha=.5, color=color)
        sns.plt.plot(mn.index, mn, linewidth=2, color=color, label=h, **kwargs)
        sns.plt.xlabel(x)
        sns.plt.ylabel(y)

    if hue is not None:
        sns.plt.legend()