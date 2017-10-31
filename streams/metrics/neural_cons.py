from __future__ import division, print_function, absolute_import

import numpy as np
import pandas
import scipy.stats
import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from streams.envs import hvm
from streams import utils
from streams.parallel import Parallel


def nfit(model_feats, neural, labels, n_splits=10, n_components=200, test_size=.25):
    if model_feats.shape[1] > n_components:
        model_feats = PCA(n_components=n_components).fit_transform(model_feats)
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    df = []
    for it, (train_idx, test_idx) in enumerate(skf.split(model_feats, labels)):
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(model_feats[train_idx], neural[train_idx])
        pred = reg.predict(model_feats[test_idx])
        rs = utils.pearsonr_matrix(neural[test_idx], pred)
        df.extend([(it, site, r) for site, r in enumerate(rs)])
    df = pandas.DataFrame(df, columns=['split', 'site', 'fit_r'])
    return df


def internal_cons(neural, labels, n_splits=10, test_size=.25):
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    dfs = []
    for it, (train_idx, test_idx) in enumerate(skf.split(neural.mean(0), labels)):
        df = splithalf_corr(neural[:, test_idx])
        df['split'] = it
        dfs.append(df)
    df = pandas.concat(dfs, ignore_index=True)
    return df


def splithalf_corr(data, niter=10, seed=None):
    rng = np.random.RandomState(seed)
    df = []
    for i in range(niter):
        split1, split2 = utils.splithalf(data, rng=rng)
        r = utils.pearsonr_matrix(split1, split2)
        rc = utils.spearman_brown_correct(r, n=2)
        df.extend([(i, site, rci) for site, rci in enumerate(rc)])
    df = pandas.DataFrame(df, columns=['splithalf', 'site', 'internal_cons'])
    return df


class NeuralFit(object):

    def __init__(self, model_feats, neural_feats_reps, labels,
                 regression=OrthogonalMatchingPursuit(),
                 n_splits=10, test_size=1/4., n_splithalves=10,
                 pca=False, n_components=None,
                 **parallel_kwargs):
        """
        Regression of model features to neurons.

        :Args:
            - model_feats
                Model responses corresponding to stimuli in dataset
            - neural_feats_reps
                Neural responses in the format of (reps, images, sites)

        :Kwargs:
            - n_splits
                Number of splits into (train, test)
            - n_splithalves
                Number of splits over reps

        :Returns:
            The fit outcome (pandas.DataFrame)
        """
        self.model_feats = model_feats
        self._neural_feats_reps = neural_feats_reps
        self.labels = labels
        self.reg = regression
        self.n_splits = n_splits
        self.n_splithalves = n_splithalves
        self.do_pca = bool(pca)
        self._pca = pca
        self.n_components = n_components
        self.pkwargs = parallel_kwargs

        self.rng = np.random.RandomState(0)
        sss = StratifiedShuffleSplit(n_splits=self.n_splits,
                                     test_size=test_size, random_state=self.rng)
        self.splits = [s for s in sss.split(self.model_feats, self.labels)]

    def pca(self, train_inds, test_inds):
        if isinstance(self._pca, PCA):
            pca = self._pca
        else:
            pca = PCA(n_components=self.n_components)
        neural_feats = np.nanmean(self._neural_feats_reps, axis=0)
        out = np.zeros_like(self._neural_feats_reps)
        pca.fit(neural_feats[train_inds])
        out[:, train_inds] = [pca.transform(r) for r in self._neural_feats_reps[:, train_inds]]
        out[:, test_inds] = [pca.transform(r) for r in self._neural_feats_reps[:, test_inds]]
        return out

    def fit(self):
        df = Parallel(self._fit, n_iter=self.n_splits, **self.pkwargs)()
        df = pandas.concat(df, axis=0, ignore_index=True)
        if self.do_pca:
            df.rename(columns={'site': 'pc'}, inplace=True)
        return df

    def _fit(self, splitno):
        train_inds, test_inds = self.splits[splitno]
        if self.do_pca:
            self.neural_feats_reps = self.pca(train_inds, test_inds)
        else:
            self.neural_feats_reps = self._neural_feats_reps
        r = self.raw_fit(train_inds, test_inds)
        res = Parallel(self._cons, n_iter=self.n_splithalves, **self.pkwargs)(r, train_inds, test_inds)
        res = pandas.concat(res, axis=0, ignore_index=True)
        res['split'] = splitno
        res['explained_var'] = res.fit_r ** 2 / (res.internal_cons * res.mapping_cons)
        return res

    def raw_fit(self, train_inds, test_inds):
        neural_feats = np.nanmean(self.neural_feats_reps, axis=0)
        rs = []
        for site in range(self.neural_feats_reps.shape[-1]):
            self.reg.fit(self.model_feats[train_inds], np.squeeze(neural_feats[train_inds, site]))
            pred = np.squeeze(self.reg.predict(self.model_feats[test_inds]))
            actual = neural_feats[test_inds, site]
            r = scipy.stats.pearsonr(actual, pred)[0]
            rs.append(r)
        return np.squeeze(rs)

    def _cons(self, splithalfno, r, train_inds, test_inds):
        mapping_cons = self.mapping_cons(splithalfno, train_inds, test_inds)
        internal_cons = self.internal_cons(splithalfno, test_inds)
        res = pandas.DataFrame({'mapping_cons': mapping_cons, 'internal_cons': internal_cons})
        res['site'] = range(len(res))
        res['splithalf'] = splithalfno
        res['fit_r'] = r
        return res

    def mapping_cons(self, splithalfno, train_inds, test_inds):
        """
        Split data in half over reps, run PLS on each half on the train set,
        get predictions for the test set, correlate the two, Spearman-Brown
        """
        rs = []
        rng = np.random.RandomState(splithalfno)
        for site in range(self.neural_feats_reps.shape[-1]):
            split1, split2 = utils.splithalf(self.neural_feats_reps[:, train_inds, site], rng=rng)
            self.reg.fit(self.model_feats[train_inds], split1)
            pred1 = self.reg.predict(self.model_feats[test_inds])
            self.reg.fit(self.model_feats[train_inds], split2)
            pred2 = self.reg.predict(self.model_feats[test_inds])
            r = scipy.stats.pearsonr(pred1, pred2)[0]
            rs.append(r)
        return np.squeeze(utils.spearman_brown_correct(rs, n=2))

    def internal_cons(self, splithalfno, test_inds):
        rng = np.random.RandomState(splithalfno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, test_inds], rng=rng)
        r = utils.pearsonr_matrix(split1, split2)
        rc = utils.spearman_brown_correct(r, n=2)
        return rc


class NeuralFitAllSites(NeuralFit):

    def __init__(self, model_feats, neural_feats_reps, labels,
                 regression=PLSRegression(n_components=25, scale=False),
                 **kwargs):
        super(NeuralFitAllSites, self).__init__(model_feats, neural_feats_reps,
                labels, regression=regression, **kwargs)

    def raw_fit(self, train_inds, test_inds):
        neural_feats = np.nanmean(self.neural_feats_reps, axis=0)
        self.reg.fit(self.model_feats[train_inds], np.squeeze(neural_feats[train_inds]))
        pred = np.squeeze(self.reg.predict(self.model_feats[test_inds]))
        actual = neural_feats[test_inds]
        rs = utils.pearsonr_matrix(actual, pred)
        return np.squeeze(rs)

    def mapping_cons(self, splithalfno, train_inds, test_inds):
        """
        Split data in half over reps, run PLS on each half on the train set,
        get predictions for the test set, correlate the two, Spearman-Brown
        """
        rng = np.random.RandomState(splithalfno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, train_inds], rng=rng)
        self.reg.fit(self.model_feats[train_inds], split1)
        pred1 = self.reg.predict(self.model_feats[test_inds])
        self.reg.fit(self.model_feats[train_inds], split2)
        pred2 = self.reg.predict(self.model_feats[test_inds])
        rs = utils.pearsonr_matrix(pred1, pred2)
        return np.squeeze(utils.spearman_brown_correct(rs, n=2))


class NeuralFitCV(object):

    REG_DICT = {
        'ElasticNetCV': {'l1_ratio': 'l1_ratio_', 'alpha': 'alpha_'},
        # 'LarsCV': {None: 'alpha_'},
        'LassoCV': {'alpha': 'alpha_'},
        'OrthogonalMatchingPursuitCV': {'n_nonzero_coefs': 'n_nonzero_coefs_'},
        'RidgeCV': {'alpha': 'alpha_'},
        }

    def __init__(self, model_feats, neural_feats_reps, labels,
                 regression=linear_model.OrthogonalMatchingPursuitCV,
                 n_splits=10, test_size=1/4., n_splithalves=10,
                 **parallel_kwargs):
        """
        Regression of model features to neurons.

        :Args:
            - model_feats
                Model responses corresponding to stimuli in dataset
            - neural_feats_reps
                Neural responses in the format of (reps, images, sites)

        :Kwargs:
            - n_splits
                Number of splits into (train, test)
            - n_splithalves
                Number of splits over reps

        :Returns:
            The fit outcome (pandas.DataFrame)
        """
        self.model_feats = model_feats
        self.neural_feats_reps = neural_feats_reps
        self.labels = labels
        self.regcv = regression
        name = self.regcv.__class__.__name__
        if name.endswith('CV'):
            self.reg = getattr(linear_model, name[:-2])()
            self.reg_dict = self.REG_DICT[name]
        else:
            self.reg = self.regcv
            self.reg_dict = None
        self.n_splits = n_splits
        self.n_splithalves = n_splithalves
        self.pkwargs = parallel_kwargs

        self.rng = np.random.RandomState(0)
        sss = StratifiedShuffleSplit(n_splits=self.n_splits,
                                     test_size=test_size, random_state=self.rng)
        self.splits = [s for s in sss.split(self.model_feats, self.labels)]

    def fit(self):
        df = Parallel(self._fit, n_iter=self.n_splits, **self.pkwargs)()
        df = pandas.concat(df, axis=0, ignore_index=True)
        return df

    def _fit(self, splitno):
        train_inds, test_inds = self.splits[splitno]
        r, reg_params = self.raw_fit(train_inds, test_inds)
        res = Parallel(self._cons, n_iter=self.n_splithalves, **self.pkwargs)(r, reg_params, train_inds, test_inds)
        res = pandas.concat(res, axis=0, ignore_index=True)
        res['split'] = splitno
        res['explained_var'] = res.fit_r ** 2 / (res.internal_cons * res.mapping_cons)
        return res

    def raw_fit(self, train_inds, test_inds):
        neural_feats = np.nanmean(self.neural_feats_reps, axis=0)
        rs = []
        reg_params = []
        for site in range(self.neural_feats_reps.shape[-1]):
            self.regcv.fit(self.model_feats[train_inds], np.squeeze(neural_feats[train_inds, site]))
            pred = np.squeeze(self.regcv.predict(self.model_feats[test_inds]))
            actual = neural_feats[test_inds, site]
            r = scipy.stats.pearsonr(actual, pred)[0]
            rs.append(r)
            if self.reg_dict is not None:
                reg_params.append({k:getattr(self.regcv, v) for k,v in self.reg_dict.items()})
        return np.squeeze(rs), reg_params

    def _cons(self, splithalfno, r, reg_params, train_inds, test_inds):
        mapping_cons = self.mapping_cons(splithalfno, train_inds, test_inds, reg_params)
        internal_cons = self.internal_cons(splithalfno, test_inds)
        res = pandas.DataFrame({'mapping_cons': mapping_cons, 'internal_cons': internal_cons})
        res['site'] = range(len(res))
        res['splithalf'] = splithalfno
        res['fit_r'] = r
        return res

    def mapping_cons(self, splithalfno, train_inds, test_inds, reg_params):
        """
        Split data in half over reps, run PLS on each half on the train set,
        get predictions for the test set, correlate the two, Spearman-Brown
        """
        rs = []
        rng = np.random.RandomState(splithalfno)
        for site in range(self.neural_feats_reps.shape[-1]):
            if self.reg_dict is not None:
                self.reg.set_params(**reg_params[site])
            split1, split2 = utils.splithalf(self.neural_feats_reps[:, train_inds, site], rng=rng)

            self.reg.fit(self.model_feats[train_inds], split1)
            pred1 = self.reg.predict(self.model_feats[test_inds])

            self.reg.fit(self.model_feats[train_inds], split2)
            pred2 = self.reg.predict(self.model_feats[test_inds])

            r = scipy.stats.pearsonr(pred1, pred2)[0]
            rs.append(r)
        return np.squeeze(utils.spearman_brown_correct(rs, n=2))

    def internal_cons(self, splithalfno, test_inds):
        rng = np.random.RandomState(splithalfno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, test_inds], rng=rng)
        r = utils.pearsonr_matrix(split1, split2)
        rc = utils.spearman_brown_correct(r, n=2)
        return rc


def _ttest_mean(data, alpha=.05):
    # fit = data.fit_r[data.fit_r.notnull()]# & (data.fit_r > 0)]
    # mc = data.mapping_cons[data.mapping_cons.notnull()]# & (data.mapping_cons > 0)]
    ic = data.internal_cons[data.internal_cons.notnull()]# & (data.internal_cons > 0)]
    # p_fit = scipy.stats.ttest_1samp(fit, 0)[1]
    # p_map = scipy.stats.ttest_1samp(mc, 0)[1]
    p_int = scipy.stats.ttest_1samp(ic, 0)[1]
    agg = data.mean()
    if p_int > alpha:
        agg[:] = np.nan
    return agg

    # cols = ['fit_r', 'mapping_cons', 'internal_cons']
    # agg = [data[cols].mean(), data[cols].std(ddof=1) / np.sqrt(data[cols].count())]
    # if p_int > alpha:
    #     agg[0][:] = np.nan
    #     agg[1][:] = np.nan
    # out = pandas.concat(agg, keys=['mean', 'stderr'])
    # return out


def stderr(data, niter=100, ci=95, rng=None):
    res = []
    if rng is None:
        rng = np.random.RandomState(None)

    mn = data.mean()
    mn['explained_var'] = mn.fit_r ** 2 / (mn.internal_cons * mn.mapping_cons)

    for i in range(niter):
        inds = rng.choice(range(data.shape[0]), size=data.shape[0], replace=True)
        mnr = data.iloc[inds].mean()
        mnr['explained_var'] = mnr.fit_r ** 2 / (mnr.internal_cons * mnr.mapping_cons)
        res.append(mnr)
    res = pandas.concat(res, axis=1).T
    ci_low = res.apply(lambda x: np.percentile(x[x.notnull()], 50-ci/2.))
    ci_high = res.apply(lambda x: np.percentile(x[x.notnull()], 50+ci/2.))
    return pandas.concat([mn, ci_low, ci_high], axis=0, keys=['mean', 'ci_low', 'ci_high'])


def analysis(df, alpha=.05):
    # ttest_mean = lambda x: _ttest_mean(x, alpha=alpha)
    # agg = df.groupby(['site', 'split'])[['fit_r', 'mapping_cons', 'internal_cons']].apply(ttest_mean)

    # stderr = lambda x: x.std(ddof=1) / np.sqrt(x.count())
    # mn = agg.reset_index().groupby('site').mean()
    # stderr = agg.reset_index().groupby('site').apply(stderr)

    agg = df.groupby(['site', 'split'])[['fit_r', 'mapping_cons', 'internal_cons']].mean()
    mn = agg.reset_index().groupby('site')[['fit_r', 'mapping_cons', 'internal_cons']].apply(stderr)
    return mn


if __name__ == '__main__':
    data = hvm.HvM6IT()
    nfit = NeuralFit(data.model_feats(), data.neural(), labels=data.meta['obj'], n_splithalves=10)
    res = nfit.fit()
    import pdb; pdb.set_trace()
