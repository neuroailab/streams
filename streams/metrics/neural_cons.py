from __future__ import division, print_function, absolute_import

import numpy as np
import pandas
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_decomposition import PLSRegression
import scipy.stats
import tensorflow as tf
import tqdm

from streams.envs import hvm
from streams import utils
from streams.parallel_new import Parallel


class NeuralFit(object):

    def __init__(self, model_feats, neural_feats_reps, labels=None,
                 n_splits=10, test_size=1/4., n_splithalves=10,
                 n_components=25, scale=False, timer=False):
        """
        PLSRegression-based fitting of model features to neurons.

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
            - n_components
                Number of components for PLSRegression
            - scale
                Whether to scale features in PLSRegression

        :Returns:
            The fit outcome (pandas.DataFrame)
        """
        self.model_feats = model_feats
        self.neural_feats_reps = neural_feats_reps
        self.labels = labels
        self.n_splits = n_splits
        self.n_splithalves = n_splithalves
        self.n_components = n_components
        self.scale = scale
        self.timer = timer

        self.rng = np.random.RandomState(0)
        sss = StratifiedShuffleSplit(n_splits=self.n_splits,
                                     test_size=test_size, random_state=self.rng)
        self.splits = [s for s in sss.split(self.model_feats, self.labels)]

    def fit(self):
        df = Parallel(self._fit, n_iter=self.n_splits, timer=self.timer)()
        # df = pandas.concat(df, axis=0, ignore_index=True)
        return df

    def _fit(self, splitno):
        train_inds, test_inds = self.splits[splitno]
        r = self.raw_fit(train_inds, test_inds)
        res = Parallel(self._cons, n_iter=self.n_splithalves)(r, train_inds, test_inds)
        # res = pandas.concat(res, axis=0, ignore_index=True)
        res['split'] = splitno
        res['explained_var'] = res.fit_r ** 2 / (res.internal_cons * res.mapping_cons)
        return res

    def raw_fit(self, train_inds, test_inds):
        neural_feats = self.neural_feats_reps.mean(axis=0)
        pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        pls.fit(self.model_feats[train_inds], neural_feats[train_inds])
        pred = pls.predict(self.model_feats[test_inds])
        actual = neural_feats[test_inds]
        r = utils.pearsonr_matrix(actual, pred)
        return r

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
        pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        rng = np.random.RandomState(splithalfno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, train_inds], rng=rng)
        pls.fit(self.model_feats[train_inds], split1)
        pred1 = pls.predict(self.model_feats[test_inds])
        pls.fit(self.model_feats[train_inds], split2)
        pred2 = pls.predict(self.model_feats[test_inds])
        r = utils.pearsonr_matrix(pred1, pred2)
        rc = utils.spearman_brown_correct(r, n=2)
        return rc

    def internal_cons(self, splithalfno, test_inds):
        rng = np.random.RandomState(splithalfno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, test_inds], rng=rng)
        r = utils.pearsonr_matrix(split1, split2)
        rc = utils.spearman_brown_correct(r, n=2)
        return rc


class HvM6IT_QuickNeural(object):

    def __init__(self, path, resize=224, batch_size=256, **kwargs):
        """
        tfutils compatible HvM reader
        """
        self.resize = resize
        self.batch_size = batch_size
        self.data = np.load(path)
        self._idx = 0

    def init_ops(self):
        sel = slice(self._idx, self._idx + self.batch_size)
        ims = self.data['images'][sel]
        batch = {'images': tf.image.resize_images(ims, [self.resize, self.resize]),
                 'labels': tf.constant(self.data['labels'][sel])}
        self._idx += self.batch_size
        return [batch]

    def get_target(self, inputs, outputs, target='fc6',
                   tensor_name='validation/hvm_neural_fit/fc6/fc/fc:0'):
        return {target: tf.get_default_graph().get_tensor_by_name(tensor_name)}

    def agg_func(self, output):
        perm = np.random.RandomState(0).permutation(output[0]['fc6'].shape[1])[:1024]
        model_feats = np.row_stack(it['fc6'] for it in output)[:, perm]
        explained_var = self.neural_fit(model_feats)
        return explained_var

    def neural_fit(self, model_feats):
        rng = np.random.RandomState(0)
        neural_feats_reps = self.data['neural']
        labels = self.data['labels']
        # np.repeat(range(64), 40)
        explained_var = []
        sss = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=rng)
        for train_inds, test_inds in tqdm.tqdm(sss.split(model_feats, labels), desc='hvm regression', total=sss.n_splits):
            neural_feats = neural_feats_reps.mean(axis=0)
            pls = PLSRegression(n_components=25, scale=False)
            pls.fit(model_feats[train_inds], neural_feats[train_inds])
            pred = pls.predict(model_feats[test_inds])
            actual = neural_feats[test_inds]
            raw_fit = utils.pearsonr_matrix(actual, pred)

            # mapping cons
            pls = PLSRegression(n_components=25, scale=False)
            split1, split2 = utils.splithalf(neural_feats_reps[:, train_inds], rng=rng)
            pls.fit(model_feats[train_inds], split1)
            pred1 = pls.predict(model_feats[test_inds])
            pls.fit(model_feats[train_inds], split2)
            pred2 = pls.predict(model_feats[test_inds])
            r = utils.pearsonr_matrix(pred1, pred2)
            mc_rc = utils.spearman_brown_correct(r, n=2)

            # internal cons
            split1, split2 = utils.splithalf(neural_feats_reps[:, test_inds], rng=rng)
            r = utils.pearsonr_matrix(split1, split2)
            ic_rc = utils.spearman_brown_correct(r, n=2)

            ev = (raw_fit / np.sqrt(mc_rc * ic_rc)) ** 2
            # import pdb; pdb.set_trace()
            explained_var.append(ev.mean())
        return {'explained_var': np.mean(explained_var)}


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
