from __future__ import division, print_function, absolute_import

import numpy as np
import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression

import hvm, utils
from parallel import Parallel


class NeuralFit(object):

    def __init__(self, model_feats, neural_feats_reps, labels=None,
                 n_splits=10, n_splithalves=10, n_components=25, scale=False):
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

        self.rng = np.random.RandomState(0)
        self.skf = StratifiedKFold(n_splits=self.n_splits,
                                   shuffle=True, random_state=self.rng)
        self.splits = [s for s in self.skf.split(self.model_feats, self.labels)]

    def fit(self):
        df = Parallel(self._fit, n_iter=self.n_splits, timer=True)()
        df = pandas.concat(df, axis=0, ignore_index=True)
        return df

    def _fit(self, iterno):
        train_inds, test_inds = self.splits[iterno]
        r = self.raw_fit(train_inds, test_inds)
        cons = Parallel(self._cons, n_iter=self.n_splithalves)(train_inds, test_inds)
        cons = np.array(cons)
        res = pandas.DataFrame([iterno] * len(r), columns=['split'])
        res['site'] = range(len(r))
        res['fit_r'] = r
        res['mapping_cons'] = np.mean(cons[:,0], axis=0)
        res['mapping_cons_std'] = np.std(cons[:,0], axis=0)
        res['internal_cons'] = np.mean(cons[:,1], axis=0)
        res['internal_cons_std'] = np.std(cons[:,1], axis=0)
        res['explained_var'] = res.fit_r ** 2 / np.sqrt(res.internal_cons * res.mapping_cons)
        return res

    def raw_fit(self, train_inds, test_inds):
        neural_feats = self.neural_feats_reps.mean(axis=0)
        pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        pls.fit(self.model_feats[train_inds], neural_feats[train_inds])
        pred = pls.predict(self.model_feats[test_inds])
        actual = neural_feats[test_inds]
        r = utils.pearsonr_matrix(actual, pred)
        return r

    def _cons(self, iterno, train_inds, test_inds):
        mapping_cons = self.mapping_cons(iterno, train_inds, test_inds)
        internal_cons = self.internal_cons(iterno, test_inds)
        return [mapping_cons, internal_cons]

    def mapping_cons(self, iterno, train_inds, test_inds):
        """
        Split data in half over reps, run PLS on each half on the train set,
        get predictions for the test set, correlate the two, Spearman-Brown
        """
        pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        rng = np.random.RandomState(iterno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, train_inds], rng=rng)
        pls.fit(self.model_feats[train_inds], split1)
        pred1 = pls.predict(self.model_feats[test_inds])
        pls.fit(self.model_feats[train_inds], split2)
        pred2 = pls.predict(self.model_feats[test_inds])
        r = utils.pearsonr_matrix(pred1, pred2)
        return utils.spearman_brown_correct(r, n=2)

    def internal_cons(self, iterno, test_inds):
        rng = np.random.RandomState(iterno)
        split1, split2 = utils.splithalf(self.neural_feats_reps[:, test_inds], rng=rng)
        r = utils.pearsonr_matrix(split1, split2)
        return utils.spearman_brown_correct(r, n=2)


if __name__ == '__main__':
    data = hvm.HvM6IT()
    nfit = NeuralFit(data.model_feats(), data.neural(), labels=data.meta['obj'], n_splithalves=10)
    res = nfit.fit()
    import pdb; pdb.set_trace()