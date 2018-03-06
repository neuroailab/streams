from collections import OrderedDict

import numpy as np
import scipy.stats
import pandas
import sklearn, sklearn.svm, sklearn.preprocessing, sklearn.linear_model

import streams.utils


class MatchToSampleClassifier(object):

    def __init__(self, norm=True, nfeats=None, seed=None, C=1):
        """
        A classifier for the Delayed Match-to-Sample task.

        It is formulated as a typical sklearn classifier with `score`, `predict_proba`
        and `fit` methods available.

        :Kwargs:
            - norm (bool, default: True)
                Whether to zscore features or not.
            - nfeats (int or None, default: None)
                The number of features to use. Useful when you want to match the
                number of features across layers. If None, all features are used.
            - seed (int or None, default: None)
                Random seed for feature selecition
        """
        self.norm = norm
        self.nfeats = nfeats
        self.seed = seed
        self.C = C

    def preproc(self, X, reset=False):
        if self.norm:
            if reset:
                self.scaler = sklearn.preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        else:
            self.scaler = None

        if self.nfeats is not None:
            if reset:
                sel = np.random.RandomState(self.seed).permutation(X.shape[1])[:self.nfeats]
            X = X[:,sel]
        return X

    def fit(self, X, y, order=None):#, decision_function_shape='ovo'):
        """
        :Kwargs:
            - order
                Label order. If None, will be sorted alphabetically
        """
        # self.decision_function_shape = decision_function_shape
        if order is None:
            order = np.unique(y)
        self.label_dict = OrderedDict([(obj,o) for o,obj in enumerate(order)])
        y = self.labels2inds(y)
        X = self.preproc(X, reset=True)
        # self.clf = sklearn.svm.SVC(kernel='linear', probability=True,
        #             decision_function_shape='ovr', C=self.C)#decision_function_shape)
        self.clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', C=self.C)
        self.clf.fit(X, y)

    def _acc(self, x, y):
        return x / (x + y)

    def _dprime(self, x, y):
        return scipy.stats.norm.ppf(x) - scipy.stats.norm.ppf(y)

    def predict_proba(self, X, targets=None, distrs=None, kind='2-way', measure='acc'):
        """
        Model classification confidence (range 0-1)
        """
        if not hasattr(self, 'clf'):
            raise Exception('Must train the classifier first')

        if measure not in ['acc', 'dprime', "d'"]:
            raise ValueError('measure {} not recognized'.format(measure))

        measure_op = self._acc if measure == 'acc' else self._dprime

        X = self.preproc(X)
        conf = self.clf.predict_proba(X)
        # conf = self.clf.decision_function(X)

        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            ti = self.labels2inds(targets)
            # target probability
            t = np.array([x[i] for x,i in zip(conf, ti)])

            if distrs is not None:
                if isinstance(distrs, str):
                    distrs = [distrs]
                dinds = self.labels2inds(distrs)
                # distractor probability
                d = np.array([c[di] for c, di in zip(conf, dinds)])
                acc = measure_op(t, d)

            elif kind == '2-way':
                acc = []
                for c,target in zip(conf,targets):
                    ti = self.label_dict[target]
                    c_tmp = []
                    # compute distractor probability for each distractor
                    for di in self.label_dict.values():
                        if di != ti:
                            tmp = measure_op(c[ti], c[di])
                            c_tmp.append(tmp)
                            # c_tmp.append(c[di])
                        else:
                            # c_tmp.append(c[ti])
                            c_tmp.append(np.nan)
                    acc.append(c_tmp)
                acc = pandas.DataFrame(acc, index=targets, columns=list(self.label_dict.keys()))

            else:
                acc = t
        else:
            acc = conf

        return acc

    def labels2inds(self, y):
        """
        Converts class labels (usually strings) to indices
        """
        return np.array([self.label_dict[x] for x in y])

    def score(self, X, y, kind='2-way', measure='dprime', cap=5):
        """
        Classification accuracy.

        Accuracy is either 0 or 1. For a 2-way classifier, this depends on
        `predict_proba` being less or more that .5. For an n-way classifier, it
        checks if argmax of `predict_proba` gives the correct or incorrect class.
        """
        if kind == '2-way':
            acc = self.predict_proba(X, targets=y, kind=kind)
            acc[~np.isnan(acc)] = acc[~np.isnan(acc)] > .5
        else:
            conf = self.predict_proba(X, kind=kind)
            y = self.labels2inds(y)
            acc = np.argmax(conf, 1) == y

        if measure == 'dprime':
            import ipdb; ipdb.set_trace()
            acc = streams.utils.hitrate_to_dprime_o1(acc, cap=cap)

        return acc


class CorrelationClassifier(object):

    def __init__(self):
        # self.clf =
        pass

    def fit(self, X, y):
        self.tokens = X
        self.labels = np.array(y)

    def predict(self, X):
        y = self.tokens
        feats = np.row_stack([y, X])
        corr = np.corrcoef(feats)
        proba = corr[:len(y), len(y):]
        proba /= proba.sum(0)  # normalize to sum=1
        proba = proba.T  # (n_samples, n_classes)
        labels = self.labels[proba.argmax(1)]
        return labels



