from __future__ import absolute_import, division, print_function

import copy
from collections import OrderedDict

import tqdm
import h5py
import numpy as np
import skimage.transform
import sklearn.decomposition


class FeatureSelector(object):

    def __call__(self, feats):
        raise NotImplementedError


class PCASelector(FeatureSelector):

    def __init__(self, ims=None, nfeats=1000, nimg=1000, random_state=0):
        self.nfeats = nfeats
        self.nimg = nimg
        self.random_state = random_state

    def _pca_fit(self, m):
        ims = self._get_imagenet_val(nimg=self.nimg)
        m.feat_sel = None
        features = m.get_features(ims)
        m._feat_sel = self

        if len(m.layers) == 1:
            features = {m.layers[0]: features}
            # features = [{m.layers[0]: f} for f in features]

        self.pca = []
        for layer, feats in tqdm.tqdm(features.items(), desc='PCA'):
            pca = sklearn.decomposition.PCA(n_components=self.nfeats,
                                            random_state=self.random_state)
            pca.fit(feats.reshape((len(feats), -1)))
            self.pca.append((layer, pca))
        self.pca = OrderedDict(self.pca)

        # self.pca = []
        # for layer in tqdm.tqdm(features[0].keys(), desc='PCA'):
        #     feats = np.array([f[layer] for f in features])
        #     pca = sklearn.decomposition.PCA(n_components=self.nfeats)
        #     pca.fit(feats)
        #     self.pca.append((layer, pca))
        # self.pca = OrderedDict(self.pca)

    def _get_imagenet_val(self, nimg=1000):
        n_img_per_class = (nimg - 1) // 1000
        base_idx = np.arange(n_img_per_class).astype(int)
        idx = []
        for i in range(1000):
            idx.extend(50 * i + base_idx)

        for i in range((nimg - 1) % 1000 + 1):
            idx.extend(50 * i + np.array([n_img_per_class]).astype(int))

        with h5py.File('/braintree/data2/active/users/qbilius/datasets/imagenet2012.hdf5', 'r') as f:
            ims = np.array([skimage.transform.resize(f['val/images'][i], (256,256)) for i in idx])
        return ims

    def __call__(self, feats, layer):
        """
        Feats is assumed to come from a multiple images
        """
        # feats = np.expand_dims(feats.ravel(), 0)
        feats = np.reshape(feats, (len(feats), -1))
        return np.squeeze(self.pca[layer].transform(feats))


class RandomSelector(FeatureSelector):

    def __init__(self, nfeats=200, seed=0):
        self.nfeats = nfeats
        self.seed = seed

    def __call__(self, feats):
        feats = feats.ravel()
        if self.nfeats is not None:
            rnd = np.random.RandomState(self.seed)
            sel = rnd.permutation(len(feats))[:self.nfeats]
            feats = feats[sel]
        return feats


class MaxPoolSelector(FeatureSelector):

    def __init__(self, nfeats=4096):
        raise NotImplementedError
        self.nfeats = nfeats

    def __call__(self, feats):
        n = self.nfeats // feats.shape[1]
        d = feats.reshape((feats.shape[1], -1))
        p = np.apply_along_axis(lambda x: x[np.argsort(x)[-n:]], 1, d)
        return p.ravel()