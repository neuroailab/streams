from __future__ import division, print_function, absolute_import
import os, hashlib, cPickle

import numpy as np
import pandas
import tables


DATA_HOME = os.path.abspath(os.path.expanduser(os.environ.get(
                'SKDATA_ROOT', os.path.join('~', '.skdata'))))


def get_id(obj):
    return hashlib.sha1(repr(obj)).hexdigest()


# class HvM6Image(object):

#     def __init__(self, id_):
#         self.name = 'HvMWithDiscfade'
#         self.id = id_
#         self._neural = HvM6Neural()

#     def home(self, *suffix_paths):
#         return os.path.join(DATA_HOME, self.name, *suffix_paths)

#     @property
#     def meta(self):
#         if not hasattr(self, '_meta'):
#             self._meta = pandas.read_pickle(self.home('meta.pkl'))
#         return self._meta[self._meta['id'] == self.id]

#     @property
#     def data(self):
#         d = {}
#         d['neural'] = self._neural.neural_data()[:, self.meta.index]
#         d['neural_time'] = self._neural.neural_data_time[:, :, self.meta.index]


class HvMImageSet(object):

    def __init__(self):
        self.name = 'HvMWithDiscfade'
        self.ids = self.meta['id']

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self._meta = pandas.read_pickle(self.home('meta.pkl'))
        return self._meta


class HvM6Neural(object):

    IT_NEURONS = range(0, 11) + range(45, 85) + range(121, 163) + range(171, 211) + range(221, 256)
    TIMEPOINTS = range(-90, 300, 10)

    def __init__(self):
        self.name = 'HvMWithDiscfade'

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self._meta = pandas.read_pickle(self.home('meta.pkl'))
        return self._meta[3200:]

    def neural_data(self, timepoint=None):
        """
        Format: (time bins, reps, images, sites)
        """
        if timepoint is None:
            path = 'Chabo_Tito_20110907_Var06a_pooled_P58.trim.wh.evoked.repr.h5'
        else:
            path = 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.wh.evoked.repr.h5'
        path = self.home(path)
        f = tables.open_file(path)
        if timepoint is None:
            nd = f.root.spk[0]  # shape is (1, reps, ...)
        else:
            nd = f.root.spk[timepoint]
        neural_order = f.root.meta.idx2iid[:]
        f.close()
        meta_order = [os.path.basename(n) for n in self.meta['filename']]
        order = np.array([neural_order.index(n) for n in meta_order])
        self._neural_data = nd[:,order]
        return self._neural_data


class HvMModel(object):

    def __init__(self):
        self.name = 'HvMWithDiscfade'
        # self.ids = self.meta['id']

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    def model_feats(self, model, layer):
        if model == 'alexnet' and layer == 'pool5':
            path = 'alexnet_pool5_feats_pca1000.npy'
            model_feats = np.load(self.home(path))
        elif model == 'hmo' and layer == 'top':
            path = 'hmo_topfeats.pkl'
            model_feats = cPickle.load(open(self.home(path)))
        return model_feats


class HvM6IT(object):

    def __init__(self, order=range(2560)):
        self.name = 'HvMWithDiscfade'
        self.order = np.array(order)
        self._imageset = HvMImageSet()
        # self.behav = HvMBehav()
        self._neural = HvM6Neural()
        self._model = HvMModel()

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    @property
    def meta(self):
        return self._imageset.meta.iloc[640 + 2560 + self.order]

    def neural(self, timepoint=None):
        d = self._neural.neural_data(timepoint=timepoint)[:, self.order]
        neural_data = d[:, :, self._neural.IT_NEURONS]
        return neural_data

    def model_feats(self, model='alexnet', layer='pool5'):
        model_feats = self._model.model_feats(model, layer)
        model_feats = model_feats[640 + 2560 + self.order]
        return model_feats
