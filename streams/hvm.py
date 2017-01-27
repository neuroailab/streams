from __future__ import division, print_function, absolute_import
import os, hashlib

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

    IT_NEURONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]

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


# class HvMModel(object):

#     def __init__(self):
#         self.name = 'HvMWithDiscfade'
#         self.ids = self.meta['id']

#     def home(self, *suffix_paths):
#         return os.path.join(DATA_HOME, self.name, *suffix_paths)


class HvM6IT(object):

    def __init__(self, order=range(2560)):
        self.name = 'HvMWithDiscfade'
        self.order = np.array(order)
        self._imageset = HvMImageSet()
        # self.behav = HvMBehav()
        self._neural = HvM6Neural()

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    @property
    def meta(self):
        return self._imageset.meta.iloc[640 + 2560 + self.order]

    def neural(self, timepoint=None):
        d = self._neural.neural_data(timepoint=timepoint)[:, self.order]
        neural_data = d[:, :, self._neural.IT_NEURONS]
        return neural_data

    def model_feats(self):
        if not hasattr(self, '_model_feats'):
            self._model_feats = np.load(self.home('alexnet_pool5_feats_pca1000.npy'))
            self._model_feats = self._model_feats[640 + 2560 + self.order]
        return self._model_feats
