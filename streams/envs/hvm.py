import sys, os, hashlib, pickle, tempfile, zipfile, glob
from collections import OrderedDict

import numpy as np
import pandas
import tables
import pymongo
import boto3
import tqdm
import skimage, skimage.io, skimage.transform

from streams.envs.dataset import Dataset
import streams.utils


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


class HvM(Dataset):

    DATA = {'meta': 'streams/hvm/meta.pkl',
            'images256': 'streams/hvm/imageset/ims256.npy',
            'images224': 'streams/hvm/imageset/ims224.npy',

            'imageset/discfade': ('diskfade_150x150.png', '0f4f617c24a58bc039d3ed5ff86cad829d2245ca', 'imageset'),
            'imageset/tfrecords/var0': 'streams/hvm/imageset/tfrecords/var0.tfrecords',
            'imageset/tfrecords/var3': 'streams/hvm/imageset/tfrecords/var3.tfrecords',
            'imageset/tfrecords/var6': 'streams/hvm/imageset/tfrecords/var6.tfrecords',
            'imageset/tfrecords/meta': 'streams/hvm/imageset/tfrecords/meta.pkl',

            # 'meta': ('streams/hvm/imageset/meta.pkl', None, 'hvm/imageset/meta.pkl'),

            'neural/averaged/var0': ('Chabo_Tito_20110907_Var00a_pooled_P58.trim.wh.evoked.repr.h5',
                              '300b4446797b6244987f3d98b38c4cb36f61086d', 'neural'),
            'neural/averaged/var3': ('Chabo_Tito_20110907_Var03a_pooled_P58.trim.wh.evoked.repr.h5',
                              'd72802384c35915ceee80a530e1d1053b086975a', 'neural'),
            'neural/averaged/var6': ('Chabo_Tito_20110907_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                              'a95c797b0b2eef56d431c8ccca4c160143a65357', 'neural'),

            'neural/temporal/var0': ('Chabo_Tito_20140307_Var00a_pooled_P58.trim.wh.evoked.repr.h5',
                              '588d6d118a45e98c65260e9226c237c72244af0d', 'neural'),
            'neural/temporal/var3': ('Chabo_Tito_20140307_Var03a_pooled_P58.trim.wh.evoked.repr.h5',
                              '52a35255b1595c29be747c5725be7d6f0e6bd037', 'neural'),
            'neural/temporal/var6': ('Chabo_Tito_20140307_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                              '8ed6ec266fd0104368121aa742038f04681f7231', 'neural'),
            # 'temporal_raw': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.raw.d.repr.h5',
            # 'temporal_evoked': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.evoked.repr.h5'

            'model/alexnet/pool5': 'streams/hvm/model/alexnet_pool5_feats_pca1000.npy',
            'model/vgg-19/fc2': 'streams/hvm/model/vgg-19_fc2.npy',
            'model/basenet6': 'streams/hvm/model/basenet6.pkl',
            'model/basenet6/pIT': 'streams/hvm/model/basenet6_pIT.npy',
            'model/basenet6/aIT': 'streams/hvm/model/basenet6_aIT.npy',
            'model/basenet6/pIT/pca': 'streams/hvm/model/basenet6_pIT_pca1000.npy',
            'model/basenet6/aIT/pca': 'streams/hvm/model/basenet6_aIT_pca1000.npy',

            'model/basenet11/pIT': 'streams/hvm/model/basenet11_pIT.npy',
            'model/basenet11/aIT': 'streams/hvm/model/basenet11_aIT.npy',
            'model/basenet11/pIT/pca': 'streams/hvm/model/basenet11_pIT_pca1000.npy',
            'model/basenet11/aIT/pca': 'streams/hvm/model/basenet11_aIT_pca1000.npy',

            'model/hmo/layer1': 'streams/hvm/model/hmo_layer1feats.npy',
            'model/hmo/layer2': 'streams/hvm/model/hmo_layer2feats.npy',
            'model/hmo/layer3': 'streams/hvm/model/hmo_layer3feats.npy',
            'model/hmo/top': 'streams/hvm/model/hmo_topfeats.pkl',
            }

    # -- which channel is which, for the standard HvM dataset
    CHANNEL_INFO = {
        'region': ['IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
            'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4'],
        # Tito:
        # -pIT: arr == 'M'
        # - cIT: arr == 'A'
        # Chabo:
        # - pIT: (arr == 'M') & (col < 5)
        # - cIT: ((arr == 'M') & (col >= 5)) | ((arr == 'A') & (col < 5))
        # - aIT: (arr == 'A') & (col >= 5)
        'subregion': ['cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'pIT', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'aIT', 'aIT', 'aIT', 'aIT', 'aIT', 'aIT', 'aIT',
                    'aIT', 'aIT', 'aIT', 'aIT', 'cIT', 'aIT', 'cIT', 'cIT', 'aIT',
                    'cIT', 'aIT', 'cIT', 'aIT', 'cIT', 'aIT', 'cIT', 'cIT', 'cIT',
                    'aIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'pIT',
                    'pIT', 'cIT', 'cIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT',
                    'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'cIT', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT', 'pIT',
                    'pIT', 'pIT', 'pIT', 'pIT', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4', 'V4',
                    'V4'],
        'animal': ['Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo', 'Chabo',
            'Chabo', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito',
            'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito', 'Tito'],
        # relative array placement, NOT IT subdivisions
        'arr': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                'A', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'P', 'P'],
        'row': [5, 6, 5, 7, 6, 7, 9, 7, 9, 8, 9, 2, 1, 3, 6, 7, 6, 6, 8, 5, 8, 5, 9,
                6, 8, 7, 9, 7, 9, 7, 9, 8, 9, 8, 9, 8, 9, 2, 2, 3, 2, 3, 2, 4, 4, 8,
                9, 6, 8, 7, 9, 7, 9, 7, 9, 8, 9, 8, 9, 8, 3, 2, 4, 4, 4, 3, 5, 4, 6,
                4, 6, 5, 5, 6, 6, 7, 7, 8, 7, 8, 4, 4, 6, 6, 7, 3, 5, 3, 5, 6, 4, 5,
                6, 6, 7, 7, 8, 7, 8, 8, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                6, 6, 7, 7, 9, 8, 9, 8, 0, 0, 0, 9, 8, 4, 5, 7, 6, 7, 6, 8, 5, 9, 6,
                8, 7, 9, 7, 7, 9, 8, 9, 8, 9, 8, 2, 3, 2, 3, 2, 4, 3, 4, 2, 5, 3, 6,
                5, 3, 8, 9, 8, 9, 7, 6, 5, 6, 2, 4, 5, 6, 7, 6, 8, 5, 8, 9, 9, 9, 9,
                9, 9, 9, 4, 6, 4, 5, 3, 5, 5, 6, 6, 7, 7, 8, 8, 1, 0, 2, 3, 4, 5, 5,
                6, 7, 9, 8, 8, 8, 8, 0, 6, 6, 7, 7, 9, 8, 4, 6, 5, 5, 6, 6, 7, 7, 8,
                1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                7, 9, 8, 8, 7, 7, 5, 9, 6, 7, 5, 5, 9, 0, 4, 5, 7, 9, 4, 0, 9, 3, 0,
                6, 5, 1, 3, 1, 3, 6, 2, 9, 6, 1, 4, 4, 7, 4, 4, 3, 7, 4, 9],
        'col': [9, 9, 8, 9, 8, 8, 7, 7, 5, 6, 4, 9, 9, 9, 9, 9, 8, 7, 9, 6, 8, 7, 8,
                6, 7, 5, 7, 7, 6, 6, 5, 6, 4, 5, 3, 4, 2, 8, 7, 7, 6, 5, 4, 5, 4, 8,
                8, 6, 7, 5, 7, 7, 6, 6, 5, 6, 4, 5, 3, 4, 5, 4, 5, 4, 6, 3, 5, 3, 4,
                2, 5, 2, 3, 2, 3, 2, 3, 2, 4, 3, 1, 0, 1, 0, 1, 3, 4, 2, 5, 4, 2, 2,
                2, 3, 2, 3, 2, 4, 1, 3, 6, 5, 4, 3, 2, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                1, 0, 0, 1, 1, 0, 3, 1, 9, 6, 3, 1, 0, 7, 8, 9, 8, 8, 7, 8, 7, 8, 6,
                7, 5, 7, 7, 6, 5, 6, 4, 5, 3, 4, 6, 6, 5, 5, 4, 5, 4, 4, 3, 4, 3, 4,
                5, 2, 6, 0, 4, 2, 2, 2, 2, 5, 9, 7, 9, 9, 9, 7, 9, 6, 8, 8, 7, 6, 5,
                4, 3, 2, 6, 5, 2, 5, 2, 2, 3, 2, 3, 2, 3, 1, 3, 3, 1, 2, 1, 1, 1, 0,
                0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0, 1, 1, 0, 2, 5, 2, 3, 2, 3, 3, 4, 1,
                6, 5, 5, 4, 4, 3, 2, 2, 1, 1, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                1, 1, 0, 5, 3, 5, 1, 3, 3, 4, 0, 4, 6, 4, 6, 6, 6, 8, 1, 5, 9, 0, 3,
                6, 7, 1, 8, 6, 5, 8, 8, 7, 7, 0, 5, 9, 8, 7, 0, 9, 7, 3, 5],
        'hemisphere': 168 * ['L']
    }

    V4_NEURONS = np.array([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                           31,32,33,34,35,36,37,38,39,40,41,42,43,44,85,86,87,88,89,90,
                           91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,
                           108,109,110,111,112,113,114,115,116,117,118,119,120,163,
                           164, 165, 166, 167, 168, 169, 170, 211, 212, 213, 214, 215,
                           216, 217, 218, 219, 220])

    V4_NEURONS_POOR = np.array([256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
                                267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277,
                                278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
                                289, 290, 291, 292, 293, 294, 295])
    # ordered by pIT, cIT, aIT x Chabo, Tito
    IT_NEURONS = np.array([ 10, 121, 122, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                           135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                           148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                           161, 162, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
                           232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
                           245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,   0,   1,
                             2,   3,   4,   5,   6,   7,   8,   9,  56,  58,  59,  61,  63,
                            65,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84, 123, 124, 171, 172, 173, 174, 175, 176,
                           177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
                           190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                           203, 204, 205, 206, 207, 208, 209, 210,  45,  46,  47,  48,  49,
                            50,  51,  52,  53,  54,  55,  57,  60,  62,  64,  66,  70])
    BOUNDS = [6, 76, 111, 151, 168, 168]  # pIT: C,T; cIT: C,T; aIT: C,T
    VAR_SLICES = {None: slice(0,5760),
                  0: slice(0, 640),
                  3: slice(640, 640+2560),
                  6: slice(640+2560, 5760)}
    TIMEPOINTS = OrderedDict(zip(range(-90, 300, 10), range(39)))

    def __init__(self, var=None, region='IT'):
        self.name = 'hvm'
        self.var = var
        self.region = region
        if self.region == 'IT':
            self.neurons = self.IT_NEURONS
        elif self.region == 'V4':
            self.neurons = self.V4_NEURONS

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
            if self.var is not None:
                self._meta = self._meta[self._meta['var'] == self.var]
        return self._meta

    def images(self, size=256):
        if not hasattr(self, '_images'):
            try:
                self._images = np.load(self.datapath('images{}'.format(size)))
                self._images = self._images[self.VAR_SLICES[self.var]]
            except:
                ims = []
                for idd in self.meta.id.values:
                    im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
                    im = skimage.transform.resize(im, (size,size))
                    im = skimage.img_as_float(im)
                    ims.append(im)
                self._images = np.array(ims)
        return self._images

    def tokens(self, size=256):
        if not hasattr(self, '_tokens'):
            tokens = []
            for idd in self.meta.obj.unique():
                im = skimage.io.imread(self.home('imageset/tokens', idd + '.png'))
                im = skimage.color.gray2rgb(im)
                im = skimage.transform.resize(im, (size,size))
                im = skimage.img_as_float(im)
                tokens.append(im)
            self._tokens = np.array(tokens)
        return self._tokens

    @property
    def discfade(self):
        if not hasattr(self, '_discfade'):
            mask = 255 - skimage.io.imread(self.datapath('imageset/discfade'))[:,:,3]
            mask = np.dstack([mask,mask,mask])
            self._discfade = skimage.transform.resize(mask, [256,256])
        return self._discfade

    def neural(self, timepoint=None):
        """
        Format: (time bins, reps, images, sites)
        """
        if timepoint is None:
            path = self.datapath('neural/averaged/var{}'.format(self.var))
        else:
            path = self.datapath('neural/temporal/var{}'.format(self.var))
        f = tables.open_file(path)
        if timepoint is None:
            nd = f.root.spk[0]  # shape is (1, reps, ...)
        else:
            nd = f.root.spk[self.TIMEPOINTS[timepoint]]
        neural_order = [n.decode('ascii') for n in f.root.meta.idx2iid]
        f.close()
        order = np.array([neural_order.index(n) for n in self.meta.filename])
        self._neural_data = nd[:, order][:, :, self.neurons]
        return self._neural_data

    def model(self, name='alexnet', layer='pool5', pca=True):
        if name == 'alexnet' and layer == 'pool5':
            model_feats = np.load(self.datapath('model/alexnet/pool5'))
        elif name == 'hmo':
            path = self.datapath('model/hmo/' + layer)
            if path.endswith('.npy'):
                model_feats = np.load(path)
            else:
                model_feats = pickle.load(open(path, 'rb'), encoding='latin1')
        else:  #if name.startswith('basenet'):
            # model_feats = pickle.load(open(self.datapath('model/' + name), 'rb'))[layer]
            if pca:
                model_feats = np.load(open(self.datapath('model/{}/{}/pca'.format(name, layer)), 'rb'))
            else:
                model_feats = np.load(open(self.datapath('model/{}/{}'.format(name, layer)), 'rb'))
        # else:
        #     raise ValueError

        if self.var is not None:
            model_feats = model_feats[self.VAR_SLICES[self.var]]
        return model_feats

    def __call__(self, kind='meta', **kwargs):
        if kind == 'meta':
            data = self.meta
        elif kind == 'neural':
            data = self.neural(**kwargs)
        elif kind == 'model':
            data = self.model(**kwargs)
        return data

    def query(self, query):
        raise NotImplementedError
        meta_query = {k:v for k,v in query.items() if '.' not in k}

        keys = {}
        for key, value in query.items():
            spl = key.split('.')
            if len(spl) == 1:
                spl = ['meta', key]
            if spl[0] not in keys:
                keys[spl[0]] = {}
            keys[spl[0]][spl[1]] = value

        # if 'meta' in keys:
        #     if 'var' in keys['meta']:

        data = {}
        for key, query in keys.items():
            if len(query) > 0:
                data[key] = getattr(self, key)(**query)

        return data

    def _neural_order(self):
        """
        Order neural data by subregion and animal
        """
        sel = np.array(sel.CHANNEL_INFO['subregion']) != 'V4'
        neural_order = []
        bounds = [0]
        for subregion in ['pIT', 'cIT', 'aIT']:
            for animal in ['Chabo', 'Tito']:
                ix = np.arange(168)[(np.array(hvm.HvM.CHANNEL_INFO['subregion'])[sel] == subregion) & (np.array(hvm.HvM.CHANNEL_INFO['animal'])[sel] == animal)].tolist()
                bounds.append(bounds[-1] + len(ix))
                neural_order += ix
        neural_order = np.array(neural_order)
        bounds = bounds[1:]
        return neural_order, bounds


class HvMImageNet(Dataset):

    DATA = {'meta': 'streams/hvm_imagenet/meta.pkl',
           'imageset/tfrecords': 'streams/hvm_imagenet/imageset/tfrecords/data.tfrecords'}

    def __init__(self):
        self.name = 'hvm_imagenet'

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta

    @property
    def images(self):
        if not hasattr(self, '_images'):
            ims = []
            for idd in self.meta.filename.values:
                im = skimage.io.imread(self.home('imageset', 'images', idd))
                im = skimage.color.gray2rgb(im)
                im = skimage.transform.resize(im, (224,224))
                # im = skimage.img_as_float(im)
                ims.append(im)
            self._images = np.array(ims)
        return self._images


class HvM10(HvM):

    DATA = dict({#'meta': 'streams/hvm10/meta.pkl',

            'human_data': 'streams/hvm10/behav/human_data.pkl',
            'human_data_timing': 'streams/hvm10/behav/human_data_timing.pkl'
            }, **HvM.DATA)

    OBJS = ['bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug',
            'TURTLE_L', 'Apple_Fruit_obj', 'f16', '_001']

    def __init__(self, region='IT'):
        super(HvM10, self).__init__(var=6, region='IT')
        meta = super(HvM10, self).meta
        self.sel = meta.obj.isin(self.OBJS)
        self._meta = meta[self.sel]
        self.name = 'hvm10'
        self._name = 'hvm10'

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self._meta = super(HvM10, self).meta[self.sel]
            # self.fetch()
            # self._meta = pandas.read_pickle(HvM(var=6).datapath('meta'))
            # self._meta = meta[self.sel]
        return self._meta

    def images(self, **kwargs):
        self.name = 'hvm'
        ims = super(HvM10, self).images(**kwargs)
        self.name = self._name
        return ims[self.sel]

    def neural(self, **kwargs):
        self.name = 'hvm'
        n = super(HvM10, self).neural(**kwargs)
        self.name = self._name
        return n[self.sel]

    def model(self, **kwargs):
        self.name = 'hvm'
        m = super(HvM10, self).model(**kwargs)
        self.name = self._name
        return m[self.sel]

    @property
    def human_data(self):
        if not hasattr(self, '_human_data'):
            try:
                self._human_data = pandas.read_pickle(self.datapath('human_data'))
            except:
                self._human_data = self._get_human_data()
        return self._human_data

    def _get_human_data(self):
        collection_name = 'hvm10_basic_2ways'
        conn = pymongo.MongoClient(host='localhost', port=22334)
        coll = conn['mturk'][collection_name]

        dfs = []
        for subjid, doc in enumerate(coll.find()):
            tmp = []
            data = zip(doc['ImgData'], doc['Response'], doc['RT'], doc['TimingInfo']['Stimdur'])
            for trial, respno, rt, act in data:
                if trial['Test'][1]['obj'] == trial['Sample']['obj']:
                    distr = trial['Test'][0]['obj']
                else:
                    distr = trial['Test'][1]['obj']

                trial['Sample'].update({'subjid': subjid,
                                        'label1': trial['Test'][0]['obj'],
                                        'label2': trial['Test'][1]['obj'],
                                        'rt': rt,
                                        'corr_resp': trial['Sample']['obj'],
                                        'subj_resp': trial['Test'][respno]['obj'],
                                        'distractor': distr,
                                        'stim_dur': 100.,
                                        'actual_stim_dur': act
                                        })
                acc = trial['Sample']['corr_resp'] == trial['Sample']['subj_resp']
                trial['Sample'].update({'acc': acc})
                tmp.append(trial['Sample'])
            dfs.append(pandas.DataFrame(tmp))
        df = pandas.concat(dfs)
        df['imgno'] = 0
        gr = df.groupby(['obj', 'id']).acc.count()
        for obj in self.OBJS:
            gr[obj] = range(len(gr[obj]))
        df.imgno = df.groupby(['obj', 'id']).imgno.apply(lambda x: gr[x.name])

        df['imgcount'] = df.groupby(['obj', 'distractor', 'id']).cumcount()
        df['intensity'] = df.stim_dur / df.stim_dur.max()
        df['uuid'] = df[['obj','distractor','id']].apply(lambda x: '-'.join(x.values), axis=1)
        df.obj = df.obj.astype('category', categories=self.OBJS, ordered=True)
        df.distractor = df.distractor.astype('category', categories=self.OBJS, ordered=True)
        df.id = df.id.astype('category', categories=self.meta['id'], ordered=True)
        df = streams.utils.clean_data(df)
        df.to_pickle(self.datapath('human_data'))
        return df

    @property
    def human_data_timing(self):
        if not hasattr(self, '_human_data_timing'):
            try:
                self._human_data_timing = pandas.read_pickle(self.datapath('human_data_timing'))
            except:
                self._human_data_timing = self._get_human_data_timing()
        return self._human_data_timing

    def _get_human_data_timing(self):
        collection_name = 'hvm10-var6-timing-nogap'
        mongo_conn = pm.MongoClient(host='localhost', port=22334)
        db = mongo_conn['mturk']
        coll = db[collection_name]

        dfs = [pandas.DataFrame(doc['ImgData']) for doc in coll.find()]
        df = pandas.concat(dfs)
        stim_durs = [doc['TimingInfo']['StimDur'] for doc in coll.find()]
        df['actual_stim_dur'] = np.hstack(stim_durs)
        df['imgcount'] = df.groupby(['obj', 'distractor', 'id', 'stim_dur']).cumcount()
        df['intensity'] = df.stim_dur / df.stim_dur.max()
        df['uuid'] = df[['obj','distractor','id']].apply(lambda x: '-'.join(x.values), axis=1)
        # NAMES = sorted(df.groupby(['objno','obj']).groups.keys())
        # NAMES = [n[1] for n in NAMES]
        df.obj = df.obj.astype('category', categories=self.OBJS, ordered=True)
        df.distractor = df.distractor.astype('category', categories=self.OBJS, ordered=True)
        df.id = df.id.astype('category', categories=self.meta['id'], ordered=True)

        # mf = pandas.DataFrame(dataset.meta)
        # sel_hvm10 = np.array(mf.obj.isin(NAMES) & (mf['var']=='V6'))
        # mf_hvm10 = mf[sel_hvm10]

        # agg = df.groupby('id').aggregate(lambda x: x.iloc[0])
        # mf_hvm10 = mf_hvm10.set_index('id')
        # mf_hvm10['objno'] = agg.objno
        # mf_hvm10['distractor'] = agg.distractor
        # mf_hvm10['distrno'] = agg.distrno
        # mf_hvm10['uuid'] = agg.uuid
        # mf_hvm10 = mf_hvm10.reset_index().set_index('uuid')

        df = streams.utils.clean_data(df)
        df.to_pickle(self.datapath('human_data_timing'))
        return df

    def human_acc(self, time=False):
        df = self.human_data_timing if time else self.human_data
        return df.pivot_table(index='id', columns='stim_dur', values='acc')


class HvM10Train(Dataset):

    DATA = {'meta': 'streams/hvm10_train/meta.pkl',
            'images224': 'streams/hvm10_train/imageset/ims224.npy',
            'model/basenet6/pIT': 'streams/hvm10_train/model/basenet6_pIT.npy',
            'model/basenet6/aIT': 'streams/hvm10_train/model/basenet6_aIT.npy',
            'model/basenet6/pIT/pca': 'streams/hvm10_train/model/basenet6_pIT_pca1000.npy',
            'model/basenet6/aIT/pca': 'streams/hvm10_train/model/basenet6_aIT_pca1000.npy',
            'model/vgg-19/fc2': 'streams/hvm10_train/model/vgg-19_fc2.npy',
            }

    def __init__(self):
        self.name = 'hvm10_train'

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta

    def images(self, size=256):
        if not hasattr(self, '_images'):
            try:
                self._images = np.load(self.datapath('images{}'.format(size)))
                self._images = self._images[self.VAR_SLICES[self.var]]
            except:
                ims = []
                for idd in self.meta.id.values:
                    im = skimage.io.imread(self.home('imageset', 'images', idd + '.png'))
                    im = skimage.transform.resize(im, (size,size))
                    ims.append(im)
                self._images = np.array(ims)
                # np.save(self.datapath('images{}'.format(size)), self._images)
        return self._images

    def model(self, name='basenet', layer='aIT', pca=True):
        if pca:
            model_feats = np.load(self.datapath('model/{}/{}/pca'.format(name, layer)))
        else:
            model_feats = np.load(self.datapath('model/{}/{}'.format(name, layer)))

        return model_feats


class HvMImageSet(Dataset):

    DATA = {'meta': 'streams/hvm/imageset/meta.pkl',
            'tfrecords/var0': 'streams/hvm/imageset/tfrecords/var0.tfrecords',
            'tfrecords/var3': 'streams/hvm/imageset/tfrecords/var3.tfrecords',
            'tfrecords/var6': 'streams/hvm/imageset/tfrecords/var6.tfrecords',
            'tfrecords/meta': 'streams/hvm/imageset/tfrecords/meta.pkl',
            }

    def __init__(self):
        self.name = 'hvm/imageset'
        self.ids = self.meta['id']

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta


class HvMNeural(Dataset):

    DATA = {'meta': ('streams/hvm/imageset/meta.pkl', None, 'hvm/imageset/meta.pkl'),

            'averaged/var0': ('Chabo_Tito_20110907_Var00a_pooled_P58.trim.wh.evoked.repr.h5',
                              '300b4446797b6244987f3d98b38c4cb36f61086d', None),
            'averaged/var3': ('Chabo_Tito_20110907_Var03a_pooled_P58.trim.wh.evoked.repr.h5',
                              'd72802384c35915ceee80a530e1d1053b086975a', None),
            'averaged/var6': ('Chabo_Tito_20110907_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                              'a95c797b0b2eef56d431c8ccca4c160143a65357', None),

            'temporal/var0': ('Chabo_Tito_20140307_Var00a_pooled_P58.trim.wh.evoked.repr.h5',
                              '588d6d118a45e98c65260e9226c237c72244af0d', None),
            'temporal/var3': ('Chabo_Tito_20140307_Var03a_pooled_P58.trim.wh.evoked.repr.h5',
                              '52a35255b1595c29be747c5725be7d6f0e6bd037', None),
            'temporal/var6': ('Chabo_Tito_20140307_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                              '8ed6ec266fd0104368121aa742038f04681f7231', None),
            # 'temporal_raw': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.raw.d.repr.h5',
            # 'temporal_evoked': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.evoked.repr.h5'
            }

    # ordered as pIT: Chabo, Tito; cIT: Chabo, Tito; aIT: Chabo, Tito (see BOUNDS)
    IT_NEURONS = np.array([ 10, 121, 122, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                           135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                           148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                           161, 162, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
                           232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
                           245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,   0,   1,
                             2,   3,   4,   5,   6,   7,   8,   9,  56,  58,  59,  61,  63,
                            65,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84, 123, 124, 171, 172, 173, 174, 175, 176,
                           177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
                           190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                           203, 204, 205, 206, 207, 208, 209, 210,  45,  46,  47,  48,  49,
                            50,  51,  52,  53,  54,  55,  57,  60,  62,  64,  66,  70])
    BOUNDS = [6, 76, 111, 151, 168, 168]
    VAR_SLICES = {0: slice(0, 640),
                  3: slice(640, 640+2560),
                  6: slice(640+2560, 5760)}
    TIMEPOINTS = OrderedDict(zip(range(-90, 300, 10), range(39)))

    def __init__(self, var=6):
        self.name = 'hvm/neural'
        self.var = var

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta[self.VAR_SLICES[self.var]]

    def neural_data(self, var=6, timepoint=None):
        """
        Format: (time bins, reps, images, sites)
        """
        if timepoint is None:
            path = self.datapath('averaged/var{}'.format(var))
        else:
            path = self.datapath('temporal/var{}'.format(var))
        f = tables.open_file(path)
        if timepoint is None:
            nd = f.root.spk[0]  # shape is (1, reps, ...)
        else:
            nd = f.root.spk[self.TIMEPOINTS[timepoint]]
        neural_order = f.root.meta.idx2iid[:]
        f.close()
        meta_order = [os.path.basename(n) for n in self.meta['filename']]
        order = np.array([neural_order.index(n) for n in meta_order])
        self._neural_data = nd[:,order]
        return self._neural_data


class HvMModel(Dataset):

    def __init__(self):
        self.name = 'hvm/model'
        # self.ids = self.meta['id']

    def model_feats(self, model, layer):
        if model == 'alexnet' and layer == 'pool5':
            path = 'alexnet_pool5_feats_pca1000.npy'
            model_feats = np.load(self.home(path))
        elif model == 'hmo' and layer == 'top':
            path = 'hmo_topfeats.pkl'
            model_feats = pickle.load(open(self.home(path)))
        return model_feats


class HvM6IT(Dataset):

    def __init__(self, order=range(2560)):
        self.name = 'hvm'
        self.order = np.array(order)
        self._imageset = HvMImageSet()
        # self.behav = HvMBehav()
        self._neural = HvMNeural()
        self._model = HvMModel()
        self.DATA = {}
        self.DATA.update(self._imageset.DATA)
        self.DATA.update(self._neural.DATA)
        # self.DATA.update(self._model.DATA)

    # def home(self, *suffix_paths):
    #     return os.path.join(DATA_HOME, self.name, *suffix_paths)

    def datapath(self, handle):
        if handle in self._imageset.DATA:
            return self._imageset.datapath(handle)
        elif handle in self._neural.DATA:
            return self._neural.datapath(handle)

    @property
    def meta(self):
        return self._imageset.meta.iloc[640 + 2560 + self.order]

    def neural(self, var=6, timepoint=None):
        d = self._neural.neural_data(var=var, timepoint=timepoint)[:, self.order]
        neural_data = d[:, :, self._neural.IT_NEURONS]
        return neural_data

    def model_feats(self, model='alexnet', layer='pool5'):
        model_feats = self._model.model_feats(model, layer)
        model_feats = model_feats[640 + 2560 + self.order]
        return model_feats


def create_tfrec():
    import skimage.io
    import streams.utils
    from streams.utils import TFWriter

    import dldata.stimulus_sets.hvm

    d = dldata.stimulus_sets.hvm.HvMWithDiscfade()
    d.meta.shape

    import skimage.io
    for row in tqdm.tqdm_notebook(d.meta):
        im = skimage.io.imread(row['filename'])
        skimage.io.imsave(os.path.join('/mindhive/dicarlolab/u/qbilius/.skdata/HvMWithDiscfade/images', row['id'] + '.png'), im)

    meta = hvm.HvMImageSet().meta
    slices = [('var0', slice(0,640)), ('var3', slice(640, 3200)), ('var6', slice(3200,5760))]
    for v, s in slices:
        writer = TFWriter(n_data=s.stop-s.start, chunk_size=s.stop-s.start, prefix=v,
                        save_path=os.path.join(PATH, '.skdata/HvMWithDiscfade/images_tf3'))
        for row_idx, row in tqdm.tqdm_notebook(meta.iloc[s].iterrows()):
            data = row.to_dict()
            data['imno'] = row.name
            data['images'] = skimage.io.imread(os.path.join(PATH, '.skdata/HvMWithDiscfade/images/' + row['id'] + '.png'))
            writer.write(data)
        writer.close()

    m = {}
    for n, d in meta.dtypes.iteritems():
        if d.name == 'object':
            m[n] = {'dtype': tf.string, 'shape': []}
        elif d.name == 'float64':
            m[n] = {'dtype': tf.float32, 'shape': []}
        elif d.name == 'int64':
            m[n] = {'dtype': tf.int64, 'shape': []}

    m['imno'] = {'dtype': tf.int64, 'shape': []}
    m['images'] = {'dtype': tf.string, 'shape': []}
    import pickle
    pickle.dump(m, open('/home/qbilius/mh17/.skdata/HvMWithDiscfade/images_tf3/meta.pkl', 'wb'))


if __name__ == '__main__':
    pass
    # HvM()._upload('meta.pkl')