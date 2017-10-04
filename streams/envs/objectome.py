import sys, os, hashlib, pickle, tempfile, zipfile, glob
from collections import OrderedDict

import numpy as np
import pandas
import tables
import boto3
import pymongo
import tqdm
import skimage, skimage.io, skimage.transform

from streams.envs.dataset import Dataset



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


class Objectome(Dataset):

    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images': 'streams/objectome/imageset/ims24s100.npy',

            'imageset/tfrecords/var0': 'streams/hvm/imageset/tfrecords/var0.tfrecords',
            'imageset/tfrecords/var3': 'streams/hvm/imageset/tfrecords/var3.tfrecords',
            'imageset/tfrecords/var6': 'streams/hvm/imageset/tfrecords/var6.tfrecords',
            'imageset/tfrecords/meta': 'streams/hvm/imageset/tfrecords/meta.pkl',

            # 'meta': ('streams/hvm/imageset/meta.pkl', None, 'hvm/imageset/meta.pkl'),

            'model/alexnet/pool5': 'streams/hvm/model/alexnet_pool5_feats_pca1000.npy',
            'model/hmo/layer1': 'streams/hvm/model/hmo_layer1feats.npy',
            'model/hmo/layer2': 'streams/hvm/model/hmo_layer2feats.npy',
            'model/hmo/layer3': 'streams/hvm/model/hmo_layer3feats.npy',
            'model/hmo/top': 'streams/hvm/model/hmo_topfeats.pkl',
            }

    OBJS = ['lo_poly_animal_RHINO_2',
            'MB30758',
            'calc01',
            'interior_details_103_4',
            'zebra',
            'MB27346',
            'build51',
            'weimaraner',
            'interior_details_130_2',
            'lo_poly_animal_CHICKDEE',
            'kitchen_equipment_knife2',
            'lo_poly_animal_BEAR_BLK',
            'MB30203',
            'antique_furniture_item_18',
            'lo_poly_animal_ELE_AS1',
            'MB29874',
            'womens_stockings_01M',
            'Hanger_02',
            'dromedary',
            'MB28699',
            'lo_poly_animal_TRANTULA',
            'flarenut_spanner',
            'womens_shorts_01M',
            '22_acoustic_guitar']

    def __init__(self, var=6):
        self.name = 'objectome'
        self.var = var

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta

    @property
    def human_data(self):
        if not hasattr(self, '_human_data'):
            self._human_data = pandas.read_pickle(self.home('hvm10_basic_2ways.pkl'))
        return self._human_data

    def human_acc(self, time=False):
        df = self.human_data_timing if time else self.human_data
        return df.pivot_table(index='id', columns='stim_dur', values='acc')

    # @property
    # def images(self):
    #     if not hasattr(self, '_images'):
    #         ims = []
    #         for idd in self.meta.id.values:
    #             im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
    #             im = skimage.img_as_float(im)
    #             ims.append(im)
    #         self._images = np.array(ims)
    #     return self._images

    @property
    def images(self):
        if not hasattr(self, '_images'):
            try:
                self._images = np.load(self.datapath('images'))
            except:
                images = []
                for idd in self.meta.id.values:
                    im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
                    im = skimage.img_as_float(im)
                    im = skimage.transform.resize(im, (256,256))
                    im = np.dstack([im,im,im])
                    images.append(im)
                self._images = np.array(images)
                # np.save(self.datapath('images'), self._images)
        return self._images

    @property
    def tokens(self):
        if not hasattr(self, '_tokens'):
            tokens = []
            for idd in self.meta.obj.unique():
                im = skimage.io.imread(self.home('imageset/tokens', idd + '.png'))
                im = skimage.color.gray2rgb(im)
                im = skimage.transform.resize(im, (256,256))
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

    def model(self, name='alexnet', layer='pool5'):
        if name == 'alexnet' and layer == 'pool5':
            model_feats = np.load(self.datapath('model/alexnet/pool5'))

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


class Objectome24s10(Objectome):

    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images': 'streams/objectome/imageset/ims24s10.npy',
            'sel240': 'streams/objectome/sel240.pkl',
            'metrics240': 'streams/objectome/metrics240.pkl'}


    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            meta = super(Objectome24s10, self).meta
            sel = pandas.read_pickle(self.datapath('sel240'))
            self._meta = meta.loc[sel]
        return self._meta

    @property
    def images(self):
        if not hasattr(self, '_images'):
            try:
                self._images = np.load(self.datapath('images'))
            except:
                images = []
                for im in super(Objectome24s10, self).images:
                    im = skimage.transform.resize(im, (224,224))
                    im = np.dstack([im,im,im])
                    images.append(im)
                self._images = np.array(images)
                np.save(self.datapath('images'), self._images)
        return self._images

    def human_data(self, kind='I2_accuracy'):
        data = pandas.read_pickle(self.datapath('metrics240'))
        return data[kind]