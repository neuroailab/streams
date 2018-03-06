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
from streams.utils import lazy_property


class Objectome(Dataset):

    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images256': 'streams/objectome/imageset/ims24s100_256.npy',
            'imageset/tfrecords': 'streams/objectome/imageset/images224.tfrecords',
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

    def __init__(self):
        self.name = 'objectome'


class Objectome24s10(Objectome):

    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images224': 'streams/objectome/imageset/ims24s10_224.npy',
            'sel240': 'streams/objectome/sel240.pkl',
            'metrics240': 'streams/objectome/metrics240.pkl'}
    OBJS = Objectome.OBJS

    @lazy_property
    def meta(self):
        meta = super(Objectome24s10, self).meta
        sel = pandas.read_pickle(self.datapath('sel240'))
        return meta.loc[sel]

    def human_data(self, kind='I2_dprime_C'):
        """
        Kind:
        - O1_hitrate, O1_accuracy, O1_dprime, O1_dprime_v2
        - O2_hitrate, O2_accuracy, O2_dprime,
        - I1_hitrate, I1_accuracy, I1_dprime, I1_dprime_C, I1_dprime_v2_C
        - I2_hitrate, I2_accuracy, I2_dprime, I2_dprime_C, I1_dprime_v2

        Rishi: "'v2' means averaging rather than pooling. So O1_dprime_v2 averages over all the distracter bins from O2, rather than pooling over all the trials."
        """
        data = pandas.read_pickle(self.datapath('metrics240'))
        # organized like: metric kind x 10 splits x 2 split halves
        return data[kind]