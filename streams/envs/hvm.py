from __future__ import division, print_function, absolute_import
import sys, os, hashlib, cPickle, tempfile, zipfile, glob
from collections import OrderedDict

import numpy as np
import pandas
import tables
import boto3
import tqdm


DATA_HOME = os.path.abspath(os.path.expanduser(os.environ.get(
                'STREAMS_ROOT', os.path.join('~', '.streams'))))


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


class Dataset(object):

    BUCKET = 'dicarlocox-datasets'
    COLL = 'streams'

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    def datapath(self, handle):
        data = self.DATA[handle]
        if isinstance(data, tuple):
            s3_path, sha1, local_path = data
            if local_path is None:
                local_path = s3_path.replace(self.COLL + '/' + self.name + '/', '', 1)
        else:
            local_path = data.replace(self.COLL + '/' + self.name + '/', '', 1)
        return self.home(local_path)

    def fetch(self):
        if not os.path.exists(self.home()):
            os.makedirs(self.home())

        session = boto3.Session()
        client = session.client('s3')

        for data in self.DATA.values():
            if isinstance(data, tuple):
                s3_path, sha1, local_path = data
                if local_path is None:
                    local_path = s3_path.replace(self.COLL + '/' + self.name + '/', '', 1)
            else:
                local_path = data.replace(self.COLL + '/' + self.name + '/', '', 1)
                s3_path = data
                sha1 = None

            local_path = self.home(local_path)
            if not os.path.exists(local_path):
                # rel_path = os.path.relpath(local_path, DATA_HOME)
                # s3_path = os.path.join(self.COLL, rel_path)
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                client.download_file(self.BUCKET, s3_path, local_path)
                if sha1 is not None:
                    with open(local_path) as f:
                        if sha1 != hashlib.sha1(f.read()).hexdigest():
                            raise IOError("File '{}': SHA-1 does not match.".format(filename))

    def upload(self, pattern='*'):
        session = boto3.Session()
        client = session.client('s3')

        uploads = []
        for root, dirs, filenames in os.walk(self.home()):
            for filename in glob.glob(os.path.join(root, pattern)):
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, DATA_HOME)
                s3_path = os.path.join(self.COLL, rel_path)
                try:
                    client.head_object(Bucket=self.BUCKET, Key=s3_path)
                except:
                    uploads.append((local_path, s3_path))

        if len(uploads) > 0:
            text = []
            for local_path, s3_path in uploads:
                with open(local_path) as f:
                    sha1 = hashlib.sha1(f.read()).hexdigest()
                    rec = '    {} (sha-1: {})'.format(s3_path, sha1)
                text.append(rec)
            text = ['Will upload:'] + text + ['Proceed? ']
            proceed = raw_input('\n'.join(text))
            if proceed == 'y':
                for local_path, s3_path in tqdm.tqdm(uploads):
                    client.upload_file(local_path, self.BUCKET, s3_path)
        else:
            print('nothing found to upload')

    def _upload(self, filename):
        session = boto3.Session()
        client = session.client('s3')
        local_path = self.home(filename)
        rel_path = os.path.relpath(local_path, DATA_HOME)
        s3_path = os.path.join(self.COLL, rel_path)
        client.upload_file(local_path, self.BUCKET, s3_path)


    # def move(self, old_path, new_path):
    #     client.copy_object(Bucket=self.BUCKET, Key=new_path,
    #                         CopySource=self.BUCKET + '/' + old_path)
    #     client.delete_object(Bucket=self.BUCKET, Key=new_path)




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


class HvM6Neural(Dataset):

    DATA = {'meta': ('streams/hvm/imageset/meta.pkl', None, 'hvm/imageset/meta.pkl'),
            'averaged': ('Chabo_Tito_20110907_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                         'a95c797b0b2eef56d431c8ccca4c160143a65357', None),
            'temporal': ('Chabo_Tito_20140307_Var06a_pooled_P58.trim.wh.evoked.repr.h5',
                         '8ed6ec266fd0104368121aa742038f04681f7231', None),
            'temporal_raw': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.raw.d.repr.h5',
            'temporal_evoked': 'Chabo_Tito_20140307_Var06a_pooled_P58.trim.evoked.repr.h5'}


    IT_NEURONS = range(0, 11) + range(45, 85) + range(121, 163) + range(171, 211) + range(221, 256)
    TIMEPOINTS = OrderedDict(zip(range(-90, 300, 10), range(39)))

    def __init__(self):
        self.name = 'hvm/neural'

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta[640 + 2560:]

    def neural_data(self, timepoint=None):
        """
        Format: (time bins, reps, images, sites)
        """
        if timepoint is None:
            path = self.datapath('averaged')
        else:
            path = self.datapath('temporal_raw')
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
            model_feats = cPickle.load(open(self.home(path)))
        return model_feats


class HvM6IT(Dataset):

    def __init__(self, order=range(2560)):
        self.name = 'hvm'
        self.order = np.array(order)
        self._imageset = HvMImageSet()
        # self.behav = HvMBehav()
        self._neural = HvM6Neural()
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

    def neural(self, timepoint=None):
        d = self._neural.neural_data(timepoint=timepoint)[:, self.order]
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
    import cPickle
    cPickle.dump(m, open('/home/qbilius/mh17/.skdata/HvMWithDiscfade/images_tf3/meta.pkl', 'wb'))


if __name__ == '__main__':
    HvM6Neural().neural_data()