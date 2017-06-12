from __future__ import division, print_function, absolute_import
import sys, os, hashlib, cPickle, tempfile, zipfile, glob
from collections import OrderedDict

import numpy as np
import pandas
import tables
import boto3
import tqdm
import skimage, skimage.io, skimage.transform

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

    def datapath(self, handle, prefix=None):
        data = self.DATA[handle]
        if isinstance(data, tuple):
            s3_path, sha1, local_path = data
            local_path = os.path.join(local_path, s3_path)
            # if local_path is None:
            #     local_path = s3_path.replace(self.COLL + '/' + self.name + '/', '', 1)
        else:
            local_path = data.replace(self.COLL + '/' + self.name + '/', '', 1)
        if prefix is not None:
            local_path = '/'.join([prefix, local_path])
        return self.home(local_path)

    def fetch(self):
        return
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


class HvM(Dataset):

    DATA = {'meta': 'streams/hvm/meta.pkl',

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
            'model/hmo/layer1': 'streams/hvm/model/hmo_layer1feats.npy',
            'model/hmo/layer2': 'streams/hvm/model/hmo_layer2feats.npy',
            'model/hmo/layer3': 'streams/hvm/model/hmo_layer3feats.npy',
            'model/hmo/top': 'streams/hvm/model/hmo_topfeats.npy',
            }

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
        self.name = 'hvm'
        self.var = var

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
            if self.var is not None:
                self._meta = self._meta[self._meta['var'] == self.var]
        return self._meta

    @property
    def images(self):
        if not hasattr(self, '_images'):
            ims = []
            for idd in self.meta.id.values:
                im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
                im = skimage.img_as_float(im)
                ims.append(im)
            self._images = np.array(ims)
        return self._images

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
        neural_order = f.root.meta.idx2iid[:]
        f.close()
        order = np.array([neural_order.index(n) for n in self.meta.filename])
        self._neural_data = nd[:, order][:, :, self.IT_NEURONS]
        return self._neural_data

    def model(self, name='alexnet', layer='pool5'):
        if name == 'alexnet' and layer == 'pool5':
            model_feats = np.load(self.datapath('model/alexnet/pool5'))
        elif name == 'hmo':
            model_feats = cPickle.load(open(self.datapath('model/hmo/' + layer)))
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
            model_feats = cPickle.load(open(self.home(path)))
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
    import cPickle
    cPickle.dump(m, open('/home/qbilius/mh17/.skdata/HvMWithDiscfade/images_tf3/meta.pkl', 'wb'))


if __name__ == '__main__':
    HvM()._upload('meta.pkl')