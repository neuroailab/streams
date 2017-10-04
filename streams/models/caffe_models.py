"""
Caffe model interface for ImageNet-trained models via a netwok
"""
from __future__ import absolute_import, division, print_function

import os, glob
from collections import OrderedDict

import numpy as np
import skimage
import tqdm

from .networking import Client, Server
import streams.models.base as base


def _import_caffe(device=0, mode='gpu'):
    # Suppress GLOG output for python bindings
    GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
    os.environ['GLOG_minloglevel'] = '5'

    if 'CAFFE_ROOT' not in os.environ:
        raise Exception('You must set an environment variable "CAFFE_ROOT" '
                    'that points to where caffe source is.')

    import caffe
    caffe.set_device(device)
    # print(mode)
    if mode == 'gpu':
        caffe.set_mode_gpu()
    elif mode == 'cpu':
        caffe.set_mode_cpu()
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    # Turn GLOG output back on for subprocess calls
    if GLOG_minloglevel is None:
        del os.environ['GLOG_minloglevel']
    else:
        os.environ['GLOG_minloglevel'] = GLOG_minloglevel
    return caffe


class Model(base.Model):

    def _get_mean(self, model_path):
        meanf = os.path.join(model_path, '*mean*.binaryproto')
        meanf = glob.glob(meanf)
        if len(meanf) > 0:
            data = open(meanf[0], 'rb').read()
            blob = self.caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(data)
            mn = np.array(self.caffe.io.blobproto_to_array(blob))[0]
        else:
            meanf = os.path.join(os.environ['CAFFE_ROOT'], 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
            mn = np.load(meanf)
        mn = mn.mean(1).mean(1)
        return mn

    def create_model(self, model_name):
        self.caffe = _import_caffe()
        # model_path = os.path.join(os.environ['CAFFE_ROOT'], 'models', model_name)
        model_path = os.path.join('/braintree/home/qbilius/models/caffe', model_name)

        path = os.path.join(model_path, '*deploy*.prototxt')
        model_file = sorted(glob.glob(path))[0]

        path = os.path.join(model_path, '*.caffemodel')
        weight_file = sorted(glob.glob(path))[0]

        self.net = self.caffe.Net(str(model_file), str(weight_file), self.caffe.TEST)
        self.net.model_name = model_name
        new_shape = (1,) + self.net.blobs['data'].data.shape[1:]
        self.net.blobs['data'].reshape(*new_shape)
        self.set_transformer()

    def set_transformer(self, mn=None):
        if mn is None:
            # model_path = os.path.join(os.environ['CAFFE_ROOT'], 'models', self.net.model_name)
            model_path = os.path.join('/braintree/home/qbilius/models/caffe', self.net.model_name)
            mn = self._get_mean(model_path)

        transformer = self.caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', mn)  # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.net.transformer = transformer

    def forward(self, im):
        self.net.blobs['data'].data[...] = self.net.transformer.preprocess('data', im)
        self.net.forward()

    def _get_features(self, im):
        self.forward(im)
        resps = OrderedDict()
        for layer in self.layers:
            resps[layer] = self.net.blobs[layer].data.copy()
        return resps

    def get_layer_names(self):
        return self.net.blobs.keys()

    def confidence(self, im, topn=1):
        """
        Return model confidence for a given model
        """
        self.forward(im)
        pred = np.squeeze(self.net.blobs['prob'].data)
        if topn is not None:
            pred.sort()
            return pred[::-1][:topn]
        else:
            return pred

    def predict(self, im, topn=5):
        """
        Return topn predicted class information
        """
        labels = self._get_labels()
        pred = self.confidence(im, topn=None)

        classno = np.argsort(pred)[::-1][:topn]
        out = []
        for n in classno:
            d = {'classno': n,
                 'synset': labels[n][0],
                 'label': labels[n][1],
                 'confidence': pred[n]}
            out.append(d)
        return out

    def _get_labels(self):
        synset_file = os.path.join(os.environ['CAFFE_ROOT'],
                                   'data/ilsvrc12/synset_words.txt')
        try:
            with open(synset_file) as f:
                lines = f.readlines()
        except:
            raise Exception('ERROR: synset file with labels not found.\n'
                            'Tried: %s' % synset_file)
        out = []
        for line in lines:
            line = line.strip('\n\r')
            out.append((line[:9], line[10:]))
        return out

    def cam(self, data, topn=5):
        """
        Access to Class Activation Mapping layers

        http://cnnlocalization.csail.mit.edu/
        """
        self.forward()
        prob = self.net.blobs['prob'].data.ravel()
        inds = np.argsort(prob)[::-1]
        act = np.squeeze(self.net.blobs['CAM_conv'].data)
        weights = self.net.params['CAM_fc'][0].data

        r = np.zeros((act.shape[1], act.shape[2]))
        for ind in inds[:topn]:
            r += np.dot(np.rollaxis(act,0,3), weights[ind])
        r = skimage.transform.resize(r, (data.shape[0], data.shape[1]))
        r = r > np.ptp(r) / 2
        return r

    def pca(self, ims):
        resps = OrderedDict([(layer, []) for layer in self.layers])
        for im in tqdm.tqdm(ims, desc='images'):
            # im = self.preprocess(im)
            self.forward(im)
            for vals, layer in zip(resps.values(), self.layers):
                vals.append(self.net.blobs[layer].data.copy())

        for layer, resp in tqdm.tqdm(resps.items(), desc='image PCA'):
            resps[layer] = self.feat_sel(resp, layer)
            # resps[layer] = np.array(resp)
        return resps


class ModelManager(object):

    def __init__(self):
        self._models = {}
        self.current_model = None

    def set_model(self, model_name, mode='gpu', device=0, **model_kwargs):
        if model_name not in self._models:
            self._models[model_name] = Model(model_name=model_name, **model_kwargs)
            self.current_model = self._models[model_name]
        else:
            self.current_model = self._models[model_name]
            self.current_model.layers = model_kwargs.get('layers', self.current_model.layers)
            self.current_model.feat_sel = model_kwargs.get('feat_sel', self.current_model.feat_sel)

        self.caffe = _import_caffe(device=device, mode=mode)
        self.current_model.set_transformer(mn=None)

    def __getattr__(self, attr):
        if hasattr(self.current_model, attr):
            return getattr(self.current_model, attr)
        elif hasattr(self, attr):
            return getattr(self, attr)
        else:
            return self.__dict__[attr]


class MServer(ModelManager, Server):

    def __init__(self, *args, **kwargs):
        ModelManager.__init__(self)
        Server.__init__(self, *args, **kwargs)


def MClient(host='localhost', port=22777,
            model_name='vgg-19', layers='fc7', feat_sel=None, mode='gpu', device=0):
    """
    """
    if host is None:  # local; don't use sockets
        client = ModelManager()
    else:  # remote communication via sockets
        client = Client(host=host, port=port)
    client.set_model(model_name=model_name, layers=layers,
                    feat_sel=feat_sel, mode=mode, device=device)
    return client
