import os, argparse
from collections import OrderedDict

import numpy as np
import tqdm
import tensorflow as tf
import skimage.transform

import streams.models.base as base


class Model(base.Model):

    # KNOWN_MODELS = {'basenet6': ('basenet6', 'BaseNet6', (224, 224)),
    #                 'basenet11': ('basenet11', 'BaseNet11', (224, 224))}

    def __init__(self, batch_size=256, model_func=None, path=None, *args, **kwargs):
        self.batch_size = batch_size
        self.model_func = model_func
        self.path = path
        super(Model, self).__init__(*args, **kwargs)

    def create_model(self, model_name):
        # specs = self.KNOWN_MODELS[model_name]
        self.shape = (224,224) #specs[2]
        self.placeholder = tf.placeholder(shape=(256,224,224,3), dtype=tf.float32)

        if self.path is None:
            self.path = '/braintree/home/qbilius/models/{}/model.ckpt-100000'.format(model_name)

        # if model_name == 'basenet6':
        #     prefix = 'hvm_nfit_and_corr'
        #     self.sess = tf.Session()
        #     input_map = {'{}/fifo_queue_DequeueMany:0'.format(prefix): self.placeholder}
        #     new_saver = tf.train.import_meta_graph(path + '.meta', input_map=input_map)
        #     new_saver.restore(self.sess, path)
        #     graph = tf.get_default_graph()
        #     self.targets = {l: graph.get_tensor_by_name('{}/{}/output:0'.format(prefix, l)) for l in self.layers}
        if self.model_func is None:
            if model_name == 'basenet6':
                self.model_func = basenet6
            elif model_name == 'basenet11':
                self.model_func = basenet11
            else:
                raise ValueError

        self.model_func(self.placeholder)

        saver = tf.train.Saver()  #var_list=restore_vars())
        self.sess = tf.Session()
        saver.restore(self.sess, save_path=self.path)
        graph = tf.get_default_graph()
        self.targets = {l: graph.get_tensor_by_name('{}/output:0'.format(l)) for l in self.layers}

        # path = getattr(keras.applications, specs[0])
        # net = getattr(path, specs[1])(weights='imagenet')
        # outputs = [net.get_layer(layer).output for layer in self.layers]
        # self.net = keras.models.Model(inputs=net.input, outputs=outputs)

    def _get_features(self, ims):
        if ims[0].shape[0] != 224:
            ims = self.preprocess(ims)
        n_batches = (len(ims) - 1) // self.batch_size  # number of FULL batches
        out = [self.sess.run(self.targets, feed_dict={self.placeholder: ims[self.batch_size*i:self.batch_size*(i+1)]}) for i in range(n_batches)]
        final = self.sess.run(self.targets, feed_dict={self.placeholder: ims[-self.batch_size:]})
        final = {k:f[-(len(ims) - self.batch_size * n_batches):] for k,f in final.items()}
        out.append(final)
        resps = OrderedDict()
        for layer in self.layers:
            resps[layer] = np.row_stack([o[layer] for o in out])
            # feats = feats.reshape((len(feats), -1))
            # if
            # pca[layer] = PCA(n_components=1000)
            # pca[layer].fit(feats)
            # out[layer] = pca[layer].transform(feats)

            # out = run_images(ims)

        # im = self.preprocess(im)
        # out = self.net.predict(im)
        # resps = OrderedDict((layer, o) for layer, o in zip(self.layers, out))
        return resps

    def get_layer_names(self):
        return ['V1', 'V2', 'V4', 'pIT', 'aIT']

    def preprocess(self, ims):
        ims = np.array([skimage.transform.resize(im, self.shape) for im in ims])
        return ims

    # @property
    # def pca_ims(self):
    #     if not hasattr(self, '_pca_ims'):
    #         self._pca_ims = np.load('../hvm/imagenet_ims.npy')
    #     return self._pca_ims

    # def pca(self, ims):
    #     # resps = OrderedDict([(layer, []) for layer in self.layers])
    #     ims = self.preprocess(ims)
    #     resps = self._get_features(ims)
    #     # for vals, o in zip(resps.values(), out):
    #     #     vals.append(o)

    #     for layer, resp in tqdm.tqdm(resps.items(), desc='image PCA'):
    #         resps[layer] = self.feat_sel(resp, layer)
    #         # resps[layer] = np.array(resp)
    #     return resps


def basenet6(inputs, reuse=None):

    conv_kwargs = {'padding': 'same',
                    'activation': tf.nn.elu,
                    'kernel_initializer': tf.contrib.layers.xavier_initializer(),
                    'kernel_regularizer': tf.contrib.layers.l2_regularizer(.0005),
                    'bias_regularizer': tf.contrib.layers.l2_regularizer(.0005),
                    'reuse': reuse}
    pool_kwargs = {'padding': 'same'}

    with tf.variable_scope('V1', reuse=reuse):
        x = tf.layers.conv2d(inputs, 64, (7, 7), strides=2, **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('V2'):
        x = tf.layers.conv2d(x, 128, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('V4'):
        x = tf.layers.conv2d(x, 256, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('pIT'):
        x = tf.layers.conv2d(x, 256, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('aIT'):
        x = tf.layers.conv2d(x, 512, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('ds'):
        # x = tf.stop_gradient(x)
        x = tf.layers.conv2d(x, 1000, (1, 1), **conv_kwargs)
        x = tf.layers.average_pooling2d(x, x.shape.as_list()[1], 1, padding='valid')
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        x = tf.identity(x, name='output')

    return x


def basenet11(inputs, train=False, reuse=None):

    conv_kwargs = {'padding': 'same',
                    'activation': tf.nn.elu,
                    'kernel_initializer': tf.contrib.layers.xavier_initializer(),
                    'kernel_regularizer': tf.contrib.layers.l2_regularizer(.0005),
                    'bias_regularizer': tf.contrib.layers.l2_regularizer(.0005),
                    'reuse': reuse}
    pool_kwargs = {'padding': 'same'}

    with tf.variable_scope('V1', reuse=reuse):
        x = tf.layers.conv2d(inputs, 32, (7, 7), strides=2, **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('V2'):
        x = tf.layers.conv2d(x, 64, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('V4'):
        with tf.variable_scope('V4/a'):
            x = tf.layers.conv2d(x, 256, (3, 3), **conv_kwargs)
        with tf.variable_scope('V4/b'):
            x = tf.layers.conv2d(x, 256, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('pIT'):
        if train:
            x = tf.nn.dropout(x, keep_prob=.25)
        with tf.variable_scope('pIT/a'):
            x = tf.layers.conv2d(x, 512, (3, 3), **conv_kwargs)
        with tf.variable_scope('pIT/b'):
            x = tf.layers.conv2d(x, 512, (3, 3), **conv_kwargs)
        if train:
            x = tf.nn.dropout(x, keep_prob=.25)
        with tf.variable_scope('pIT/c'):
            x = tf.layers.conv2d(x, 512, (3, 3), **conv_kwargs)
        with tf.variable_scope('pIT/d'):
            x = tf.layers.conv2d(x, 512, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('aIT'):
        if train:
            x = tf.nn.dropout(x, keep_prob=.25)
        with tf.variable_scope('aIT/a'):
            x = tf.layers.conv2d(x, 1024, (3, 3), **conv_kwargs)
        with tf.variable_scope('aIT/b'):
            x = tf.layers.conv2d(x, 1024, (3, 3), **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('ds'):
        x = tf.layers.conv2d(x, 1000, (1, 1), **conv_kwargs)
        x = tf.layers.average_pooling2d(x, x.shape.as_list()[1], 1, padding='valid')
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        x = tf.identity(x, name='output')

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', type=str, help='which gpu to use')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
