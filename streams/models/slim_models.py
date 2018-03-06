import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from preprocessing import preprocessing_factory
from nets import nets_factory

import streams.models.base as base

MODEL_PATH = os.path.join(os.environ['STORE'], 'models', 'slim')


class Model(base.Model):

    def __init__(self, batch_size=256, path=None, targets=None, labels_offset=0, *args, **kwargs):
        self.batch_size = batch_size
        self.targets = targets
        self.path = path
        self.labels_offset = labels_offset
        if self.labels_offset > 0:
            print('NOTE that output labels will be incorrectly offset by {}'.format(self.labels_offset))
        super(Model, self).__init__(*args, **kwargs)

    def create_model(self, model_name):
        if self.path is None:
            self.path = os.path.join(MODEL_PATH, self.model_name, self.model_name + '.ckpt')
        self.model_func = nets_factory.get_network_fn(self.model_name,
                                                      num_classes=1001 - self.labels_offset,
                                                      is_training=False)
        preproc_func = preprocessing_factory.get_preprocessing(
                                self.model_name, is_training=False)

        self.placeholder = tf.placeholder(shape=(self.batch_size,224,224,3), dtype=tf.float32)

        # with slim.arg_scope(self.model_func.arg_scope):
        im_size = self.model_func.default_image_size
        ims_batch = tf.map_fn(lambda im: preproc_func(tf.image.convert_image_dtype(im, dtype=tf.uint8), im_size, im_size),
                              self.placeholder, dtype=tf.float32)
        logits, self.endpoints = self.model_func(ims_batch) #, num_classes=1001, is_training)

        restorer = tf.train.Saver()
        self.sess = tf.Session()
        restorer.restore(self.sess, save_path=self.path)
        # restore = slim.assign_from_checkpoint_fc(self.path,
        #     slim.get_model_variables(self.model_name))
        # restore(self.sess)

        # graph = tf.get_default_graph()
        if self.targets is None:
            self.targets = {l: self.endpoints[self.model_name + '/' + l] for l in self.layers}

    # def preprocess(self, ims):
    #     im_size = self.model_func.default_image_size
    #     return tf.stack([self.preproc_func(im, im_size, im_size) for im in ims])

    def _get_features(self, ims):
        if ims[0].shape[1] != self.model_func.default_image_size:
            import ipdb; ipdb.set_trace()
        n_batches = (len(ims) - 1) // self.batch_size  # number of FULL batches
        output = []
        for i in range(n_batches):
            ims_batch = ims[self.batch_size*i: self.batch_size*(i+1)]
            out = self.sess.run(self.targets, feed_dict={self.placeholder: ims_batch})
            output.append(out)
        final = self.sess.run(self.targets, feed_dict={self.placeholder: ims[-self.batch_size:]})
        final = {k:f[-(len(ims) - self.batch_size * n_batches):] for k,f in final.items()}
        output.append(final)

        resps = OrderedDict()
        for layer in self.layers:
            resps[layer] = np.row_stack([o[layer] for o in output])

        return resps

    def get_layer_names(self):
        return list(self.endpoints.keys())

