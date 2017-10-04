import os, argparse
from collections import OrderedDict

import numpy as np
import tqdm
import keras
import keras.preprocessing.image

import streams.models.base as base


class Model(base.Model):

    KNOWN_MODELS = {'vgg-16': ('vgg16', 'VGG16', (224, 224)),
                    'vgg-19': ('vgg19', 'VGG19', (224, 224)),
                    'resnet-50': ('resnet50', 'ResNet50', (224, 224)),
                    'inception_v3': ('inception_v3', 'InceptionV3', (299, 299)),
                    'xception': ('xception', 'Xception', (299, 299)),
                    'mobilenet': ('mobilenet', 'MobileNet', (224, 224))}

    def create_model(self, model_name):
        specs = self.KNOWN_MODELS[model_name]
        path = getattr(keras.applications, specs[0])
        net = getattr(path, specs[1])(weights='imagenet')
        outputs = [net.get_layer(layer).output for layer in self.layers]
        self.net = keras.models.Model(inputs=net.input, outputs=outputs)

    def preprocess(self, im):
        specs = self.KNOWN_MODELS[self.model_name]
        im = keras.preprocessing.image.array_to_img(im)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        hw_tuple = (specs[2][1], specs[2][0])
        if im.size != hw_tuple:
            im = im.resize(hw_tuple)
        # im = keras.preprocessing.image.load_img(im, target_size=specs[2])
        im = keras.preprocessing.image.img_to_array(im)
        im = np.expand_dims(im, axis=0)
        path = getattr(keras.applications, specs[0])
        im = getattr(path, 'preprocess_input')(im)
        return im

    def _get_features(self, ims):
        ims_preproc = np.concatenate([self.preprocess(im) for im in ims], axis=0)
        out = self.net.predict(ims_preproc)
        # specs = self.KNOWN_MODELS[self.model_name]
        # path = getattr(keras.applications, specs[0])
        # ims = getattr(path, 'preprocess_input')(ims)
        # out = self.net.predict(ims)

        if len(self.layers) == 1:
            resps = {self.layers[0]: out}
        else:
            resps = OrderedDict((layer, o) for layer, o in zip(self.layers, out))
        return resps

    def get_layer_names(self):
        return self.net.layers

    def pca(self, ims):
        resps = OrderedDict([(layer, []) for layer in self.layers])
        for im in ims:
            im = self.preprocess(im)
            out = self.net.predict(im)
            for vals, o in zip(resps.values(), out):
                vals.append(o)

        for layer, resp in tqdm.tqdm(resps.items(), desc='image PCA'):
            resps[layer] = self.feat_sel(resp, layer)
            # resps[layer] = np.array(resp)
        return resps



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', type=str, help='which gpu to use')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
