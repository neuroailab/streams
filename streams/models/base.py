from collections import OrderedDict
import numpy as np


class Model(object):

    def __init__(self, model_name='vgg-19', layers=None, feat_sel=None):
        """
        Generic model interface
        """
        self.model_name = model_name
        self.layers = layers
        self.create_model(model_name)
        self.feat_sel = feat_sel

    @property
    def feat_sel(self):
        return self._feat_sel

    @feat_sel.setter
    def feat_sel(self, feat_sel):
        self._feat_sel = feat_sel
        if self._feat_sel is not None:
            if self._feat_sel.__class__.__name__ == 'PCASelector':
                # if len(self.layers) != 1:
                #     raise ValueError('PCA feature selector can only work with a '
                #                      'single layer at a time but got {} layers '
                #                      'instead'.format(len(self.layers)))
                self._feat_sel._pca_fit(self)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        if not isinstance(layers, (tuple, list, np.ndarray)):
            layers = [layers]
        self._layers = layers
        print('Using layers {}'.format(self._layers))

    def create_model(self, model_name):
        raise NotImplementedError

    def _get_features(self, ims):
        raise NotImplementedError

    def get_features(self, ims):
        resps = self._get_features(ims)

        if self.feat_sel is not None:
            feats = np.zeros((len(self.layers), len(ims), self.feat_sel.nfeats))
            for i, (layer, resp) in enumerate(resps.items()):
                feats[i] = self.feat_sel(resp, layer)
        else:
            if len(self.layers) == 1:
                feats = resps[self.layers[0]]
            else:
                feats = OrderedDict([(l, resps[l]) for l in self.layers])

        return feats

    def get_layer_names(self):
        raise NotImplementedError