import os, functools, glob
from collections import OrderedDict

import tqdm
import numpy as np
import pandas
import skimage, skimage.io, skimage.transform

from streams.envs.dataset import Dataset, DATA_HOME


COCO_PATH = '/braintree/data2/active/common/coco/'


def lazy_property(function):
    """
    From: https://danijar.com/structuring-your-tensorflow-models/
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class COCO(Dataset):
    IMAGEDIR = 'images'
    MASKDIR = 'masks'  # phase-scrambled masks
    LABELDIR = 'labels'  # response images

    def __init__(self):
        self.name = 'coco'

    @lazy_property
    def coco(self):
        import pycocotools.coco
        train = pycocotools.coco.COCO('/braintree/data2/active/common/coco/annotations/instances_train2014.json')
        val = pycocotools.coco.COCO('/braintree/data2/active/common/coco/annotations/instances_val2014.json')
        return {'train': train, 'val': val}

    @lazy_property
    def db(self):
        try:
            db = pandas.read_pickle(self.home('cocodb.pkl'))
        except:
            db = self.create_db
        return db

    @lazy_property
    def create_db(self):
        df = []
        for kind in self.coco.keys():
            db = self.coco[kind]
            for im in db.imgs.values():
                annids = db.getAnnIds(imgIds=im['id'])
                for annid in annids:
                    ann = db.anns[annid]
                    cat = db.cats[ann['category_id']]
                    im_meta = [(k,v) for k,v in im.items() if k != 'id']
                    extra = [('kind', kind),
                             ('category', cat['name']),
                             ('supercategory', cat['supercategory'])]
                    df.append(dict(im_meta + list(ann.items()) + extra))

        df = pandas.DataFrame(df)
        if not os.path.isdir(self.home()):
            os.makedirs(self.home())
        df.to_pickle(self.home('cocodb.pkl'))
        return df

    def get_single_cat(self, cat):
        # dfe = df[df.category.isin(self.NAMES)]
        def f(x, cat):
            names = [n for n in self.NAMES if n != cat]
            if all([i not in names for i in x.values]) and cat in x.values:
                return True
            else:
                return False
        imsel = self.db.groupby('image_id').category.aggregate(f, cat=cat)
        imids = imsel.index[imsel]
        return imids

    def get_entry_from_filename(self, filename):
        if filename.split('_')[1].startswith('val'):
            kind = 'val'
        else:
            kind = 'train'
        try:
            entry = self.coco[kind]
        except:
            self._set_dbs()
            entry = self.coco[kind]
        return entry

    def get_meta_from_filename(self, filename, name):
        coco = self.get_entry_from_filename(filename)
        imid = int(filename.split('_')[-1])
        catid = coco.getCatIds(catNms=name)[0]

        return OrderedDict([('id', str(filename)),
                            ('imid', imid),
                            ('category_id', catid),
                            ('category', str(name)),
                            ('supercategory', coco.cats[catid]['supercategory']),
                            # ('filename', os.path.join(self.home(self.SUBDIR, self.IMAGEDIR),
                            #                           filename + '.jpg'))
                            ])


class COCO10(COCO):
    NAMES = ['bear', 'elephant', 'person', 'car', 'dog', 'apple', 'chair', 'airplane', 'bird', 'zebra']
    DATA = {'meta': 'streams/coco/meta.pkl'}

    def create_meta(self):
        recs = [self.db[self.db.category == n].iloc[:200] for n in self.NAMES]
        recs = pandas.concat(recs, ignore_index=True)
        recs.loc[recs.category == 'person', 'category'] = 'face'
        recs.to_pickle(self.datapath('meta'))

    @lazy_property
    def meta(self):
        meta = pandas.read_pickle(self.datapath('meta'))
        return meta

    @lazy_property
    def images(self):
        ims = []
        for idx, row in tqdm.tqdm(self.meta.iterrows()):
            im = skimage.io.imread(os.path.join(COCO_PATH, row.kind + '2014', row.file_name))
            im = skimage.color.gray2rgb(im)
            im = skimage.transform.resize(im, (256,256))
            im = skimage.img_as_float(im)
            ims.append(im)
        return np.array(ims)