import os, glob
from collections import OrderedDict

import tqdm
import numpy as np
import pandas
import skimage, skimage.io, skimage.transform

from streams.envs.dataset import Dataset, DATA_HOME
from streams.utils import lazy_property


COCO_PATH = '/braintree/data2/active/common/coco/'


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
    DATA = {'meta': 'streams/coco10/meta.pkl'}

    def __init__(self):
        self.name = 'coco10'

    def create_meta(self):

        def obj_area(df, x):
            sel = df.loc[x.index]
            out = sel[sel.category == name]
            return out.area.max()

        recs = []
        self.db
        for name in tqdm.tqdm(self.NAMES):
            df = self.db[(self.db.category == name) & (self.db.iscrowd == 0)].copy()
            assert len(df) >= 1000
            ims = self.db[self.db.image_id.isin(df.image_id)]
            df['obj_area'] = ims.groupby('image_id').area.transform(lambda x: obj_area(df, x))
            df['n_objs'] = ims.groupby('image_id').category.transform(len)
            df['n_rep_obj'] = ims.groupby('image_id').category.transform(lambda x: (x == name).sum())
            df = df.sort_values(by=['area', 'n_objs', 'n_rep_obj'], ascending=[False, True, True])
            recs.append(df.iloc[:1000])
        recs = pandas.concat(recs, ignore_index=True)
        recs.to_pickle(self.datapath('meta'))

    @lazy_property
    def meta(self):
        meta = pandas.read_pickle(self.datapath('meta'))
        return meta

    def images(self, size=224):
        ims = []
        for idx, row in tqdm.tqdm(self.meta.iterrows()):
            im = skimage.io.imread(os.path.join(COCO_PATH, row.kind + '2014', row.file_name))
            im = skimage.color.gray2rgb(im)
            im = skimage.transform.resize(im, (size,size))
            im = skimage.img_as_float(im)
            ims.append(im)
        return np.array(ims)