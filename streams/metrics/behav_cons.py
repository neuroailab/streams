from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats
import pandas
import tqdm

from sklearn.model_selection import StratifiedKFold

from streams import utils
from streams.metrics.classifiers import MatchToSampleClassifier
from streams.envs import objectome


def internal_cons(df, kind='dprime', time=False, niter=100,
                        corr_method='pearson', cons_kind='i1'):
    """
    Spearman-Brown corrected split-half behavioral consistency.

    :Kwargs:
        - kind (dprime or acc default: dprime)
            Compute consistency using accuracy or d'
        - time (bool, default: False)
            Over time or using the 70-170 time window.
        - niter (int, default: 100)
            How many times to split in half.
        - cons_kind (i1, or i2, default: i1)
            The kind of consistency.
    :Returns:
        `pandas.DataFrame` with split number and Spearman-Brown corrected consistency
    """
    index = 'id' if cons_kind == 'i1' else 'uuid'
    groupby = [index, 'obj', 'distractor', 'imgno']
    if time:
        groupby += ['stim_dur']
    sdf = utils.splithalf(df, groupby=groupby)

    if kind in ["d'", 'dprime']:
        groups = ['split', 'iterno']
        if time: groups += ['stim_dur']
        sdf, _ = _human_dprime_i1(sdf, groups=groups)
        kind = "d'"
    else:
        kind = 'acc'
    cols = ['split', 'iterno']
    if time:
        cols += ['stim_dur']

    pv_all = sdf.pivot_table(index=index, columns=cols, values=kind)
    if corr_method == 'pearson':
        r_all = pv_all[0].corrwith(pv_all[1]).reset_index()
        r_all.rename(columns={0: 'r'}, inplace=True)
    else:
        r_all = []
        for col in pv_all[0]:
            corr = pv_all[0][col].corr(pv_all[1][col], method='spearman')
            if pv_all.columns.nlevels > 2:
                r_all.append(list(col) + [corr])
            else:
                r_all.append([col, corr])
        r_all = pandas.DataFrame(r_all, columns=['iterno', 'r'])

    r_all.r = utils.spearman_brown(r_all.r)
    return r_all


def i1(human_data, model_conf, kind='dprime', time=None, niter=10,
                   corr_method='pearson'):
    if not isinstance(model_conf, pandas.Series):
        model_conf = pandas.Series(model_conf, index=self.meta['id'])
    ic = internal_cons(kind=kind, time=time,
                        niter=niter, corr_method=corr_method,
                        cons_kind='i1')

    if kind == 'acc':
        pv = df.pivot_table(index='id', columns='stim_dur', values='acc')
        groups = ['stim_dur'] if time else []
    elif kind in ["d'", 'dprime']:
        kind = "d'"
        groups = ['stim_dur'] if time else []
        human, _ = _human_dprime_i1(human_data, groups=groups, ikind='i1')
        pv = human.pivot_table(index='id', columns=groups, values=kind)
    else:
        raise ValueError("'kind' %s not recognized." % kind)

    pv = human.pivot_table(index='id', columns=groups, values=kind)
    if len(groups) == 0:
        if isinstance(pv, pandas.Series):
            r = pv.corr(model_conf, method=corr_method)
        else:
            import ipdb; ipdb.set_trace()  #TODO: why is this else here?
            r = pv[pv.columns[0]].corr(model_conf, method=corr_method)
        r = pandas.DataFrame(r / np.sqrt(internal_cons.r))
        r['iterno'] = ic.iterno
    else:
        corr_acc = []
        for dur in pv:
            r = pv[dur].corr(model_conf, method=corr_method)
            relh = ic[ic.stim_dur == dur]
            r = pandas.DataFrame(r / np.sqrt(relh.r))
            r['iterno'] = relh.iterno
            r['stim_dur'] = dur
            corr_acc.append(r)
        r = pandas.concat(corr_acc, ignore_index=True)
    return r


def _human_dprime_i1(df, ikind='i1', groups=[], ceiling=5):
    # I1 hits are the average accuracy for each image (that is, each object & imgno pair). uuid is here for convenience only
    # groups = ['obj', 'imgno', 'uuid']
    cons_gr = 'id' if ikind == 'i1' else 'uuid'
    hits_gr = ['obj', 'imgno'] + [cons_gr] + groups
    # if 'stim_dur' in df: groups += ['stim_dur']
    # hits_gr += groups
    hits = df.groupby(hits_gr).acc.mean()

    distr = df.groupby(['obj'] + groups).acc.mean()
    for obj in df.obj.unique():
        if len(groups) > 0:
            distr[obj] = 1 - df[df.distractor == obj].groupby(groups).acc.mean()
        else:
            distr[obj] = 1 - df[df.distractor == obj].acc.mean()

    if len(groups) > 0:
        def f(x):
            sel = tuple([x.obj] + [x[n] for n in x.index if n in groups])
            return distr[sel]
    else:
        def f(x):
            return distr.loc[x.obj]

    fas_tmp = hits.reset_index().apply(f, axis='columns')
    fas = hits.copy()
    fas[:] = fas_tmp.values

    return _dprime(hits, fas, ceiling=ceiling)


def _human_dprime_i2(df, ceiling=5):
    """
    NOTE: NOT WORKING!!
    """
    raise NotImplementedError
    # I2 hits are the average accuracy for each image (that is, each object & imgno pair) for each distractor.
    groups = ['obj', 'distractor', 'imgno', 'uuid']
    if 'stim_dur' in df: groups += ['stim_dur']
    hits = df.groupby(groups).acc.mean()

    # I2 false alarms are one minus the average accuracy for images that have the same object & distractor pair but the correct answer is distractor
    groups = ['distractor', 'obj']
    if 'stim_dur' in df: groups += ['stim_dur']
    distr = df.groupby(groups).acc.mean()
    if 'stim_dur' in df:
        f = lambda x: distr[(x.name[1], x.name[0], x.name[2])]
    else:
        f = lambda x: distr[(x.name[1], x.name[0])]

    groups = ['obj', 'distractor']
    if 'stim_dur' in df: groups += ['stim_dur']
    fas_tmp = 1 - hits.reset_index().groupby(groups).acc.transform(f)
    fas = hits.copy()
    fas[:] = fas_tmp.values

    dprime = hits.apply(scipy.stats.norm.ppf) - fas.apply(scipy.stats.norm.ppf)
    dprime = dprime.reset_index()
    dprime.rename(columns={'acc': "d'"}, inplace=True)
    dprime.loc[dprime["d'"] > ceiling, "d'"] = ceiling

    c = .5 * (hits.apply(scipy.stats.norm.ppf) + fas.apply(scipy.stats.norm.ppf))
    c = c.reset_index()
    c.rename(columns={'acc': 'bias'}, inplace=True)
    return dprime, c


def _dprime(hits, fas, cap=5):
    # dprime = bangmetric.dprime(hits.values, fas.values, mode='rate', max_value=ceiling)
    # dprime = pandas.Series(dprime, index=hits.index)
    # dprime = dprime.reset_index()
    # dprime.rename(columns={0: "d'"}, inplace=True)
    dprime = hits.apply(scipy.stats.norm.ppf) - fas.apply(scipy.stats.norm.ppf)
    dprime = dprime.reset_index()
    dprime.rename(columns={'acc': "d'"}, inplace=True)

    # bias
    c = .5 * (hits.apply(scipy.stats.norm.ppf) + fas.apply(scipy.stats.norm.ppf))
    c = c.reset_index()
    c.rename(columns={'acc': 'bias'}, inplace=True)

    return dprime, c


def human_acc(df):
    return df.pivot_table(index='id', columns='stim_dur', values='acc')


def _to_c(model_feats, labels, order):
    df = pandas.DataFrame(model_feats, index=labels, columns=order)
    out = np.zeros_like(model_feats)
    for (i,j), hit in np.ndenumerate(df.values):
        target = labels[i]
        distr = order[j]
        if target == distr:
            c = np.nan
        else:
            c = np.mean(df.loc[df.index == target, distr])
        out[i,j] = model_feats[i,j] - c

    return out


def o1():
    df = pandas.DataFrame(preds, index=labels[test_idx], columns=order)
    df = df.stack().reset_index()
    df = df.rename(columns={'level_1': 'distr', 0: 'acc'})
    df.obj = df.obj.astype('category', ordered=True, categories=order)
    df.distr = df.distr.astype('category', ordered=True, categories=order)
    acc = df.groupby('obj').acc.mean()


# def hitrate_to_dprime(df, metric, target, distr, imid, value, kind='dprime',
#                       cap=20, normalize=True):

#     if target is None:
#         raise ValueError('target column name must be specified')



    # dfi = df.set_index(indices[metric])
    # out = pandas.Series(np.zeros(len(dfi)), index=dfi.index)






    # for idx, row in dfi.iterrows():
    #     hit_rate = row[value]

    #     if metric == 'o1':  # idx: target
    #         rej = df.loc[(df[target] != idx) & (df[distr] == idx), value]
    #     elif metric == 'o2':  # idx: (target, distr)
    #         rej = df.loc[(df[target] == idx[1]) & (df[distr] == idx[0]), value]
    #     elif metric == 'i1':  # idx: (target, imid)
    #         rej = df.loc[(df[target] != idx[0]) & (df[distr] == idx[0]), value]
    #         import ipdb; ipdb.set_trace()
    #     elif metric == 'i2':  # idx: (target, imid, distr)
    #         rej = df.loc[(df[target] == idx[2]) & (df[distr] == idx[0]), value]

    #     fa_rate = 1 - np.nanmean(rej)

    #     if kind == 'dprime':
    #         dprime = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
    #         out.loc[idx] = np.clip(dprime, -cap, cap)
    #     elif kind == 'acc':
    #         raise NotImplementedError

    # if normalize:
    #     if metric == 'i1':
    #         by = target
    #     elif metric == 'i2':
    #         by = [target, distr]
    #     else:
    #         raise ValueError(f'normalization only defined for i1 and i2, got {metric}')
    #     out[value] -= out.groupby(by)[value].transform(lambda x: x.mean())

    # return out#.reset_index()


def objectome_cons(model_feats, metric='i2n', kind='dprime',
                   target='obj', distr='distr', imid='id', value='acc', cap=20):
    normalize = metric[-1] == 'n'
    metric = metric.rstrip('n')

    obj = objectome.Objectome()
    obj24 = objectome.Objectome24s10()
    if normalize:
        hkind = f'{metric.upper()}_{kind}_C'
    else:
        hkind = f'{metric.upper()}_{kind}'
    human_data = obj24.human_data(kind=hkind)
    # import ipdb; ipdb.set_trace()
    test_idx = pandas.read_pickle(obj24.datapath('sel240'))

    clf = MatchToSampleClassifier(C=1e-3)
    train_idx = [i for i in range(len(obj.meta.obj)) if i not in test_idx]

    clf.fit(model_feats[train_idx], obj.meta.obj.iloc[train_idx], order=obj.OBJS)
    preds = clf.predict_proba(model_feats[test_idx],
                              targets=obj.meta.obj.iloc[test_idx], kind='2-way')
    df = pandas.DataFrame(preds, index=obj.meta.obj.iloc[test_idx], columns=obj.OBJS).reset_index()
    df['id'] = obj.meta.id.iloc[test_idx].values
    df = df.set_index(['obj', 'id'])
    df = df.stack().reset_index()
    df = df.rename(columns={'level_2': 'distr', 0: 'acc'})

    df = df[['obj', 'id', 'distr', 'acc']]

    # obj_order = ['lo_poly_animal_RHINO_2', 'calc01', 'womens_shorts_01M', 'zebra', 'MB27346', 'build51', 'weimaraner', 'interior_details_130_2', 'lo_poly_animal_CHICKDEE', 'kitchen_equipment_knife2', 'interior_details_103_4', 'lo_poly_animal_BEAR_BLK', 'MB30203', 'antique_furniture_item_18', 'lo_poly_animal_ELE_AS1', 'MB29874', 'womens_stockings_01M', 'Hanger_02', 'dromedary', 'MB28699', 'lo_poly_animal_TRANTULA', 'flarenut_spanner', 'MB30758', '22_acoustic_guitar']
    # df.obj = df.obj.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj_order))
    # df.distr = df.distr.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj.OBJS))
    # mm = obj.meta.iloc[test_idx]
    # id_order = np.concatenate([mm[mm.obj == o].id for o in obj_order])
    # df.id = df.id.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=id_order))
    # df = df.sort_values('id')

    # indices = {
    #     'o1': ['obj'],
    #     'o2': ['obj', 'distr'],
    #     'i1': ['obj', 'id'],
    #     'i2': ['obj', 'id', 'distr']
    # }
    indices = {
        'o1': [target],
        'o2': [target, distr],
        'i1': [target, imid],
        'i2': [target, imid, distr]
    }

    def hitrate_to_dprime(x):
        idx = x.name
        hit_rate = np.nanmean(x)

        if metric == 'o1':  # idx: target
            rej = df.loc[(df[target] != idx) & (df[distr] == idx), value]
        elif metric == 'o2':  # idx: (target, distr)
            rej = df.loc[(df[target] == idx[1]) & (df[distr] == idx[0]), value]
        elif metric == 'i1':  # idx: (target, imid)
            rej = df.loc[(df[target] != idx[0]) & (df[distr] == idx[0]), value]
            # import ipdb; ipdb.set_trace()
        elif metric == 'i2':  # idx: (target, imid, distr)
            rej = df.loc[(df[target] == idx[2]) & (df[distr] == idx[0]), value]

        fa_rate = 1 - np.nanmean(rej)

        if kind == 'dprime':
            output = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
            output = np.clip(output, -cap, cap)
        elif kind == 'acc':
            raise NotImplementedError
        return output

    dprime = df.groupby(indices[metric])['acc'].apply(hitrate_to_dprime)
    dprime = dprime.reset_index()
    if normalize:
        if metric == 'i1':
            by = target
        elif metric == 'i2':
            by = [target, distr]
        else:
            raise ValueError(f'normalization only defined for i1 and i2, got {metric}')
        # idx = dprime.index
        # import ipdb; ipdb.set_trace()
        # dprime = dprime.sort_values(imid)
        dprime[value] = dprime.groupby(by)[value].transform(lambda x: x - x.mean())
        # dprime = dprime.set_index(indices[metric])

    # dprime = hitrate_to_dprime(df, metric=metric, kind=kind,
    #                            target='obj', distr='distr',
    #                            imid='id', value='acc', normalize=normalize)
    obj_order = ['lo_poly_animal_RHINO_2', 'calc01', 'womens_shorts_01M', 'zebra', 'MB27346', 'build51', 'weimaraner', 'interior_details_130_2', 'lo_poly_animal_CHICKDEE', 'kitchen_equipment_knife2', 'interior_details_103_4', 'lo_poly_animal_BEAR_BLK', 'MB30203', 'antique_furniture_item_18', 'lo_poly_animal_ELE_AS1', 'MB29874', 'womens_stockings_01M', 'Hanger_02', 'dromedary', 'MB28699', 'lo_poly_animal_TRANTULA', 'flarenut_spanner', 'MB30758', '22_acoustic_guitar']

    if metric in ['o1', 'o2']:
        dprime.obj = dprime.obj.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj.OBJS))
    else:
        dprime.obj = dprime.obj.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj_order))

    if metric in ['o2', 'i2']:
        dprime.distr = dprime.distr.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj.OBJS))
    if metric in ['i1', 'i2']:
        mm = obj.meta.iloc[test_idx]
        id_order = np.concatenate([mm[mm.obj == o].id for o in obj_order])
        dprime.id = dprime.id.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=id_order))

    if metric == 'o1':
        dprime = dprime.sort_values('obj')
        preds = pandas.DataFrame(dprime.set_index(['obj']))
    elif metric == 'o2':
        dprime = dprime.sort_values('obj')
        preds = dprime.set_index(['obj','distr']).unstack('distr')
    elif metric == 'i1':
        dprime = dprime.sort_values('id')
        preds = pandas.DataFrame(dprime.set_index(['id','obj']))
    elif metric == 'i2':
        dprime = dprime.sort_values('id')
        preds = dprime.set_index(['id','obj','distr']).unstack('distr')

    # if metric in ['o2', 'i2']:
    #     preds = preds.unstack(distr)
    # else:
    #     preds = pandas.DataFrame(preds)

    preds = preds.fillna(np.nan).values

    df = []
    for iterno, split in enumerate(tqdm.tqdm(human_data)):
        inds = np.isfinite(split[0]) & np.isfinite(split[1]) & np.isfinite(preds)
        c0 = np.corrcoef(preds[inds], split[0][inds])[0,1]
        c1 = np.corrcoef(preds[inds], split[1][inds])[0,1]
        corr = (c0 + c1) / 2
        ic = np.corrcoef(split[0][inds], split[1][inds])[0,1]
        df.append([iterno, ic, corr, corr / np.sqrt(ic)])
        # import ipdb; ipdb.set_trace()
    df = pandas.DataFrame(df, columns=['split', 'internal_cons', 'r', 'cons'])
    # import ipdb; ipdb.set_trace()
    return df


def objectome_i2(model_feats, human_acc, meta, test_idx, order=None,
                 kind='i2'):
    # skf = StratifiedKFold(n_splits=10)
    # labels = np.array(labels)

    # preds = np.zeros((model_feats.shape[0], len(np.unique(labels))))
    # for i, (train_idx, test_idx) in enumerate(skf.split(model_feats, labels)):
    #     clf = MatchToSampleClassifier()
    #     clf.fit(model_feats[train_idx], labels[train_idx], order=order,
    #             decision_function_shape='ovr')
    #     pred = clf.predict_proba(model_feats[test_idx],
    #                              targets=labels[test_idx], kind='2-way')
    #     preds[test_idx] = pred

    import matplotlib.pyplot as plt
    import os

    clf = MatchToSampleClassifier()
    train_idx = [i for i in range(len(meta.obj)) if i not in test_idx]
    # train_idx2 = np.array([   9,   13,   25,   30,   34,   39,   47,   50,   52,   55,   56,
    #      58,   61,   65,   67,   68,   74,   77,   82,   84,  100,  104,
    #     114,  120,  123,  130,  140,  153,  155,  157,  163,  164,  165,
    #     167,  174,  179,  183,  196,  197,  199,  200,  201,  202,  204,
    #     213,  215,  217,  220,  231,  234,  245,  247,  256,  260,  261,
    #     274,  275,  277,  286,  287,  302,  304,  314,  315,  317,  320,
    #     325,  330,  336,  346,  349,  352,  353,  366,  372,  374,  380,
    #     383,  384,  399,  406,  424,  431,  433,  436,  445,  449,  452,
    #     460,  467,  469,  472,  473,  483,  484,  488,  492,  494,  495,
    #     498,  500,  501,  506,  507,  521,  524,  525,  528,  537,  544,
    #     548,  550,  553,  554,  558,  570,  572,  586,  591,  599,  602,
    #     604,  613,  614,  620,  623,  631,  642,  644,  647,  649,  655,
    #     657,  664,  668,  669,  678,  687,  691,  692,  702,  710,  718,
    #     720,  721,  723,  725,  730,  731,  737,  739,  744,  747,  760,
    #     765,  772,  774,  778,  789,  797,  811,  824,  827,  828,  832,
    #     836,  842,  850,  852,  854,  855,  875,  877,  881,  884,  889,
    #     891,  892,  895,  896,  902,  907,  910,  922,  929,  932,  933,
    #     938,  941,  948,  949,  958,  967,  968,  978,  980,  985,  988,
    #     993,  995, 1002, 1003, 1005, 1018, 1021, 1028, 1030, 1049, 1055,
    #    1059, 1064, 1065, 1066, 1071, 1073, 1076, 1078, 1081, 1088, 1098,
    #    1105, 1106, 1115, 1117, 1123, 1124, 1133, 1137, 1143, 1144, 1146,
    #    1152, 1157, 1163, 1180, 1182, 1184, 1187, 1191, 1199, 1207, 1214,
    #    1218, 1227, 1230, 1236, 1247, 1249, 1252, 1254, 1257, 1267, 1268,
    #    1274, 1275, 1279, 1280, 1286, 1289, 1294, 1305, 1306, 1321, 1324,
    #    1333, 1339, 1341, 1351, 1357, 1359, 1360, 1361, 1377, 1378, 1385,
    #    1386, 1387, 1389, 1394, 1395, 1402, 1408, 1409, 1415, 1418, 1423,
    #    1424, 1426, 1428, 1432, 1435, 1457, 1458, 1460, 1472, 1474, 1477,
    #    1486, 1491, 1494, 1503, 1504, 1512, 1517, 1519, 1523, 1538, 1541,
    #    1546, 1550, 1554, 1558, 1575, 1577, 1579, 1580, 1583, 1587, 1598,
    #    1599, 1603, 1604, 1605, 1606, 1611, 1615, 1626, 1628, 1630, 1646,
    #    1647, 1656, 1657, 1659, 1670, 1671, 1675, 1689, 1690, 1699, 1703,
    #    1709, 1713, 1726, 1727, 1729, 1731, 1732, 1737, 1751, 1752, 1760,
    #    1761, 1762, 1768, 1769, 1775, 1787, 1796, 1797, 1815, 1818, 1820,
    #    1822, 1830, 1831, 1834, 1835, 1836, 1841, 1843, 1844, 1850, 1853,
    #    1860, 1875, 1884, 1885, 1887, 1888, 1902, 1910, 1915, 1916, 1919,
    #    1922, 1924, 1934, 1936, 1939, 1940, 1957, 1959, 1967, 1972, 1976,
    #    1985, 1990, 1993, 1994, 2004, 2008, 2010, 2015, 2021, 2026, 2031,
    #    2035, 2038, 2041, 2043, 2049, 2050, 2053, 2066, 2070, 2071, 2089,
    #    2090, 2099, 2102, 2105, 2112, 2113, 2116, 2123, 2125, 2126, 2127,
    #    2136, 2144, 2146, 2147, 2155, 2170, 2177, 2182, 2192, 2194, 2195,
    #    2214, 2216, 2219, 2220, 2224, 2233, 2234, 2237, 2242, 2245, 2249,
    #    2255, 2258, 2262, 2265, 2272, 2283, 2285, 2288, 2295, 2305, 2315,
    #    2318, 2323, 2339, 2343, 2344, 2348, 2349, 2350, 2361, 2363, 2367,
    #    2372, 2376, 2381, 2383, 2391, 2392, 2395])
    # train_idx3 = np.random.choice(train_idx, size=1000, replace=False)
    clf.fit(model_feats[train_idx], meta.obj.iloc[train_idx], order=order)
            # decision_function_shape='ovr')
    # test_idx = np.sort(test_idx)
    preds = clf.predict_proba(model_feats[test_idx],
                              targets=meta.obj.iloc[test_idx], kind='2-way')
    # kk = preds.values.ravel()
    # plt.figure();plt.hist(kk[np.isfinite(kk)]);plt.show()
    # import ipdb; ipdb.set_trace()
    df = pandas.DataFrame(preds, index=meta.obj.iloc[test_idx], columns=order).reset_index()
    df['id'] = meta.id.iloc[test_idx].values
    df = df.set_index(['obj', 'id'])
    df = df.stack().reset_index()
    df = df.rename(columns={'level_2': 'distr', 0: 'acc'})
    # df.obj = df.obj.astype('category', ordered=True, categories=order)
    # df.distr = df.distr.astype('category', ordered=True, categories=order)

    # df = pandas.read_pickle('trials-from-rishi.pkl')
    # # df = pandas.read_pickle('trials.pkl')
    # df = df.drop(labels=['WorkerID', 'AssignmentID'], axis=1)
    # # df = df.drop(labels=['WorkerID', 'AssignmentID', 'DecisionScore'], axis=1)
    # df = df.rename(columns={'sample_obj': 'obj', 'dist_obj': 'distr', 'prob_choice': 'acc'})
    # df.acc = df.acc.astype(float)
    # # for i in range(10):
    # # dfr = df.iloc[5520*i:5520*(i+1)]
    # dfr = df.iloc[:5520]

    # pv = df.pivot_table(index=['obj', 'id'], columns='distr', values='acc')

    # pv = pv.reset_index()
    # import ipdb; ipdb.set_trace()
    # pv = pv.drop('id', axis='columns').set_index('obj')

    # dprime = utils.hitrate_to_dprime_o1(pv).reset_index()
    df = df[['obj', 'id', 'distr', 'acc']]

    # trials = pandas.read_pickle('/mindhive/dicarlolab/common/forJonas/trials_GOOGLENET_pool5_multicls20softmax_pandas.pkl')
    # import ipdb; ipdb.set_trace()
    # mtj = pandas.read_pickle('/mindhive/dicarlolab/u/qbilius/tmp_rishi_i2/objectome_utils/metrics240_full_jonas.pkl')

    dprime = utils.hitrate_to_dprime(df, kind='i2', target='obj', distr='distr',
                                     imid='id', value='acc', normalize=True)#.reset_index()
    obj_order = ['lo_poly_animal_RHINO_2', 'calc01', 'womens_shorts_01M', 'zebra', 'MB27346', 'build51', 'weimaraner', 'interior_details_130_2', 'lo_poly_animal_CHICKDEE', 'kitchen_equipment_knife2', 'interior_details_103_4', 'lo_poly_animal_BEAR_BLK', 'MB30203', 'antique_furniture_item_18', 'lo_poly_animal_ELE_AS1', 'MB29874', 'womens_stockings_01M', 'Hanger_02', 'dromedary', 'MB28699', 'lo_poly_animal_TRANTULA', 'flarenut_spanner', 'MB30758', '22_acoustic_guitar']
    mm = meta.iloc[test_idx]
    id_order = np.concatenate([mm[mm.obj==o].id for o in obj_order])
    # dprime.obj = dprime.obj.astype('category', ordered=True, categories=obj_order)
    dprime.distr = dprime.distr.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=order))
    dprime.id = dprime.id.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=id_order))
    dprime = dprime.sort_values('id')
    preds = dprime.set_index(['id','obj','distr']).unstack('distr').values
    # import ipdb; ipdb.set_trace()

    # import os
    # mt = pandas.read_pickle('GOOGLENET_pool5_multicls20softmax.pkl')
    # mtj = pandas.read_pickle('/mindhive/dicarlolab/u/qbilius/tmp_rishi_i2/objectome_utils/metrics240_full_jonas.pkl')
    # mtj['I2_dprime'] = [(mtj['I2_dprime'], mtj['I2_dprime']), (mtj['I2_dprime'], mtj['I2_dprime'])]
    # human_acc = pandas.read_pickle(os.path.expanduser('~/.streams/objectome/metrics240.pkl'))
    # df = []
    # for preds, split in zip(mtj['I2_dprime'], human_acc['I2_dprime']):
    #     inds = np.isfinite(split[0]) & np.isfinite(preds[0]) & np.isfinite(split[1]) & np.isfinite(preds[1])
    #     c0 = np.corrcoef(preds[0][inds], split[0][inds])[0,1]
    #     c1 = np.corrcoef(preds[1][inds], split[1][inds])[0,1]
    #     corr = (c0 + c1) / 2
    #     ich = np.corrcoef(split[0][inds], split[1][inds])[0,1]
    #     icm = np.corrcoef(preds[0][inds], preds[1][inds])[0,1]
    #     out = corr / np.sqrt(ich * icm)
    #     df.append(out)
    # print(np.mean(df))
    # import ipdb; ipdb.set_trace()

    # ht = pandas.read_pickle(os.path.expanduser('~/.streams/objectome/metrics240.pkl'))
    # for key in mt.keys():
    #     corr = np.corrcoef(mt[key][0][0].ravel(), ht[key][0][0].ravel())[0,1]
    #     ic = np.corrcoef(ht[key][0][0].ravel(), ht[key][0][1].ravel())[0,1]
    #     print(key, corr, corr / np.sqrt(ic))
    # import ipdb; ipdb.set_trace()

    # acc = df.groupby('obj').acc.mean()

    # df.groupby()
    # acc = clf.score(model_feats[test_idx], labels[test_idx], kind='2-way')
    # import ipdb; ipdb.set_trace()
    # if kind == 'dprime':
    #     preds = _feats_to_dprime(preds, np.array(labels[test_idx]), order)
    # elif kind == 'dprime_c':
    #     preds = _feats_to_dprime(preds, np.array(labels[test_idx]), order)
    #     preds = _to_c(preds, np.array(labels[test_idx]), order)
    # elif kind == 'acc':
    #     preds = _to_c(preds, np.array(labels[test_idx]), order)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=[10,6]);plt.imshow(preds.T); plt.show()
    # plt.figure(figsize=[10,6]);plt.imshow(np.mean(human_acc,0).mean(0).T); plt.show()
    # import ipdb; ipdb.set_trace()

    # df = []
    # for iterno, split in enumerate(tqdm.tqdm(human_acc)):
    #     split0 = split[0][~np.isnan(split[0])]
    #     split1 = split[1][~np.isnan(split[1])]

    #     # plt.figure(figsize=(18,5));plt.imshow(preds.T);plt.show()
    #     # import ipdb; ipdb.set_trace()
    #     c1 = np.corrcoef(preds[~np.isnan(split[0])], split0)[0,1]
    #     c2 = np.corrcoef(preds[~np.isnan(split[1])], split1)[0,1]
    #     corr = (c1 + c2) / 2
    #     ic = np.corrcoef(split0, split1)[0,1]
    #     df.append([iterno, ic, corr, corr / np.sqrt(ic)])
    #     import ipdb; ipdb.set_trace()
    # df = pandas.DataFrame(df, columns=['split', 'internal_cons', 'r', 'cons'])

    df = []
    for iterno, split in enumerate(tqdm.tqdm(human_acc)):
        inds = np.isfinite(split[0]) & np.isfinite(split[1]) & np.isfinite(preds)
        c0 = np.corrcoef(preds[inds], split[0][inds])[0,1]
        c1 = np.corrcoef(preds[inds], split[1][inds])[0,1]
        corr = (c0 + c1) / 2
        ic = np.corrcoef(split[0][inds], split[1][inds])[0,1]
        df.append([iterno, ic, corr, corr / np.sqrt(ic)])
    df = pandas.DataFrame(df, columns=['split', 'internal_cons', 'r', 'cons'])
    # print(df)
    import ipdb; ipdb.set_trace()
    return df


###
# Rishi's:
# O1_hitrate 0.80352847689
# O1_accuracy 0.86396145369
# O1_dprime 0.870613549616
# O1_dprime_v2 0.835565862977

# O2_accuracy 0.822552874127
# O2_dprime 0.819807171725
# O2_hitrate 0.75551248105

# I1_hitrate 0.599541880761
# I1_accuracy 0.643152468507
# I1_dprime 0.697413854086
# I1_dprime_v2 0.67724399873
# I1_dprime_C 0.577332128854
# I1_dprime_v2_C 0.590692856277

# I2_hitrate 0.523947662884
# I2_accuracy 0.649239539486
# I2_dprime 0.667768961395
# I2_dprime_C 0.535381501732


if __name__ == "__main__":
    df = hvm.human_data if not time else hvm.human_data_timing