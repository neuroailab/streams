from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats
import pandas

import bangmetric

from streams import utils


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
        human = df.pivot_table(index='id', columns='stim_dur', values='acc')
    elif kind in ["d'", 'dprime']:
        kind = "d'"
        groups = ['stim_dur'] if time else []
        human, _ = _human_dprime_i1(human_data, groups=groups, ikind='i1')

    pv = human.pivot_table(index='id', columns=groups, values=kind)
    if len(groups) == 0:
        r = pv.corr(model_conf, method=corr_method)
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


def _dprime(hits, fas, ceiling=5):
    dprime = bangmetric.dprime(hits.values, fas.values, mode='rate', max_value=ceiling)
    dprime = pandas.Series(dprime, index=hits.index)
    dprime = dprime.reset_index()
    dprime.rename(columns={0: "d'"}, inplace=True)

    # bias
    c = .5 * (hits.apply(scipy.stats.norm.ppf) + fas.apply(scipy.stats.norm.ppf))
    c = c.reset_index()
    c.rename(columns={'acc': 'bias'}, inplace=True)

    return dprime, c


def human_acc(df):
    return df.pivot_table(index='id', columns='stim_dur', values='acc')


if __name__ == "__main__":
    df = hvm.human_data if not time else hvm.human_data_timing