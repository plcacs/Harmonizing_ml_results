import os
import re
import string
import hashlib
import warnings
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nevergrad.common.typing as tp
from . import utils
from .exporttable import export_table
_DPI = 250
no_limit = False
pure_algorithms = []

def compactize(name):
    if len(name) < 70:
        return name
    hashcode = hashlib.md5(bytes(name, 'utf8')).hexdigest()
    name = re.sub('\\([^()]*\\)', '', name)
    mid = 35
    name = name[:mid] + hashcode + name[-mid:]
    return name

def _make_style_generator():
    lines = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle('ov^<>8sp*hHDd')
    colors = itertools.cycle('bgrcmyk')
    return (l + m + c for l, m, c in zip(lines, markers, colors))

class NameStyle(tp.Dict[str, tp.Any]):
    """Provides a style for each name, and keeps to it"""

    def __init__(self):
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name):
        if name not in self:
            super().__setitem__(name, next(self._gen))
        return super().__getitem__(name)

def _make_winners_df(df, all_optimizers):
    """Finds mean loss over all runs for each of the optimizers, and creates a matrix
    winner_ij = 1 if opt_i is better (lower loss) then opt_j (and .5 for ties)
    """
    if not isinstance(df, utils.Selector):
        df = utils.Selector(df)
    all_optim_set = set(all_optimizers)
    assert all((x in all_optim_set for x in df.unique('optimizer_name')))
    assert all((x in df.columns for x in ['optimizer_name', 'loss']))
    winners = utils.Selector(index=all_optimizers, columns=all_optimizers, data=0.0)
    grouped = df.loc[:, ['optimizer_name', 'loss']].groupby(['optimizer_name']).mean()
    df_optimizers = list(grouped.index)
    values = np.array(grouped)
    diffs = values - values.T
    winners.loc[df_optimizers, df_optimizers] = (diffs < 0) + 0.5 * (diffs == 0)
    return winners

def aggregate_winners(df, categories, all_optimizers):
    """Computes the sum of winning rates on all cases corresponding to the categories

    Returns
    -------
    Selector
        the aggregate
    int
        the total number of cases
    """
    if not categories:
        return (_make_winners_df(df, all_optimizers), 1)
    subcases = df.unique(categories[0])
    if len(subcases) == 1:
        return aggregate_winners(df, categories[1:], all_optimizers)
    iterdf, iternum = zip(*(aggregate_winners(df.loc[df.loc[:, categories[0]] == val], categories[1:], all_optimizers) for val in subcases))
    return (sum(iterdf), sum(iternum))

def _make_sorted_winrates_df(victories):
    """Converts a dataframe counting number of victories into a sorted
    winrate dataframe. The algorithm which performs better than all other
    algorithms comes first. When you do not play in a category, you are
    considered as having lost all comparisons in that category.
    """
    assert all((x == y for x, y in zip(victories.index, victories.columns)))
    winrates = victories / (victories + victories.T).max(axis=1)
    mean_win = winrates.mean(axis=1).sort_values(ascending=False)
    return winrates.loc[mean_win.index, mean_win.index]

def remove_errors(df):
    df = utils.Selector(df)
    if 'error' not in df.columns:
        return df
    nandf = df.select(loss=np.isnan)
    for row in nandf.itertuples():
        msg = f'Removing "{row.optimizer_name}"'
        msg += f' with dimension {row.dimension}' if hasattr(row, 'dimension') else ''
        msg += f': got error "{row.error}"' if isinstance(row.error, str) else 'recommended a nan'
        warnings.warn(msg)
    handlederrordf = df.select(error=lambda x: isinstance(x, str) and x, loss=lambda x: not np.isnan(x))
    for row in handlederrordf.itertuples():
        warnings.warn(f'Keeping non-optimal recommendation of "{row.optimizer_name}" with dimension {(row.dimension if hasattr(row, 'dimension') else 'UNKNOWN')} which raised "{row.error}".')
    err_inds = set(nandf.index)
    output = df.loc[[i for i in df.index if i not in err_inds], [c for c in df.columns if c != 'error']]
    df.loc[np.isnan(df.loss), 'loss'] = float('inf')
    assert not output.loc[:, 'loss'].isnull().values.any(), 'Some nan values remain while there should not be any!'
    output = utils.Selector(output.reset_index(drop=True))
    return output

class PatternAggregate:

    def __init__(self, pattern):
        self._pattern = pattern

    def __call__(self, df):
        return self._pattern.format(**df.to_dict())
_PARAM_MERGE_PATTERN = '{optimizer_name},{parametrization}'

def merge_optimizer_name_pattern(df, pattern, merge_parametrization=False, remove_suffix=False):
    """Merge the optimizer name with other descriptors based on a pattern
    Nothing happens if merge_parametrization is false and pattern is empty string
    """
    if merge_parametrization:
        if pattern:
            raise ValueError("Cannot specify both merge-pattern and merge-parametrization (merge-parametrization is equivalent to merge-pattern='{optimizer_name},{parametrization}')")
        pattern = _PARAM_MERGE_PATTERN
    if not pattern:
        return df
    df = df.copy()
    okey = 'optimizer_name'
    elements = [tup[1] for tup in string.Formatter().parse(pattern) if tup[1] is not None]
    assert okey in elements, f'Missing optimizer key {okey!r} in merge pattern.\nEg: ' + 'pattern="{optimizer_name}_{parametrization}"'
    others = [x for x in elements if x != okey]
    aggregate = PatternAggregate(pattern)
    sub = df.loc[:, elements].fillna('')
    if len(sub.unique(others)) > 1:
        for optim in sub.unique(okey):
            inds = sub.loc[:, okey] == optim
            if len(sub.loc[inds, :].unique(others)) > 1:
                df.loc[inds, okey] = sub.loc[inds, elements].agg(aggregate, axis=1)
    if remove_suffix:
        df['optimizer_name'] = df['optimizer_name'].replace('[0-9\\.\\-]*$', '', regex=True)
    return df.drop(columns=others)

def normalized_losses(df, descriptors):
    df = utils.Selector(df.copy())
    cases = df.unique(descriptors)
    if not cases:
        cases = [()]
    for case in cases:
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        losses = np.array(subdf.loc[:, 'loss'])
        m = min(losses)
        M = max(losses[losses < (float('inf') if no_limit else float('1e26'))])
        df.loc[subdf.index, 'loss'] = (df.loc[subdf.index, 'loss'] - m) / (M - m) if M != m else 1
    return df

def create_plots(df, output_folder, max_combsize=1, xpaxis='budget', competencemaps=False, nomanyxp=False):
    """Saves all representing plots to the provided folder

    Parameters
    ----------
    df: pd.DataFrame
        the experiment data
    output_folder: PathLike
        path of the folder where the plots should be saved
    max_combsize: int
        maximum number of parameters to fix (combinations) when creating experiment plots
    xpaxis: str
        x-axis for xp plots (either budget or pseudotime)
    """
    assert xpaxis in ['budget', 'pseudotime']
    if 'non_proxy_function' in df.columns:
        print('removing non_proxy_function')
        df.drop(columns=['non_proxy_function'], inplace=True)
    df = remove_errors(df)
    df.loc[:, 'loss'] = pd.to_numeric(df.loc[:, 'loss'])
    if not no_limit:
        loss = pd.to_numeric(df.loc[:, 'loss'])
        upper = np.max(loss[loss < 1e+26])
        df.loc[:, 'loss'] = df.loc[:, 'loss'].clip(lower=-1e+26, upper=upper)
    df = df.loc[:, [x for x in df.columns if not x.startswith('info/')]]
    for col in df.columns:
        print(' Working on ', col)
        failed_indices = []
        if 'max_irr' in col:
            df[col] = df[col].round(decimals=4)
        if col in ('budget', 'num_workers', 'dimension', 'useful_dimensions', 'num_blocks', 'block_dimension', 'num_objectives'):
            for _ in range(2):
                try:
                    df[col] = df[col].astype(float).astype(int)
                    print(col, ' is converted to int')
                    continue
                except Exception as e1:
                    for i in range(len(df[col])):
                        try:
                            float(df[col][i])
                        except Exception as e2:
                            failed_indices += [i]
                            assert len(failed_indices) < 100, f'Fails at row {i + 2}, Exceptions: {e1}, {e2}. Failed-indices = {failed_indices}'
                print('Dropping ', failed_indices)
                df.drop(df.index[failed_indices], inplace=True)
                failed_indices = []
        elif col != 'loss':
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('\\.[0]*$', '', regex=True)
            try:
                df.loc[:, col] = pd.to_numeric(df.loc[:, col])
            except:
                pass
    if 'num_objectives' in df.columns:
        df = df[df.num_objectives != 0]
    if 'instrum_str' in set(df.columns):
        df.loc[:, 'optimizer_name'] = df.loc[:, 'optimizer_name'] + df.loc[:, 'instrum_str']
        df = df.drop(columns='instrum_str')
        df = df.drop(columns='dimension')
        if 'parametrization' in set(df.columns):
            df = df.drop(columns='parametrization')
        if 'instrumentation' in set(df.columns):
            df = df.drop(columns='instrumentation')
    df = utils.Selector(df.fillna('N-A'))
    assert not any(('Unnamed: ' in x for x in df.columns)), f'Remove the unnamed index column:  {df.columns}'
    assert 'error ' not in df.columns, f'Remove error rows before plotting'
    required = {'optimizer_name', 'budget', 'loss', 'elapsed_time', 'elapsed_budget'}
    missing = required - set(df.columns)
    assert not missing, f'Missing fields: {missing}'
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    descriptors = sorted(set(df.columns) - (required | {'instrum_str', 'seed', 'pseudotime'}))
    to_drop = [x for x in descriptors if len(df.unique(x)) == 1]
    df = utils.Selector(df.loc[:, [x for x in df.columns if x not in to_drop]])
    all_descriptors = sorted(set(df.columns) - (required | {'instrum_str', 'seed', 'pseudotime'}))
    print(f'Descriptors: {all_descriptors}')
    print('# Fight plots')
    fight_descriptors = all_descriptors + ['budget']
    combinable = [x for x in fight_descriptors if len(df.unique(x)) > 1]
    descriptors = []
    for d in all_descriptors:
        acceptable = False
        for b in df.budget.unique():
            if len(df.loc[df['budget'] == b][d].unique()) > 1:
                acceptable = True
                break
        if acceptable:
            descriptors += [d]
    num_rows = 6
    if competencemaps:
        max_combsize = max(max_combsize, 2)
    for fixed in list(itertools.chain.from_iterable((itertools.combinations(combinable, order) for order in range(max_combsize + 1)))):
        orders = [len(c) for c in df.unique(fixed)]
        if orders:
            assert min(orders) == max(orders)
            order = min(orders)
        else:
            order = 0
        best_algo = []
        if competencemaps and order == 2:
            print('\n#trying to competence-map')
            if all([len(c) > 1 for c in df.unique(fixed)]):
                try:
                    xindices = sorted(set((c[0] for c in df.unique(fixed))))
                except TypeError:
                    xindices = list(set((c[0] for c in df.unique(fixed))))
                try:
                    yindices = sorted(set((c[1] for c in df.unique(fixed))))
                except TypeError:
                    yindices = list(set((c[1] for c in df.unique(fixed))))
                for _ in range(len(xindices)):
                    best_algo += [[]]
                for i in range(len(xindices)):
                    for _ in range(len(yindices)):
                        best_algo[i] += ['none']
        for case in df.unique(fixed) if fixed else [()]:
            print('\n# new case #', fixed, case)
            casedf = df.select(**dict(zip(fixed, case)))
            data_df = FightPlotter.winrates_from_selection(casedf, fight_descriptors, num_rows=num_rows, num_cols=350)
            fplotter = FightPlotter(data_df)
            if order == 2 and competencemaps and best_algo:
                print('\n#storing data for competence-map')
                best_algo[xindices.index(case[0])][yindices.index(case[1])] = fplotter.winrates.index[0]
            name = 'fight_' + ','.join(('{}{}'.format(x, y) for x, y in zip(fixed, case))) + '.png'
            name = 'fight_all.png' if name == 'fight_.png' else name
            name = compactize(name)
            fullname = name
            if name == 'fight_all.png':
                with open(str(output_folder / name) + '.cp.txt', 'w') as f:
                    f.write(fullname)
                    f.write('ranking:\n')
                    for i, algo in enumerate(data_df.columns[:158]):
                        f.write(f'  algo {i}: {algo}\n')
            if len(name) > 240:
                hashcode = hashlib.md5(bytes(name, 'utf8')).hexdigest()
                name = re.sub('\\([^()]*\\)', '', name)
                mid = 120
                name = name[:mid] + hashcode + name[-mid:]
            fplotter.save(str(output_folder / name), dpi=_DPI)
            data_df = FightPlotter.winrates_from_selection(casedf, fight_descriptors, num_rows=num_rows, complete_runs_only=True)
            fplotter = FightPlotter(data_df)
            if name == 'fight_all.png':
                global pure_algorithms
                pure_algorithms = list(data_df.columns[:])
            if name == 'fight_all.png':
                fplotter.save(str(output_folder / 'fight_all_pure.png'), dpi=_DPI)
            else:
                fplotter.save(str(output_folder / name) + '_pure.png', dpi=_DPI)
                print(f'# {len(data_df.columns[:])}  {data_df.columns[:]}')
            if order == 2 and competencemaps and best_algo:
                print('\n# Competence map')
                name = 'competencemap_' + ','.join(('{}'.format(x) for x in fixed)) + '.tex'
                export_table(str(output_folder / name), xindices, yindices, best_algo)
                print('Competence map data:', fixed, case, best_algo)
    plt.close('all')
    print('# Xp plots')
    name_style = NameStyle()
    cases = df.unique(descriptors)
    if not cases:
        cases = [()]
    out_filepath = output_folder / 'xpresults_all.png'
    try:
        data = XpPlotter.make_data(df, normalized_loss=True)
        for pure_only in [False, True]:
            xpplotter = XpPlotter(data, title=os.path.basename(output_folder), name_style=name_style, xaxis=xpaxis, pure_only=pure_only)
    except Exception as e:
        lower = 0
        upper = len(df)
        while upper > lower + 1:
            middle = (lower + upper) // 2
            small_df = df.head(middle)
            try:
                print('Testing ', middle)
                _ = XpPlotter.make_data(small_df, normalized_loss=True)
                xpplotter = XpPlotter(data, title=os.path.basename(output_folder), name_style=name_style, xaxis=xpaxis, pure_only=True)
                print('Work with ', middle)
                lower = middle
            except:
                print('Failing with ', middle)
                upper = middle
        assert False, f'Big failure {e} at line {middle}'
    xpplotter.save(out_filepath)
    for case in cases:
        if nomanyxp:
            continue
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        description = ','.join(('{}:{}'.format(x, y) for x, y in zip(descriptors, case)))
        full_description = description
        description = compactize(description)
        if len(description) > 280:
            hash_ = hashlib.md5(bytes(description, 'utf8')).hexdigest()
            description = description[:140] + hash_ + description[-140:]
        out_filepath = output_folder / 'xpresults{}{}.png'.format('_' if description else '', description.replace(':', ''))
        txt_out_filepath = output_folder / 'xpresults{}{}.leaderboard.txt'.format('_' if description else '', description.replace(':', ''))
        data = XpPlotter.make_data(subdf)
        try:
            xpplotter = XpPlotter(data, title=description, name_style=name_style, xaxis=xpaxis)
        except Exception as e:
            warnings.warn(f'Bypassing error in xpplotter:\n{e}', RuntimeWarning)
        else:
            xpplotter.save(out_filepath)
            xpplotter.save_txt(txt_out_filepath, data, full_description)
    plt.close('all')

def gp_sota():
    gp = {}
    gp['CartPole-v1'] = (-500.0, 100000.0)
    gp['Acrobot-v1'] = (83.17, 200000.0)
    gp['MountainCarContinuous-v0'] = (-99.31, 900000.0)
    gp['Pendulum-v0'] = (154.36, 1100000.0)
    gp['InvertedPendulumSwingupBulletEnv-v0'] = (-893.35, 400000.0)
    gp['BipedalWalker-v3'] = (-268.85, 1100000.0)
    gp['BipedalWalkerHardcore-v3'] = (-9.25, 1100000.0)
    gp['HopperBulletEnv-v0'] = (-999.19, 1000000.0)
    gp['InvertedDoublePendulumBulletEnv-v0'] = (-9092.17, 300000.0)
    gp['LunarLanderContinuous-v2'] = (-287.58, 1000000.0)
    return gp

def ceviche_sota():
    ceviche = {}
    ceviche['waveguide-bend'] = (0.0681388, 1000000)
    ceviche['beam-splitter'] = (0.496512, 1000000)
    ceviche['mode-converter'] = (0.181592, 1000000)
    ceviche['wdm'] = (0.982352, 100000)
    return ceviche

class LegendInfo(tp.NamedTuple):
    """Handle for information used to create a legend."""

class XpPlotter:
    """Creates a xp result plot out of the given dataframe: regret with respect to budget for
    each optimizer after averaging on all experiments (it is good practice to use a df
    which is filtered out for one set of input parameters)

    Parameters
    ----------
    optim_vals: dict
        output of the make_data static method, containing all information necessary for plotting
    title: str
        title of the plot
    name_style: dict
        a dict or dict-like object providing a line style for each optimizer name.
        (can be helpful for consistency across plots)
    """

    def __init__(self, optim_vals, title, name_style=None, xaxis='budget', pure_only=False):
        if name_style is None:
            name_style = NameStyle()
        upperbound = max((np.max(vals['loss']) for vals in optim_vals.values() if np.max(vals['loss']) < np.inf))
        for optim, vals in optim_vals.items():
            if optim.lower() in ['stupid', 'idiot'] or optim in ['Zero', 'StupidRandom']:
                upperbound = min(upperbound, np.max(vals['loss']))
        lowerbound = np.inf
        sorted_optimizers = sorted(optim_vals, key=lambda x: optim_vals[x]['loss'][-1], reverse=True)
        if pure_only:
            assert len(pure_algorithms) > 0
            sorted_optimizers = [o for o in sorted_optimizers if o + ' ' in [p[:len(o) + 1] for p in pure_algorithms]]
        with open(('rnk__' if not pure_only else 'rnkpure__') + str(title) + '.cp.txt', 'w') as f:
            f.write(compactize(title))
            f.write('ranking:\n')
            for i, algo in reversed(list(enumerate(sorted_optimizers))):
                f.write(f'  algo {i}: {algo} (x)\n')
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        logplot = not any((x <= 0 or x > 10 ** 8 for ov in optim_vals.values() for x in ov['loss']))
        if logplot:
            self._ax.set_yscale('log')
            for ov in optim_vals.values():
                if ov['loss'].size:
                    ov['loss'] = np.maximum(1e-30, ov['loss'])
        self._ax.autoscale(enable=False)
        self._ax.set_xscale('log')
        self._ax.set_xlabel(xaxis)
        self._ax.set_ylabel('loss')
        self._ax.grid(True, which='both')
        self._overlays = []
        legend_infos = []
        title_addendum = f'({len(sorted_optimizers)} algos)'
        for optim_name in sorted_optimizers[:1] + sorted_optimizers[-35:] if len(sorted_optimizers) > 35 else sorted_optimizers:
            vals = optim_vals[optim_name]
            indices = np.where(vals['num_eval'] > 0)
            lowerbound = min(lowerbound, np.min(vals['loss']))
            for sota_name, sota in [('GP', gp_sota()), ('ceviche', ceviche_sota())]:
                for k in sota.keys():
                    if k in title:
                        th = sota[k][0]
                        cost = sota[k][1]
                        title_addendum = f'({sota_name}:{th})'
                        lowerbound = min(lowerbound, th, 0.9 * th, 1.1 * th)
                        plt.plot(vals[xaxis][indices], th + 0 * vals['loss'][indices], name_style[optim_name], label=sota_name)
                        plt.plot([cost] * 3, [min(vals['loss'][indices]), sum(vals['loss'][indices]) / len(indices), max(vals['loss'][indices])], name_style[optim_name], label=sota_name)
            line = plt.plot(vals[xaxis], vals['loss'], name_style[optim_name], label=optim_name)
            for conf in self._get_confidence_arrays(vals, log=logplot):
                plt.plot(vals[xaxis], conf, name_style[optim_name], label=optim_name, alpha=0.1)
            text = '{} ({:.3g} <{:.3g}>)'.format(optim_name, vals['loss'][-1], vals['loss'][-2] if len(vals['loss']) > 1 else float('nan'))
            if vals[xaxis].size:
                legend_infos.append(LegendInfo(vals[xaxis][-1], vals['loss'][-1], line, text))
        if not (np.isnan(upperbound) or np.isinf(upperbound)):
            upperbound_up = upperbound
            if not (np.isnan(lowerbound) or np.isinf(lowerbound)):
                self._ax.set_ylim(bottom=lowerbound)
                upperbound_up += 0.02 * (upperbound - lowerbound)
                if logplot:
                    upperbound_up = 10 ** (np.log10(upperbound) + 0.02 * (np.log10(upperbound) - np.log10(lowerbound)))
            self._ax.set_ylim(top=upperbound_up)
        all_x = [v for vals in optim_vals.values() for v in vals[xaxis]]
        try:
            all_x = [float(a_) for a_ in all_x]
            self._ax.set_xlim([min(all_x), max(all_x)])
        except TypeError:
            print(f'TypeError for minimum or maximum or {all_x}')
        self.add_legends(legend_infos)
        if 'tmp' not in title:
            self._ax.set_title(split_long_title(title + title_addendum))
        self._ax.tick_params(axis='both', which='both')

    @staticmethod
    def _get_confidence_arrays(vals, log=False):
        loss = vals['loss']
        conf = vals['loss_std'] / np.sqrt(vals['loss_nums'] - 1)
        if not log:
            return (loss - conf, loss + conf)
        lloss = np.log10(loss)
        lstd = 0.434 * conf / loss
        return tuple((10 ** (lloss + x) for x in [-lstd, lstd]))

    def add_legends(self, legend_infos):
        """Adds the legends"""
        ax = self._ax
        trans = ax.transScale + ax.transLimits
        fontsize = 10.0
        display_y = (ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0)))[1]
        shift = (2.0 + fontsize) / display_y
        legend_infos = legend_infos[::-1]
        values = [float(np.clip(trans.transform((0, i.y))[1], -0.01, 1.01)) for i in legend_infos]
        placements = compute_best_placements(values, min_diff=shift)
        for placement, info in zip(placements, legend_infos):
            self._overlays.append(Legend(ax, info.line, [info.text], loc='center left', bbox_to_anchor=(1, placement), frameon=False, fontsize=fontsize))
            ax.add_artist(self._overlays[-1])

    @staticmethod
    def make_data(df, normalized_loss=False):
        """Process raw xp data and process it to extract relevant information for xp plots:
        regret with respect to budget for each optimizer after averaging on all experiments (it is good practice to use a df
        which is filtered out for one set of input parameters)

        Parameters
        ----------
        df: pd.DataFrame
            run data
        normalized_loss: bool
            whether we should normalize each data (for each budget and run) between 0 and 1. Convenient when we consider
            averages over several distinct functions that can have very different ranges - then we return data which are rescaled to [0,1].
            Warning: then even if algorithms converge (i.e. tend to minimize), the value can increase, because the normalization
            is done separately for each budget.
        """
        if normalized_loss:
            descriptors = sorted(set(df.columns) - {'pseudotime', 'time', 'budget', 'elapsed_time', 'elapsed_budget', 'loss', 'optimizer_name', 'seed'})
            df = normalized_losses(df, descriptors=descriptors)
        df = utils.Selector(df.loc[:, ['optimizer_name', 'budget', 'loss'] + (['pseudotime'] if 'pseudotime' in df.columns else [])])
        groupeddf = df.groupby(['optimizer_name', 'budget'])
        means = groupeddf.mean() if no_limit else groupeddf.median()
        stds = groupeddf.std()
        nums = groupeddf.count()
        optim_vals = {}
        for optim in df.unique('optimizer_name'):
            optim_vals[optim] = {}
            optim_vals[optim]['budget'] = np.array(means.loc[optim, :].index)
            optim_vals[optim]['loss'] = np.array(means.loc[optim, 'loss'])
            optim_vals[optim]['loss_std'] = np.array(stds.loc[optim, 'loss'])
            optim_vals[optim]['loss_nums'] = np.array(nums.loc[optim, 'loss'])
            num_eval = np.array(groupeddf.count().loc[optim, 'loss'])
            optim_vals[optim]['num_eval'] = num_eval
            if 'pseudotime' in means.columns:
                optim_vals[optim]['pseudotime'] = np.array(means.loc[optim, 'pseudotime'])
        return optim_vals

    @staticmethod
    def save_txt(output_filepath, optim_vals, addendum=''):
        """Saves a list of best performances.

        output_filepath: Path or str
            path where the figure must be saved
        optim_vals: dict
            dict of losses obtained by a given optimizer.
        """
        best_performance = defaultdict(lambda: (float('inf'), 'none'))
        for optim in optim_vals.keys():
            for i, l in zip(optim_vals[optim]['budget'], optim_vals[optim]['loss']):
                if l < best_performance[i][0]:
                    best_performance[i] = (l, optim)
        with open(output_filepath, 'w') as f:
            f.write(addendum)
            f.write('Best performance:\n')
            for i in best_performance.keys():
                f.write(f'  budget {i}: {best_performance[i][0]} ({best_performance[i][1]}) ({output_filepath})\n')

    def save(self, output_filepath):
        """Saves the xp plot

        Parameters
        ----------
        output_filepath: Path or str
            path where the figure must be saved
        """
        try:
            self._fig.savefig(str(output_filepath), bbox_extra_artists=self._overlays, bbox_inches='tight', dpi=_DPI)
        except ValueError as v:
            print(f'We catch {v} which means that image = too big.')
            self._fig.savefig(str(output_filepath), bbox_extra_artists=self._overlays, bbox_inches='tight', dpi=_DPI / 5)

    def __del__(self):
        plt.close(self._fig)

def split_long_title(title):
    """Splits a long title around the middle comma"""
    if len(title) <= 60:
        return title
    comma_indices = np.where(np.array(list(title)) == ',')[0]
    if not comma_indices.size:
        return title
    best_index = comma_indices[np.argmin(abs(comma_indices - len(title) // 2))]
    title = title[:best_index + 1] + '\n' + title[best_index + 1:]
    return title

class FightPlotter:
    """Creates a fight plot out of the given dataframe, by iterating over all cases with fixed category variables.

    Parameters
    ----------
    winrates_df: pd.DataFrame
        winrate data as a dataframe
    """

    def __init__(self, winrates_df):
        self.winrates = winrates_df
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        max_cols = 25
        self._cax = self._ax.imshow(100 * np.array(self.winrates)[:, :max_cols], cmap='seismic', interpolation='none', vmin=0, vmax=100)
        x_names = self.winrates.columns[:max_cols]
        self._ax.set_xticks(list(range(len(x_names))))
        self._ax.set_xticklabels(x_names, rotation=45, ha='right', fontsize=7)
        y_names = self.winrates.index
        self._ax.set_yticks(list(range(len(y_names))))
        self._ax.set_yticklabels(y_names, rotation=45, fontsize=7)
        divider = make_axes_locatable(self._ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self._fig.colorbar(self._cax, cax=cax)
        plt.tight_layout()

    @staticmethod
    def winrates_from_selection(df, categories, num_rows=5, num_cols=350, complete_runs_only=False):
        """Creates a fight plot win rate data out of the given run dataframe,
        by iterating over all cases with fixed category variables.

        Parameters
        ----------
        df: pd.DataFrame
            run data
        categories: list
            List of variables to fix for obtaining similar run conditions
        num_rows: int
            number of rows to plot (best algorithms)
        complete_runs_only: bool
            if we want a plot with only algorithms which have run on all settings
        """
        all_optimizers = list(df.unique('optimizer_name'))
        num_rows = min(num_rows, len(all_optimizers))
        victories, total = aggregate_winners(df, categories, all_optimizers)
        if complete_runs_only:
            max_num = max([int(2 * victories.loc[n, n]) for n in all_optimizers])
            new_all_optimizers = [n for n in all_optimizers if int(2 * victories.loc[n, n]) == max_num]
            if len(new_all_optimizers) > 0:
                df = df[df['optimizer_name'].isin(new_all_optimizers)]
                victories, total = aggregate_winners(df, categories, new_all_optimizers)
        winrates = _make_sorted_winrates_df(victories)
        mean_win = winrates.mean(axis=1)
        winrates.fillna(0.5)
        sorted_names = winrates.index
        sorted_names = ['{} ({}/{})'.format(n, int(2 * victories.loc[n, n]), total) for n in sorted_names]
        num_names = len(sorted_names)
        sorted_names = [sorted_names[i] for i in range(min(num_cols, num_names))]
        data = np.array(winrates.iloc[:num_rows, :len(sorted_names)])
        best_names = [f'{name} ({i + 1}/{num_names}:{100 * val:2.1f}% +- {25 * np.sqrt(val * (1 - val) / int(2 * victories.loc[name, name])):2.1f})'.replace('Search', '') for i, (name, val) in enumerate(zip(mean_win.index[:num_rows], mean_win))]
        return pd.DataFrame(index=best_names, columns=sorted_names, data=data)

    def save(self, *args, **kwargs):
        """Shortcut to the figure savefig method"""
        self._fig.savefig(*args, **kwargs)

    def __del__(self):
        plt.close(self._fig)

class LegendGroup:
    """Class used to compute legend best placements.
    Each group contains at least one legend, and has a position and span (with bounds). LegendGroup are then
    responsible for providing each of its legends' position (non-overlapping)


    Parameters
    ----------
    indices: List[int]
        identifying index of each of the legends
    init_position: List[float]
        best position for each of the legends (if there was no overlapping)
    min_diff: float
        minimal distance between two legends so that they do not overlap
    """

    def __init__(self, indices, init_positions, min_diff):
        assert all((x2 - x1 == 1 for x2, x1 in zip(indices[1:], indices[:-1])))
        assert all((v2 >= v1 for v2, v1 in zip(init_positions[1:], init_positions[:-1])))
        assert len(indices) == len(init_positions)
        self.indices = indices
        self.init_positions = init_positions
        self.min_diff = min_diff
        self.position = float(np.mean(init_positions))

    def combine_with(self, other):
        assert self.min_diff == other.min_diff
        return LegendGroup(self.indices + other.indices, self.init_positions + other.init_positions, self.min_diff)

    def get_positions(self):
        first_position = self.bounds[0] + self.min_diff / 2.0
        return [first_position + k * self.min_diff for k in range(len(self.indices))]

    @property
    def bounds(self):
        half_span = len(self.indices) * self.min_diff / 2.0
        return (self.position - half_span, self.position + half_span)

    def __repr__(self):
        return f'LegendGroup({self.indices}, {self.init_positions}, {self.min_diff})'

def compute_best_placements(positions, min_diff):
    """Provides a list of new positions from a list of initial position, with a minimal
    distance between each position.

    Parameters
    ----------
    positions: List[float]
        best positions if minimal distance were 0.
    min_diff: float
        minimal distance allowed between two positions

    Returns
    -------
    new_positions: List[float]
        positions after taking into account the minimal distance constraint

    Note
    ----
    This function is probably not optimal, but seems a very good heuristic
    """
    assert all((v2 >= v1 for v2, v1 in zip(positions[1:], positions[:-1])))
    groups = [LegendGroup([k], [pos], min_diff) for k, pos in enumerate(positions)]
    new_groups = []
    ready = False
    while not ready:
        ready = True
        for k in range(len(groups)):
            if k < len(groups) - 1 and groups[k + 1].bounds[0] < groups[k].bounds[1]:
                new_groups.append(groups[k].combine_with(groups[k + 1]))
                new_groups.extend(groups[k + 2:])
                groups = new_groups
                new_groups = []
                ready = False
                break
            new_groups.append(groups[k])
    new_positions = np.array(positions, copy=True)
    for group in groups:
        new_positions[group.indices] = group.get_positions()
    return new_positions.tolist()

def main():
    parser = argparse.ArgumentParser(description='Create plots from an experiment data file')
    parser.add_argument('filepath', type=str, help='filepath containing the experiment data')
    parser.add_argument('--output', type=str, default=None, help='Output path for the CSV file (default: a folder <filename>_plots next to the data file.')
    parser.add_argument('--max_combsize', type=int, default=0, help='maximum number of parameters to fix (combinations) when creating experiment plots')
    parser.add_argument('--pseudotime', nargs='?', default=False, const=True, help='Plots with respect to pseudotime instead of budget')
    parser.add_argument('--competencemaps', type=bool, default=False, help='whether we should export only competence maps')
    parser.add_argument('--nomanyxp', type=bool, default=False, help='whether we should remove the export of detailed convergence curves')
    parser.add_argument('--merge-parametrization', action='store_true', help='if present, parametrization is merge into the optimizer name')
    parser.add_argument('--remove-suffix', action='store_true', help='if present, remove numerical suffixes in fight plots')
    parser.add_argument('--merge-pattern', type=str, default='', help=f'if present, optimizer name is updated according to the pattern as an f-string. --merge-parametrization is equivalent to using --merge-pattern with {_PARAM_MERGE_PATTERN!r}')
    args = parser.parse_args()
    exp_df = merge_optimizer_name_pattern(utils.Selector.read_csv(args.filepath), args.merge_pattern, args.merge_parametrization, args.remove_suffix)
    exp_df.replace('CSEC11', 'NGIohTuned', inplace=True)
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.filepath).with_suffix('')) + '_plots'
    create_plots(exp_df, output_folder=output_dir, max_combsize=args.max_combsize if not args.competencemaps else 2, xpaxis='pseudotime' if args.pseudotime else 'budget', competencemaps=args.competencemaps, nomanyxp=args.nomanyxp)
if __name__ == '__main__':
    main()