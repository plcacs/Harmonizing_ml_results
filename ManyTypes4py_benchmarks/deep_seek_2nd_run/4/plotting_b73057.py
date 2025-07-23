import os
import re
import string
import hashlib
import warnings
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union, cast
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
pure_algorithms: List[str] = []

def compactize(name: str) -> str:
    if len(name) < 70:
        return name
    hashcode = hashlib.md5(bytes(name, 'utf8')).hexdigest()
    name = re.sub('\\([^()]*\\)', '', name)
    mid = 35
    name = name[:mid] + hashcode + name[-mid:]
    return name

def _make_style_generator() -> itertools.cycle[str]:
    lines = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle('ov^<>8sp*hHDd')
    colors = itertools.cycle('bgrcmyk')
    return (l + m + c for l, m, c in zip(lines, markers, colors))

class NameStyle(tp.Dict[str, str]):
    """Provides a style for each name, and keeps to it"""

    def __init__(self) -> None:
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name: str) -> str:
        if name not in self:
            super().__setitem__(name, next(self._gen))
        return super().__getitem__(name)

def _make_winners_df(df: Union[pd.DataFrame, utils.Selector], all_optimizers: List[str]) -> utils.Selector:
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

def aggregate_winners(df: utils.Selector, categories: List[str], all_optimizers: List[str]) -> Tuple[utils.Selector, int]:
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

def _make_sorted_winrates_df(victories: utils.Selector) -> pd.DataFrame:
    """Converts a dataframe counting number of victories into a sorted
    winrate dataframe. The algorithm which performs better than all other
    algorithms comes first. When you do not play in a category, you are
    considered as having lost all comparisons in that category.
    """
    assert all((x == y for x, y in zip(victories.index, victories.columns)))
    winrates = victories / (victories + victories.T).max(axis=1)
    mean_win = winrates.mean(axis=1).sort_values(ascending=False)
    return winrates.loc[mean_win.index, mean_win.index]

def remove_errors(df: Union[pd.DataFrame, utils.Selector]) -> utils.Selector:
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

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    def __call__(self, df: pd.Series) -> str:
        return self._pattern.format(**df.to_dict())

_PARAM_MERGE_PATTERN = '{optimizer_name},{parametrization}'

def merge_optimizer_name_pattern(df: pd.DataFrame, pattern: str, merge_parametrization: bool = False, remove_suffix: bool = False) -> pd.DataFrame:
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

def normalized_losses(df: utils.Selector, descriptors: List[str]) -> utils.Selector:
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

def create_plots(df: pd.DataFrame, output_folder: Union[str, Path], max_combsize: int = 1, xpaxis: str = 'budget', competencemaps: bool = False, nomanyxp: bool = False) -> None:
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
        df.loc[:, 'loss'] = df.loc[:, 'loss'].clip(lower=-1e26, upper=upper)
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
    if