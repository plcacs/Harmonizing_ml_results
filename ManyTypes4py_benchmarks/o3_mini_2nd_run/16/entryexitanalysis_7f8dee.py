import logging
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Any, Optional, Union
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, load_backtest_data, load_backtest_stats, load_exit_signal_candles, load_rejected_signals, load_signal_candles
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.util import print_df_rich_table

logger = logging.getLogger(__name__)


def _process_candles_and_indicators(
    pairlist: List[str],
    strategy_name: str,
    trades: DataFrame,
    signal_candles: Dict[str, Dict[str, DataFrame]],
    date_col: str = 'open_date'
) -> Dict[str, Dict[str, DataFrame]]:
    analysed_trades_dict: Dict[str, Dict[str, DataFrame]] = {strategy_name: {}}
    try:
        logger.info(f'Processing {strategy_name} : {len(pairlist)} pairs')
        for pair in pairlist:
            if pair in signal_candles[strategy_name]:
                analysed_trades_dict[strategy_name][pair] = _analyze_candles_and_indicators(
                    pair, trades, signal_candles[strategy_name][pair], date_col
                )
    except Exception as e:
        print(f'Cannot process entry/exit reasons for {strategy_name}: ', e)
    return analysed_trades_dict


def _analyze_candles_and_indicators(
    pair: str,
    trades: DataFrame,
    signal_candles: DataFrame,
    date_col: str = 'open_date'
) -> DataFrame:
    buyf = signal_candles
    if len(buyf) > 0:
        buyf = buyf.set_index('date', drop=False)
        trades_red = trades.loc[trades['pair'] == pair].copy()
        trades_inds = pd.DataFrame()
        if trades_red.shape[0] > 0 and buyf.shape[0] > 0:
            for t, v in trades_red.iterrows():
                allinds = buyf.loc[buyf['date'] < v[date_col]]
                if allinds.shape[0] > 0:
                    tmp_inds = allinds.iloc[[-1]]
                    trades_red.loc[t, 'signal_date'] = tmp_inds['date'].values[0]
                    trades_red.loc[t, 'enter_reason'] = trades_red.loc[t, 'enter_tag']
                    tmp_inds.index.rename('signal_date', inplace=True)
                    trades_inds = pd.concat([trades_inds, tmp_inds])
            if 'signal_date' in trades_red:
                trades_red['signal_date'] = pd.to_datetime(trades_red['signal_date'], utc=True)
                trades_red.set_index('signal_date', inplace=True)
                try:
                    trades_red = pd.merge(trades_red, trades_inds, on='signal_date', how='outer')
                except Exception as e:
                    raise e
        return trades_red
    else:
        return pd.DataFrame()


def _do_group_table_output(
    bigdf: DataFrame,
    glist: List[str],
    csv_path: Path,
    to_csv: bool = False
) -> None:
    for g in glist:
        if g == '0':
            group_mask = ['enter_reason']
            wins = bigdf.loc[bigdf['profit_abs'] >= 0].groupby(group_mask).agg({'profit_abs': ['sum']})
            wins.columns = ['profit_abs_wins']
            loss = bigdf.loc[bigdf['profit_abs'] < 0].groupby(group_mask).agg({'profit_abs': ['sum']})
            loss.columns = ['profit_abs_loss']
            new = bigdf.groupby(group_mask).agg({'profit_abs': ['count', lambda x: sum(x > 0), lambda x: sum(x <= 0)]})
            new = pd.concat([new, wins, loss], axis=1).fillna(0)
            new['profit_tot'] = new['profit_abs_wins'] - abs(new['profit_abs_loss'])
            new['wl_ratio_pct'] = (new.iloc[:, 1] / new.iloc[:, 0] * 100).fillna(0)
            new['avg_win'] = (new['profit_abs_wins'] / new.iloc[:, 1]).fillna(0)
            new['avg_loss'] = (new['profit_abs_loss'] / new.iloc[:, 2]).fillna(0)
            new['exp_ratio'] = ((1 + new['avg_win'] / abs(new['avg_loss'])) * (new['wl_ratio_pct'] / 100) - 1).fillna(0)
            new.columns = ['total_num_buys', 'wins', 'losses', 'profit_abs_wins', 'profit_abs_loss', 'profit_tot', 'wl_ratio_pct', 'avg_win', 'avg_loss', 'exp_ratio']
            sortcols = ['total_num_buys']
            _print_table(new, sortcols, show_index=True, name='Group 0:', to_csv=to_csv, csv_path=csv_path)
        else:
            agg_mask = {'profit_abs': ['count', 'sum', 'median', 'mean'], 'profit_ratio': ['median', 'mean', 'sum']}
            agg_cols = ['num_buys', 'profit_abs_sum', 'profit_abs_median', 'profit_abs_mean', 'median_profit_pct', 'mean_profit_pct', 'total_profit_pct']
            sortcols = ['profit_abs_sum', 'enter_reason']
            group_mask: Optional[List[str]] = None
            if g == '1':
                group_mask = ['enter_reason']
            if g == '2':
                group_mask = ['enter_reason', 'exit_reason']
            if g == '3':
                group_mask = ['pair', 'enter_reason']
            if g == '4':
                group_mask = ['pair', 'enter_reason', 'exit_reason']
            if g == '5':
                group_mask = ['exit_reason']
                sortcols = ['exit_reason']
            if group_mask:
                new = bigdf.groupby(group_mask).agg(agg_mask).reset_index()
                new.columns = group_mask + agg_cols
                new['median_profit_pct'] = new['median_profit_pct'] * 100
                new['mean_profit_pct'] = new['mean_profit_pct'] * 100
                new['total_profit_pct'] = new['total_profit_pct'] * 100
                _print_table(new, sortcols, name=f'Group {g}:', to_csv=to_csv, csv_path=csv_path)
            else:
                logger.warning('Invalid group mask specified.')


def _do_rejected_signals_output(
    rejected_signals_df: DataFrame,
    to_csv: bool = False,
    csv_path: Optional[Path] = None
) -> None:
    cols = ['pair', 'date', 'enter_tag']
    sortcols = ['date', 'pair', 'enter_tag']
    _print_table(rejected_signals_df[cols], sortcols, show_index=False, name='Rejected Signals:', to_csv=to_csv, csv_path=csv_path)  # type: ignore


def _select_rows_within_dates(
    df: DataFrame,
    timerange: Optional[TimeRange] = None,
    df_date_col: str = 'date'
) -> DataFrame:
    if timerange:
        if timerange.starttype == 'date':
            df = df.loc[df[df_date_col] >= timerange.startdt]
        if timerange.stoptype == 'date':
            df = df.loc[df[df_date_col] < timerange.stopdt]
    return df


def _select_rows_by_tags(
    df: DataFrame,
    enter_reason_list: List[str],
    exit_reason_list: List[str]
) -> DataFrame:
    if enter_reason_list and 'all' not in enter_reason_list:
        df = df.loc[df['enter_reason'].isin(enter_reason_list)]
    if exit_reason_list and 'all' not in exit_reason_list:
        df = df.loc[df['exit_reason'].isin(exit_reason_list)]
    return df


def prepare_results(
    analysed_trades: Dict[str, Dict[str, DataFrame]],
    stratname: str,
    enter_reason_list: List[str],
    exit_reason_list: List[str],
    timerange: Optional[TimeRange] = None
) -> DataFrame:
    res_df = pd.DataFrame()
    for pair, trades in analysed_trades[stratname].items():
        if trades.shape[0] > 0:
            trades.dropna(subset=['close_date'], inplace=True)
            res_df = pd.concat([res_df, trades], ignore_index=True)
    res_df = _select_rows_within_dates(res_df, timerange)
    if res_df is not None and res_df.shape[0] > 0 and ('enter_reason' in res_df.columns):
        res_df = _select_rows_by_tags(res_df, enter_reason_list, exit_reason_list)
    return res_df


def print_results(
    res_df: DataFrame,
    exit_df: Optional[DataFrame],
    analysis_groups: List[str],
    indicator_list: List[str],
    entry_only: bool,
    exit_only: bool,
    csv_path: Path,
    rejected_signals: Optional[DataFrame] = None,
    to_csv: bool = False
) -> None:
    if res_df.shape[0] > 0:
        if analysis_groups:
            _do_group_table_output(res_df, analysis_groups, csv_path, to_csv=to_csv)
        if rejected_signals is not None:
            if rejected_signals.empty:
                print('There were no rejected signals.')
            else:
                _do_rejected_signals_output(rejected_signals, to_csv=to_csv, csv_path=csv_path)
        if 'all' in indicator_list:
            _print_table(res_df, show_index=False, name='Indicators:', to_csv=to_csv, csv_path=csv_path)
        elif indicator_list is not None and indicator_list:
            available_inds: List[str] = []
            for ind in indicator_list:
                if ind in res_df:
                    available_inds.append(ind)
            merged_df = _merge_dfs(res_df, exit_df, available_inds, entry_only, exit_only)
            _print_table(merged_df, sortcols=['exit_reason'], show_index=False, name='Indicators:', to_csv=to_csv, csv_path=csv_path)
    else:
        print('\\No trades to show')


def _merge_dfs(
    entry_df: DataFrame,
    exit_df: Optional[DataFrame],
    available_inds: List[str],
    entry_only: bool,
    exit_only: bool
) -> DataFrame:
    merge_on = ['pair', 'open_date']
    signal_wide_indicators: List[str] = list(set(available_inds) - set(BT_DATA_COLUMNS))
    columns_to_keep = merge_on + ['enter_reason', 'exit_reason']
    if exit_df is None or exit_df.empty or entry_only is True:
        return entry_df[columns_to_keep + available_inds]
    if exit_only is True:
        return pd.merge(
            entry_df[columns_to_keep],
            exit_df[merge_on + signal_wide_indicators],
            on=merge_on,
            suffixes=(' (entry)', ' (exit)')
        )
    return pd.merge(
        entry_df[columns_to_keep + available_inds],
        exit_df[merge_on + signal_wide_indicators],
        on=merge_on,
        suffixes=(' (entry)', ' (exit)')
    )


def _print_table(
    df: DataFrame,
    sortcols: Optional[List[str]] = None,
    *,
    show_index: bool = False,
    name: Optional[str] = None,
    to_csv: bool = False,
    csv_path: Union[Path, None]
) -> None:
    if sortcols is not None:
        data = df.sort_values(sortcols)
    else:
        data = df
    if to_csv:
        safe_name = Path(csv_path, name.lower().replace(' ', '_').replace(':', '') + '.csv')  # type: ignore
        data.to_csv(safe_name)
        print(f'Saved {name} to {safe_name}')
    else:
        if name is not None:
            print(name)
        print_df_rich_table(data, list(data.keys()), show_index=show_index)


def process_entry_exit_reasons(config: Dict[str, Any]) -> None:
    try:
        analysis_groups: List[str] = config.get('analysis_groups', [])
        enter_reason_list: List[str] = config.get('enter_reason_list', ['all'])
        exit_reason_list: List[str] = config.get('exit_reason_list', ['all'])
        indicator_list: List[str] = config.get('indicator_list', [])
        entry_only: bool = config.get('entry_only', False)
        exit_only: bool = config.get('exit_only', False)
        do_rejected: bool = config.get('analysis_rejected', False)
        to_csv: bool = config.get('analysis_to_csv', False)
        csv_path: Path = Path(config.get('analysis_csv_path', config['exportfilename']))
        if entry_only is True and exit_only is True:
            raise OperationalException('Cannot use --entry-only and --exit-only at the same time. Please choose one.')
        if to_csv and (not csv_path.is_dir()):
            raise OperationalException(f'Specified directory {csv_path} does not exist.')
        timerange: Optional[TimeRange] = TimeRange.parse_timerange(
            None if config.get('timerange') is None else str(config.get('timerange'))
        )
        try:
            backtest_stats = load_backtest_stats(config['exportfilename'])
        except ValueError as e:
            raise ConfigurationError(e) from e
        for strategy_name, results in backtest_stats['strategy'].items():
            trades: DataFrame = load_backtest_data(config['exportfilename'], strategy_name)
            if trades is not None and (not trades.empty):
                signal_candles: Dict[str, Dict[str, DataFrame]] = load_signal_candles(config['exportfilename'])
                exit_signals: Dict[str, Dict[str, DataFrame]] = load_exit_signal_candles(config['exportfilename'])
                rej_df: Optional[DataFrame] = None
                if do_rejected:
                    rejected_signals_dict: Dict[str, Dict[str, DataFrame]] = load_rejected_signals(config['exportfilename'])
                    rej_df = prepare_results(rejected_signals_dict, strategy_name, enter_reason_list, exit_reason_list, timerange=timerange)
                entry_df: DataFrame = _generate_dfs(
                    config['exchange']['pair_whitelist'], enter_reason_list, exit_reason_list,
                    signal_candles, strategy_name, timerange, trades, 'open_date'
                )
                exit_df: DataFrame = _generate_dfs(
                    config['exchange']['pair_whitelist'], enter_reason_list, exit_reason_list,
                    exit_signals, strategy_name, timerange, trades, 'close_date'
                )
                print_results(entry_df, exit_df, analysis_groups, indicator_list, entry_only, exit_only, csv_path, rejected_signals=rej_df, to_csv=to_csv)
    except ValueError as e:
        raise OperationalException(e) from e


def _generate_dfs(
    pairlist: List[str],
    enter_reason_list: List[str],
    exit_reason_list: List[str],
    signal_candles: Dict[str, Dict[str, DataFrame]],
    strategy_name: str,
    timerange: Optional[TimeRange],
    trades: DataFrame,
    date_col: str
) -> DataFrame:
    analysed_trades_dict: Dict[str, Dict[str, DataFrame]] = _process_candles_and_indicators(pairlist, strategy_name, trades, signal_candles, date_col)
    res_df: DataFrame = prepare_results(analysed_trades_dict, strategy_name, enter_reason_list, exit_reason_list, timerange=timerange)
    return res_df
