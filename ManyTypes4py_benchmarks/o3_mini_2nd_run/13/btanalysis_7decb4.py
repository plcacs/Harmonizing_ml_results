#!/usr/bin/env python3
"""
Helpers when analyzing backtest data
"""
import logging
import zipfile
from copy import copy
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Union, Literal, Optional, List, Dict
import numpy as np
import pandas as pd
from freqtrade.constants import LAST_BT_RESULT_FN, IntOrInf
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.ft_types import BacktestHistoryEntryType, BacktestResultType
from freqtrade.misc import file_dump_json, json_load
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename
from freqtrade.persistence import LocalTrade, Trade, init_db

logger = logging.getLogger(__name__)

BT_DATA_COLUMNS = ['pair', 'stake_amount', 'max_stake_amount', 'amount', 'open_date', 'close_date', 'open_rate', 'close_rate', 'fee_open', 'fee_close', 'trade_duration', 'profit_ratio', 'profit_abs', 'exit_reason', 'initial_stop_loss_abs', 'initial_stop_loss_ratio', 'stop_loss_abs', 'stop_loss_ratio', 'min_rate', 'max_rate', 'is_open', 'enter_tag', 'leverage', 'is_short', 'open_timestamp', 'close_timestamp', 'orders']


def get_latest_optimize_filename(directory: Union[str, Path], variant: Literal['backtest', 'hyperopt']) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :param variant: 'backtest' or 'hyperopt' - the method to return
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist.")
    filename = directory / LAST_BT_RESULT_FN
    if not filename.is_file():
        raise ValueError(f"Directory '{directory}' does not seem to contain backtest statistics yet.")
    with filename.open() as file:
        data = json_load(file)
    if f'latest_{variant}' not in data:
        raise ValueError(f"Invalid '{LAST_BT_RESULT_FN}' format.")
    return data[f'latest_{variant}']


def get_latest_backtest_filename(directory: Union[str, Path]) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    return get_latest_optimize_filename(directory, 'backtest')


def get_latest_hyperopt_filename(directory: Union[str, Path]) -> str:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest hyperopt result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    try:
        return get_latest_optimize_filename(directory, 'hyperopt')
    except ValueError:
        return 'hyperopt_results.pickle'


def get_latest_hyperopt_file(directory: Union[str, Path], predef_filename: Optional[str] = None) -> Path:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :param predef_filename: Optional predefined filename.
    :return: Path of the latest hyperopt result file.
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if predef_filename:
        if Path(predef_filename).is_absolute():
            raise ConfigurationError('--hyperopt-filename expects only the filename, not an absolute path.')
        return directory / predef_filename
    return directory / get_latest_hyperopt_filename(directory)


def load_backtest_metadata(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Read metadata dictionary from backtest results file without reading and deserializing entire
    file.
    :param filename: path to backtest results file.
    :return: metadata dict or empty dict if metadata is not present.
    """
    filename = get_backtest_metadata_filename(filename)
    try:
        with filename.open() as fp:
            return json_load(fp)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise OperationalException('Unexpected error while loading backtest metadata.') from e


def load_backtest_stats(filename: Union[str, Path]) -> BacktestResultType:
    """
    Load backtest statistics file.
    :param filename: pathlib.Path object, or string pointing to the file.
    :return: a dictionary containing the resulting file.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.is_dir():
        filename = filename / get_latest_backtest_filename(filename)
    if not filename.is_file():
        raise ValueError(f'File {filename} does not exist.')
    logger.info(f'Loading backtest result from {filename}')
    if filename.suffix == '.zip':
        data = json_load(StringIO(load_file_from_zip(filename, filename.with_suffix('.json').name).decode('utf-8')))
    else:
        with filename.open() as file:
            data = json_load(file)
    if isinstance(data, dict):
        data['metadata'] = load_backtest_metadata(filename)
    return data


def load_and_merge_backtest_result(strategy_name: str, filename: Path, results: Dict[str, Any]) -> None:
    """
    Load one strategy from multi-strategy result and merge it with results
    :param strategy_name: Name of the strategy contained in the result
    :param filename: Backtest-result-filename to load
    :param results: dict to merge the result to.
    """
    bt_data = load_backtest_stats(filename)
    for k in ('metadata', 'strategy'):
        results[k][strategy_name] = bt_data[k][strategy_name]
    results['metadata'][strategy_name]['filename'] = filename.stem
    comparison = bt_data['strategy_comparison']
    for i in range(len(comparison)):
        if comparison[i]['key'] == strategy_name:
            results['strategy_comparison'].append(comparison[i])
            break


def _get_backtest_files(dirname: Path) -> List[Path]:
    json_files = dirname.glob('backtest-result-*-[0-9][0-9]*.json')
    zip_files = dirname.glob('backtest-result-*-[0-9][0-9]*.zip')
    return list(reversed(sorted(list(json_files) + list(zip_files))))


def _extract_backtest_result(filename: Path) -> List[Dict[str, Any]]:
    metadata: Dict[str, Any] = load_backtest_metadata(filename)
    return [
        {
            'filename': filename.stem,
            'strategy': s,
            'run_id': v['run_id'],
            'notes': v.get('notes', ''),
            'backtest_start_time': v['backtest_start_time'],
            'backtest_start_ts': v.get('backtest_start_ts', None),
            'backtest_end_ts': v.get('backtest_end_ts', None),
            'timeframe': v.get('timeframe', None),
            'timeframe_detail': v.get('timeframe_detail', None)
        }
        for s, v in metadata.items()
    ]


def get_backtest_result(filename: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Get backtest result read from metadata file
    """
    if isinstance(filename, str):
        filename = Path(filename)
    return _extract_backtest_result(filename)


def get_backtest_resultlist(dirname: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Get list of backtest results read from metadata files
    """
    if isinstance(dirname, str):
        dirname = Path(dirname)
    return [result for filename in _get_backtest_files(dirname) for result in _extract_backtest_result(filename)]


def delete_backtest_result(file_abs: Path) -> None:
    """
    Delete backtest result file and corresponding metadata file.
    """
    logger.info(f'Deleting backtest result file: {file_abs.name}')
    for file in file_abs.parent.glob(f'{file_abs.stem}*'):
        logger.info(f'Deleting file: {file}')
        file.unlink()


def update_backtest_metadata(filename: Union[str, Path], strategy: str, content: Dict[str, Any]) -> None:
    """
    Updates backtest metadata file with new content.
    :raises: ValueError if metadata file does not exist, or strategy is not in this file.
    """
    metadata = load_backtest_metadata(filename)
    if not metadata:
        raise ValueError('File does not exist.')
    if strategy not in metadata:
        raise ValueError('Strategy not in metadata.')
    metadata[strategy].update(content)
    file_dump_json(get_backtest_metadata_filename(filename), metadata)


def get_backtest_market_change(filename: Union[str, Path], include_ts: bool = True) -> pd.DataFrame:
    """
    Read backtest market change file.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.suffix == '.zip':
        data = load_file_from_zip(filename, f'{filename.stem}_market_change.feather')
        df = pd.read_feather(BytesIO(data))
    else:
        df = pd.read_feather(filename)
    if include_ts:
        df.loc[:, '__date_ts'] = df.loc[:, 'date'].astype(np.int64) // 1000 // 1000
    return df


def find_existing_backtest_stats(dirname: Union[str, Path], run_ids: Dict[str, str],
                                 min_backtest_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Find existing backtest stats that match specified run IDs and load them.
    :param dirname: pathlib.Path object, or string pointing to the file.
    :param run_ids: {strategy_name: id_string} dictionary.
    :param min_backtest_date: do not load a backtest older than specified date.
    :return: results dict.
    """
    run_ids = copy(run_ids)
    if isinstance(dirname, str):
        dirname = Path(dirname)
    results: Dict[str, Any] = {'metadata': {}, 'strategy': {}, 'strategy_comparison': []}
    for filename in _get_backtest_files(dirname):
        metadata = load_backtest_metadata(filename)
        if not metadata:
            break
        for strategy_name, run_id in list(run_ids.items()):
            strategy_metadata = metadata.get(strategy_name, None)
            if not strategy_metadata:
                continue
            if min_backtest_date is not None:
                backtest_date = strategy_metadata['backtest_start_time']
                backtest_date = datetime.fromtimestamp(backtest_date, tz=timezone.utc)
                if backtest_date < min_backtest_date:
                    del run_ids[strategy_name]
                    continue
            if strategy_metadata['run_id'] == run_id:
                del run_ids[strategy_name]
                load_and_merge_backtest_result(strategy_name, filename, results)
        if len(run_ids) == 0:
            break
    return results


def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibility support for older backtest data.
    """
    df['open_date'] = pd.to_datetime(df['open_date'], utc=True)
    df['close_date'] = pd.to_datetime(df['close_date'], utc=True)
    if 'is_short' not in df.columns:
        df['is_short'] = False
    if 'leverage' not in df.columns:
        df['leverage'] = 1.0
    if 'enter_tag' not in df.columns:
        df['enter_tag'] = df['buy_tag']
        df = df.drop(['buy_tag'], axis=1)
    if 'max_stake_amount' not in df.columns:
        df['max_stake_amount'] = df['stake_amount']
    if 'orders' not in df.columns:
        df['orders'] = None
    return df


def load_backtest_data(filename: Union[str, Path], strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Load backtest data file.
    :param filename: pathlib.Path object, or string pointing to a file or directory
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests.
                     Can also serve as protection to load the correct result.
    :return: a dataframe with the analysis results
    :raise: ValueError if loading goes wrong.
    """
    data = load_backtest_stats(filename)
    if not isinstance(data, list):
        if 'strategy' not in data:
            raise ValueError('Unknown dataformat.')
        if not strategy:
            if len(data['strategy']) == 1:
                strategy = list(data['strategy'].keys())[0]
            else:
                raise ValueError('Detected backtest result with more than one strategy. Please specify a strategy.')
        if strategy not in data['strategy']:
            raise ValueError(f"Strategy {strategy} not available in the backtest result. Available strategies are '{','.join(list(data['strategy'].keys()))}'")
        data_trades = data['strategy'][strategy]['trades']
        df = pd.DataFrame(data_trades)
        if not df.empty:
            df = _load_backtest_data_df_compatibility(df)
    else:
        raise OperationalException('Backtest-results with only trades data are no longer supported.')
    if not df.empty:
        df = df.sort_values('open_date').reset_index(drop=True)
    return df


def load_file_from_zip(zip_path: Union[str, Path], filename: str) -> bytes:
    """
    Load a file from a zip file
    :param zip_path: Path to the zip file
    :param filename: Name of the file to load
    :return: Bytes of the file
    :raises: ValueError if loading goes wrong.
    """
    if isinstance(zip_path, str):
        zip_path = Path(zip_path)
    try:
        with zipfile.ZipFile(zip_path) as zipf:
            try:
                with zipf.open(filename) as file:
                    return file.read()
            except KeyError:
                logger.error(f'File {filename} not found in zip: {zip_path}')
                raise ValueError(f'File {filename} not found in zip: {zip_path}') from None
    except FileNotFoundError:
        raise ValueError(f'Zip file {zip_path} not found.')
    except zipfile.BadZipFile:
        logger.error(f'Bad zip file: {zip_path}.')
        raise ValueError(f'Bad zip file: {zip_path}.') from None


def load_backtest_analysis_data(backtest_dir: Union[str, Path], name: str) -> Any:
    """
    Load backtest analysis data either from a pickle file or from within a zip file
    :param backtest_dir: Directory containing backtest results
    :param name: Name of the analysis data to load (signals, rejected, exited)
    :return: Analysis data
    """
    import joblib
    if isinstance(backtest_dir, str):
        backtest_dir = Path(backtest_dir)
    if backtest_dir.is_dir():
        lbf = Path(get_latest_backtest_filename(backtest_dir))
        zip_path = backtest_dir / lbf
    else:
        zip_path = backtest_dir
    if zip_path.suffix == '.zip':
        analysis_name = f'{zip_path.stem}_{name}.pkl'
        data = load_file_from_zip(zip_path, analysis_name)
        if not data:
            return None
        loaded_data = joblib.load(BytesIO(data))
        logger.info(f'Loaded {name} candles from zip: {str(zip_path)}:{analysis_name}')
        return loaded_data
    else:
        if backtest_dir.is_dir():
            scpf = Path(backtest_dir, f'{zip_path.stem}_{name}.pkl')
        else:
            scpf = Path(backtest_dir.parent / f'{backtest_dir.stem}_{name}.pkl')
        try:
            with scpf.open('rb') as scp:
                loaded_data = joblib.load(scp)
                logger.info(f'Loaded {name} candles: {str(scpf)}')
                return loaded_data
        except Exception:
            logger.exception(f'Cannot load {name} data from pickled results.')
            return None


def load_rejected_signals(backtest_dir: Union[str, Path]) -> Any:
    """
    Load rejected signals from backtest directory
    """
    return load_backtest_analysis_data(backtest_dir, 'rejected')


def load_signal_candles(backtest_dir: Union[str, Path]) -> Any:
    """
    Load signal candles from backtest directory
    """
    return load_backtest_analysis_data(backtest_dir, 'signals')


def load_exit_signal_candles(backtest_dir: Union[str, Path]) -> Any:
    """
    Load exit signal candles from backtest directory
    """
    return load_backtest_analysis_data(backtest_dir, 'exited')


def analyze_trade_parallelism(results: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps.
    :param results: Results DataFrame - can be loaded
    :param timeframe: Timeframe used for backtest
    :return: DataFrame with open-counts per time-period in timeframe
    """
    from freqtrade.exchange import timeframe_to_resample_freq
    timeframe_freq: str = timeframe_to_resample_freq(timeframe)
    dates_list = [pd.Series(pd.date_range(row['open_date'], row['close_date'], freq=timeframe_freq, inclusive='left'))
                  for _, row in results[['open_date', 'close_date']].iterrows()]
    deltas: List[int] = [len(x) for x in dates_list]
    dates = pd.Series(pd.concat(dates_list).values, name='date')
    df2 = pd.DataFrame(np.repeat(results.values, deltas, axis=0), columns=results.columns)
    df2 = pd.concat([dates, df2], axis=1)
    df2 = df2.set_index('date')
    df_final = df2.resample(timeframe_freq)[['pair']].count()
    df_final = df_final.rename({'pair': 'open_trades'}, axis=1)
    return df_final


def evaluate_result_multi(results: pd.DataFrame, timeframe: str, max_open_trades: int) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps
    :param results: Results DataFrame - can be loaded
    :param timeframe: Frequency used for the backtest
    :param max_open_trades: parameter max_open_trades used during backtest run
    :return: DataFrame with open-counts per time-period in frequency
    """
    df_final = analyze_trade_parallelism(results, timeframe)
    return df_final[df_final['open_trades'] > max_open_trades]


def trade_list_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """
    Convert list of Trade objects to pandas DataFrame
    :param trades: List of trade objects
    :return: DataFrame with BT_DATA_COLUMNS
    """
    df = pd.DataFrame.from_records([t.to_json(True) for t in trades], columns=BT_DATA_COLUMNS)
    if len(df) > 0:
        df['close_date'] = pd.to_datetime(df['close_date'], utc=True)
        df['open_date'] = pd.to_datetime(df['open_date'], utc=True)
        df['close_rate'] = df['close_rate'].astype('float64')
    return df


def load_trades_from_db(db_url: str, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Load trades from a DB (using dburl)
    :param db_url: Sqlite url (default format sqlite:///tradesv3.dry-run.sqlite)
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests.
                     Can also serve as protection to load the correct result.
    :return: DataFrame containing Trades
    """
    init_db(db_url)
    filters = []
    if strategy:
        from freqtrade.persistence import Trade  # ensure Trade is imported
        filters.append(Trade.strategy == strategy)
    trades = trade_list_to_dataframe(list(Trade.get_trades(filters).all()))
    return trades


def load_trades(source: str, db_url: str, exportfilename: Union[str, Path],
                no_trades: bool = False, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Based on configuration option 'trade_source':
    * loads data from DB (using `db_url`)
    * loads data from backtest file (using `exportfilename`)
    :param source: "DB" or "file" - specify source to load from
    :param db_url: sqlalchemy formatted url to a database
    :param exportfilename: Json file generated by backtesting
    :param no_trades: Skip using trades, only return backtesting data columns
    :param strategy: Optional strategy name to filter
    :return: DataFrame containing trades
    """
    if no_trades:
        df = pd.DataFrame(columns=BT_DATA_COLUMNS)
        return df
    if source == 'DB':
        return load_trades_from_db(db_url, strategy)
    elif source == 'file':
        return load_backtest_data(exportfilename, strategy)
    else:
        raise ValueError("Invalid trade source. Expected 'DB' or 'file'.")


def extract_trades_of_period(dataframe: pd.DataFrame, trades: pd.DataFrame, date_index: bool = False) -> pd.DataFrame:
    """
    Compare trades and backtested pair DataFrames to get trades performed on backtested period
    :param dataframe: DataFrame with backtested data with a date column or index
    :param trades: DataFrame of trades
    :param date_index: Use index for date instead of 'date' column in dataframe
    :return: DataFrame of trades in the period
    """
    if date_index:
        trades_start = dataframe.index[0]
        trades_stop = dataframe.index[-1]
    else:
        trades_start = dataframe.iloc[0]['date']
        trades_stop = dataframe.iloc[-1]['date']
    filtered_trades = trades.loc[(trades['open_date'] >= trades_start) & (trades['close_date'] <= trades_stop)]
    return filtered_trades
