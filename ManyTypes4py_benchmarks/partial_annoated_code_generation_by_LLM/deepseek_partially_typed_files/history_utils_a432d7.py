import logging
import operator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pandas import DataFrame, concat
from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT, DL_DATA_TIMEFRAMES, DOCS_LINK, Config
from freqtrade.data.converter import clean_ohlcv_dataframe, convert_trades_to_ohlcv, trades_df_remove_duplicates, trades_list_to_df
from freqtrade.data.history.datahandlers import IDataHandler, get_datahandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.util import dt_now, dt_ts, format_ms_time, format_ms_time_det
from freqtrade.util.migrations import migrate_data
from freqtrade.util.progress_tracker import CustomProgress, retrieve_progress_tracker

logger = logging.getLogger(__name__)


def load_pair_history(
    pair: str,
    timeframe: str,
    datadir: Path,
    *,
    timerange: Optional[TimeRange] = None,
    fill_up_missing: bool = True,
    drop_incomplete: bool = False,
    startup_candles: int = 0,
    data_format: Optional[str] = None,
    data_handler: Optional[IDataHandler] = None,
    candle_type: CandleType = CandleType.SPOT
) -> DataFrame:
    """
    Load cached ohlcv history for the given pair.

    :param pair: Pair to load data for
    :param timeframe: Timeframe (e.g. "5m")
    :param datadir: Path to the data storage location.
    :param data_format: Format of the data. Ignored if data_handler is set.
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param drop_incomplete: Drop last candle assuming it may be incomplete.
    :param startup_candles: Additional candles to load at the start of the period
    :param data_handler: Initialized data-handler to use.
                         Will be initialized from data_format if not set
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :return: DataFrame with ohlcv data, or empty DataFrame
    """
    data_handler = get_datahandler(datadir, data_format, data_handler)
    return data_handler.ohlcv_load(
        pair=pair,
        timeframe=timeframe,
        timerange=timerange,
        fill_missing=fill_up_missing,
        drop_incomplete=drop_incomplete,
        startup_candles=startup_candles,
        candle_type=candle_type
    )


def load_data(
    datadir: Path,
    timeframe: str,
    pairs: List[str],
    *,
    timerange: Optional[TimeRange] = None,
    fill_up_missing: bool = True,
    startup_candles: int = 0,
    fail_without_data: bool = False,
    data_format: str = 'feather',
    candle_type: CandleType = CandleType.SPOT,
    user_futures_funding_rate: Optional[int] = None
) -> Dict[str, DataFrame]:
    """
    Load ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param startup_candles: Additional candles to load at the start of the period
    :param fail_without_data: Raise OperationalException if no data is found.
    :param data_format: Data format which should be used. Defaults to json
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :return: dict(<pair>:<Dataframe>)
    """
    result: Dict[str, DataFrame] = {}
    if startup_candles > 0 and timerange:
        logger.info(f'Using indicator startup period: {startup_candles} ...')
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        hist = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            fill_up_missing=fill_up_missing,
            startup_candles=startup_candles,
            data_handler=data_handler,
            candle_type=candle_type
        )
        if not hist.empty:
            result[pair] = hist
        elif candle_type is CandleType.FUNDING_RATE and user_futures_funding_rate is not None:
            logger.warning(f'{pair} using user specified [{user_futures_funding_rate}]')
        elif candle_type not in (CandleType.SPOT, CandleType.FUTURES):
            result[pair] = DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    if fail_without_data and (not result):
        raise OperationalException('No data found. Terminating.')
    return result


def refresh_data(
    *,
    datadir: Path,
    timeframe: str,
    pairs: List[str],
    exchange: Exchange,
    data_format: Optional[str] = None,
    timerange: Optional[TimeRange] = None,
    candle_type: CandleType
) -> None:
    """
    Refresh ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param exchange: Exchange object
    :param data_format: dataformat to use
    :param timerange: Limit data to be loaded to this timerange
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    """
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        _download_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            exchange=exchange,
            data_handler=data_handler,
            candle_type=candle_type
        )


def _load_cached_data_for_updating(
    pair: str,
    timeframe: str,
    timerange: Optional[TimeRange],
    data_handler: IDataHandler,
    candle_type: CandleType,
    prepend: bool = False
) -> Tuple[DataFrame, Optional[int], Optional[int]]:
    """
    Load cached data to download more data.
    If timerange is passed in, checks whether data from an before the stored data will be
    downloaded.
    If that's the case then what's available should be completely overwritten.
    Otherwise downloads always start at the end of the available data to avoid data gaps.
    Note: Only used by download_pair_history().
    """
    start = None
    end = None
    if timerange:
        if timerange.starttype == 'date':
            start = timerange.startdt
        if timerange.stoptype == 'date':
            end = timerange.stopdt
    data = data_handler.ohlcv_load(
        pair,
        timeframe=timeframe,
        timerange=None,
        fill_missing=False,
        drop_incomplete=True,
        warn_no_data=False,
        candle_type=candle_type
    )
    if not data.empty:
        if prepend:
            end = data.iloc[0]['date']
        else:
            if start and start < data.iloc[0]['date']:
                logger.info(f"{pair}, {timeframe}, {candle_type}: Requested start date {start:{DATETIME_PRINT_FORMAT}} earlier than local data start date {data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}. Use `--prepend` to download data prior to {data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}, or `--erase` to redownload all data.")
            start = data.iloc[-1]['date']
    start_ms = int(start.timestamp() * 1000) if start else None
    end_ms = int(end.timestamp() * 1000) if end else None
    return (data, start_ms, end_ms)


def _download_pair_history(
    pair: str,
    *,
    datadir: Path,
    exchange: Exchange,
    timeframe: str = '5m',
    new_pairs_days: int = 30,
    data_handler: Optional[IDataHandler] = None,
    timerange: Optional[TimeRange] = None,
    candle_type: CandleType,
    erase: bool = False,
    prepend: bool = False
) -> bool:
    """
    Download latest candles from the exchange for the pair and timeframe passed in parameters
    The data is downloaded starting from the last correct data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    :param pair: pair to download
    :param timeframe: Timeframe (e.g "5m")
    :param timerange: range of time to download
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :param erase: Erase existing data
    :return: bool with success state
    """
    data_handler = get_datahandler(datadir, data_handler=data_handler)
    try:
        if erase:
            if data_handler.ohlcv_purge(pair, timeframe, candle_type=candle_type):
                logger.info(f'Deleting existing data for pair {pair}, {timeframe}, {candle_type}.')
        (data, since_ms, until_ms) = _load_cached_data_for_updating(pair, timeframe, timerange, data_handler=data_handler, candle_type=candle_type, prepend=prepend)
        logger.info(f'''Download history data for "{pair}", {timeframe}, {candle_type} and store in {datadir}. From {(format_ms_time(since_ms) if since_ms else 'start')} to {(format_ms_time(until_ms) if until_ms else 'now')}''')
        logger.debug('Current Start: %s', f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else 'None')
        logger.debug('Current End: %s', f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else 'None')
        new_dataframe = exchange.get_historic_ohlcv(
            pair=pair,
            timeframe=timeframe,
            since_ms=since_ms if since_ms else int((datetime.now() - timedelta(days=new_pairs_days)).timestamp()) * 1000,
            is_new_pair=data.empty,
            candle_type=candle_type,
            until_ms=until_ms if until_ms else None
        )
        logger.info(f'Downloaded data for {pair} with length {len(new_dataframe)}.')
        if data.empty:
            data = new_dataframe
        else:
            data = clean_ohlcv_dataframe(concat([data, new_dataframe], axis=0), timeframe, pair, fill_missing=False, drop_incomplete=False)
        logger.debug('New Start: %s', f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else 'None')
        logger.debug('New End: %s', f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else 'None')
        data_handler.ohlcv_store(pair, timeframe, data=data, candle_type=candle_type)
        return True
    except Exception:
        logger.exception(f'Failed to download history data for pair: "{pair}", timeframe: {timeframe}.')
        return False


def refresh_backtest_ohlcv_data(
    exchange: Exchange,
    pairs: List[str],
    timeframes: List[str],
    datadir: Path,
    trading_mode: str,
    timerange: Optional[TimeRange] = None,
    new_pairs_days: int = 30,
    erase: bool = False,
    data_format: Optional[str] = None,
    prepend: bool = False,
    progress_tracker: Optional[CustomProgress] = None
) -> List[str]:
    """
    Refresh stored ohlcv data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    progress_tracker = retrieve_progress_tracker(progress_tracker)
    pairs_not_available: List[str] = []
    data_handler = get_datahandler(datadir, data_format)
    candle_type = CandleType.get_default(trading_mode)
    with progress_tracker as progress:
        tf_length = len(timeframes) if trading_mode != 'futures' else len(timeframes) + 2
        timeframe_task = progress.add_task('Timeframe', total=tf_length)
        pair_task = progress.add_task('Downloading data...', total=len(pairs))
        for pair in pairs:
            progress.update(pair_task, description=f'Downloading {pair}')
            progress.update(timeframe_task, completed=0)
            if pair not in exchange.markets:
                pairs_not_available.append(f'{pair}: Pair not available on exchange.')
                logger.info(f'Skipping pair {pair}...')
                continue
            for timeframe in timeframes:
                progress.update(timeframe_task, description=f'Timeframe {timeframe}')
                logger.debug(f'Downloading pair {pair}, {candle_type}, interval {timeframe}.')
                _download_pair_history(
                    pair=pair,
                    datadir=datadir,
                    exchange=exchange,
                    timerange=timerange,
                    data_handler=data_handler,
                    timeframe=str(timeframe),
                    new_pairs_days=new_pairs_days,
                    candle_type=candle_type,
                    erase=erase,
                    prepend=prepend
                )
                progress.update(timeframe_task, advance=1)
            if trading_mode == 'futures':
                tf_mark = exchange.get_option('mark_ohlcv_timeframe')
                tf_funding_rate = exchange.get_option('funding_fee_timeframe')
                fr_candle_type = CandleType.from_string(exchange.get_option('mark_ohlcv_price'))
                combs = ((CandleType.FUNDING_RATE, tf_funding_rate), (fr_candle_type, tf_mark))
                for (candle_type_f, tf) in combs:
                    logger.debug(f'Downloading pair {pair}, {candle_type_f}, interval {tf}.')
                    _download_pair_history(
                        pair=pair,
                        datadir=datadir,
                        exchange=exchange,
                        timerange=timerange,
                        data_handler=data_handler,
                        timeframe=str(tf),
                        new_pairs_days=new_pairs_days,
                        candle_type=candle_type_f,
                        erase=erase,
                        prepend=prepend
                    )
                    progress.update(timeframe_task, advance=1, description=f'Timeframe {candle_type_f}, {tf}')
            progress.update(pair_task, advance=1)
            progress.update(timeframe_task, description='Timeframe')
    return pairs_not_available


def _download_trades_history(
    exchange: Exchange,
    pair: str,
    *,
    new_pairs_days: int = 30,
    timerange: Optional[TimeRange] = None,
    data_handler: IDataHandler,
    trading_mode: TradingMode
) -> bool:
    """
    Download trade history from the exchange.
    Appends to previously downloaded trades data.
    """
    until = None
    since = 0
    if timerange:
        if timerange.starttype == 'date':
            since = timerange.startts * 1000
        if timerange.stoptype == 'date':
            until = timerange.stopts * 1000
    trades = data_handler.trades_load(pair, trading_mode)
    if not trades.empty and since > 0 and (since + 1000 < trades.iloc[0]['timestamp']):
        raise ValueError(f"Start {format_ms_time_det(since)} earlier than available data ({format_ms_time_det(trades.iloc[0]['timestamp'])}). Please use `--erase` if you'd like to redownload {pair}.")
    from_id = trades.iloc[-1]['id'] if not trades.empty else None
    if not trades.empty and since < trades.iloc[-1]['timestamp']:
        since = int(trades.iloc[-1]['timestamp'] - 5 * 1000)
        logger.info(f'Using last trade date -5s - Downloading trades for {pair} since: {format_ms_time(since)}.')
    if not since:
        since = dt_ts(dt_now() - timedelta(days=new_pairs_days))
    logger.debug('Current Start: %s', 'None' if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}")
    logger.debug('Current End: %s', 'None' if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}")
    logger.info(f'Current Amount of trades: {len(trades)}')
    new_trades = exchange.get_historic_trades(pair=pair, since=since, until=until, from_id=from_id)
    new_trades_df = trades_list_to_df(new_trades[1])
    trades = concat([trades, new_trades_df], axis=0)
    trades = trades_df_remove_duplicates(trades)
    data_handler.trades_store(pair, trades, trading_mode)
    logger.debug('New Start: %s', 'None' if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}")
    logger.debug('New End: %s', 'None' if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}")
    logger.info(f'New Amount of trades: {len(trades)}')
    return True


def refresh_backtest_trades_data(
    exchange: Exchange,
    pairs: List[str],
    datadir: Path,
    timerange: TimeRange,
    trading_mode: TradingMode,
    new_pairs_days: int = 30,
    erase: bool = False,
    data