"""
Functions to convert data from one format to another
"""
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, Config
from freqtrade.enums import CandleType, TradingMode

logger = logging.getLogger(__name__)

def ohlcv_to_dataframe(
    ohlcv: list,
    timeframe: str,
    pair: str,
    *,
    fill_missing: bool = True,
    drop_incomplete: bool = True
) -> DataFrame:
    """
    Converts a list with candle (OHLCV) data (in format returned by ccxt.fetch_ohlcv)
    to a Dataframe
    """
    logger.debug(f'Converting candle (OHLCV) data to dataframe for pair {pair}.')
    cols = DEFAULT_DATAFRAME_COLUMNS
    df = DataFrame(ohlcv, columns=cols)
    df['date'] = to_datetime(df['date'], unit='ms', utc=True)
    df = df.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    return clean_ohlcv_dataframe(df, timeframe, pair, fill_missing=fill_missing, drop_incomplete=drop_incomplete)

def clean_ohlcv_dataframe(
    data: DataFrame,
    timeframe: str,
    pair: str,
    *,
    fill_missing: bool,
    drop_incomplete: bool
) -> DataFrame:
    """
    Cleanse a OHLCV dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
    """
    data = data.groupby(by='date', as_index=False, sort=True).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'max'})
    if drop_incomplete:
        data.drop(data.tail(1).index, inplace=True)
        logger.debug('Dropping last candle')
    if fill_missing:
        return ohlcv_fill_up_missing_data(data, timeframe, pair)
    else:
        return data

def ohlcv_fill_up_missing_data(
    dataframe: DataFrame,
    timeframe: str,
    pair: str
) -> DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high", "low" and "close", volume is set to 0
    """
    from freqtrade.exchange import timeframe_to_resample_freq
    ohlcv_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resample_interval = timeframe_to_resample_freq(timeframe)
    df = dataframe.resample(resample_interval, on='date').agg(ohlcv_dict)
    df['close'] = df['close'].ffill()
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(value={'open': df['close'], 'high': df['close'], 'low': df['close']})
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0
    if len_before != len_after:
        message = f'Missing data fillup for {pair}, {timeframe}: before: {len_before} - after: {len_after} - {pct_missing:.2%}'
        if pct_missing > 0.01:
            logger.info(message)
        else:
            logger.debug(message)
    return df

def trim_dataframe(
    df: DataFrame,
    timerange: object,
    *,
    df_date_col: str = 'date',
    startup_candles: int = 0
) -> DataFrame:
    """
    Trim dataframe based on given timerange
    """
    if startup_candles:
        df = df.iloc[startup_candles:, :]
    elif timerange.starttype == 'date':
        df = df.loc[df[df_date_col] >= timerange.startdt, :]
    if timerange.stoptype == 'date':
        df = df.loc[df[df_date_col] <= timerange.stopdt, :]
    return df

def trim_dataframes(
    preprocessed: dict,
    timerange: object,
    startup_candles: int
) -> dict:
    """
    Trim startup period from analyzed dataframes
    """
    processed = {}
    for pair, df in preprocessed.items():
        trimed_df = trim_dataframe(df, timerange, startup_candles=startup_candles)
        if not trimed_df.empty:
            processed[pair] = trimed_df
        else:
            logger.warning(f'{pair} has no data left after adjusting for startup candles, skipping.')
    return processed

def order_book_to_dataframe(
    bids: list,
    asks: list
) -> DataFrame:
    """
    TODO: This should get a dedicated test
    Gets order book list, returns dataframe with below format per suggested by creslin
    """
    cols = ['bids', 'b_size']
    bids_frame = DataFrame(bids, columns=cols)
    bids_frame['b_sum'] = bids_frame['b_size'].cumsum()
    cols2 = ['asks', 'a_size']
    asks_frame = DataFrame(asks, columns=cols2)
    asks_frame['a_sum'] = asks_frame['a_size'].cumsum()
    frame = pd.concat([bids_frame['b_sum'], bids_frame['b_size'], bids_frame['bids'], asks_frame['asks'], asks_frame['a_size'], asks_frame['a_sum']], axis=1, keys=['b_sum', 'b_size', 'bids', 'asks', 'a_size', 'a_sum'])
    return frame

def convert_ohlcv_format(
    config: Config,
    convert_from: str,
    convert_to: str,
    erase: bool
) -> None:
    """
    Convert OHLCV from one format to another
    """
    from freqtrade.data.history import get_datahandler
    src = get_datahandler(config['datadir'], convert_from)
    trg = get_datahandler(config['datadir'], convert_to)
    timeframes = config.get('timeframes', [config.get('timeframe')])
    logger.info(f'Converting candle (OHLCV) for timeframe {timeframes}')
    candle_types = [CandleType.from_string(ct) for ct in config.get('candle_types', [c.value for c in CandleType])]
    logger.info(candle_types)
    paircombs = src.ohlcv_get_available_data(config['datadir'], TradingMode.SPOT)
    paircombs.extend(src.ohlcv_get_available_data(config['datadir'], TradingMode.FUTURES))
    if 'pairs' in config:
        paircombs = [comb for comb in paircombs if comb[0] in config['pairs']]
    if 'timeframes' in config:
        paircombs = [comb for comb in paircombs if comb[1] in config['timeframes']]
    paircombs = [comb for comb in paircombs if comb[2] in candle_types]
    paircombs = sorted(paircombs, key=lambda x: (x[0], x[1], x[2].value))
    formatted_paircombs = '\n'.join([f'{pair}, {timeframe}, {candle_type}' for pair, timeframe, candle_type in paircombs])
    logger.info(f'Converting candle (OHLCV) data for the following pair combinations:\n{formatted_paircombs}')
    for pair, timeframe, candle_type in paircombs:
        data = src.ohlcv_load(pair=pair, timeframe=timeframe, timerange=None, fill_missing=False, drop_incomplete=False, startup_candles=0, candle_type=candle_type)
        logger.info(f'Converting {len(data)} {timeframe} {candle_type} candles for {pair}')
        if len(data) > 0:
            trg.ohlcv_store(pair=pair, timeframe=timeframe, data=data, candle_type=candle_type)
            if erase and convert_from != convert_to:
                logger.info(f'Deleting source data for {pair} / {timeframe}')
                src.ohlcv_purge(pair=pair, timeframe=timeframe, candle_type=candle_type)

def reduce_dataframe_footprint(
    df: DataFrame
) -> DataFrame:
    """
    Ensure all values are float32 in the incoming dataframe.
    """
    logger.debug(f'Memory usage of dataframe is {df.memory_usage().sum() / 1024 ** 2:.2f} MB')
    df_dtypes = df.dtypes
    for column, dtype in df_dtypes.items():
        if column in ['open', 'high', 'low', 'close', 'volume']:
            continue
        if dtype == np.float64:
            df_dtypes[column] = np.float32
        elif dtype == np.int64:
            df_dtypes[column] = np.int32
    df = df.astype(df_dtypes)
    logger.debug(f'Memory usage after optimization is: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')
    return df
