import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Literal, Optional, Union, overload
import numpy as np
import pandas as pd
from pandas.core.base import PandasObject
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def numpy_rolling_window(data: np.ndarray, window: int) -> np.ndarray:
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def numpy_rolling_series(func: Callable) -> Callable:

    def func_wrapper(data: Union[pd.Series, np.ndarray], window: int, as_source: bool = False) -> Union[pd.Series, np.ndarray]:
        series = data.values if isinstance(data, pd.Series) else data
        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated):] = calculated
        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)
        return new_series
    return func_wrapper

@numpy_rolling_series
def numpy_rolling_mean(data: np.ndarray, window: int, as_source: bool = False) -> np.ndarray:
    return np.mean(numpy_rolling_window(data, window), axis=-1)

@numpy_rolling_series
def numpy_rolling_std(data: np.ndarray, window: int, as_source: bool = False) -> np.ndarray:
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)

def session(df: pd.DataFrame, start: str = '17:00', end: str = '16:00') -> pd.DataFrame:
    """remove previous globex day from df"""
    if df.empty:
        return df
    int_start = list(map(int, start.split(':')))
    int_start = int_start[0] + int_start[1] - 1 / 100 - 0.0001
    int_end = list(map(int, end.split(':')))
    int_end = int_end[0] + int_end[1] / 100
    int_now = df[-1:].index.hour[0] + df[:1].index.minute[0] / 100
    is_same_day = int_end > int_start
    curr = prev = df[-1:].index[0].strftime('%Y-%m-%d')
    if not is_same_day:
        prev = (datetime.strptime(curr, '%Y-%m-%d') - timedelta(1)).strftime('%Y-%m-%d')
    if int_now >= int_start:
        df = df[df.index >= curr + ' ' + start]
    else:
        df = df[df.index >= prev + ' ' + start]
    return df.copy()

def heikinashi(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars['ha_close'] = (bars['open'] + bars['high'] + bars['low'] + bars['close']) / 4
    bars.at[0, 'ha_open'] = (bars.at[0, 'open'] + bars.at[0, 'close']) / 2
    for i in range(1, len(bars)):
        bars.at[i, 'ha_open'] = (bars.at[i - 1, 'ha_open'] + bars.at[i - 1, 'ha_close']) / 2
    bars['ha_high'] = bars.loc[:, ['high', 'ha_open', 'ha_close']].max(axis=1)
    bars['ha_low'] = bars.loc[:, ['low', 'ha_open', 'ha_close']].min(axis=1)
    return pd.DataFrame(index=bars.index, data={'open': bars['ha_open'], 'high': bars['ha_high'], 'low': bars['ha_low'], 'close': bars['ha_close']})

def tdi(series: pd.Series, rsi_lookback: int = 13, rsi_smooth_len: int = 2, rsi_signal_len: int = 7, bb_lookback: int = 34, bb_std: float = 1.6185) -> pd.DataFrame:
    rsi_data = rsi(series, rsi_lookback)
    rsi_smooth = sma(rsi_data, rsi_smooth_len)
    rsi_signal = sma(rsi_data, rsi_signal_len)
    bb_series = bollinger_bands(rsi_data, bb_lookback, bb_std)
    return pd.DataFrame(index=series.index, data={'rsi': rsi_data, 'rsi_signal': rsi_signal, 'rsi_smooth': rsi_smooth, 'rsi_bb_upper': bb_series['upper'], 'rsi_bb_lower': bb_series['lower'], 'rsi_bb_mid': bb_series['mid']})

def awesome_oscillator(df: pd.DataFrame, weighted: bool = False, fast: int = 5, slow: int = 34) -> pd.Series:
    midprice = (df['high'] + df['low']) / 2
    if weighted:
        ao = (midprice.ewm(fast).mean() - midprice.ewm(slow).mean()).values
    else:
        ao = numpy_rolling_mean(midprice, fast) - numpy_rolling_mean(midprice, slow)
    return pd.Series(index=df.index, data=ao)

def nans(length: int = 1) -> np.ndarray:
    mtx = np.empty(length)
    mtx[:] = np.nan
    return mtx

def typical_price(bars: pd.DataFrame) -> pd.Series:
    res = (bars['high'] + bars['low'] + bars['close']) / 3.0
    return pd.Series(index=bars.index, data=res)

def mid_price(bars: pd.DataFrame) -> pd.Series:
    res = (bars['high'] + bars['low']) / 2.0
    return pd.Series(index=bars.index, data=res)

def ibs(bars: pd.DataFrame) -> pd.Series:
    """Internal bar strength"""
    res = np.round((bars['close'] - bars['low']) / (bars['high'] - bars['low']), 2)
    return pd.Series(index=bars.index, data=res)

def true_range(bars: pd.DataFrame) -> pd.Series:
    return pd.DataFrame({'hl': bars['high'] - bars['low'], 'hc': abs(bars['high'] - bars['close'].shift(1)), 'lc': abs(bars['low'] - bars['close'].shift(1))}).max(axis=1)

def atr(bars: pd.DataFrame, window: int = 14, exp: bool = False) -> pd.Series:
    tr = true_range(bars)
    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)
    return pd.Series(res)

@overload
def crossed(series1: pd.Series, series2: Union[pd.Series, float, int, np.ndarray], direction: Literal['above']) -> pd.Series: ...
@overload
def crossed(series1: pd.Series, series2: Union[pd.Series, float, int, np.ndarray], direction: Literal['below']) -> pd.Series: ...
@overload
def crossed(series1: pd.Series, series2: Union[pd.Series, float, int, np.ndarray], direction: None = None) -> pd.Series: ...

def crossed(series1: Union[pd.Series, np.ndarray], series2: Union[pd.Series, float, int, np.ndarray], direction: Optional[str] = None) -> pd.Series:
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)
    if isinstance(series2, (float, int, np.ndarray, np.integer, np.floating)):
        series2 = pd.Series(index=series1.index, data=series2)
    if direction is None or direction == 'above':
        above = pd.Series((series1 > series2) & (series1.shift(1) <= series2.shift(1)))
    if direction is None or direction == 'below':
        below = pd.Series((series1 < series2) & (series1.shift(1) >= series2.shift(1)))
    if direction is None:
        return above | below
    return above if direction == 'above' else below

def crossed_above(series1: Union[pd.Series, np.ndarray], series2: Union[pd.Series, float, int, np.ndarray]) -> pd.Series:
    return crossed(series1, series2, 'above')

def crossed_below(series1: Union[pd.Series, np.ndarray], series2: Union[pd.Series, float, int, np.ndarray]) -> pd.Series:
    return crossed(series1, series2, 'below')

def rolling_std(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_std(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).std()

def rolling_mean(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()

def rolling_min(series: Union[pd.Series, np.ndarray], window: int = 14, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()

def rolling_max(series: Union[pd.Series, np.ndarray], window: int = 14, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).max()
    except Exception as e:
        return pd.Series(series).rolling(window=window, min_periods=min_periods).max()

def rolling_weighted_mean(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:
        return pd.ewma(series, span=window, min_periods=min_periods)

def hull_moving_average(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    ma = 2 * rolling_weighted_mean(series, window / 2, min_periods) - rolling_weighted_mean(series, window, min_periods)
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)

def sma(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    return rolling_mean(series, window=window, min_periods=min_periods)

def wma(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    return rolling_weighted_mean(series, window=window, min_periods=min_periods)

def hma(series: Union[pd.Series, np.ndarray], window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    return hull_moving_average(series, window=window, min_periods=min_periods)

def vwap(bars: pd.DataFrame) -> None:
    """
    calculate vwap of entire time series
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    raise ValueError('using `qtpylib.vwap` facilitates lookahead bias. Please use `qtpylib.rolling_vwap` instead, which calculates vwap in a rolling manner.')

def rolling_vwap(bars: pd.DataFrame, window: int = 200, min_periods: Optional[int] = None) -> pd.Series:
    """
    calculate vwap using moving window
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    min_periods = window if min_periods is None else min_periods
    typical = (bars['high'] + bars['low'] + bars['close']) / 3
    volume = bars['volume']
    left = (volume * typical).rolling(window=window, min_periods=min_periods).sum()
    right = volume.rolling(window=window, min_periods=min_periods).sum()
    return pd.Series(index=bars.index, data=left / right).replace([np.inf, -np.inf], float('NaN')).ffill()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    compute the n period relative strength indicator
    """
    deltas = np.diff(series)
    seed = deltas[:window + 1]
    ups = seed[seed > 0].sum() / window
    downs = -seed[seed < 0].sum() / window
    rsival = np.zeros_like(series)
    rsival[:window] = 100.0 - 100.0 / (1.0 + ups / downs)
    for i in range(window, len(series)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        ups = (ups * (window - 1) + upval) / window
        downs = (downs * (window - 1.0) + downval) / window
        rsival[i] = 100.0 - 100.0 / (1.0 + ups / downs)
    return pd.Series(index=series.index, data=rsival)

def macd(series: pd.Series, fast: int = 3, slow: int = 10, smooth: int = 16) -> pd.DataFrame:
    """
    compute the MACD (Moving Average Convergence/Divergence)
    using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    macd_line = rolling_weighted_mean(series, window=fast) - rolling_weighted_mean(series, window=slow)
    signal = rolling_weighted_mean(macd_line, window=smooth)
    histogram = macd_line - signal
    return pd.DataFrame(index=series.index, data={'macd': macd_line.values, 'signal': signal.values, 'histogram': histogram.values})

def bollinger_bands(series: pd.Series, window: int = 20, stds: int = 2) -> pd.DataFrame:
    ma = rolling_mean(series, window=window, min_periods=1)
    std = rolling_std(series, window=window, min_periods=1)
    upper = ma + std * stds
    lower = ma - std * stds
    return pd.DataFrame(index=series.index, data={'upper': upper, 'mid': ma, 'lower': lower})

def weighted_bollinger_bands(series: pd.Series, window: int = 20, stds: int = 2) -> pd.DataFrame:
    ema = rolling_weighted_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = ema + std * stds
    lower = ema - std * stds
    return pd.DataFrame(index=series.index, data={'upper': upper.values, 'mid': ema.values, 'lower': lower.values})

def returns(series: pd.Series) -> pd.Series:
    try:
        res = (series / series.shift(1) - 1).replace([np.inf, -np.inf], float('NaN'))
    except Exception as e:
        res = nans(len(series))
    return pd.Series(index=series.index, data=res)

def log_returns(series: pd.Series) -> pd.Series:
    try:
        res = np.log(series / series.shift(1)).replace([np.inf, -np.inf], float('NaN'))
    except Exception as e:
       