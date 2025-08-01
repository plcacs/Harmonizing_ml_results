from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Callable as TCallable, Dict, Optional, Union
from pandas import DataFrame
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import merge_informative_pair

PopulateIndicators = TCallable[[Any, DataFrame, Dict[str, Any]], DataFrame]

@dataclass
class InformativeData:
    asset: str
    timeframe: str
    fmt: Optional[Union[str, TCallable[..., str]]]
    ffill: bool
    candle_type: Optional[CandleType]

def informative(
    timeframe: str,
    asset: str = '',
    fmt: Optional[Union[str, TCallable[..., str]]] = None,
    *,
    candle_type: Optional[str] = None,
    ffill: bool = True
) -> TCallable[[PopulateIndicators], PopulateIndicators]:
    """
    A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to
    define informative indicators.

    Example usage:

        @informative('1h')
        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

    :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.
    :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use
                  current pair. Also supports limited pair format strings (see below)
    :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
    specified, defaults to:
    * {base}_{quote}_{column}_{timeframe} if asset is specified.
    * {column}_{timeframe} if asset is not specified.
    Pair format supports these format variables:
    * {base} - base currency in lower case, for example 'eth'.
    * {BASE} - same as {base}, except in upper case.
    * {quote} - quote currency in lower case, for example 'usdt'.
    * {QUOTE} - same as {quote}, except in upper case.
    Format string additionally supports this variables.
    * {asset} - full name of the asset, for example 'BTC/USDT'.
    * {column} - name of dataframe column.
    * {timeframe} - timeframe of informative dataframe.
    :param ffill: ffill dataframe after merging informative pair.
    :param candle_type: '', mark, index, premiumIndex, or funding_rate
    """
    _asset: str = asset
    _timeframe: str = timeframe
    _fmt: Optional[Union[str, TCallable[..., str]]] = fmt
    _ffill: bool = ffill
    _candle_type: Optional[CandleType] = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn: PopulateIndicators) -> PopulateIndicators:
        informative_pairs = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator

def __get_pair_formats(market: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not market:
        return {}
    base: str = market['base']
    quote: str = market['quote']
    return {'base': base.lower(), 'BASE': base.upper(), 'quote': quote.lower(), 'QUOTE': quote.upper()}

def _format_pair_name(config: Dict[str, Any], pair: str, market: Optional[Dict[str, Any]] = None) -> str:
    return pair.format(stake_currency=config['stake_currency'], stake=config['stake_currency'], **__get_pair_formats(market)).upper()

def _create_and_merge_informative_pair(
    strategy: Any,
    dataframe: DataFrame,
    metadata: Dict[str, Any],
    inf_data: InformativeData,
    populate_indicators: PopulateIndicators
) -> DataFrame:
    asset: str = inf_data.asset or ''
    timeframe: str = inf_data.timeframe
    fmt: Optional[Union[str, TCallable[..., str]]] = inf_data.fmt
    candle_type: Optional[CandleType] = inf_data.candle_type
    config: Dict[str, Any] = strategy.config
    if asset:
        market1: Dict[str, Any] = strategy.dp.market(metadata['pair'])
        asset = _format_pair_name(config, asset, market1)
    else:
        asset = metadata['pair']
    market: Optional[Dict[str, Any]] = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f'Market {asset} is not available.')
    if not fmt:
        fmt = '{column}_{timeframe}'
        if inf_data.asset:
            fmt = '{base}_{quote}_' + fmt
    inf_metadata: Dict[str, Any] = {'pair': asset, 'timeframe': timeframe}
    inf_dataframe: DataFrame = strategy.dp.get_pair_dataframe(asset, timeframe, candle_type)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)
    formatter: TCallable[..., str]
    if callable(fmt):
        formatter = fmt
    else:
        formatter = fmt.format
    fmt_args: Dict[str, Any] = {**__get_pair_formats(market), 'asset': asset, 'timeframe': timeframe}
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)
    date_column: str = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(f'Duplicate column name {date_column} exists in dataframe! Ensure column names are unique!')
    dataframe = merge_informative_pair(
        dataframe,
        inf_dataframe,
        strategy.timeframe,
        timeframe,
        ffill=inf_data.ffill,
        append_timeframe=False,
        date_column=date_column
    )
    return dataframe