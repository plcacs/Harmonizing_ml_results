from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pandas import DataFrame
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import merge_informative_pair
PopulateIndicators = Callable[[Any, DataFrame, Dict[str, Any]], DataFrame]

@dataclass
class InformativeData:
    asset: Optional[str]
    timeframe: str
    fmt: Optional[Union[str, Callable[[Any], str]]]
    ffill: bool
    candle_type: Optional[CandleType]

def informative(timeframe, asset='', fmt=None, *, candle_type: Optional[Union[CandleType, str]]=None, ffill: bool=True):
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill
    _candle_type = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn):
        informative_pairs: List[InformativeData] = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator

def __get_pair_formats(market):
    if not market:
        return {}
    base = market['base']
    quote = market['quote']
    return {'base': base.lower(), 'BASE': base.upper(), 'quote': quote.lower(), 'QUOTE': quote.upper()}

def _format_pair_name(config, pair, market=None):
    return pair.format(stake_currency=config['stake_currency'], stake=config['stake_currency'], **__get_pair_formats(market)).upper()

def _create_and_merge_informative_pair(strategy, dataframe, metadata, inf_data, populate_indicators):
    asset = inf_data.asset or ''
    timeframe = inf_data.timeframe
    fmt = inf_data.fmt
    candle_type = inf_data.candle_type
    config = strategy.config
    if asset:
        market1 = strategy.dp.market(metadata['pair'])
        asset = _format_pair_name(config, asset, market1)
    else:
        asset = metadata['pair']
    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f'Market {asset} is not available.')
    if not fmt:
        fmt = '{column}_{timeframe}'
        if inf_data.asset:
            fmt = '{base}_{quote}_' + fmt
    inf_metadata = {'pair': asset, 'timeframe': timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe, candle_type)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)
    formatter: Any = None
    if callable(fmt):
        formatter = fmt
    else:
        formatter = fmt.format
    fmt_args = {**__get_pair_formats(market), 'asset': asset, 'timeframe': timeframe}
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)
    date_column = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(f'Duplicate column name {date_column} exists in dataframe! Ensure column names are unique!')
    dataframe = merge_informative_pair(dataframe, inf_dataframe, strategy.timeframe, timeframe, ffill=inf_data.ffill, append_timeframe=False, date_column=date_column)
    return dataframe