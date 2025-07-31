from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, Optional, Dict
from pandas import DataFrame
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import merge_informative_pair

PopulateIndicators = Callable[[Any, DataFrame, Dict[str, Any]], DataFrame]

@dataclass
class InformativeData:
    asset: str
    timeframe: str
    fmt: Optional[Union[str, Callable[..., str]]] = None
    ffill: bool = True
    candle_type: Optional[CandleType] = None

def informative(
    timeframe: str,
    asset: str = '',
    fmt: Optional[Union[str, Callable[..., str]]] = None,
    *,
    candle_type: Optional[str] = None,
    ffill: bool = True
) -> Callable[[PopulateIndicators], PopulateIndicators]:
    _asset: str = asset
    _timeframe: str = timeframe
    _fmt: Optional[Union[str, Callable[..., str]]] = fmt
    _ffill: bool = ffill
    _candle_type: Optional[CandleType] = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn: PopulateIndicators) -> PopulateIndicators:
        informative_pairs = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator

def __get_pair_formats(market: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not market:
        return {}
    base = market['base']
    quote = market['quote']
    return {
        'base': base.lower(),
        'BASE': base.upper(),
        'quote': quote.lower(),
        'QUOTE': quote.upper()
    }

def _format_pair_name(config: Dict[str, Any], pair: str, market: Optional[Dict[str, Any]] = None) -> str:
    return pair.format(
        stake_currency=config['stake_currency'],
        stake=config['stake_currency'],
        **__get_pair_formats(market)
    ).upper()

def _create_and_merge_informative_pair(
    strategy: Any,
    dataframe: DataFrame,
    metadata: Dict[str, Any],
    inf_data: InformativeData,
    populate_indicators: PopulateIndicators
) -> DataFrame:
    asset: str = inf_data.asset or ''
    timeframe: str = inf_data.timeframe
    fmt: Optional[Union[str, Callable[..., str]]] = inf_data.fmt
    candle_type: Optional[CandleType] = inf_data.candle_type
    config: Dict[str, Any] = strategy.config
    if asset:
        market1 = strategy.dp.market(metadata['pair'])
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
    formatter: Callable[..., str]
    if callable(fmt):
        formatter = fmt
    else:
        formatter = fmt.format
    fmt_args: Dict[str, Any] = {**__get_pair_formats(market), 'asset': asset, 'timeframe': timeframe}
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)
    date_column: str = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(
            f'Duplicate column name {date_column} exists in dataframe! Ensure column names are unique!'
        )
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