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


def informative(
    timeframe: str,
    asset: str = "",
    fmt: Optional[Union[str, Callable[[Any], str]]] = None,
    *,
    candle_type: Optional[Union[CandleType, str]] = None,
    ffill: bool = True,
) -> Callable[[PopulateIndicators], PopulateIndicators]:
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill
    _candle_type = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn: PopulateIndicators) -> PopulateIndicators:
        informative_pairs: List[InformativeData] = getattr(fn, "_ft_informative", [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, "_ft_informative", informative_pairs)  # noqa: B010
        return fn

    return decorator


def __get_pair_formats(market: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not market:
        return {}
    base = market["base"]
    quote = market["quote"]
    return {
        "base": base.lower(),
        "BASE": base.upper(),
        "quote": quote.lower(),
        "QUOTE": quote.upper(),
    }


def _format_pair_name(config: Dict[str, Any], pair: str, market: Optional[Dict[str, Any]] = None) -> str:
    return pair.format(
        stake_currency=config["stake_currency"],
        stake=config["stake_currency"],
        **__get_pair_formats(market),
    ).upper()


def _create_and_merge_informative_pair(
    strategy: Any,
    dataframe: DataFrame,
    metadata: Dict[str, Any],
    inf_data: InformativeData,
    populate_indicators: PopulateIndicators,
) -> DataFrame:
    asset = inf_data.asset or ""
    timeframe = inf_data.timeframe
    fmt = inf_data.fmt
    candle_type = inf_data.candle_type

    config = strategy.config

    if asset:
        # Insert stake currency if needed.
        market1 = strategy.dp.market(metadata["pair"])
        asset = _format_pair_name(config, asset, market1)
    else:
        # Not specifying an asset will define informative dataframe for current pair.
        asset = metadata["pair"]

    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f"Market {asset} is not available.")

    # Default format. This optimizes for the common case: informative pairs using same stake
    # currency. When quote currency matches stake currency, column name will omit base currency.
    # This allows easily reconfiguring strategy to use different base currency. In a rare case
    # where it is desired to keep quote currency in column name at all times user should specify
    # fmt='{base}_{quote}_{column}_{timeframe}' format or similar.
    if not fmt:
        fmt = "{column}_{timeframe}"  # Informatives of current pair
        if inf_data.asset:
            fmt = "{base}_{quote}_" + fmt  # Informatives of other pairs

    inf_metadata = {"pair": asset, "timeframe": timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe, candle_type)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)

    formatter: Any = None
    if callable(fmt):
        formatter = fmt  # A custom user-specified formatter function.
    else:
        formatter = fmt.format  # A default string formatter.

    fmt_args = {
        **__get_pair_formats(market),
        "asset": asset,
        "timeframe": timeframe,
    }
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)

    date_column = formatter(column="date", **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(
            f"Duplicate column name {date_column} exists in "
            f"dataframe! Ensure column names are unique!"
        )
    dataframe = merge_informative_pair(
        dataframe,
        inf_dataframe,
        strategy.timeframe,
        timeframe,
        ffill=inf_data.ffill,
        append_timeframe=False,
        date_column=date_column,
    )
    return dataframe
