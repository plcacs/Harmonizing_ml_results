from typing import Any

# === Internal dependency: freqtrade.constants ===
DEFAULT_TRADES_COLUMNS: Any

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.orderflow import populate_dataframe_with_trades

# === Internal dependency: freqtrade.data.converter.orderflow ===
def timeframe_to_DateOffset(timeframe: str) -> pd.DateOffset: ...
def trades_to_volumeprofile_with_total_delta_bid_ask(trades: pd.DataFrame, scale: float) -> pd.DataFrame: ...
def stacked_imbalance(df: pd.DataFrame, label: str, stacked_imbalance_range: int) -> Any: ...
# re-export: from freqtrade.constants import ORDERFLOW_ADDED_COLUMNS

# === Internal dependency: freqtrade.data.converter.trade_converter ===
def trades_list_to_df(trades: TradeList, convert: bool = ...) -> Any: ...

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider:
    def __init__(self, config: Config, exchange: Exchange | None, pairlists = ..., rpc: RPCManager | None = ...) -> None: ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, DateOffset, read_csv, read_feather, read_json, to_datetime

# === Third-party dependency: pytest ===
# Used symbols: approx, fixture

# === Internal dependency: tests.strategy.strats.strategy_test_v3 ===
class StrategyTestV3(IStrategy):
    ...