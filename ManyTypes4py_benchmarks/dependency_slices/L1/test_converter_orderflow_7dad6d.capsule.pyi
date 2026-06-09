# === Internal dependency: freqtrade.constants ===
DEFAULT_TRADES_COLUMNS = ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost']

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.orderflow import populate_dataframe_with_trades

# === Internal dependency: freqtrade.data.converter.orderflow ===
def timeframe_to_DateOffset(timeframe): ...
def trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale): ...
def stacked_imbalance(df, label, stacked_imbalance_range): ...
from freqtrade.constants import ORDERFLOW_ADDED_COLUMNS

# === Internal dependency: freqtrade.data.converter.trade_converter ===
def trades_list_to_df(trades, convert=...): ...

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider:
    def __init__(self, config, exchange, pairlists=..., rpc=...): ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, DateOffset, read_csv, read_feather, read_json, to_datetime

# === Third-party dependency: pytest ===
# Used symbols: approx, fixture

# === Internal dependency: tests.strategy.strats.strategy_test_v3 ===
class StrategyTestV3(IStrategy):
    ...