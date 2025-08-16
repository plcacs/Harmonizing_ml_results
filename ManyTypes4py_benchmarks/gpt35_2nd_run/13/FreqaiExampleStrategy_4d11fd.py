import logging
from functools import reduce
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy
from freqtrade.typing import DataFrame, Any, Dict, Tuple

logger: logging.Logger = logging.getLogger(__name__)

class FreqaiExampleStrategy(IStrategy):
    minimal_roi: Dict[str, float] = {'0': 0.1, '240': -1}
    plot_config: Dict[str, Dict[str, Dict[str, str]]] = {'main_plot': {}, 'subplots': {'&-s_close': {'&-s_close': {'color': 'blue'}}, 'do_predict': {'do_predict': {'color': 'brown'}}}
    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 40
    can_short: bool = True

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        ...

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        ...

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        ...

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        ...

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: Any, entry_tag: str, side: str, **kwargs: Any) -> bool:
        ...
