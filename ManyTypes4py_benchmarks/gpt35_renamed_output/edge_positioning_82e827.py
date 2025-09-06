from typing import Any, NamedTuple, Dict, List
import numpy as np
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config
from freqtrade.data.history import get_timerange, load_data, refresh_data
from freqtrade.enums import CandleType, ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.strategy.interface import IStrategy
from freqtrade.util import dt_now

class PairInfo(NamedTuple):
    stoploss: float
    winrate: float
    risk_reward_ratio: float
    required_risk_reward: float
    expectancy: float
    nb_trades: int
    avg_trade_duration: float

class Edge:
    _cached_pairs: Dict[str, PairInfo] = {}

    def __init__(self, config: Config, exchange: Any, strategy: IStrategy) -> None:
        self.config: Config = config
        self.exchange: Any = exchange
        self.strategy: IStrategy = strategy
        self.edge_config: Dict[str, Any] = self.config.get('edge', {})
        self._cached_pairs: Dict[str, PairInfo] = {}
        self._final_pairs: List[str] = []
        self._capital_ratio: float = self.config['tradable_balance_ratio']
        self._allowed_risk: float = self.edge_config.get('allowed_risk')
        self._since_number_of_days: int = self.edge_config.get('calculate_since_number_of_days', 14)
        self._last_updated: int = 0
        self._refresh_pairs: bool = True
        self._stoploss_range_min: float = float(self.edge_config.get('stoploss_range_min', -0.01))
        self._stoploss_range_max: float = float(self.edge_config.get('stoploss_range_max', -0.05))
        self._stoploss_range_step: float = float(self.edge_config.get('stoploss_range_step', -0.001))
        self._stoploss_range: np.ndarray = np.arange(self._stoploss_range_min, self._stoploss_range_max, self._stoploss_range_step)
        self._timerange: TimeRange = TimeRange.parse_timerange(f'{(dt_now() - timedelta(days=self._since_number_of_days)).strftime("%Y%m%d")}-')
        if config.get('fee'):
            self.fee: float = config['fee']
        else:
            try:
                self.fee = self.exchange.get_fee(symbol=expand_pairlist(self.config['exchange']['pair_whitelist'], list(self.exchange.markets))[0])
            except IndexError:
                self.fee = None

    def calculate(self, pairs: List[str]) -> bool:
        ...

    def stake_amount(self, pair: str, free_capital: float, total_capital: float, capital_in_trade: float) -> float:
        ...

    def get_stoploss(self, pair: str) -> float:
        ...

    def adjust(self, pairs: List[str]) -> List[str]:
        ...

    def accepted_pairs(self) -> List[Dict[str, Any]]:
        ...

    def _fill_calculable_fields(self, result: DataFrame) -> DataFrame:
        ...

    def _process_expectancy(self, results: DataFrame) -> Dict[str, PairInfo]:
        ...

    def _find_trades_for_stoploss_range(self, df: DataFrame, pair: str, stoploss_range: np.ndarray) -> List[Dict[str, Any]]:
        ...

    def _detect_next_stop_or_sell_point(self, buy_column: np.ndarray, sell_column: np.ndarray, date_column: np.ndarray, ohlc_columns: np.ndarray, stoploss: float, pair: str) -> List[Dict[str, Any]]:
        ...
