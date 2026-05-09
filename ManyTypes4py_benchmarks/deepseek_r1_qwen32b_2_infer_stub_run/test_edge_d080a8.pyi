import logging
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from pandas import DataFrame

class BTrade:
    exit_reason: ExitType
    open_tick: int
    close_tick: int

class BTContainer:
    data: List[List[Any]]
    stop_loss: float
    roi: Dict[str, float]
    profit_perc: float
    trades: List[BTrade]

class Edge:
    def __init__(self, config: Dict[str, Any], exchange: Any, strategy: Any) -> None:
        ...

    def adjust(self, pairs: List[str]) -> List[str]:
        ...

    def get_stoploss(self, pair: str) -> float:
        ...

    def stake_amount(self, pair: str, free_capital: float, total_capital: float, capital_in_trade: float) -> float:
        ...

    def calculate(self, pair_whitelist: List[str]) -> bool:
        ...

    def _find_trades_for_stoploss_range(self, frame: DataFrame, pair: str, stoploss_range: List[float]) -> List[Dict[str, Any]]:
        ...

    def _fill_calculable_fields(self, trades_df: DataFrame) -> DataFrame:
        ...

    def _process_expectancy(self, trades_df: DataFrame) -> Dict[str, Any]:
        ...

    _cached_pairs: Dict[str, PairInfo]
    _capital_ratio: float
    _allowed_risk: float
    fee: Optional[float]
    _last_updated: float

class PairInfo:
    stoploss: float
    winrate: float
    risk_reward_ratio: float
    required_risk_reward: float
    expectancy: float
    nb_trades: int
    max_drawdown: float
    trade_duration: float
```