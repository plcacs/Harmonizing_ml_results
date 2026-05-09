from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.util.datetime_helpers import dt_ts, dt_utc

tests_start_time = dt_utc(2018, 10, 3)
timeframe_in_minute = 60

class BTContainer:
    def __init__(self, data: List[List[float]], stop_loss: float, roi: Dict[str, float], profit_perc: float, trades: List['BTrade']):
        self.data = data
        self.stop_loss = stop_loss
        self.roi = roi
        self.profit_perc = profit_perc
        self.trades = trades

class BTrade:
    def __init__(self, exit_reason: ExitType, open_tick: int, close_tick: int):
        self.exit_reason = exit_reason
        self.open_tick = open_tick
        self.close_tick = close_tick

@pytest.fixture
def edge_conf() -> dict:
    return {}

@pytest.fixture
def get_patched_freqtradebot(mocker, edge_conf) -> 'freqtrade.freqtradebot.FreqtradeBot':
    return mocker.patch('freqtrade.freqtradebot.FreqtradeBot')

@pytest.fixture
def edge(mocker, edge_conf) -> 'Edge':
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    return Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)

def test_edge_results(edge: Edge, mocker, caplog, data: BTContainer) -> None:
    ...

def test_edge_stake_amount(edge: Edge, mocker) -> None:
    ...

def test_edge_heartbeat_calculate(edge: Edge) -> None:
    ...

def test_edge_process_downloaded_data(edge: Edge, mocker) -> None:
    ...

def test_edge_process_no_data(edge: Edge, caplog) -> None:
    ...

def test_edge_process_no_trades(edge: Edge, caplog) -> None:
    ...

def test_edge_process_no_pairs(edge: Edge, caplog) -> None:
    ...

def test_edge_init_error(edge_conf: dict, fee: float, risk_reward_ratio: float, expectancy: float) -> None:
    ...

def test_process_expectancy(edge: Edge, fee: float, risk_reward_ratio: float, expectancy: float) -> None:
    ...

def test_process_expectancy_remove_pumps(edge: Edge, fee: float) -> None:
    ...

def test_process_expectancy_only_wins(edge: Edge, fee: float) -> None:
    ...
