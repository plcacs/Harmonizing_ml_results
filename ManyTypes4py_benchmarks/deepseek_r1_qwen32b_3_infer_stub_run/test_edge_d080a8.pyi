from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from freqtrade.enums import ExitType
from pandas import DataFrame
from freqtrade.edge import Edge, PairInfo
from freqtrade.optimize import BTContainer, BTrade

tests_start_time: datetime = ...
timeframe_in_minute: int = ...
TESTS: List[BTContainer] = ...

class BTContainer:
    def __init__(self, data: List[List[int]], stop_loss: float, roi: Dict[str, float], profit_perc: float, trades: List[BTrade]):
        ...
    data: List[List[int]]
    stop_loss: float
    roi: Dict[str, float]
    profit_perc: float
    trades: List[BTrade]

class BTrade:
    def __init__(self, exit_reason: ExitType, open_tick: int, close_tick: int):
        ...
    exit_reason: ExitType
    open_tick: int
    close_tick: int

def test_edge_results(edge_conf: Dict[str, Any], mocker: Any, caplog: Any, data: BTContainer) -> None:
    ...

def test_adjust(mocker: Any, edge_conf: Dict[str, Any]) -> List[str]:
    ...

def test_edge_get_stoploss(mocker: Any, edge_conf: Dict[str, Any]) -> float:
    ...

def test_nonexisting_get_stoploss(mocker: Any, edge_conf: Dict[str, Any]) -> float:
    ...

def test_edge_stake_amount(mocker: Any, edge_conf: Dict[str, Any]) -> float:
    ...

def test_nonexisting_stake_amount(mocker: Any, edge_conf: Dict[str, Any]) -> float:
    ...

def test_edge_heartbeat_calculate(mocker: Any, edge_conf: Dict[str, Any]) -> bool:
    ...

def mocked_load_data(datadir: str, pairs: Optional[List[str]] = ..., timeframe: str = ..., timerange: Optional[Any] = ..., *args: Any, **kwargs: Any) -> Dict[str, DataFrame]:
    ...

def test_edge_process_downloaded_data(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    ...

def test_edge_process_no_data(mocker: Any, edge_conf: Dict[str, Any], caplog: Any) -> None:
    ...

def test_edge_process_no_trades(mocker: Any, edge_conf: Dict[str, Any], caplog: Any) -> None:
    ...

def test_edge_process_no_pairs(mocker: Any, edge_conf: Dict[str, Any], caplog: Any) -> None:
    ...

def test_edge_init_error(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    ...

def test_process_expectancy(mocker: Any, edge_conf: Dict[str, Any], fee: float, risk_reward_ratio: float, expectancy: float) -> None:
    ...

def test_process_expectancy_remove_pumps(mocker: Any, edge_conf: Dict[str, Any], fee: float) -> None:
    ...

def test_process_expectancy_only_wins(mocker: Any, edge_conf: Dict[str, Any], fee: float) -> None:
    ...