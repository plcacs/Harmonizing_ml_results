import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import Exchange
from pandas import DataFrame
from tests.conftest import EXMS
from tests.optimize import BTContainer, BTrade

tests_start_time: datetime = ...
timeframe_in_minute: int = ...
tc0: BTContainer = ...
tc1: BTContainer = ...
tc2: BTContainer = ...
tc3: BTContainer = ...
tc4: BTContainer = ...
TESTS: List[BTContainer] = ...

@pytest.mark.parametrize('data', TESTS)
def test_edge_results(
    edge_conf: Dict[str, Any],
    mocker: MagicMock,
    caplog: Any,
    data: BTContainer
) -> None: ...

def test_adjust(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_edge_get_stoploss(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_nonexisting_get_stoploss(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_edge_stake_amount(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_nonexisting_stake_amount(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_edge_heartbeat_calculate(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def mocked_load_data(
    datadir: str,
    pairs: Optional[List[str]] = None,
    timeframe: str = '0m',
    timerange: Optional[str] = None,
    *args: Any,
    **kwargs: Any
) -> Dict[str, DataFrame]: ...

def test_edge_process_downloaded_data(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

def test_edge_process_no_data(mocker: MagicMock, edge_conf: Dict[str, Any], caplog: Any) -> None: ...

def test_edge_process_no_trades(mocker: MagicMock, edge_conf: Dict[str, Any], caplog: Any) -> None: ...

def test_edge_process_no_pairs(mocker: MagicMock, edge_conf: Dict[str, Any], caplog: Any) -> None: ...

def test_edge_init_error(mocker: MagicMock, edge_conf: Dict[str, Any]) -> None: ...

@pytest.mark.parametrize('fee,risk_reward_ratio,expectancy', [(0.0005, 306.5384615384, 101.5128205128), (0.001, 152.6923076923, 50.2307692308)])
def test_process_expectancy(
    mocker: MagicMock,
    edge_conf: Dict[str, Any],
    fee: float,
    risk_reward_ratio: float,
    expectancy: float
) -> None: ...

def test_process_expectancy_remove_pumps(
    mocker: MagicMock,
    edge_conf: Dict[str, Any],
    fee: float
) -> None: ...

def test_process_expectancy_only_wins(
    mocker: MagicMock,
    edge_conf: Dict[str, Any],
    fee: float
) -> None: ...