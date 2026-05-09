from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
from pandas import DataFrame
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.util.datetime_helpers import datetime as dt_ts
from freqtrade.tests.conftest import EXMS, BTContainer, BTrade

tests_start_time: datetime = ...
timeframe_in_minute: int = ...

@pytest.mark.parametrize('data', TESTS)
def test_edge_results(edge_conf: Dict[str, Any], mocker: pytest_mock.MockFixture, caplog: pytest.LogCaptureFixture, data: BTContainer) -> None:
    ...

def test_adjust(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_edge_get_stoploss(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_nonexisting_get_stoploss(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_edge_stake_amount(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_nonexisting_stake_amount(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_edge_heartbeat_calculate(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def mocked_load_data(datadir: str, pairs: Optional[List[str]] = None, timeframe: str = '0m', timerange: Optional[str] = None, *args: Any, **kwargs: Any) -> Dict[str, DataFrame]:
    ...

def test_edge_process_downloaded_data(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

def test_edge_process_no_data(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_edge_process_no_trades(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_edge_process_no_pairs(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_edge_init_error(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('fee,risk_reward_ratio,expectancy', [(0.0005, 306.5384615384, 101.5128205128), (0.001, 152.6923076923, 50.2307692308)])
def test_process_expectancy(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], fee: float, risk_reward_ratio: float, expectancy: float) -> None:
    ...

def test_process_expectancy_remove_pumps(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], fee: float) -> None:
    ...

def test_process_expectancy_only_wins(mocker: pytest_mock.MockFixture, edge_conf: Dict[str, Any], fee: float) -> None:
    ...