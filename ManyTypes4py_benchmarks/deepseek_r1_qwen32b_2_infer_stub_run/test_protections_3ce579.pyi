from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
import pytest
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager

AVAILABLE_PROTECTIONS: List[str] = ...

def generate_mock_trade(
    pair: str,
    fee: float,
    is_open: bool,
    exit_reason: ExitType = ExitType.EXIT_SIGNAL,
    min_ago_open: Optional[int] = None,
    min_ago_close: Optional[int] = None,
    profit_rate: float = 0.9,
    is_short: bool = False
) -> Trade:
    ...

def test_protectionmanager(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any]
) -> None:
    ...

@pytest.mark.parametrize('protconf,expected', [
    (List[Dict[str, Any]], Optional[str])
])
def test_validate_protections(
    protconf: List[Dict[str, Any]],
    expected: Optional[str]
) -> None:
    ...

@pytest.mark.parametrize('timeframe,expected_lookback,expected_stop,protconf', [
    (str, int, Union[int, str], List[Dict[str, Any]])
])
def test_protections_init(
    default_conf: Dict[str, Any],
    timeframe: str,
    expected_lookback: int,
    expected_stop: Union[int, str],
    protconf: List[Dict[str, Any]]
) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture,
    is_short: bool
) -> None:
    ...

@pytest.mark.parametrize('only_per_pair,only_per_side', [[False, False], [False, True], [True, False], [True, True]])
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard_perpair(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture,
    only_per_pair: bool,
    only_per_side: bool
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod_unlock_at(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture,
    time_machine: pytest.fixture
) -> None:
    ...

@pytest.mark.parametrize('only_per_side', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_LowProfitPairs(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture,
    only_per_side: bool
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
def test_MaxDrawdown(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    fee: pytest.fixture,
    caplog: pytest.fixture
) -> None:
    ...

@pytest.mark.parametrize('protectionconf,desc_expected,exception_expected', [
    (Dict[str, Any], str, Optional[Exception])
])
def test_protection_manager_desc(
    mocker: pytest.fixture,
    default_conf: Dict[str, Any],
    protectionconf: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[Exception]
) -> None:
    ...