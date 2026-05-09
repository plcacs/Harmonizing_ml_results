from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union
import pytest
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager

AVAILABLE_PROTECTIONS: list[str] = ...

def generate_mock_trade(
    pair: str,
    fee: float,
    is_open: bool,
    exit_reason: ExitType = ExitType.EXIT_SIGNAL,
    min_ago_open: Optional[int] = None,
    min_ago_close: Optional[int] = None,
    profit_rate: float = 0.9,
    is_short: bool = False
) -> Trade: ...

def test_protectionmanager(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any]
) -> None: ...

def test_validate_protections(
    protconf: List[Dict[str, Any]],
    expected: Optional[str]
) -> None: ...

def test_protections_init(
    default_conf: Dict[str, Any],
    timeframe: str,
    expected_lookback: Union[int, str],
    expected_stop: Union[int, str],
    protconf: List[Dict[str, Any]]
) -> None: ...

def test_stoploss_guard(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture,
    is_short: bool
) -> None: ...

def test_stoploss_guard_perpair(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture,
    only_per_pair: bool,
    only_per_side: bool
) -> None: ...

def test_CooldownPeriod(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture
) -> None: ...

def test_CooldownPeriod_unlock_at(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture,
    time_machine: Any
) -> None: ...

def test_LowProfitPairs(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture,
    only_per_side: bool
) -> None: ...

def test_MaxDrawdown(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    fee: Callable[[], float],
    caplog: pytest.LogCaptureFixture
) -> None: ...

def test_protection_manager_desc(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    protectionconf: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[Exception]
) -> None: ...