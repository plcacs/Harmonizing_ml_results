import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union, List, Dict
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager

AVAILABLE_PROTECTIONS: List[str] = ['CooldownPeriod', 'LowProfitPairs', 'MaxDrawdown', 'StoplossGuard']

def generate_mock_trade(
    pair: str,
    fee: float,
    is_open: bool,
    exit_reason: Union[ExitType, str] = ExitType.EXIT_SIGNAL,
    min_ago_open: Optional[int] = None,
    min_ago_close: Optional[int] = None,
    profit_rate: float = 0.9,
    is_short: bool = False
) -> Trade: ...

def test_protectionmanager(mocker: Any, default_conf: Dict[str, Any]) -> None: ...

def test_validate_protections(protconf: List[Dict[str, Any]], expected: Optional[str]) -> None: ...

def test_protections_init(
    default_conf: Dict[str, Any],
    timeframe: str,
    expected_lookback: int,
    expected_stop: Union[int, str],
    protconf: List[Dict[str, Any]]
) -> None: ...

def test_stoploss_guard(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, is_short: bool) -> None: ...

def test_stoploss_guard_perpair(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_pair: bool,
    only_per_side: bool
) -> None: ...

def test_CooldownPeriod(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None: ...

def test_CooldownPeriod_unlock_at(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, time_machine: Any) -> None: ...

def test_LowProfitPairs(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, only_per_side: bool) -> None: ...

def test_MaxDrawdown(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None: ...

def test_protection_manager_desc(
    mocker: Any,
    default_conf: Dict[str, Any],
    protectionconf: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[Any]
) -> None: ...