```pyi
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from freqtrade.enums import ExitType
from freqtrade.persistence import Trade
from freqtrade.persistence.trade_model import Order

AVAILABLE_PROTECTIONS: list[str]

def generate_mock_trade(
    pair: str,
    fee: Any,
    is_open: bool,
    exit_reason: str = ...,
    min_ago_open: Optional[int] = ...,
    min_ago_close: Optional[int] = ...,
    profit_rate: float = ...,
    is_short: bool = ...,
) -> Trade: ...

def test_protectionmanager(mocker: Any, default_conf: dict[str, Any]) -> None: ...

def test_validate_protections(protconf: list[dict[str, Any]], expected: Optional[str]) -> None: ...

def test_protections_init(
    default_conf: dict[str, Any],
    timeframe: str,
    expected_lookback: int,
    expected_stop: int | str,
    protconf: list[dict[str, Any]],
) -> None: ...

def test_stoploss_guard(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
    is_short: bool,
) -> None: ...

def test_stoploss_guard_perpair(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_pair: bool,
    only_per_side: bool,
) -> None: ...

def test_CooldownPeriod(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
) -> None: ...

def test_CooldownPeriod_unlock_at(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
    time_machine: Any,
) -> None: ...

def test_LowProfitPairs(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_side: bool,
) -> None: ...

def test_MaxDrawdown(
    mocker: Any,
    default_conf: dict[str, Any],
    fee: Any,
    caplog: Any,
) -> None: ...

def test_protection_manager_desc(
    mocker: Any,
    default_conf: dict[str, Any],
    protectionconf: dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[str],
) -> None: ...
```