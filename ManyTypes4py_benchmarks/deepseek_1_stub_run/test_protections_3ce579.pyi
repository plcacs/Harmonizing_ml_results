```python
import pytest
from datetime import datetime
from typing import Any, List, Optional, Union
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.plugins.protectionmanager import ProtectionManager

AVAILABLE_PROTECTIONS: List[str] = ...

def generate_mock_trade(
    pair: str,
    fee: Any,
    is_open: bool,
    exit_reason: ExitType = ...,
    min_ago_open: Optional[int] = None,
    min_ago_close: Optional[int] = None,
    profit_rate: float = ...,
    is_short: bool = ...
) -> Trade: ...

def test_protectionmanager(
    mocker: Any,
    default_conf: Any
) -> None: ...

def test_validate_protections(
    protconf: Any,
    expected: Optional[str]
) -> None: ...

def test_protections_init(
    default_conf: Any,
    timeframe: str,
    expected_lookback: int,
    expected_stop: Union[int, str],
    protconf: Any
) -> None: ...

def test_stoploss_guard(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any,
    is_short: bool
) -> None: ...

def test_stoploss_guard_perpair(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any,
    only_per_pair: bool,
    only_per_side: bool
) -> None: ...

def test_CooldownPeriod(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any
) -> None: ...

def test_CooldownPeriod_unlock_at(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any,
    time_machine: Any
) -> None: ...

def test_LowProfitPairs(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any,
    only_per_side: bool
) -> None: ...

def test_MaxDrawdown(
    mocker: Any,
    default_conf: Any,
    fee: Any,
    caplog: Any
) -> None: ...

def test_protection_manager_desc(
    mocker: Any,
    default_conf: Any,
    protectionconf: Any,
    desc_expected: str,
    exception_expected: Optional[Any]
) -> None: ...
```