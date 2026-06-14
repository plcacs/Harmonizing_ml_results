from typing import Any, Dict, List, Optional, Union
from freqtrade.persistence import Trade

AVAILABLE_PROTECTIONS: List[str] ...

def generate_mock_trade(
    pair: str,
    fee: float,
    is_open: bool,
    exit_reason: object = ...,
    min_ago_open: Optional[int] = ...,
    min_ago_close: Optional[int] = ...,
    profit_rate: float = ...,
    is_short: bool = ...
) -> Trade: ...

def test_protectionmanager(mocker: Any, default_conf: Dict[str, Any]) -> None: ...

def test_validate_protections(protconf: List[Dict[str, Any]], expected: Optional[str]) -> None: ...

def test_protections_init(
    default_conf: Dict[str, Any],
    timeframe: str,
    expected_lookback: int,
    expected_stop: Union[int, str],
    protconf: List[Dict[str, Any]],
) -> None: ...

def test_stoploss_guard(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, is_short: bool) -> None: ...

def test_stoploss_guard_perpair(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_pair: bool,
    only_per_side: bool,
) -> None: ...

def test_CooldownPeriod(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None: ...

def test_CooldownPeriod_unlock_at(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    time_machine: Any,
) -> None: ...

def test_LowProfitPairs(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, only_per_side: bool) -> None: ...

def test_MaxDrawdown(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None: ...

def test_protection_manager_desc(
    mocker: Any,
    default_conf: Dict[str, Any],
    protectionconf: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[object],
) -> None: ...