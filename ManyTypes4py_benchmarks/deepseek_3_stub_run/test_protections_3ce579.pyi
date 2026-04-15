import pytest
from datetime import datetime, timedelta
from typing import Any, Optional, Union, List, Dict, Tuple
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager

AVAILABLE_PROTECTIONS: List[str] = ...

def generate_mock_trade(
    pair: str,
    fee: Any,
    is_open: bool,
    exit_reason: str = ...,
    min_ago_open: Optional[int] = None,
    min_ago_close: Optional[int] = None,
    profit_rate: float = ...,
    is_short: bool = ...
) -> Trade: ...

def test_protectionmanager(
    mocker: Any,
    default_conf: Dict[str, Any]
) -> None: ...

@pytest.mark.parametrize('protconf,expected', ...)
def test_validate_protections(
    protconf: List[Dict[str, Any]],
    expected: Optional[str]
) -> None: ...

@pytest.mark.parametrize('timeframe,expected_lookback,expected_stop,protconf', ...)
def test_protections_init(
    default_conf: Dict[str, Any],
    timeframe: str,
    expected_lookback: int,
    expected_stop: Union[int, str],
    protconf: List[Dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', ...)
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    is_short: bool
) -> None: ...

@pytest.mark.parametrize('only_per_pair', ...)
@pytest.mark.parametrize('only_per_side', ...)
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard_perpair(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_pair: bool,
    only_per_side: bool
) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any
) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod_unlock_at(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    time_machine: Any
) -> None: ...

@pytest.mark.parametrize('only_per_side', ...)
@pytest.mark.usefixtures('init_persistence')
def test_LowProfitPairs(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any,
    only_per_side: bool
) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_MaxDrawdown(
    mocker: Any,
    default_conf: Dict[str, Any],
    fee: Any,
    caplog: Any
) -> None: ...

@pytest.mark.parametrize('protectionconf,desc_expected,exception_expected', ...)
def test_protection_manager_desc(
    mocker: Any,
    default_conf: Dict[str, Any],
    protectionconf: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[Any]
) -> None: ...