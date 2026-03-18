```python
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from _pytest.fixtures import FixtureRequest
from pytest import MonkeyPatch
from unittest.mock import MagicMock

def test_sync_wallet_at_boot(mocker: MonkeyPatch, default_conf: Dict[str, Any]) -> None: ...

def test_sync_wallet_missing_data(mocker: MonkeyPatch, default_conf: Dict[str, Any]) -> None: ...

def test_get_trade_stake_amount_no_stake_amount(default_conf: Dict[str, Any], mocker: MonkeyPatch) -> None: ...

def test_get_trade_stake_amount_unlimited_amount(
    default_conf: Dict[str, Any],
    ticker: Any,
    balance_ratio: float,
    capital: Optional[float],
    result1: float,
    result2: float,
    limit_buy_order_open: Dict[str, Any],
    fee: Any,
    mocker: MonkeyPatch
) -> None: ...

def test_validate_stake_amount(
    mocker: MonkeyPatch,
    default_conf: Dict[str, Any],
    stake_amount: float,
    min_stake: Optional[float],
    stake_available: float,
    max_stake: float,
    trade_amount: Optional[float],
    expected: float
) -> None: ...

def test_get_starting_balance(
    mocker: MonkeyPatch,
    default_conf: Dict[str, Any],
    available_capital: Optional[float],
    closed_profit: float,
    open_stakes: float,
    free: float,
    expected: float
) -> None: ...

def test_sync_wallet_futures_live(mocker: MonkeyPatch, default_conf: Dict[str, Any]) -> None: ...

def test_sync_wallet_dry(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: Any) -> None: ...

def test_sync_wallet_futures_dry(mocker: MonkeyPatch, default_conf: Dict[str, Any], fee: Any) -> None: ...

def test_check_exit_amount(mocker: MonkeyPatch, default_conf: Dict[str, Any], fee: Any) -> None: ...

def test_check_exit_amount_futures(mocker: MonkeyPatch, default_conf: Dict[str, Any], fee: Any) -> None: ...

def test_dry_run_wallet_initialization(
    mocker: MonkeyPatch,
    default_conf_usdt: Dict[str, Any],
    config: Dict[str, Any],
    wallets: Dict[str, Dict[str, Any]]
) -> None: ...
```