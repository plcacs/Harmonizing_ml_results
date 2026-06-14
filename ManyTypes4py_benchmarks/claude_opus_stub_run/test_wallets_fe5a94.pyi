from typing import Any
from unittest.mock import MagicMock

def test_sync_wallet_at_boot(mocker: Any, default_conf: dict[str, Any]) -> None: ...
def test_sync_wallet_missing_data(mocker: Any, default_conf: dict[str, Any]) -> None: ...
def test_get_trade_stake_amount_no_stake_amount(default_conf: dict[str, Any], mocker: Any) -> None: ...
def test_get_trade_stake_amount_unlimited_amount(
    default_conf: dict[str, Any],
    ticker: Any,
    balance_ratio: float,
    capital: float | None,
    result1: float,
    result2: float,
    limit_buy_order_open: dict[str, Any],
    fee: Any,
    mocker: Any,
) -> None: ...
def test_validate_stake_amount(
    mocker: Any,
    default_conf: dict[str, Any],
    stake_amount: float,
    min_stake: float | None,
    stake_available: float,
    max_stake: float,
    trade_amount: float | None,
    expected: float,
) -> None: ...
def test_get_starting_balance(
    mocker: Any,
    default_conf: dict[str, Any],
    available_capital: float | None,
    closed_profit: float,
    open_stakes: float,
    free: float,
    expected: float,
) -> None: ...
def test_sync_wallet_futures_live(mocker: Any, default_conf: dict[str, Any]) -> None: ...
def test_sync_wallet_dry(mocker: Any, default_conf_usdt: dict[str, Any], fee: Any) -> None: ...
def test_sync_wallet_futures_dry(mocker: Any, default_conf: dict[str, Any], fee: Any) -> None: ...
def test_check_exit_amount(mocker: Any, default_conf: dict[str, Any], fee: Any) -> None: ...
def test_check_exit_amount_futures(mocker: Any, default_conf: dict[str, Any], fee: Any) -> None: ...
def test_dry_run_wallet_initialization(
    mocker: Any,
    default_conf_usdt: dict[str, Any],
    config: dict[str, Any],
    wallets: dict[str, dict[str, Any]],
) -> None: ...