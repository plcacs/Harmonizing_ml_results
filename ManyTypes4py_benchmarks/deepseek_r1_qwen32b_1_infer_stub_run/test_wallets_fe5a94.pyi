from unittest.mock import Mocker
from pytest import fixture
from typing import Dict, List, Optional, Tuple, Union
from freqtrade.persistence import Trade
from freqtrade.wallets import Wallets

@fixture
def default_conf() -> Dict:
    ...

@fixture
def default_conf_usdt() -> Dict:
    ...

@fixture
def fee() -> Dict:
    ...

@fixture
def ticker() -> Dict:
    ...

@fixture
def limit_buy_order_open() -> Dict:
    ...

def test_sync_wallet_at_boot(mocker: Mocker, default_conf: Dict) -> None:
    ...

def test_sync_wallet_missing_data(mocker: Mocker, default_conf: Dict) -> None:
    ...

def test_get_trade_stake_amount_no_stake_amount(mocker: Mocker, default_conf: Dict) -> None:
    ...

def test_get_trade_stake_amount_unlimited_amount(
    default_conf: Dict,
    ticker: Dict,
    balance_ratio: float,
    capital: Optional[float],
    result1: float,
    result2: float,
    limit_buy_order_open: Dict,
    fee: Dict,
    mocker: Mocker
) -> None:
    ...

def test_validate_stake_amount(
    mocker: Mocker,
    default_conf: Dict,
    stake_amount: float,
    min_stake: Optional[float],
    stake_available: float,
    max_stake: Optional[float],
    trade_amount: Optional[float],
    expected: float
) -> float:
    ...

def test_get_starting_balance(
    mocker: Mocker,
    default_conf: Dict,
    available_capital: Optional[float],
    closed_profit: float,
    open_stakes: float,
    free: float,
    expected: float
) -> float:
    ...

def test_sync_wallet_futures_live(mocker: Mocker, default_conf: Dict) -> None:
    ...

def test_sync_wallet_dry(mocker: Mocker, default_conf_usdt: Dict, fee: Dict) -> None:
    ...

def test_sync_wallet_futures_dry(mocker: Mocker, default_conf: Dict, fee: Dict) -> None:
    ...

def test_check_exit_amount(mocker: Mocker, default_conf: Dict, fee: Dict) -> bool:
    ...

def test_check_exit_amount_futures(mocker: Mocker, default_conf: Dict, fee: Dict) -> bool:
    ...

def test_dry_run_wallet_initialization(
    mocker: Mocker,
    default_conf_usdt: Dict,
    config: Dict,
    wallets: Dict
) -> None:
    ...