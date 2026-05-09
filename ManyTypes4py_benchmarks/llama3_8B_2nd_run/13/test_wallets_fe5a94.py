from copy import deepcopy
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import Trade
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, patch_wallet
from typing import Dict, Any, Optional

def test_sync_wallet_at_boot(mocker: Any, default_conf: Dict[str, Any]) -> None:
    # ... rest of the function ...

def test_get_trade_stake_amount_no_stake_amount(default_conf: Dict[str, Any], mocker: Any) -> None:
    # ... rest of the function ...

def test_validate_stake_amount(mocker: Any, default_conf: Dict[str, Any], stake_amount: float, min_stake: float, stake_available: float, max_stake: float, trade_amount: float, expected: float) -> None:
    # ... rest of the function ...

def test_get_starting_balance(mocker: Any, default_conf: Dict[str, Any], available_capital: Optional[float], closed_profit: float, open_stakes: float, free: float, expected: float) -> None:
    # ... rest of the function ...

def test_sync_wallet_futures_live(mocker: Any, default_conf: Dict[str, Any]) -> None:
    # ... rest of the function ...

def test_sync_wallet_dry(mocker: Any, default_conf_usdt: Dict[str, Any], fee: float) -> None:
    # ... rest of the function ...

def test_check_exit_amount(mocker: Any, default_conf: Dict[str, Any], fee: float) -> None:
    # ... rest of the function ...

def test_check_exit_amount_futures(mocker: Any, default_conf: Dict[str, Any], fee: float) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('config,wallets', [
    ({'stake_currency': 'USDT', 'dry_run_wallet': 1000.0}, {'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}}),
    ({'stake_currency': 'USDT', 'dry_run_wallet': {'USDT': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}}),
    # ... rest of the parameterized test ...
])
def test_dry_run_wallet_initialization(mocker: Any, default_conf_usdt: Dict[str, Any], config: Dict[str, Any], wallets: Dict[str, Any]) -> None:
    # ... rest of the function ...
