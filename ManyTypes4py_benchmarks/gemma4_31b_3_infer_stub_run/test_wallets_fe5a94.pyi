from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock
from freqtrade.persistence import Trade

def test_sync_wallet_at_boot(mocker: Any, default_conf: Dict[str, Any]) -> None: ...

def test_sync_wallet_missing_data(mocker: Any, default_conf: Dict[str, Any]) -> None: ...

def test_get_trade_stake_amount_no_stake_amount(default_conf: Dict[str, Any], mocker: Any) -> None: ...

@pytest.mark.parametrize('balance_ratio,capital,result1,result2', [
    (1, None, 50, 66.66666), 
    (0.99, None, 49.5, 66.0), 
    (0.5, None, 25, 33.3333), 
    (1, 100, 50, 0.0), 
    (0.99, 200, 50, 66.66666), 
    (0.99, 150, 50, 50), 
    (0.5, 50, 25, 0.0), 
    (0.5, 10, 5, 0.0)
])
def test_get_trade_stake_amount_unlimited_amount(
    default_conf: Dict[str, Any], 
    ticker: Dict[str, Any], 
    balance_ratio: float, 
    capital: Optional[float], 
    result1: float, 
    result2: float, 
    limit_buy_order_open: Any, 
    fee: float, 
    mocker: Any
) -> None: ...

@pytest.mark.parametrize('stake_amount,min_stake,stake_available,max_stake,trade_amount,expected', [
    (22, 11, 50, 10000, None, 22), 
    (100, 11, 500, 10000, None, 100), 
    (1000, 11, 500, 10000, None, 500), 
    (700, 11, 1000, 400, None, 400), 
    (20, 15, 10, 10000, None, 0), 
    (9, 11, 100, 10000, None, 11), 
    (1, 15, 10, 10000, None, 0), 
    (20, 50, 100, 10000, None, 0), 
    (1000, None, 1000, 10000, None, 1000), 
    (2000, 15, 2000, 3000, 1500, 1500)
])
def test_validate_stake_amount(
    mocker: Any, 
    default_conf: Dict[str, Any], 
    stake_amount: float, 
    min_stake: Optional[float], 
    stake_available: float, 
    max_stake: Optional[float], 
    trade_amount: Optional[float], 
    expected: float
) -> None: ...

@pytest.mark.parametrize('available_capital,closed_profit,open_stakes,free,expected', [
    (None, 10, 100, 910, 1000), 
    (None, 0, 0, 2500, 2500), 
    (None, 500, 0, 2500, 2000), 
    (None, 500, 0, 2500, 2000), 
    (None, -70, 0, 1930, 2000), 
    (100, 0, 0, 0, 100), 
    (1000, 0, 2, 5, 1000), 
    (1235, 2250, 2, 5, 1235), 
    (1235, -2250, 2, 5, 1235)
])
def test_get_starting_balance(
    mocker: Any, 
    default_conf: Dict[str, Any], 
    available_capital: Optional[float], 
    closed_profit: float, 
    open_stakes: float, 
    free: float, 
    expected: float
) -> None: ...

def test_sync_wallet_futures_live(mocker: Any, default_conf: Dict[str, Any]) -> None: ...

def test_sync_wallet_dry(mocker: Any, default_conf_usdt: Dict[str, Any], fee: float) -> None: ...

def test_sync_wallet_futures_dry(mocker: Any, default_conf: Dict[str, Any], fee: float) -> None: ...

def test_check_exit_amount(mocker: Any, default_conf: Dict[str, Any], fee: float) -> None: ...

def test_check_exit_amount_futures(mocker: Any, default_conf: Dict[str, Any], fee: float) -> None: ...

@pytest.mark.parametrize('config,wallets', [
    ({'stake_currency': 'USDT', 'dry_run_wallet': 1000.0}, {'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}}), 
    ({'stake_currency': 'USDT', 'dry_run_wallet': {'USDT': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}}), 
    ({'stake_currency': 'USDT', 'margin_mode': 'cross', 'dry_run_wallet': {'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 4200.0, 'used': 0.0, 'total': 0.0}, 'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}}), 
    ({'stake_currency': 'USDT', 'margin_mode': 'cross', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 4700.0, 'used': 0.0, 'total': 500.0}, 'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}}), 
    ({'stake_currency': 'USDT', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 500.0, 'used': 0.0, 'total': 500.0}, 'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}}), 
    ({'stake_currency': 'USDT', 'margin_mode': 'cross', 'trading_mode': 'futures', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}}, {'USDT': {'currency': 'USDT', 'free': 4700.0, 'used': 0.0, 'total': 500.0}, 'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}, 'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}, 'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}})
])
def test_dry_run_wallet_initialization(
    mocker: Any, 
    default_conf_usdt: Dict[str, Any], 
    config: Dict[str, Any], 
    wallets: Dict[str, Dict[str, Any]]
) -> None: ...