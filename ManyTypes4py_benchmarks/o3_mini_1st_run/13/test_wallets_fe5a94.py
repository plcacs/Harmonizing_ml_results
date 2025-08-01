from copy import deepcopy
from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import Trade
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, patch_wallet

def test_sync_wallet_at_boot(mocker: pytest.MockerFixture, default_conf: Dict[str, Any]) -> None:
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            'BNT': {'free': 1.0, 'used': 2.0, 'total': 3.0},
            'GAS': {'free': 0.260739, 'used': 0.0, 'total': 0.260739},
            'USDT': {'free': 20, 'used': 20, 'total': 40}
        })
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 3
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used == 0.0
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('BNT') == 1.0
    assert 'USDT' in freqtrade.wallets._wallets
    assert freqtrade.wallets._last_wallet_refresh is not None

    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            'BNT': {'free': 1.2, 'used': 1.9, 'total': 3.5},
            'GAS': {'free': 0.270739, 'used': 0.1, 'total': 0.260439}
        })
    )
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.2
    assert freqtrade.wallets._wallets['BNT'].used == 1.9
    assert freqtrade.wallets._wallets['BNT'].total == 3.5
    assert freqtrade.wallets._wallets['GAS'].free == 0.270739
    assert freqtrade.wallets._wallets['GAS'].used == 0.1
    assert freqtrade.wallets._wallets['GAS'].total == 0.260439
    assert freqtrade.wallets.get_free('GAS') == 0.270739
    assert freqtrade.wallets.get_used('GAS') == 0.1
    assert freqtrade.wallets.get_total('GAS') == 0.260439
    assert freqtrade.wallets.get_owned('GAS/USDT', 'GAS') == 0.260439
    update_mock = mocker.patch('freqtrade.wallets.Wallets._update_live')
    freqtrade.wallets.update(False)
    assert update_mock.call_count == 0
    freqtrade.wallets.update()
    assert update_mock.call_count == 1
    assert freqtrade.wallets.get_free('NOCURRENCY') == 0
    assert freqtrade.wallets.get_used('NOCURRENCY') == 0
    assert freqtrade.wallets.get_total('NOCURRENCY') == 0
    assert freqtrade.wallets.get_owned('NOCURRENCY/USDT', 'NOCURRENCY') == 0

def test_sync_wallet_missing_data(mocker: pytest.MockerFixture, default_conf: Dict[str, Any]) -> None:
    default_conf['dry_run'] = False
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={
            'BNT': {'free': 1.0, 'used': 2.0, 'total': 3.0},
            'GAS': {'free': 0.260739, 'total': 0.260739}
        })
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used == 0.0
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('GAS') == 0.260739

def test_get_trade_stake_amount_no_stake_amount(default_conf: Dict[str, Any], mocker: pytest.MockerFixture) -> None:
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.wallets.get_trade_stake_amount('ETH/BTC', 1)

@pytest.mark.parametrize(
    'balance_ratio,capital,result1,result2',
    [
        (1, None, 50, 66.66666),
        (0.99, None, 49.5, 66.0),
        (0.5, None, 25, 33.3333),
        (1, 100, 50, 0.0),
        (0.99, 200, 50, 66.66666),
        (0.99, 150, 50, 50),
        (0.5, 50, 25, 0.0),
        (0.5, 10, 5, 0.0)
    ]
)
def test_get_trade_stake_amount_unlimited_amount(
    default_conf: Dict[str, Any],
    ticker: Any,
    balance_ratio: float,
    capital: Optional[Union[int, float]],
    result1: Union[int, float],
    result2: Union[int, float],
    limit_buy_order_open: Any,
    fee: Any,
    mocker: pytest.MockerFixture
) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        create_order=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee
    )
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['stake_amount'] = UNLIMITED_STAKE_AMOUNT
    conf['dry_run_wallet'] = 100
    conf['tradable_balance_ratio'] = balance_ratio
    if capital is not None:
        conf['available_capital'] = capital
    freqtrade = get_patched_freqtradebot(mocker, conf)
    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 2)
    assert result == result1
    freqtrade.execute_entry('ETH/USDT', result)
    result = freqtrade.wallets.get_trade_stake_amount('LTC/USDT', 2)
    assert result == result1
    freqtrade.execute_entry('LTC/BTC', result)
    result = freqtrade.wallets.get_trade_stake_amount('XRP/USDT', 2)
    assert result == 0
    freqtrade.config['dry_run_wallet'] = 200
    freqtrade.wallets._start_cap['BTC'] = 200
    result = freqtrade.wallets.get_trade_stake_amount('XRP/USDT', 3)
    assert round(result, 4) == round(result2, 4)
    result = freqtrade.wallets.get_trade_stake_amount('NEO/USDT', 0)
    assert result == 0

@pytest.mark.parametrize(
    'stake_amount,min_stake,stake_available,max_stake,trade_amount,expected',
    [
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
    ]
)
def test_validate_stake_amount(
    mocker: pytest.MockerFixture,
    default_conf: Dict[str, Any],
    stake_amount: Union[int, float],
    min_stake: Optional[Union[int, float]],
    stake_available: Union[int, float],
    max_stake: Union[int, float],
    trade_amount: Optional[Union[int, float]],
    expected: Union[int, float]
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.wallets.Wallets.get_available_stake_amount', return_value=stake_available)
    res = freqtrade.wallets.validate_stake_amount('XRP/USDT', stake_amount, min_stake, max_stake, trade_amount)
    assert res == expected

@pytest.mark.parametrize(
    'available_capital,closed_profit,open_stakes,free,expected',
    [
        (None, 10, 100, 910, 1000),
        (None, 0, 0, 2500, 2500),
        (None, 500, 0, 2500, 2000),
        (None, 500, 0, 2500, 2000),
        (None, -70, 0, 1930, 2000),
        (100, 0, 0, 0, 100),
        (1000, 0, 2, 5, 1000),
        (1235, 2250, 2, 5, 1235),
        (1235, -2250, 2, 5, 1235)
    ]
)
def test_get_starting_balance(
    mocker: pytest.MockerFixture,
    default_conf: Dict[str, Any],
    available_capital: Optional[Union[int, float]],
    closed_profit: Union[int, float],
    open_stakes: Union[int, float],
    free: Union[int, float],
    expected: Union[int, float]
) -> None:
    if available_capital:
        default_conf['available_capital'] = available_capital
    mocker.patch('freqtrade.persistence.models.Trade.get_total_closed_profit', return_value=closed_profit)
    mocker.patch('freqtrade.persistence.models.Trade.total_open_trades_stakes', return_value=open_stakes)
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=free)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    multiplier = 1 if available_capital else 0.99
    assert freqtrade.wallets.get_starting_balance() == expected * multiplier

def test_sync_wallet_futures_live(mocker: pytest.MockerFixture, default_conf: Dict[str, Any]) -> None:
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    mock_result = [
        {'symbol': 'ETH/USDT:USDT', 'timestamp': None, 'datetime': None, 'initialMargin': 0.0,
         'initialMarginPercentage': None, 'maintenanceMargin': 0.0, 'maintenanceMarginPercentage': 0.005,
         'entryPrice': 0.0, 'notional': 100.0, 'leverage': 5.0, 'unrealizedPnl': 0.0, 'contracts': 100.0,
         'contractSize': 1, 'marginRatio': None, 'liquidationPrice': 0.0, 'markPrice': 2896.41,
         'collateral': 20, 'marginType': 'isolated', 'side': 'short', 'percentage': None},
        {'symbol': 'ADA/USDT:USDT', 'timestamp': None, 'datetime': None, 'initialMargin': 0.0,
         'initialMarginPercentage': None, 'maintenanceMargin': 0.0, 'maintenanceMarginPercentage': 0.005,
         'entryPrice': 0.0, 'notional': 100.0, 'leverage': 5.0, 'unrealizedPnl': 0.0, 'contracts': 100.0,
         'contractSize': 1, 'marginRatio': None, 'liquidationPrice': 0.0, 'markPrice': 0.91,
         'collateral': 20, 'marginType': 'isolated', 'side': 'short', 'percentage': None},
        {'symbol': 'SOL/BUSD:BUSD', 'timestamp': None, 'datetime': None, 'initialMargin': 0.0,
         'initialMarginPercentage': None, 'maintenanceMargin': 0.0, 'maintenanceMarginPercentage': 0.005,
         'entryPrice': 0.0, 'notional': 0.0, 'leverage': 5.0, 'unrealizedPnl': 0.0, 'contracts': 0.0,
         'contractSize': 1, 'marginRatio': None, 'liquidationPrice': 0.0, 'markPrice': 15.41,
         'collateral': 0.0, 'marginType': 'isolated', 'side': 'short', 'percentage': None}
    ]
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value={'USDT': {'free': 900, 'used': 100, 'total': 1000}}),
        fetch_positions=MagicMock(return_value=mock_result)
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 2
    assert 'USDT' in freqtrade.wallets._wallets
    assert 'ETH/USDT:USDT' in freqtrade.wallets._positions
    assert freqtrade.wallets._last_wallet_refresh is not None
    assert freqtrade.wallets.get_owned('ETH/USDT:USDT', 'ETH') == 1000
    assert freqtrade.wallets.get_owned('SOL/USDT:USDT', 'SOL') == 0
    del mock_result[0]
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._positions) == 1
    assert 'ETH/USDT:USDT' not in freqtrade.wallets._positions

def test_sync_wallet_dry(mocker: pytest.MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any) -> None:
    default_conf_usdt['dry_run'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 0
    assert freqtrade.wallets.get_total('USDT') == 1000
    create_mock_trades_usdt(fee, is_short=None)
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._wallets) == 5
    assert len(freqtrade.wallets._positions) == 0
    bal: Dict[str, Any] = freqtrade.wallets.get_all_balances()
    assert bal['NEO'].total == 0
    assert bal['XRP'].total == 10
    assert bal['LTC'].total == 2
    usdt_bal = bal['USDT']
    assert usdt_bal.free == 922.74
    assert usdt_bal.total == 942.74
    assert usdt_bal.used == 20.0
    assert usdt_bal.total == usdt_bal.free + usdt_bal.used
    assert freqtrade.wallets.get_starting_balance() == default_conf_usdt['dry_run_wallet'] * default_conf_usdt['tradable_balance_ratio']
    total = freqtrade.wallets.get_total('LTC')
    free = freqtrade.wallets.get_free('LTC')
    used = freqtrade.wallets.get_used('LTC')
    assert used != 0
    assert free + used == total

def test_sync_wallet_futures_dry(mocker: pytest.MockerFixture, default_conf: Dict[str, Any], fee: Any) -> None:
    default_conf['dry_run'] = True
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 0
    create_mock_trades(fee, is_short=None)
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 4
    positions = freqtrade.wallets.get_all_positions()
    assert positions['ETH/BTC'].side == 'short'
    assert positions['ETC/BTC'].side == 'long'
    assert positions['XRP/BTC'].side == 'long'
    assert positions['LTC/BTC'].side == 'short'
    assert freqtrade.wallets.get_starting_balance() == default_conf['dry_run_wallet'] * default_conf['tradable_balance_ratio']
    total = freqtrade.wallets.get_total('BTC')
    free = freqtrade.wallets.get_free('BTC')
    used = freqtrade.wallets.get_used('BTC')
    assert free + used == total

def test_check_exit_amount(mocker: pytest.MockerFixture, default_conf: Dict[str, Any], fee: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    update_mock = mocker.patch('freqtrade.wallets.Wallets.update')
    total_mock = mocker.patch('freqtrade.wallets.Wallets.get_total', return_value=50.0)
    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.amount == 50.0
    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert update_mock.call_count == 0
    assert total_mock.call_count == 1
    update_mock.reset_mock()
    total_mock = mocker.patch('freqtrade.wallets.Wallets.get_total', return_value=40)
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert update_mock.call_count == 1
    assert total_mock.call_count == 2

def test_check_exit_amount_futures(mocker: pytest.MockerFixture, default_conf: Dict[str, Any], fee: Any) -> None:
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    total_mock = mocker.patch('freqtrade.wallets.Wallets.get_total', return_value=50)
    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    trade.trading_mode = 'futures'
    assert trade.amount == 50
    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert total_mock.call_count == 0
    update_mock = mocker.patch('freqtrade.wallets.Wallets.update')
    trade.amount = 150
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert total_mock.call_count == 0
    assert update_mock.call_count == 1

@pytest.mark.parametrize(
    'config,wallets',
    [
        (
            {'stake_currency': 'USDT', 'dry_run_wallet': 1000.0},
            {
                'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0}
            }
        ),
        (
            {'stake_currency': 'USDT', 'dry_run_wallet': {'USDT': 1000.0, 'BTC': 0.1, 'ETH': 2.0}},
            {
                'USDT': {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1},
                'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}
            }
        ),
        (
            {'stake_currency': 'USDT', 'margin_mode': 'cross', 'dry_run_wallet': {'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}},
            {
                'USDT': {'currency': 'USDT', 'free': 4200.0, 'used': 0.0, 'total': 0.0},
                'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1},
                'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}
            }
        ),
        (
            {'stake_currency': 'USDT', 'margin_mode': 'cross', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}},
            {
                'USDT': {'currency': 'USDT', 'free': 4700.0, 'used': 0.0, 'total': 500.0},
                'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1},
                'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}
            }
        ),
        (
            {'stake_currency': 'USDT', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}},
            {
                'USDT': {'currency': 'USDT', 'free': 500.0, 'used': 0.0, 'total': 500.0},
                'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1},
                'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}
            }
        ),
        (
            {'stake_currency': 'USDT', 'margin_mode': 'cross', 'trading_mode': 'futures', 'dry_run_wallet': {'USDT': 500, 'USDC': 1000.0, 'BTC': 0.1, 'ETH': 2.0}},
            {
                'USDT': {'currency': 'USDT', 'free': 4700.0, 'used': 0.0, 'total': 500.0},
                'USDC': {'currency': 'USDC', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                'BTC': {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1},
                'ETH': {'currency': 'ETH', 'free': 2.0, 'used': 0.0, 'total': 2.0}
            }
        )
    ]
)
def test_dry_run_wallet_initialization(
    mocker: pytest.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    config: Dict[str, Any],
    wallets: Dict[str, Dict[str, Union[str, float]]]
) -> None:
    default_conf_usdt.update(config)
    mocker.patch(f'{EXMS}.get_tickers', return_value={
        'USDC/USDT': {'last': 1.0},
        'BTC/USDT': {'last': 20000.0},
        'ETH/USDT': {'last': 1100.0}
    })
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    stake_currency: str = config['stake_currency']
    for currency, expected_wallet in wallets.items():
        wallet = freqtrade.wallets._wallets[currency]
        assert wallet.currency == expected_wallet['currency']
        assert wallet.free == expected_wallet['free']
        assert wallet.used == expected_wallet['used']
        assert wallet.total == expected_wallet['total']
    assert len(freqtrade.wallets._wallets) == len(wallets)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.0)
    mocker.patch(f'{EXMS}.get_rate', return_value=2.22)
    mocker.patch(f'{EXMS}.fetch_ticker', return_value={'bid': 0.2, 'ask': 0.22, 'last': 0.22})
    assert freqtrade.wallets.get_collateral() == freqtrade.wallets.get_free(stake_currency)
    freqtrade.execute_entry('NEO/USDT', 100.0)
    freqtrade.wallets.update()
    if default_conf_usdt.get('trading_mode') != 'futures':
        assert 'NEO' in freqtrade.wallets._wallets
        assert freqtrade.wallets._wallets['NEO'].total == 45.04504504
        assert freqtrade.wallets._wallets['NEO'].used == 0.0
        assert freqtrade.wallets._wallets['NEO'].free == 45.04504504
        assert freqtrade.wallets.get_collateral() == freqtrade.wallets.get_free(stake_currency)
        assert pytest.approx(freqtrade.wallets._wallets[stake_currency].total) == wallets[stake_currency]['total'] - 100.0
        assert len(freqtrade.wallets._wallets) == len(wallets) + 1
    else:
        assert 'NEO' not in freqtrade.wallets._wallets
        assert freqtrade.wallets._positions['NEO/USDT'].position == 45.04504504
        assert pytest.approx(freqtrade.wallets._positions['NEO/USDT'].collateral) == 100
        assert pytest.approx(freqtrade.wallets.get_collateral()) == freqtrade.wallets.get_free(stake_currency) + 100
        assert pytest.approx(freqtrade.wallets._wallets[stake_currency].free) == wallets[stake_currency]['free'] - 100.0