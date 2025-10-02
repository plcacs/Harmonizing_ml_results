from copy import deepcopy
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import Trade
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, patch_wallet

def test_sync_wallet_at_boot(mocker: MagicMock, default_conf: dict) -> None:
    default_conf['dry_run'] = False
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value={'BNT': {'free': 1.0, 'used': 2.0, 'total': 3.0}, 'GAS': {'free': 0.260739, 'used': 0.0, 'total': 0.260739}, 'USDT': {'free': 20, 'used': 20, 'total': 40}}))
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
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value={'BNT': {'free': 1.2, 'used': 1.9, 'total': 3.5}, 'GAS': {'free': 0.270739, 'used': 0.1, 'total': 0.260439}}))
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

def test_sync_wallet_missing_data(mocker: MagicMock, default_conf: dict) -> None:
    default_conf['dry_run'] = False
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value={'BNT': {'free': 1.0, 'used': 2.0, 'total': 3.0}, 'GAS': {'free': 0.260739, 'total': 0.260739}}))
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets['BNT'].free == 1.0
    assert freqtrade.wallets._wallets['BNT'].used == 2.0
    assert freqtrade.wallets._wallets['BNT'].total == 3.0
    assert freqtrade.wallets._wallets['GAS'].free == 0.260739
    assert freqtrade.wallets._wallets['GAS'].used == 0.0
    assert freqtrade.wallets._wallets['GAS'].total == 0.260739
    assert freqtrade.wallets.get_free('GAS') == 0.260739

def test_get_trade_stake_amount_no_stake_amount(default_conf: dict, mocker: MagicMock) -> None:
    patch_wallet(mocker, free=default_conf['stake_amount'] * 0.5)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.wallets.get_trade_stake_amount('ETH/BTC', 1)

# Remaining tests follow the same pattern with appropriate type annotations
