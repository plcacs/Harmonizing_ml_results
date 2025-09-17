#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import MagicMock
import pytest
from freqtrade.enums.marginmode import MarginMode
from freqtrade.enums.tradingmode import TradingMode
from tests.conftest import EXMS, get_mock_coro, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_additional_exchange_init_bybit(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    api_mock: Any = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={'dualSidePosition': False})
    api_mock.is_unified_enabled = MagicMock(return_value=[False, False])
    exchange: Any = get_patched_exchange(mocker, default_conf, exchange='bybit', api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 1
    assert api_mock.is_unified_enabled.call_count == 1
    assert exchange.unified_account is False
    assert log_has('Bybit: Standard account.', caplog)
    api_mock.set_position_mode.reset_mock()
    api_mock.is_unified_enabled = MagicMock(return_value=[False, True])
    exchange = get_patched_exchange(mocker, default_conf, exchange='bybit', api_mock=api_mock)
    assert log_has('Bybit: Unified account. Assuming dedicated subaccount for this bot.', caplog)
    assert api_mock.set_position_mode.call_count == 1
    assert api_mock.is_unified_enabled.call_count == 1
    assert exchange.unified_account is True
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'bybit', 'additional_exchange_init', 'set_position_mode')


async def test_bybit_fetch_funding_rate(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    api_mock: Any = MagicMock()
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=[])
    exchange: Any = get_patched_exchange(mocker, default_conf, exchange='bybit', api_mock=api_mock)
    limit: int = 200
    await exchange._fetch_funding_rate_history(pair='BTC/USDT:USDT', timeframe='4h', limit=limit)
    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == 'BTC/USDT:USDT'
    kwargs: Dict[str, Any] = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs['since'] is None
    api_mock.fetch_funding_rate_history.reset_mock()
    since_ms: int = 1610000000000
    await exchange._fetch_funding_rate_history(pair='BTC/USDT:USDT', timeframe='4h', limit=limit, since_ms=since_ms)
    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == 'BTC/USDT:USDT'
    kwargs = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs['since'] == since_ms


def test_bybit_get_funding_fees(default_conf: Dict[str, Any], mocker: Any) -> None:
    now: datetime = datetime.now(timezone.utc)
    exchange: Any = get_patched_exchange(mocker, default_conf, exchange='bybit')
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees('BTC/USDT:USDT', 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 0
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='bybit')
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees('BTC/USDT:USDT', 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1


def test_bybit_fetch_orders(default_conf: Dict[str, Any], mocker: Any, limit_order: Dict[str, Any]) -> None:
    api_mock: Any = MagicMock()
    api_mock.fetch_orders = MagicMock(return_value=[limit_order['buy'], limit_order['sell']])
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order['buy']])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order['buy']])
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    start_time: datetime = datetime.now(timezone.utc) - timedelta(days=20)
    exchange: Any = get_patched_exchange(mocker, default_conf, api_mock, exchange='bybit')
    assert exchange.fetch_orders('mocked', start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='bybit')
    res = exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == 3
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert len(res) == 2 * 3


def test_bybit_fetch_order_canceled_empty(default_conf_usdt: Dict[str, Any], mocker: Any) -> None:
    default_conf_usdt['dry_run'] = False
    api_mock: Any = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'symbol': 'BTC/USDT', 'status': 'canceled', 'filled': 0.0, 'remaining': 0.0, 'amount': 20.0})
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange: Any = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange='bybit')
    res: Dict[str, Any] = exchange.fetch_order('123', 'BTC/USDT')
    assert res['remaining'] is None
    assert res['filled'] == 0.0
    assert res['amount'] == 20.0
    assert res['status'] == 'canceled'
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'symbol': 'BTC/USDT', 'status': 'canceled', 'filled': 0.0, 'remaining': 20.0, 'amount': 20.0})
    res1: Dict[str, Any] = exchange.fetch_order('123', 'BTC/USDT')
    assert res1['remaining'] == 20.0
    assert res1['filled'] == 0.0
    assert res1['amount'] == 20.0
    assert res1['status'] == 'canceled'
    api_mock.fetch_order = MagicMock(return_value={'id': '124', 'symbol': 'BTC/USDT', 'status': 'open', 'filled': 0.0, 'remaining': 20.0, 'amount': 20.0})
    res2: Dict[str, Any] = exchange.fetch_order('123', 'BTC/USDT')
    assert res2['remaining'] == 20.0
    assert res2['filled'] == 0.0
    assert res2['amount'] == 20.0
    assert res2['status'] == 'open'


@pytest.mark.parametrize('side,order_type,uta,tradingmode,expected',
                         [('buy', 'limit', False, 'spot', True),
                          ('buy', 'limit', False, 'futures', True),
                          ('sell', 'limit', False, 'spot', True),
                          ('sell', 'limit', False, 'futures', True),
                          ('buy', 'market', False, 'spot', True),
                          ('buy', 'market', False, 'futures', False),
                          ('buy', 'market', True, 'spot', False),
                          ('buy', 'market', True, 'futures', False)])
def test_bybit__order_needs_price(default_conf: Dict[str, Any], mocker: Any,
                                  side: str, order_type: str, uta: bool, tradingmode: str, expected: bool) -> None:
    exchange: Any = get_patched_exchange(mocker, default_conf, exchange='bybit')
    exchange.trading_mode = tradingmode
    exchange.unified_account = uta
    assert exchange._order_needs_price(side, order_type) == expected
