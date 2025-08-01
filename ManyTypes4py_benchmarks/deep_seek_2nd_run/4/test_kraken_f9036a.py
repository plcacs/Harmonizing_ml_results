from random import randint
from unittest.mock import MagicMock
import ccxt
import pytest
from freqtrade.exceptions import DependencyException, InvalidOrderException
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers
from typing import Any, Dict, List, Tuple, Union, Optional

STOPLOSS_ORDERTYPE: str = 'stop-loss'
STOPLOSS_LIMIT_ORDERTYPE: str = 'stop-loss-limit'

@pytest.mark.parametrize('order_type,time_in_force,expected_params', [('limit', 'ioc', {'timeInForce': 'IOC', 'trading_agreement': 'agree'}), ('limit', 'PO', {'postOnly': True, 'trading_agreement': 'agree'}), ('market', None, {'trading_agreement': 'agree'})])
def test_kraken_trading_agreement(default_conf: Dict[str, Any], mocker: Any, order_type: str, time_in_force: Optional[str], expected_params: Dict[str, Any]) -> None:
    api_mock: MagicMock = MagicMock()
    order_id: str = f'test_prod_{order_type}_{randint(0, 10 ** 6)}'
    api_mock.options: Dict[str, Any] = {}
    api_mock.create_order: MagicMock = MagicMock(return_value={'id': order_id, 'symbol': 'ETH/BTC', 'info': {'foo': 'bar'}})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='kraken')
    order: Dict[str, Any] = exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force=time_in_force)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == (200 if order_type == 'limit' else None)
    assert api_mock.create_order.call_args[0][5] == expected_params

def test_get_balances_prod_kraken(default_conf: Dict[str, Any], mocker: Any) -> None:
    balance_item: Dict[str, float] = {'free': 0.0, 'total': 10.0, 'used': 0.0}
    kraken = ccxt.kraken()
    api_mock: MagicMock = MagicMock()
    api_mock.commonCurrencies: Dict[str, Any] = kraken.commonCurrencies
    api_mock.fetch_balance: MagicMock = MagicMock(return_value={'1ST': {'free': 0.0, 'total': 0.0, 'used': 0.0}, '1ST.F': balance_item.copy(), '2ND': balance_item.copy(), '3RD': balance_item.copy(), '4TH': balance_item.copy(), 'EUR': balance_item.copy(), 'BTC': {'free': 0.0, 'total': 0.0, 'used': 0.0}, 'XBT.F': balance_item.copy(), 'timestamp': 123123})
    kraken_open_orders: List[Dict[str, Any]] = [{'symbol': '1ST/EUR', 'type': 'limit', 'side': 'sell', 'price': 20, 'cost': 0.0, 'amount': 1.0, 'filled': 0.0, 'average': 0.0, 'remaining': 1.0}, {'status': 'open', 'symbol': '2ND/EUR', 'type': 'limit', 'side': 'sell', 'price': 20.0, 'cost': 0.0, 'amount': 2.0, 'filled': 0.0, 'average': 0.0, 'remaining': 2.0}, {'status': 'open', 'symbol': '2ND/USD', 'type': 'limit', 'side': 'sell', 'price': 20.0, 'cost': 0.0, 'amount': 2.0, 'filled': 0.0, 'average': 0.0, 'remaining': 2.0}, {'status': 'open', 'symbol': '3RD/EUR', 'type': 'limit', 'side': 'buy', 'price': 0.02, 'cost': 0.0, 'amount': 100.0, 'filled': 0.0, 'average': 0.0, 'remaining': 100.0}]
    api_mock.fetch_open_orders: MagicMock = MagicMock(return_value=kraken_open_orders)
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='kraken')
    balances: Dict[str, Dict[str, float]] = exchange.get_balances()
    assert len(balances) == 7
    assert balances['1ST']['free'] == 9.0
    assert balances['1ST']['total'] == 10.0
    assert balances['1ST']['used'] == 1.0
    assert balances['2ND']['free'] == 6.0
    assert balances['2ND']['total'] == 10.0
    assert balances['2ND']['used'] == 4.0
    assert balances['3RD']['free'] == 10.0
    assert balances['3RD']['total'] == 10.0
    assert balances['3RD']['used'] == 0.0
    assert balances['4TH']['free'] == 10.0
    assert balances['4TH']['total'] == 10.0
    assert balances['4TH']['used'] == 0.0
    assert balances['EUR']['free'] == 8.0
    assert balances['EUR']['total'] == 10.0
    assert balances['EUR']['used'] == 2.0
    assert balances['BTC']['free'] == 10.0
    assert balances['BTC']['total'] == 10.0
    assert balances['BTC']['used'] == 0.0
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'kraken', 'get_balances', 'fetch_balance')

@pytest.mark.parametrize('ordertype', ['market', 'limit'])
@pytest.mark.parametrize('side,adjustedprice', [('sell', 217.8), ('buy', 222.2)])
def test_create_stoploss_order_kraken(default_conf: Dict[str, Any], mocker: Any, ordertype: str, side: str, adjustedprice: float) -> None:
    api_mock: MagicMock = MagicMock()
    order_id: str = f'test_prod_buy_{randint(0, 10 ** 6)}'
    api_mock.create_order: MagicMock = MagicMock(return_value={'id': order_id, 'info': {'foo': 'bar'}})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
    order: Dict[str, Any] = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, side=side, order_types={'stoploss': ordertype, 'stoploss_on_exchange_limit_ratio': 0.99}, leverage=1.0)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args_list[0][1]['symbol'] == 'ETH/BTC'
    assert api_mock.create_order.call_args_list[0][1]['type'] == ordertype
    assert api_mock.create_order.call_args_list[0][1]['params'] == {'trading_agreement': 'agree', 'stopLossPrice': 220}
    assert api_mock.create_order.call_args_list[0][1]['side'] == side
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    if ordertype == 'limit':
        assert api_mock.create_order.call_args_list[0][1]['price'] == adjustedprice
    else:
        assert api_mock.create_order.call_args_list[0][1]['price'] is None
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds('0 balance'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
        exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)
    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('kraken Order would trigger immediately.'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
        exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'kraken', 'create_stoploss', 'create_order', retries=1, pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)

@pytest.mark.parametrize('side', ['buy', 'sell'])
def test_create_stoploss_order_dry_run_kraken(default_conf: Dict[str, Any], mocker: Any, side: str) -> None:
    api_mock: MagicMock = MagicMock()
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'kraken')
    api_mock.create_order.reset_mock()
    order: Dict[str, Any] = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)
    assert 'id' in order
    assert 'info' in order
    assert 'type' in order
    assert order['type'] == 'market'
    assert order['price'] == 220
    assert order['amount'] == 1

@pytest.mark.parametrize('sl1,sl2,sl3,side', [(1501, 1499, 1501, 'sell'), (1499, 1501, 1499, 'buy')])
def test_stoploss_adjust_kraken(mocker: Any, default_conf: Dict[str, Any], sl1: float, sl2: float, sl3: float, side: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='kraken')
    order: Dict[str, Any] = {'type': 'market', 'stopLossPrice': 1500}
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)
    order['type'] = 'limit'
    assert exchange.stoploss_adjust(sl3, order, side=side)

@pytest.mark.parametrize('trade_id, expected', [('1234', False), ('170544369512007228', False), ('1705443695120072285', True), ('170544369512007228555', True)])
def test__valid_trade_pagination_id_kraken(mocker: Any, default_conf_usdt: Dict[str, Any], trade_id: str, expected: bool) -> None:
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange='kraken')
    assert exchange._valid_trade_pagination_id('XRP/USDT', trade_id) == expected
