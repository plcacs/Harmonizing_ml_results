from datetime import datetime, timedelta
from random import randint
from unittest.mock import MagicMock, PropertyMock
from typing import Any, Dict, List, Optional, Tuple, Union
import ccxt
import pandas as pd
import pytest
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers

@pytest.mark.parametrize('side,order_type,time_in_force,expected', [('buy', 'limit', 'gtc', {'timeInForce': 'GTC'}), ('buy', 'limit', 'IOC', {'timeInForce': 'IOC'}), ('buy', 'market', 'IOC', {}), ('buy', 'limit', 'PO', {'timeInForce': 'PO'}), ('sell', 'limit', 'PO', {'timeInForce': 'PO'}), ('sell', 'market', 'PO', {})])
def test__get_params_binance(default_conf: Dict[str, Any], mocker: Any, side: str, order_type: str, time_in_force: str, expected: Dict[str, str]) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    assert exchange._get_params(side, order_type, 1, False, time_in_force) == expected

@pytest.mark.parametrize('trademode', [TradingMode.FUTURES, TradingMode.SPOT])
@pytest.mark.parametrize('limitratio,expected,side', [(None, 220 * 0.99, 'sell'), (0.99, 220 * 0.99, 'sell'), (0.98, 220 * 0.98, 'sell'), (None, 220 * 1.01, 'buy'), (0.99, 220 * 1.01, 'buy'), (0.98, 220 * 1.02, 'buy')])
def test_create_stoploss_order_binance(default_conf: Dict[str, Any], mocker: Any, limitratio: Optional[float], expected: float, side: str, trademode: TradingMode) -> None:
    api_mock = MagicMock()
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'
    order_type = 'stop_loss_limit' if trademode == TradingMode.SPOT else 'stop'
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'info': {'foo': 'bar'}})
    default_conf['dry_run'] = False
    default_conf['margin_mode'] = MarginMode.ISOLATED
    default_conf['trading_mode'] = trademode
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=190, side=side, order_types={'stoploss': 'limit', 'stoploss_on_exchange_limit_ratio': 1.05}, leverage=1.0)
    api_mock.create_order.reset_mock()
    order_types = {'stoploss': 'limit', 'stoploss_price_type': 'mark'}
    if limitratio is not None:
        order_types.update({'stoploss_on_exchange_limit_ratio': limitratio})
    order = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types=order_types, side=side, leverage=1.0)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args_list[0][1]['symbol'] == 'ETH/BTC'
    assert api_mock.create_order.call_args_list[0][1]['type'] == order_type
    assert api_mock.create_order.call_args_list[0][1]['side'] == side
    assert api_mock.create_order.call_args_list[0][1]['amount'] == 1
    assert api_mock.create_order.call_args_list[0][1]['price'] == expected
    if trademode == TradingMode.SPOT:
        params_dict = {'stopPrice': 220}
    else:
        params_dict = {'stopPrice': 220, 'reduceOnly': True, 'workingType': 'MARK_PRICE'}
    assert api_mock.create_order.call_args_list[0][1]['params'] == params_dict
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds('0 balance'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)
    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('binance Order would trigger immediately.'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
        exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'binance', 'create_stoploss', 'create_order', retries=1, pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side=side, leverage=1.0)

def test_create_stoploss_order_dry_run_binance(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock = MagicMock()
    order_type = 'stop_loss_limit'
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, 'binance')
    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=190, side='sell', order_types={'stoploss_on_exchange_limit_ratio': 1.05}, leverage=1.0)
    api_mock.create_order.reset_mock()
    order = exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side='sell', leverage=1.0)
    assert 'id' in order
    assert 'info' in order
    assert 'type' in order
    assert order['type'] == order_type
    assert order['price'] == 220
    assert order['amount'] == 1

@pytest.mark.parametrize('sl1,sl2,sl3,side', [(1501, 1499, 1501, 'sell'), (1499, 1501, 1499, 'buy')])
def test_stoploss_adjust_binance(mocker: Any, default_conf: Dict[str, Any], sl1: float, sl2: float, sl3: float, side: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    order = {'type': 'stop_loss_limit', 'price': 1500, 'stopPrice': 1500, 'info': {'stopPrice': 1500}}
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)

@pytest.mark.parametrize('pair, is_short, trading_mode, margin_mode, wallet_balance, maintenance_amt, amount, open_rate, open_trades,mm_ratio, expected', [('ETH/USDT:USDT', False, 'futures', 'isolated', 1535443.01, 135365.0, 3683.979, 1456.84, [], 0.1, 1114.78), ('ETH/USDT:USDT', False, 'futures', 'isolated', 1535443.01, 16300.0, 109.488, 32481.98, [], 0.025, 18778.73), ('ETH/USDT:USDT', False, 'futures', 'cross', 1535443.01, 135365.0, 3683.979, 1456.84, [{'pair': 'BTC/USDT:USDT', 'open_rate': 32481.98, 'amount': 109.488, 'stake_amount': 3556387.02624, 'mark_price': 31967.27, 'mm_ratio': 0.025, 'maintenance_amt': 16300.0}, {'pair': 'ETH/USDT:USDT', 'open_rate': 1456.84, 'amount': 3683.979, 'stake_amount': 5366967.96, 'mark_price': 1335.18, 'mm_ratio': 0.1, 'maintenance_amt': 135365.0}], 0.1, 1153.26), ('BTC/USDT:USDT', False, 'futures', 'cross', 1535443.01, 16300.0, 109.488, 32481.98, [{'pair': 'BTC/USDT:USDT', 'open_rate': 32481.98, 'amount': 109.488, 'stake_amount': 3556387.02624, 'mark_price': 31967.27, 'mm_ratio': 0.025, 'maintenance_amt': 16300.0}, {'pair': 'ETH/USDT:USDT', 'open_rate': 1456.84, 'amount': 3683.979, 'stake_amount': 5366967.96, 'mark_price': 1335.18, 'mm_ratio': 0.1, 'maintenance_amt': 135365.0}], 0.025, 26316.89)])
def test_liquidation_price_binance(mocker: Any, default_conf: Dict[str, Any], pair: str, is_short: bool, trading_mode: str, margin_mode: str, wallet_balance: float, maintenance_amt: float, amount: float, open_rate: float, open_trades: List[Dict[str, Any]], mm_ratio: float, expected: float) -> None:
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = margin_mode
    default_conf['liquidation_buffer'] = 0.0
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')

    def get_maint_ratio(pair_: str, stake_amount: float) -> Tuple[float, float]:
        if pair_ != pair:
            oc = [c for c in open_trades if c['pair'] == pair_][0]
            return (oc['mm_ratio'], oc['maintenance_amt'])
        return (mm_ratio, maintenance_amt)

    def fetch_funding_rates(*args, **kwargs) -> Dict[str, Dict[str, Any]]:
        return {t['pair']: {'symbol': t['pair'], 'markPrice': t['mark_price']} for t in open_trades}
    exchange.get_maintenance_ratio_and_amt = get_maint_ratio
    exchange.fetch_funding_rates = fetch_funding_rates
    open_trade_objects = [Trade(pair=t['pair'], open_rate=t['open_rate'], amount=t['amount'], stake_amount=t['stake_amount'], fee_open=0) for t in open_trades]
    assert pytest.approx(round(exchange.get_liquidation_price(pair=pair, open_rate=open_rate, is_short=is_short, wallet_balance=wallet_balance, amount=amount, stake_amount=open_rate * amount, leverage=5, open_trades=open_trade_objects), 2)) == expected

def test_fill_leverage_tiers_binance(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock = MagicMock()
    api_mock.fetch_leverage_tiers = MagicMock(return_value={'ADA/BUSD': [{'tier': 1, 'minNotional': 0, 'maxNotional': 100000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'info': {'bracket': '1', 'initialLeverage': '20', 'maxNotional': '100000', 'minNotional': '0', 'maintMarginRatio': '0.025', 'cum': '0.0'}}, {'tier': 2, 'minNotional': 100000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'info': {'bracket': '2', 'initialLeverage': '10', 'maxNotional': '500000', 'minNotional': '100000', 'maintMarginRatio': '0.05', 'cum': '2500.0'}}, {'tier': 3, 'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'info': {'bracket': '3', 'initialLeverage': '5', 'maxNotional': '1000000', 'minNotional': '500000', 'maintMarginRatio': '0.1', 'cum': '27500.0'}}, {'tier': 4, 'minNotional': 1000000, 'maxNotional': 2000000, 'maintenanceMarginRate': 0.15, 'maxLeverage': 3, 'info': {'bracket': '4', 'initialLeverage': '3', 'maxNotional': '2000000', 'minNotional': '1000000', 'maintMarginRatio': '0.15', 'cum': '77500.0'}}, {'tier': 5, 'minNotional': 2000000, 'maxNotional': 5000000, 'maintenanceMarginRate': 0.25, 'maxLeverage': 2, 'info': {'bracket': '5', 'initialLeverage': '2', 'maxNotional': '5000000', 'minNotional': '2000000', 'maintMarginRatio': '0.25', 'cum': '277500.0'}}, {'tier': 6, 'minNotional': 5000000, 'maxNotional': 30000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'info': {'bracket': '6', 'initialLeverage': '1', 'maxNotional': '30000000', 'minNotional': '5000000', 'maintMarginRatio': '0.5', 'cum': '1527500.0'}}], 'ZEC/USDT': [{'tier': 1, 'minNotional': 0, 'maxNotional': 50000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'info': {'bracket': '1', 'initialLeverage': '50', 'maxNotional': '50000', 'minNotional': '0', 'maintMarginRatio': '0.01', 'cum': '0.0'}}, {'tier': 2, 'minNotional': 50000, 'maxNotional': 150000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'info': {'bracket': '2', 'initialLeverage': '20', 'maxNotional': '150000', 'minNotional': '50000', 'maintMarginRatio': '0.025', 'cum': '750.0'}}, {'tier': 3, 'minNotional': 150000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'info': {'bracket': '3', 'initialLeverage': '10', 'maxNotional': '250000', 'minNotional': '150000', 'maintMarginRatio': '0.05', 'cum': '4500.0'}}, {'tier': 4, 'minNotional': 250000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'info': {'bracket': '4', 'initialLeverage': '5', 'maxNotional': '500000', 'minNotional': '250000', 'maintMarginRatio': '0.1', 'cum': '17000.0'}}, {'tier': 5, 'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.125, 'maxLeverage': 4, 'info': {'bracket': '5', 'initialLeverage': '4', 'maxNotional': '1000000', 'minNotional': '500000', 'maintMarginRatio': '0.125', 'cum': '29500.0'}}, {'t