from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, PropertyMock
import ccxt
import pytest
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.exchange.exchange import timeframe_to_minutes
from tests.conftest import EXMS, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers

def test_okx_ohlcv_candle_limit(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    timeframes: Tuple[str, ...] = ('1m', '5m', '1h')
    start_time: int = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 100
        one_call: int = int((datetime.now(timezone.utc) - timedelta(minutes=290 * timeframe_to_minutes(timeframe))).timestamp() * 1000)
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        one_call = int((datetime.now(timezone.utc) - timedelta(minutes=320 * timeframe_to_minutes(timeframe))).timestamp() * 1000)
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 100

def test_get_maintenance_ratio_and_amt_okx(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock: MagicMock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['dry_run'] = False
    mocker.patch.multiple('freqtrade.exchange.okx.Okx', exchange_has=MagicMock(return_value=True), load_leverage_tiers=MagicMock(return_value={'ETH/USDT:USDT': [{'tier': 1, 'minNotional': 0, 'maxNotional': 2000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 75, 'info': {'baseMaxLoan': '', 'imr': '0.013', 'instId': '', 'maxLever': '75', 'maxSz': '2000', 'minSz': '0', 'mmr': '0.01', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '1', 'uly': 'ETH-USDT'}}, {'tier': 2, 'minNotional': 2001, 'maxNotional': 4000, 'maintenanceMarginRate': 0.015, 'maxLeverage': 50, 'info': {'baseMaxLoan': '', 'imr': '0.02', 'instId': '', 'maxLever': '50', 'maxSz': '4000', 'minSz': '2001', 'mmr': '0.015', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '2', 'uly': 'ETH-USDT'}}, {'tier': 3, 'minNotional': 4001, 'maxNotional': 8000, 'maintenanceMarginRate': 0.02, 'maxLeverage': 20, 'info': {'baseMaxLoan': '', 'imr': '0.05', 'instId': '', 'maxLever': '20', 'maxSz': '8000', 'minSz': '4001', 'mmr': '0.02', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '3', 'uly': 'ETH-USDT'}}], 'ADA/USDT:USDT': [{'tier': 1, 'minNotional': 0, 'maxNotional': 500, 'maintenanceMarginRate': 0.02, 'maxLeverage': 75, 'info': {'baseMaxLoan': '', 'imr': '0.013', 'instId': '', 'maxLever': '75', 'maxSz': '500', 'minSz': '0', 'mmr': '0.01', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '1', 'uly': 'ADA-USDT'}}, {'tier': 2, 'minNotional': 501, 'maxNotional': 1000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 50, 'info': {'baseMaxLoan': '', 'imr': '0.02', 'instId': '', 'maxLever': '50', 'maxSz': '1000', 'minSz': '501', 'mmr': '0.015', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '2', 'uly': 'ADA-USDT'}}, {'tier': 3, 'minNotional': 1001, 'maxNotional': 2000, 'maintenanceMarginRate': 0.03, 'maxLeverage': 20, 'info': {'baseMaxLoan': '', 'imr': '0.05', 'instId': '', 'maxLever': '20', 'maxSz': '2000', 'minSz': '1001', 'mmr': '0.02', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '3', 'uly': 'ADA-USDT'}}]}))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 2000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 2001) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 4001) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('ETH/USDT:USDT', 8000) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('ADA/USDT:USDT', 1) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt('ADA/USDT:USDT', 2000) == (0.03, None)

def test_get_max_pair_stake_amount_okx(default_conf: Dict[str, Any], mocker: Any, leverage_tiers: Dict[str, List[Dict[str, Any]]]) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    assert exchange.get_max_pair_stake_amount('BNB/BUSD', 1.0) == float('inf')
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    exchange._leverage_tiers = leverage_tiers
    assert exchange.get_max_pair_stake_amount('XRP/USDT:USDT', 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount('BNB/USDT:USDT', 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT:USDT', 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount('BTC/USDT:USDT', 1.0, 10.0) == 100000000
    assert exchange.get_max_pair_stake_amount('TTT/USDT:USDT', 1.0) == float('inf')

@pytest.mark.parametrize('mode,side,reduceonly,result', [('net', 'buy', False, 'net'), ('net', 'sell', True, 'net'), ('net', 'sell', False, 'net'), ('net', 'buy', True, 'net'), ('longshort', 'buy', False, 'long'), ('longshort', 'sell', True, 'long'), ('longshort', 'sell', False, 'short'), ('longshort', 'buy', True, 'short')])
def test__get_posSide(default_conf: Dict[str, Any], mocker: Any, mode: str, side: str, reduceonly: bool, result: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    exchange.net_only = mode == 'net'
    assert exchange._get_posSide(side, reduceonly) == result

def test_additional_exchange_init_okx(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock: MagicMock = MagicMock()
    api_mock.fetch_accounts = MagicMock(return_value=[{'id': '2555', 'type': '2', 'currency': None, 'info': {'acctLv': '2', 'autoLoan': False, 'ctIsoMode': 'automatic', 'greeksType': 'PA', 'level': 'Lv1', 'levelTmp': '', 'mgnIsoMode': 'automatic', 'posMode': 'long_short_mode', 'uid': '2555'}}])
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx', api_mock=api_mock)
    assert api_mock.fetch_accounts.call_count == 0
    exchange.trading_mode = TradingMode.FUTURES
    assert exchange.net_only
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert not exchange.net_only
    api_mock.fetch_accounts = MagicMock(return_value=[{'id': '2555', 'type': '2', 'currency': None, 'info': {'acctLv': '2', 'autoLoan': False, 'ctIsoMode': 'automatic', 'greeksType': 'PA', 'level': 'Lv1', 'levelTmp': '', 'mgnIsoMode': 'automatic', 'posMode': 'net_mode', 'uid': '2555'}}])
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert exchange.net_only
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'okx', 'additional_exchange_init', 'fetch_accounts')

def test_load_leverage_tiers_okx(default_conf: Dict[str, Any], mocker: Any, markets: Dict[str, Any], tmp_path: Any, caplog: Any, time_machine: Any) -> None:
    default_conf['datadir'] = tmp_path
    api_mock: MagicMock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'fetchLeverageTiers': False, 'fetchMarketLeverageTiers': True})
    api_mock.fetch_market_leverage_tiers = AsyncMock(side_effect=[[{'tier': 1, 'minNotional': 0, 'maxNotional': 500, 'maintenanceMarginRate': 0.02, 'maxLeverage': 75, 'info': {'baseMaxLoan': '', 'imr': '0.013', 'instId': '', 'maxLever': '75', 'maxSz': '500', 'minSz': '0', 'mmr': '0.01', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '1', 'uly': 'ADA-USDT'}}, {'tier': 2, 'minNotional': 501, 'maxNotional': 1000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 50, 'info': {'baseMaxLoan': '', 'imr': '0.02', 'instId': '', 'maxLever': '50', 'maxSz': '1000', 'minSz': '501', 'mmr': '0.015', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '2', 'uly': 'ADA-USDT'}}, {'tier': 3, 'minNotional': 1001, 'maxNotional': 2000, 'maintenanceMarginRate': 0.03, 'maxLeverage': 20, 'info': {'baseMaxLoan': '', 'imr': '0.05', 'instId': '', 'maxLever': '20', 'maxSz': '2000', 'minSz': '1001', 'mmr': '0.02', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '3', 'uly': 'ADA-USDT'}}], TemporaryError('this Failed'), [{'tier': 1, 'minNotional': 0, 'maxNotional': 2000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 75, 'info': {'baseMaxLoan': '', 'imr': '0.013', 'instId': '', 'maxLever': '75', 'maxSz': '2000', 'minSz': '0', 'mmr': '0.01', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '1', 'uly': 'ETH-USDT'}}, {'tier': 2, 'minNotional': 2001, 'maxNotional': 4000, 'maintenanceMarginRate': 0.015, 'maxLeverage': 50, 'info': {'baseMaxLoan': '', 'imr': '0.02', 'instId': '', 'maxLever': '50', 'maxSz': '4000', 'minSz': '2001', 'mmr': '0.015', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '2', 'uly': 'ETH-USDT'}}, {'tier': 3, 'minNotional': 4001, 'maxNotional': 8000, 'maintenanceMarginRate': 0.02, 'maxLeverage': 20, 'info': {'baseMaxLoan': '', 'imr': '0.05', 'instId': '', 'maxLever': '20', 'maxSz': '8000', 'minSz': '4001', 'mmr': '0.02', 'optMgnFactor': '0', 'quoteMaxLoan': '', 'tier': '3', 'uly': 'ETH-USDT'}}]])
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    default_conf['stake_currency'] = 'USDT'
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange.markets = markets
    assert exchange._leverage_tiers == {'ADA/USDT:USDT': [{'minNotional': 0, 'maxNotional': 500, 'maintenanceMarginRate': 0.02, 'maxLeverage': 75, 'maintAmt': None}, {'minNotional': 501, 'maxNotional': 1000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 50, 'maintAmt': None}, {'minNotional': 1001, 'maxNotional': 2000, 'maintenanceMarginRate': 0.03, 'maxLeverage': 20, 'maintAmt': None}], 'ETH/USDT:USDT': [{'minNotional': 0, 'maxNotional': 2000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 75, 'maintAmt': None}, {'minNotional': 2001, 'maxNotional': 4000, 'maintenanceMarginRate': 0.015, 'maxLeverage': 50, 'maintAmt': None}, {'minNotional': 4001, 'maxNotional': 8000, 'maintenanceMarginRate': 0.02, 'maxLeverage': 20, 'maintAmt': None}]}
    filename = default_conf['datadir'] / f'futures/leverage_tiers_{default_conf["stake_currency"]}.json'
    assert filename.is_file()
    logmsg = 'Cached leverage tiers are outdated. Will update.'
    assert not log_has(logmsg, caplog)
    api_mock.fetch_market_leverage_tiers.reset_mock()
    exchange.load_leverage_tiers()
    assert not log_has(logmsg, caplog)
    assert api_mock.fetch_market_leverage_tiers.call_count == 0
    time_machine.move_to(datetime.now() + timedelta(weeks=5))
    exchange.load_leverage_tiers()
    assert log_has(logmsg, caplog)

def test__set_leverage_okx(mocker: Any, default_conf: Dict[str, Any]) -> None:
    api_mock: MagicMock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'setLeverage': True})
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    exchange._lev_prep('BTC/USDT:USDT', 3.2, 'buy')
    assert api_mock.set_leverage.call_count == 1
    assert api_mock.set_leverage.call_args_list[0][1]['leverage'] == 3.2
    assert api_mock.set_leverage.call_args_list[0][1]['symbol'] == 'BTC/USDT:USDT'
    assert api_mock.set_leverage.call_args_list[0][1]['params'] == {'mgnMode': 'isolated', 'posSide': 'net'}
    api_mock.set_leverage = MagicMock(side_effect=ccxt.NetworkError())
    exchange._lev_prep('BTC/USDT:USDT', 3.2, 'buy')
    assert api_mock.fetch_leverage.call_count == 1
    api_mock.fetch_leverage = MagicMock(side_effect=ccxt.NetworkError())
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'okx', '_lev_prep', 'set_leverage', pair='XRP/USDT:USDT', leverage=5.0, side='buy')

@pytest.mark.usefixtures('init_persistence')
def test_fetch_stoploss_order_okx(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf['dry_run'] = False
    mocker.patch('freqtrade.exchange.common.time.sleep')
    api_mock: MagicMock = MagicMock()
    api_mock.fetch_order = MagicMock()
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    exchange.fetch_stoploss_order('1234', 'ETH/BTC')
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_order.call_args_list[0][0][0] == '1234'
    assert api_mock.fetch_order.call_args_list[0][0][1] == 'ETH/BTC'
    assert api_mock.fetch_order.call_args_list[0][1]['params'] == {'stop': True}
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound)
    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(creturn_value=[])
    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order('1234', 'ETH/BTC')
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_closed_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_orders.call_count == API_RETRY_COUNT + 1
    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()
    api_mock.fetch_closed_orders = MagicMock(return_value=[{'id': '1234', 'status': 'closed', 'info': {'ordId': '123455'}}])
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value={'id': '123455'}))
    resp = exchange.fetch_stoploss_order('1234', 'ETH/BTC')
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert api_mock.fetch_canceled_orders.call_count == 0
    assert resp['id'] == '1234'
    assert resp['id_stop'] == '123455'
    assert resp['type'] == 'stoploss'
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    dro_mock = mocker.patch(f'{EXMS}.fetch_dry_run_order', MagicMock(return_value={'id': '123455'}))
    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()
    resp = exchange.fetch_stoploss_order('1234', 'ETH/BTC')
    assert api_mock.fetch_order.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert api_mock.fetch_canceled_orders.call_count == 0
    assert dro_mock.call_count == 1

def test_fetch_stoploss_order_okx_exceptions(default_conf_usdt: Dict[str, Any], mocker: Any) -> None:
    default_conf_usdt['dry_run'] = False
    api_mock: MagicMock = MagicMock()
    ccxt_exceptionhandlers(mocker, default_conf_usdt, api_mock, 'okx', 'fetch_stoploss_order', 'fetch_order', retries=API_RETRY_COUNT + 1, order_id='12345', pair='ETH/USDT')
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound())
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(return_value=[])
    ccxt_exceptionhandlers(mocker, default_conf_usdt, api_mock, 'okx', 'fetch_stoploss_order', 'fetch_open_orders', retries=API_RETRY_COUNT + 1, order_id='12345', pair='ETH/USDT')

@pytest.mark.parametrize('sl1,sl2,sl3,side', [(1501, 1499, 1501, 'sell'), (1499, 1501, 1499, 'buy')])
def test_stoploss_adjust_okx(mocker: Any, default_conf: Dict[str, Any], sl1: float, sl2: float, sl3: float, side: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    order: Dict[str, float] = {'type': 'stoploss', 'price': 1500, 'stopLossPrice': 1500}
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)

def test_stoploss_cancel_okx(mocker: Any, default_conf: Dict[str, Any]) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    exchange.cancel_order = MagicMock()
    exchange.cancel_stoploss_order('1234', 'ETH/USDT')
    assert exchange.cancel_order.call_count == 1
    assert exchange.cancel_order.call_args_list[0][1]['order_id'] == '1234'
    assert exchange.cancel_order.call_args_list[0][1]['pair'] == 'ETH/USDT'
    assert exchange.cancel_order.call_args_list[0][1]['params'] == {'stop': True}

def test__get_stop_params_okx(mocker: Any, default_conf: Dict[str, Any]) -> None:
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    params: Dict[str, str] = exchange._get_stop_params('sell', 'market', 1500)
    assert params['tdMode'] == 'isolated'
    assert params['posSide'] == 'net'

def test_fetch_orders_okx(default_conf: Dict[str, Any], mocker: Any, limit_order: Dict[str, Dict[str, Any]]) -> None:
    api_mock: MagicMock = MagicMock()
    api_mock.fetch_orders = MagicMock(return_value=[limit_order['buy'], limit_order['sell']])
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order['buy']])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order['buy']])
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    start_time: datetime = datetime.now(timezone.utc) - timedelta(days=20)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')
    assert exchange.fetch_orders('mocked', start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='okx')

    def has_resp(_, endpoint: str) -> bool:
        if endpoint == 'fetchOrders':
            return False
        if endpoint == 'fetchClosedOrders':
            return True
        if endpoint == 'fetchOpenOrders':
            return True
        return False
    mocker.patch(f'{EXMS}.exchange_has', has_resp)
    history_params: Dict[str, str] = {'method': 'privateGetTradeOrdersHistoryArchive'}
    exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert 'params' not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]['params'] == history_params
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    exchange.fetch_orders('mocked', datetime.now(timezone.utc) - timedelta(days=6))
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert 'params' not in api_mock.fetch_closed_orders.call_args_list[0][1]
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert 'params' not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]['params'] == history_params
