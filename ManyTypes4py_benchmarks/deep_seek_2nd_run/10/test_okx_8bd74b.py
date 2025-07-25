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

def test_okx_ohlcv_candle_limit(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    timeframes = ('1m', '5m', '1h')
    start_time = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 100
        one_call = int((datetime.now(timezone.utc) - timedelta(minutes=290 * timeframe_to_minutes(timeframe)).timestamp() * 1000)
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        one_call = int((datetime.now(timezone.utc) - timedelta(minutes=320 * timeframe_to_minutes(timeframe)).timestamp() * 1000)
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 100

def test_get_maintenance_ratio_and_amt_okx(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    api_mock = MagicMock()
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

def test_get_max_pair_stake_amount_okx(default_conf: Dict[str, Any], mocker: MagicMock, leverage_tiers: Dict[str, Any]) -> None:
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
def test__get_posSide(default_conf: Dict[str, Any], mocker: MagicMock, mode: str, side: str, reduceonly: bool, result: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='okx')
    exchange.net_only = mode == 'net'
    assert exchange._get_posSide(side, reduceonly) == result

def test_additional_exchange_init_okx(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    api_mock = MagicMock()
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

def test_load_leverage_tiers_okx(default_conf: Dict[str, Any], mocker: MagicMock, markets: Dict[str, Any], tmp_path: Any, caplog: Any, time_machine: Any) -> None:
    default_conf['datadir'] = tmp_path
    api_mock = MagicMock()
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
    filename = default_conf['datadir'] / f'futures/leverage_tiers_{default_conf['stake_currency']}.json'
    assert filename.is_file()
    logmsg = 'Cached leverage tiers are outdated. Will update.'
    assert not log_has(logmsg, caplog)
    api_mock.fetch_market_leverage_tiers.reset_mock()
    exchange.load_leverage_tiers()
    assert not log_has(logmsg, caplog)
    assert api_mock.fetch