import ccxt
import copy
import logging
from datetime import datetime, timedelta, timezone
from random import randint
from unittest.mock import MagicMock, Mock, PropertyMock, patch
import ccxt
import pytest
from numpy import nan
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError, DDosProtection, DependencyException, ExchangeError, InsufficientFundsError, InvalidOrderException, OperationalException, PricingError, TemporaryError
from freqtrade.exchange import Binance, Bybit, Exchange, Kraken, market_is_active, timeframe_to_prev_date
from freqtrade.exchange.common import API_FETCH_ORDER_RETRY_COUNT, API_RETRY_COUNT, calculate_backoff, remove_exchange_credentials
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from tests.conftest import EXMS, generate_test_data_raw, get_mock_coro, get_patched_exchange, log_has, log_has_re, num_log_has_re
EXCHANGES = ['binance', 'kraken', 'gate', 'kucoin', 'bybit', 'okx']

def ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, fun, mock_ccxt_fun, retries=API_RETRY_COUNT + 1, **kwargs):
    with patch('freqtrade.exchange.common.time.sleep'):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection('DDos'))
            exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
            getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.OperationFailed('DeaDBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1

async def async_ccxt_exception(mocker, default_conf, api_mock, fun, mock_ccxt_fun, retries=API_RETRY_COUNT + 1, **kwargs):
    with patch('freqtrade.exchange.common.asyncio.sleep', get_mock_coro(None)):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection('Dooh'))
            exchange = get_patched_exchange(mocker, default_conf, api_mock)
            await getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()
    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()
    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1
    exchange.close()

def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog)

def test_remove_exchange_credentials(default_conf):
    conf = copy.deepcopy(default_conf)
    remove_exchange_credentials(conf['exchange'], False)
    assert conf['exchange']['key'] != ''
    assert conf['exchange']['secret'] != ''
    remove_exchange_credentials(conf['exchange'], True)
    assert conf['exchange']['key'] == ''
    assert conf['exchange']['secret'] == ''
    assert conf['exchange']['password'] == ''
    assert conf['exchange']['uid'] == ''

def test_init_ccxt_kwargs(default_conf, mocker, caplog):
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    aei_mock = mocker.patch(f'{EXMS}.additional_exchange_init')
    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_async_config'] = {'aiohttp_trust_env': True, 'asyncio_loop': True}
    ex = Exchange(conf)
    assert log_has("Applying additional ccxt config: {'aiohttp_trust_env': True, 'asyncio_loop': True}", caplog)
    assert ex._api_async.aiohttp_trust_env
    assert not ex._api.aiohttp_trust_env
    assert aei_mock.call_count == 1
    caplog.clear()
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_config'] = {'TestKWARG': 11}
    conf['exchange']['ccxt_sync_config'] = {'TestKWARG44': 11}
    conf['exchange']['ccxt_async_config'] = {'asyncio_loop': True}
    asynclogmsg = "Applying additional ccxt config: {'TestKWARG': 11, 'asyncio_loop': True}"
    ex = Exchange(conf)
    assert not ex._api_async.aiohttp_trust_env
    assert hasattr(ex._api, 'TestKWARG')
    assert ex._api.TestKWARG == 11
    assert not hasattr(ex._api_async, 'TestKWARG44')
    assert hasattr(ex._api_async, 'TestKWARG')
    assert log_has("Applying additional ccxt config: {'TestKWARG': 11, 'TestKWARG44': 11}", caplog)
    assert log_has(asynclogmsg, caplog)
    Exchange._ccxt_params = {'hello': 'world'}
    ex = Exchange(conf)
    assert log_has("Applying additional ccxt config: {'TestKWARG': 11, 'TestKWARG44': 11}", caplog)
    assert ex._api.hello == 'world'
    assert ex._ccxt_config == {}
    Exchange._headers = {}

def test_destroy(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog)

def test_init_exception(default_conf, mocker):
    default_conf['exchange']['name'] = 'wrong_exchange_name'
    with pytest.raises(OperationalException, match=f'Exchange {default_conf["exchange"]["name"]} is not supported'):
        Exchange(default_conf)
    default_conf['exchange']['name'] = 'binance'
    with pytest.raises(OperationalException, match=f'Exchange {default_conf["exchange"]["name"]} is not supported'):
        mocker.patch('ccxt.binance', MagicMock(side_effect=AttributeError))
        Exchange(default_conf)
    with pytest.raises(OperationalException, match='Initialization of ccxt failed. Reason: DeadBeef'):
        mocker.patch('ccxt.binance', MagicMock(side_effect=ccxt.BaseError('DeadBeef')))
        Exchange(default_conf)

def test_exchange_resolver(default_conf, mocker, caplog):
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=MagicMock()))
    mocker.patch(f'{EXMS}._load_async_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    default_conf['exchange']['name'] = 'zaif'
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert log_has_re('No .* specific subclass found. Using the generic class instead.', caplog)
    caplog.clear()
    default_conf['exchange']['name'] = 'Bybit'
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Bybit)
    assert not log_has_re('No .* specific subclass found. Using the generic class instead.', caplog)
    caplog.clear()
    default_conf['exchange']['name'] = 'kraken'
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Kraken)
    assert not isinstance(exchange, Binance)
    assert not log_has_re('No .* specific subclass found. Using the generic class instead.', caplog)
    default_conf['exchange']['name'] = 'binance'
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)
    assert not log_has_re('No .* specific subclass found. Using the generic class instead.', caplog)
    default_conf['exchange']['name'] = 'binanceus'
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)

def test_validate_order_time_in_force(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf, exchange='bybit')
    tif = {'buy': 'gtc', 'sell': 'gtc'}
    ex.validate_order_time_in_force(tif)
    tif2 = {'buy': 'fok', 'sell': 'ioc22'}
    with pytest.raises(OperationalException, match='Time in force.*not supported for .*'):
        ex.validate_order_time_in_force(tif2)
    tif2 = {'buy': 'fok', 'sell': 'ioc'}
    ex._ft_has.update({'order_time_in_force': ['GTC', 'FOK', 'IOC']})
    ex.validate_order_time_in_force(tif2)

def test_validate_orderflow(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf, exchange='bybit')
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    ex.validate_orderflow({'use_public_trades': False})
    with pytest.raises(ConfigurationError, match='Trade data not available for.*'):
        ex.validate_orderflow({'use_public_trades': True})
    ex = get_patched_exchange(mocker, default_conf, exchange='binance')
    ex.validate_orderflow({'use_public_trades': False})
    ex.validate_orderflow({'use_public_trades': True})

def test_validate_freqai_compat(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf, exchange='kraken')
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    default_conf['freqai'] = {'enabled': False}
    ex.validate_freqai(default_conf)
    default_conf['freqai'] = {'enabled': True}
    with pytest.raises(ConfigurationError, match='Historic OHLCV data not available for.*'):
        ex.validate_freqai(default_conf)
    ex = get_patched_exchange(mocker, default_conf, exchange='binance')
    default_conf['freqai'] = {'enabled': True}
    ex.validate_freqai(default_conf)
    default_conf['freqai'] = {'enabled': False}
    ex.validate_freqai(default_conf)

@pytest.mark.parametrize('price,precision_mode,precision,expected', [(2.34559, 2, 4, 0.0001), (2.34559, 2, 5, 1e-05), (2.34559, 2, 3, 0.001), (2.9999, 2, 3, 0.001), (200.0511, 2, 3, 0.001), (2.34559, 4, 0.0001, 0.0001), (2.34559, 4, 1e-05, 1e-05), (2.34559, 4, 0.0025, 0.0025), (2.9909, 4, 0.0025, 0.0025), (234.43, 4, 0.5, 0.5), (234.43, 4, 0.0025, 0.0025), (234.43, 4, 0.00013, 0.00013)])
def test_price_get_one_pip(default_conf, mocker, price, precision_mode, precision, expected):
    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'price': precision}}})
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    mocker.patch(f'{EXMS}.markets', markets)
    mocker.patch(f'{EXMS}.precisionMode', PropertyMock(return_value=precision_mode))
    mocker.patch(f'{EXMS}.precision_mode_price', PropertyMock(return_value=precision_mode))
    pair = 'ETH/BTC'
    assert pytest.approx(exchange.price_get_one_pip(pair, price)) == expected

def test__get_stake_amount_limit(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    with pytest.raises(ValueError, match='.*get market information.*'):
        exchange.get_min_pair_stake_amount('BNB/BTC', 1, stoploss)
    markets['ETH/BTC']['limits'] = {'cost': {'min': None, 'max': None}, 'amount': {'min': None, 'max': None}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 1)
    assert result == float('inf')
    markets['ETH/BTC']['limits'] = {'cost': {'min': 2, 'max': 4000}, 'amount': {'min': None, 'max': None}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    expected_result = 2 * (1 + 0.05) / (1 - abs(stoploss))
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss, 3.0)
    assert pytest.approx(result) == expected_result / 3
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 4000
    markets['ETH/BTC']['limits'] = {'cost': {'min': None, 'max': None}, 'amount': {'min': 2, 'max': 10000}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    expected_result = 2 * 2 * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss, 5.0)
    assert pytest.approx(result) == expected_result / 5
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 20000
    markets['ETH/BTC']['limits'] = {'cost': {'min': 2, 'max': None}, 'amount': {'min': 2, 'max': None}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    expected_result = max(2, 2 * 2) * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss, 10)
    assert pytest.approx(result) == expected_result / 10
    markets['ETH/BTC']['limits'] = {'cost': {'min': 8, 'max': 10000}, 'amount': {'min': 2, 'max': 500}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    expected_result = max(8, 2 * 2) * (1 + 0.05) / (1 - abs(stoploss))
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss, 7.0)
    assert pytest.approx(result) == expected_result / 7.0
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 1000
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -0.4)
    expected_result = max(8, 2 * 2) * 1.5
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -0.4, 8.0)
    assert pytest.approx(result) == expected_result / 8.0
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 1000
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -1)
    expected_result = max(8, 2 * 2) * 1.5
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -1, 12.0)
    assert pytest.approx(result) == expected_result / 12
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 1000
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2, 12.0)
    assert result == 1000 / 12
    markets['ETH/BTC']['contractSize'] = '0.01'
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -1)
    assert pytest.approx(result) == expected_result * 0.01
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 10
    markets['ETH/BTC']['contractSize'] = '10'
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, -1, 12.0)
    assert pytest.approx(result) == expected_result / 12 * 10.0
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 10000

def test_get_min_pair_stake_amount_real_data(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}
    markets['ETH/BTC']['limits'] = {'cost': {'min': 0.0001, 'max': 4000}, 'amount': {'min': 0.001, 'max': 10000}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 0.020405, stoploss)
    expected_result = max(0.0001, 0.001 * 0.020405) * (1 + 0.05) / (1 - abs(stoploss))
    assert round(result, 8) == round(expected_result, 8)
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2.0)
    assert result == 4000
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 0.020405, stoploss, 3.0)
    assert round(result, 8) == round(expected_result / 3, 8)
    markets['ETH/BTC']['contractSize'] = 0.1
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 0.020405, stoploss, 3.0)
    assert round(result, 8) == round(expected_result / 3, 8)
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 12.0)
    assert result == 4000

def test__load_async_markets(default_conf, mocker, caplog):
    mocker.patch(f'{EXMS}._init_ccxt')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    exchange = Exchange(default_conf)
    exchange._api_async.load_markets = get_mock_coro(None)
    exchange._load_async_markets()
    assert exchange._api_async.load_markets.call_count == 1
    caplog.set_level(logging.DEBUG)
    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.BaseError('deadbeef'))
    with pytest.raises(TemporaryError, match='deadbeef'):
        exchange._load_async_markets()
    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.DDoSProtection('deadbeef'))
    with pytest.raises(DDosProtection, match='deadbeef'):
        exchange._load_async_markets()
    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.OperationFailed('deadbeef'))
    with pytest.raises(TemporaryError, match='deadbeef'):
        exchange._load_async_markets()

def test__load_markets(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.BaseError('SomeError'))
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    Exchange(default_conf)
    assert log_has('Could not load markets.', caplog)
    expected_return = {'ETH/BTC': 'available'}
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(return_value=expected_return)
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    default_conf['exchange']['pair_whitelist'] = ['ETH/BTC']
    ex = Exchange(default_conf)
    assert ex.markets == expected_return

def test_reload_markets(default_conf, mocker, caplog, time_machine):
    caplog.set_level(logging.DEBUG)
    initial_markets = {'ETH/BTC': {}}
    updated_markets = {'ETH/BTC': {}, 'LTC/BTC': {}}
    start_dt = dt_now()
    time_machine.move_to(start_dt, tick=False)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(return_value=initial_markets)
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='binance', mock_markets=False)
    lam_spy = mocker.spy(exchange, '_load_async_markets')
    assert exchange._last_markets_refresh == dt_ts()
    assert exchange.markets == initial_markets
    time_machine.move_to(start_dt + timedelta(minutes=8), tick=False)
    exchange.reload_markets()
    assert exchange.markets == initial_markets
    assert lam_spy.call_count == 0
    api_mock.load_markets = get_mock_coro(return_value=updated_markets)
    time_machine.move_to(start_dt + timedelta(minutes=11), tick=False)
    exchange.reload_markets()
    assert exchange.markets == updated_markets
    assert lam_spy.call_count == 1
    assert log_has('Performing scheduled market reload..', caplog)
    lam_spy.reset_mock()
    exchange.reload_markets()
    assert lam_spy.call_count == 0
    time_machine.move_to(start_dt + timedelta(minutes=51), tick=False)
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError('LoadError'))
    exchange.reload_markets(force=False)
    assert exchange.markets == updated_markets
    assert lam_spy.call_count == 1
    lam_spy.reset_mock()
    exchange.reload_markets(force=True)
    assert lam_spy.call_count == 4
    assert exchange.markets == updated_markets

def test_reload_markets_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError('LoadError'))
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange='binance', mock_markets=False)
    exchange._last_markets_refresh = 2
    exchange.reload_markets()
    assert exchange._last_markets_refresh == 2
    assert log_has_re('Could not load markets\\..*', caplog)

@pytest.mark.parametrize('stake_currency', ['ETH', 'BTC', 'USDT'])
def test_validate_stakecurrency(default_conf, stake_currency, mocker, caplog):
    default_conf['stake_currency'] = stake_currency
    api_mock = MagicMock()
    type(api_mock).load_markets = get_mock_coro(return_value={'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'}, 'XRP/ETH': {'quote': 'ETH'}, 'NEO/USDT': {'quote': 'USDT'}})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_pricing')
    Exchange(default_conf)

def test_validate_stakecurrency_error(default_conf, mocker, caplog):
    default_conf['stake_currency'] = 'XRP'
    api_mock = MagicMock()
    type(api_mock).load_markets = get_mock_coro(return_value={'ETH/BTC': {'quote': 'BTC'}, 'LTC/BTC': {'quote': 'BTC'}, 'XRP/ETH': {'quote': 'ETH'}, 'NEO/USDT': {'quote': 'USDT'}})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.validate_timeframes')
    with pytest.raises(ConfigurationError, match='XRP is not available as stake on .*Available currencies are: BTC, ETH, USDT'):
        Exchange(default_conf)
    type(api_mock).load_markets = get_mock_coro(side_effect=ccxt.NetworkError('No connection.'))
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    with pytest.raises(OperationalException, match='Could not load markets, therefore cannot start\\. Please.*'):
        Exchange(default_conf)

def test_get_quote_currencies(default_conf, mocker):
    ex = get_patched_exchange(mocker, default_conf)
    assert set(ex.get_quote_currencies()) == set(['USD', 'ETH', 'BTC', 'USDT', 'BUSD'])

@pytest.mark.parametrize('pair,expected', [('XRP/BTC', 'BTC'), ('LTC/USD', 'USD'), ('ETH/USDT', 'USDT'), ('XLTCUSDT', 'USDT'), ('XRP/NOCURRENCY', '')])
def test_get_pair_quote_currency(default_conf, mocker, pair, expected):
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_quote_currency(pair) == expected

@pytest.mark.parametrize('pair,expected', [('XRP/BTC', 'XRP'), ('LTC/USD', 'LTC'), ('ETH/USDT', 'ETH'), ('XLTCUSDT', 'LTC'), ('XRP/NOCURRENCY', '')])
def test_get_pair_base_currency(default_conf, mocker, pair, expected):
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_base_currency(pair) == expected

@pytest.mark.parametrize('timeframe', ['5m', '1m', '15m', '1h'])
def test_validate_timeframes(default_conf, mocker, timeframe):
    default_conf['timeframe'] = timeframe
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'})
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    Exchange(default_conf)

def test_validate_timeframes_failed(default_conf, mocker):
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'15s': '15s', '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'})
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    with pytest.raises(ConfigurationError, match="Invalid timeframe '3m'. This exchange supports.*"):
        Exchange(default_conf)
    default_conf['timeframe'] = '15s'
    with pytest.raises(ConfigurationError, match='Timeframes < 1m are currently not supported by Freqtrade.'):
        Exchange(default_conf)
    default_conf['runmode'] = RunMode.UTIL_EXCHANGE
    Exchange(default_conf)

def test_validate_timeframes_emulated_ohlcv_1(default_conf, mocker):
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    del api_mock.timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    with pytest.raises(OperationalException, match='The ccxt library does not provide the list of timeframes for the exchange .* and this exchange is therefore not supported. *'):
        Exchange(default_conf)

def test_validate_timeframes_emulated_ohlcvi_2(default_conf, mocker):
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    del api_mock.timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    with pytest.raises(OperationalException, match='The ccxt library does not provide the list of timeframes for the exchange .* and this exchange is therefore not supported. *'):
        Exchange(default_conf)

def test_validate_timeframes_not_in_config(default_conf, mocker):
    del default_conf['timeframe']
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'})
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    mocker.patch(f'{EXMS}.validate_required_startup_candles')
    Exchange(default_conf)

def test_validate_pricing(default_conf, mocker):
    api_mock = MagicMock()
    has = {'fetchL2OrderBook': True, 'fetchTicker': True}
    type(api_mock).has = PropertyMock(return_value=has)
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_trading_mode_and_margin_mode')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.name', 'Binance')
    default_conf['exchange']['name'] = 'binance'
    ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchTicker': False})
    with pytest.raises(OperationalException, match='Ticker pricing not available for .*'):
        ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchTicker': True})
    default_conf['exit_pricing']['use_order_book'] = True
    ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchL2OrderBook': False})
    with pytest.raises(OperationalException, match='Orderbook not available for .*'):
        ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchL2OrderBook': True})
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    with pytest.raises(OperationalException, match='Ticker pricing not available for .*'):
        ExchangeResolver.load_exchange(default_conf)

def test_validate_ordertypes(default_conf, mocker):
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': True})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    default_conf['order_types'] = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    Exchange(default_conf)
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': False})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    default_conf['order_types'] = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    with pytest.raises(OperationalException, match='Exchange .* does not support market orders.'):
        Exchange(default_conf)
    default_conf['order_types'] = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': True}
    with pytest.raises(OperationalException, match='On exchange stoploss is not supported for .*'):
        Exchange(default_conf)

@pytest.mark.parametrize('exchange_name,stopadv, expected', [('binance', 'last', True), ('binance', 'mark', True), ('binance', 'index', False), ('bybit', 'last', True), ('bybit', 'mark', True), ('bybit', 'index', True), ('okx', 'last', True), ('okx', 'mark', True), ('okx', 'index', True), ('gate', 'last', True), ('gate', 'mark', True), ('gate', 'index', True)])
def test_validate_ordertypes_stop_advanced(default_conf, mocker, exchange_name, stopadv, expected):
    api_mock = MagicMock()
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': True})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    default_conf['order_types'] = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': True, 'stoploss_price_type': stopadv}
    default_conf['exchange']['name'] = exchange_name
    if expected:
        ExchangeResolver.load_exchange(default_conf)
    else:
        with pytest.raises(OperationalException, match='On exchange stoploss price type is not supported for .*'):
            ExchangeResolver.load_exchange(default_conf)

def test_validate_order_types_not_in_config(default_conf, mocker):
    api_mock = MagicMock()
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_pricing')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    conf = copy.deepcopy(default_conf)
    Exchange(conf)

def test_validate_required_startup_candles(default_conf, mocker, caplog):
    api_mock = MagicMock()
    mocker.patch(f'{EXMS}.name', PropertyMock(return_value='Binance'))
    mocker.patch(f'{EXMS}._init_ccxt', api_mock)
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}._load_async_markets')
    mocker.patch(f'{EXMS}.validate_pricing')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    default_conf['startup_candle_count'] = 20
    ex = Exchange(default_conf)
    assert ex
    assert ex.validate_required_startup_candles(200, '5m') == 1
    assert ex.validate_required_startup_candles(499, '5m') == 1
    assert ex.validate_required_startup_candles(600, '5m') == 2
    assert ex.validate_required_startup_candles(501, '5m') == 2
    assert ex.validate_required_startup_candles(499, '5m') == 1
    assert ex.validate_required_startup_candles(1000, '5m') == 3
    assert ex.validate_required_startup_candles(2499, '5m') == 5
    assert log_has_re('Using 5 calls to get OHLCV. This.*', caplog)
    with pytest.raises(OperationalException, match='This strategy requires 2500.*'):
        ex.validate_required_startup_candles(2500, '5m')
    default_conf['startup_candle_count'] = 6000
    with pytest.raises(OperationalException, match='This strategy requires 6000.*'):
        Exchange(default_conf)
    ex._ft_has['ohlcv_has_history'] = False
    with pytest.raises(OperationalException, match='This strategy requires 2500.*, which is more than the amount.*'):
        ex.validate_required_startup_candles(2500, '5m')

def test_exchange_has(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf)
    assert not exchange.exchange_has('ASDFASDF')
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'deadbeef': True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.exchange_has('deadbeef')
    type(api_mock).has = PropertyMock(return_value={'deadbeef': False})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert not exchange.exchange_has('deadbeef')
    exchange._ft_has['exchange_has_overrides'] = {'deadbeef': True}
    assert exchange.exchange_has('deadbeef')

@pytest.mark.parametrize('side,leverage', [('buy', 1), ('buy', 5), ('sell', 1.0), ('sell', 5.0)])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order(default_conf, mocker, side, exchange_name, leverage):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_dry_run_order(pair='ETH/BTC', ordertype='limit', side=side, amount=1, rate=200, leverage=leverage)
    assert 'id' in order
    assert f'dry_run_{side}_' in order['id']
    assert order['side'] == side
    assert order['type'] == 'limit'
    assert order['symbol'] == 'ETH/BTC'
    assert order['amount'] == 1
    assert order['cost'] == 1 * 200

@pytest.mark.parametrize('side,is_short,order_reason', [('buy', False, 'entry'), ('sell', False, 'exit'), ('buy', True, 'exit'), ('sell', True, 'entry')])
@pytest.mark.parametrize('order_type,price_side,fee', [('limit', 'same', 1.0), ('limit', 'other', 2.0), ('market', 'same', 2.0), ('market', 'other', 2.0)])
def test_create_dry_run_order_fees(default_conf, mocker, side, order_type, is_short, order_reason, price_side, fee):
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(f'{EXMS}.get_fee', side_effect=lambda symbol, taker_or_maker: 2.0 if taker_or_maker == 'taker' else 1.0)
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=price_side == 'other')
    order = exchange.create_dry_run_order(pair='LTC/USDT', ordertype=order_type, side=side, amount=10, rate=2.0, leverage=1.0)
    if price_side == 'other' or order_type == 'market':
        assert order['fee']['rate'] == fee
        return
    else:
        assert order['fee'] is None
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=price_side != 'other')
    order1 = exchange.fetch_dry_run_order(order['id'])
    assert order1['fee']['rate'] == fee

@pytest.mark.parametrize('side,price,filled,converted', [('buy', 25.563, False, False), ('buy', 25.566, True, False), ('sell', 25.566, False, False), ('sell', 25.563, True, False), ('buy', 29.563, True, True), ('sell', 21.563, True, True)])
@pytest.mark.parametrize('leverage', [1, 2, 5])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order_limit_fill(default_conf, mocker, side, price, filled, caplog, exchange_name, order_book_l2_usd, converted, leverage):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch.multiple(EXMS, exchange_has=MagicMock(return_value=True), fetch_l2_order_book=order_book_l2_usd)
    order = exchange.create_order(pair='LTC/USDT', ordertype='limit', side=side, amount=1, rate=price, leverage=leverage)
    assert order_book_l2_usd.call_count == 1
    assert 'id' in order
    assert f'dry_run_{side}_' in order['id']
    assert order['side'] == side
    if not converted:
        assert order['average'] == price
        assert order['type'] == 'limit'
    else:
        assert order['type'] == 'market'
        assert 25.5 < order['average'] < 25.6
        assert log_has_re('Converted .* to market order.*', caplog)
    assert order['symbol'] == 'LTC/USDT'
    assert order['status'] == 'open' if not filled else 'closed'
    order_book_l2_usd.reset_mock()
    order_closed = exchange.fetch_dry_run_order(order['id'])
    assert order_book_l2_usd.call_count == (1 if not filled else 0)
    assert order_closed['status'] == ('open' if not filled else 'closed')
    assert order_closed['filled'] == (0 if not filled else 1)
    assert order_closed['cost'] == 1 * order_closed['average']
    order_book_l2_usd.reset_mock()
    mocker.patch(f'{EXMS}.fetch_l2_order_book', return_value={'asks': [], 'bids': []})
    exchange._dry_run_open_orders[order['id']]['status'] = 'open'
    order_closed = exchange.fetch_dry_run_order(order['id'])

@pytest.mark.parametrize('side,rate,amount,endprice', [('buy', 25.564, 1, 25.566), ('buy', 25.564, 100, 25.5672), ('buy', 25.59, 100, 25.5672), ('buy', 25.564, 1000, 25.575), ('buy', 24.0, 100000, 25.2), ('sell', 25.564, 1, 25.563), ('sell', 25.564, 100, 25.5625), ('sell', 25.51, 100, 25.5625), ('sell', 25.564, 1000, 25.5555), ('sell', 27, 10000, 25.65)])
@pytest.mark.parametrize('leverage', [1, 2, 5])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order_market_fill(default_conf, mocker, side, rate, amount, endprice, exchange_name, order_book_l2_usd, leverage):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch.multiple(EXMS, exchange_has=MagicMock(return_value=True), fetch_l2_order_book=order_book_l2_usd)
    order = exchange.create_order(pair='LTC/USDT', ordertype='market', side=side, amount=amount, rate=rate, leverage=leverage)
    assert 'id' in order
    assert f'dry_run_{side}_' in order['id']
    assert order['side'] == side
    assert order['type'] == 'market'
    assert order['symbol'] == 'LTC/USDT'
    assert order['status'] == 'closed'
    assert order['filled'] == amount
    assert order['amount'] == amount
    assert pytest.approx(order['cost']) == amount * order['average']
    assert round(order['average'], 4) == round(endprice, 4)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_order(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f'test_prod_{randint(0, 10 ** 6)}'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'symbol': 'XLTCUSDT', 'info': {'foo': 'bar'}, 'amount': 1})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()
    price_req = exchange._ft_has.get('marketOrderRequiresPrice', False)
    order = exchange.create_order(pair='XLTCUSDT', ordertype='market', side='buy', amount=1, rate=200, leverage=1.0)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert order['amount'] == 1
    assert api_mock.create_order.call_args[0][0] == 'XLTCUSDT'
    assert api_mock.create_order.call_args[0][1] == 'market'
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price('buy', order_type='market'):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None
    api_mock.create_order.reset_mock()
    order_type = 'limit'
    order = exchange.create_order(pair='XLTCUSDT', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0)
    assert api_mock.create_order.call_args[0][0] == 'XLTCUSDT'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds('Not enough funds'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='XLTCUSDT', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0)
    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='XLTCUSDT', ordertype='limit', side='buy', amount=1, rate=200, leverage=1.0)
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='XLTCUSDT', ordertype='market', side='buy', amount=1, rate=200, leverage=1.0)
    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError('No Connection'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='XLTCUSDT', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0)
    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='XLTCUSDT', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_buy_dry_run(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_order(pair='ETH/BTC', ordertype='limit', side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    assert 'id' in order
    assert 'dry_run_buy_' in order['id']

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_buy_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'
    order_type = 'market'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'symbol': 'ETH/BTC', 'info': {'foo': 'bar'}})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price('buy', order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None
    api_mock.create_order.reset_mock()
    order_type = 'limit'
    order = exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    with pytest.raises(InsufficientFundsError):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds('0 balance'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='ETH/BTC', ordertype='limit', side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='ETH/BTC', ordertype='market', side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError('No Connection'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')
    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force='gtc')

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_buy_considers_time_in_force(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'symbol': 'ETH/BTC', 'info': {'foo': 'bar'}})
    api_mock.create_order.reset_mock()
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order_type = 'limit'
    time_in_force = 'ioc'
    order = exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force=time_in_force)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert 'timeInForce' in api_mock.create_order.call_args[0][5]
    assert api_mock.create_order.call_args[0][5]['timeInForce'] == time_in_force.upper()
    order_type = 'market'
    time_in_force = 'IOC'
    order = exchange.create_order(pair='ETH/BTC', ordertype=order_type, side='buy', amount=1, rate=200, leverage=1.0, time_in_force=time_in_force)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'ETH/BTC'
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == 'buy'
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price('buy', order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None
    assert 'timeInForce' not in api_mock.create_order.call_args[0][5]

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_balances_prod(default_conf, mocker, exchange_name):
    balance_item = {'free': 10.0, 'total': 10.0, 'used': 0.0}
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={'1ST': balance_item, '2ND': balance_item, '3RD': balance_item})
    api_mock.commonCurrencies = {}
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert len(exchange.get_balances()) == 3
    assert exchange.get_balances()['1ST']['free'] == 10.0
    assert exchange.get_balances()['1ST']['total'] == 10.0
    assert exchange.get_balances()['1ST']['used'] == 0.0
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'get_balances', 'fetch_balance')

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_positions(default_conf, mocker, exchange_name):
    mocker.patch(f'{EXMS}.validate_trading_mode_and_margin_mode')
    api_mock = MagicMock()
    api_mock.fetch_positions = MagicMock(return_value=[{'symbol': 'ETH/USDT:USDT', 'leverage': 5}, {'symbol': 'XRP/USDT:USDT', 'leverage': 5}])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_positions() == []
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = 'futures'
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.fetch_positions()
    assert len(res) == 2
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_positions', 'fetch_positions')

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_orders(default_conf, mocker, exchange_name, limit_order):
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(return_value=[limit_order['buy'], limit_order['sell']])
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order['buy']])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order['buy']])
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    start_time = datetime.now(timezone.utc) - timedelta(days=20)
    expected = 1
    if exchange_name == 'bybit':
        expected = 3
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_orders('mocked', start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert len(res) == 2 * expected
    res = exchange.fetch_orders('mocked', start_time)
    api_mock.fetch_orders.reset_mock()

    def has_resp(_, endpoint):
        if endpoint == 'fetchOrders':
            return False
        if endpoint == 'fetchClosedOrders':
            return True
        if endpoint == 'fetchOpenOrders':
            return True
    if exchange_name == 'okx':
        return
    mocker.patch(f'{EXMS}.exchange_has', has_resp)
    exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == expected
    assert api_mock.fetch_closed_orders.call_count == expected
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_orders', 'fetch_orders', retries=1, pair='mocked', since=start_time)
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == expected
    assert api_mock.fetch_closed_orders.call_count == expected

def test_fetch_trading_fees(default_conf, mocker):
    api_mock = MagicMock()
    tick = {'1INCH/USDT:USDT': {'info': {'user_id': '', 'taker_fee': '0.0018', 'maker_fee': '0.0018', 'gt_discount': False, 'gt_taker_fee': '0', 'gt_maker_fee': '0', 'loan_fee': '0.18', 'point_type': '1', 'futures_taker_fee': '0.0005', 'futures_maker_fee': '0'}, 'symbol': '1INCH/USDT:USDT', 'maker': 0.0, 'taker': 0.0, 'futures_taker': 0.0, 'futures_maker': 0.0}, 'ETH/USDT:USDT': {'info': {'user_id': '', 'taker_fee': '0.0018', 'maker_fee': '0.0018', 'gt_discount': False, 'gt_taker_fee': '0', 'gt_maker_fee': '0', 'loan_fee': '0.18', 'point_type': '1', 'futures_taker_fee': '0.0005', 'futures_maker_fee': '0'}, 'symbol': 'ETH/USDT:USDT', 'maker': 0.0, 'taker': 0.0, 'futures_taker': 0.0, 'futures_maker': 0.0}}
    exchange_name = 'gate'
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    api_mock.fetch_trading_fees = MagicMock(return_value=tick)
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert '1INCH/USDT:USDT' in exchange._trading_fees
    assert 'ETH/USDT:USDT' in exchange._trading_fees
    assert api_mock.fetch_trading_fees.call_count == 1
    api_mock.fetch_trading_fees.reset_mock()
    mocker.patch(f'{EXMS}.reload_markets')
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_trading_fees', 'fetch_trading_fees')
    api_mock.fetch_trading_fees = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_trading_fees()
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    assert exchange.fetch_trading_fees() == {}

def test_fetch_bids_asks(default_conf, mocker):
    api_mock = MagicMock()
    tick = {'ETH/BTC': {'symbol': 'ETH/BTC', 'bid': 0.5, 'ask': 1, 'last': 42}, 'BCH/BTC': {'symbol': 'BCH/BTC', 'bid': 0.6, 'ask': 0.5, 'last': 41}}
    exchange_name = 'binance'
    api_mock.fetch_bids_asks = MagicMock(return_value=tick)
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    bidsasks = exchange.fetch_bids_asks()
    assert 'ETH/BTC' in bidsasks
    assert 'BCH/BTC' in bidsasks
    assert bidsasks['ETH/BTC']['bid'] == 0.5
    assert bidsasks['ETH/BTC']['ask'] == 1
    assert bidsasks['BCH/BTC']['bid'] == 0.6
    assert bidsasks['BCH/BTC']['ask'] == 0.5
    assert api_mock.fetch_bids_asks.call_count == 1
    api_mock.fetch_bids_asks.reset_mock()
    tickers2 = exchange.fetch_bids_asks(cached=True)
    assert tickers2 == bidsasks
    assert api_mock.fetch_bids_asks.call_count == 0
    tickers2 = exchange.fetch_bids_asks(cached=False)
    assert api_mock.fetch_bids_asks.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_bids_asks', 'fetch_bids_asks')
    with pytest.raises(OperationalException):
        api_mock.fetch_bids_asks = MagicMock(side_effect=ccxt.NotSupported('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_bids_asks()
    api_mock.fetch_bids_asks = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_bids_asks()
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    assert exchange.fetch_bids_asks() == {}

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_tickers(default_conf, mocker, exchange_name, caplog):
    api_mock = MagicMock()
    tick = {'ETH/BTC': {'symbol': 'ETH/BTC', 'bid': 0.5, 'ask': 1, 'last': 42}, 'BCH/BTC': {'symbol': 'BCH/BTC', 'bid': 0.6, 'ask': 0.5, 'last': 41}}
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    api_mock.fetch_tickers = MagicMock(return_value=tick)
    api_mock.fetch_bids_asks = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    tickers = exchange.get_tickers()
    assert 'ETH/BTC' in tickers
    assert 'BCH/BTC' in tickers
    assert tickers['ETH/BTC']['bid'] == 0.5
    assert tickers['ETH/BTC']['ask'] == 1
    assert tickers['BCH/BTC']['bid'] == 0.6
    assert tickers['BCH/BTC']['ask'] == 0.5
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == 0
    api_mock.fetch_tickers.reset_mock()
    tickers2 = exchange.get_tickers(cached=True)
    assert tickers2 == tickers
    assert api_mock.fetch_tickers.call_count == 0
    assert api_mock.fetch_bids_asks.call_count == 0
    tickers2 = exchange.get_tickers(cached=False)
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == 0
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'get_tickers', 'fetch_tickers')
    with pytest.raises(OperationalException):
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.NotSupported('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.get_tickers()
    caplog.clear()
    api_mock.fetch_tickers = MagicMock(side_effect=[ccxt.BadSymbol('SomeSymbol'), []])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    x = exchange.get_tickers()
    assert x == []
    assert log_has_re('Could not load tickers due to BadSymbol\\..*SomeSymbol', caplog)
    caplog.clear()
    api_mock.fetch_tickers = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.get_tickers()
    api_mock.fetch_tickers.reset_mock()
    api_mock.fetch_bids_asks.reset_mock()
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.get_tickers()
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == (1 if exchange_name == 'binance' else 0)
    api_mock.fetch_tickers.reset_mock()
    api_mock.fetch_bids_asks.reset_mock()
    mocker.patch(f'{EXMS}.exchange_has', return_value=False)
    assert exchange.get_tickers() == {}

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_conversion_rate(default_conf_usdt, mocker, exchange_name):
    api_mock = MagicMock()
    tick = {'ETH/USDT': {'last': 42}, 'BCH/USDT': {'last': 41}, 'ETH/BTC': {'last': 250}}
    tick2 = {'ADA/USDT:USDT': {'last': 2.5}}
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    api_mock.fetch_tickers = MagicMock(side_effect=[tick, tick2])
    api_mock.fetch_bids_asks = MagicMock(return_value={})
    default_conf_usdt['trading_mode'] = 'futures'
    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange=exchange_name)
    assert exchange.get_conversion_rate('USDT', 'USDT') == 1
    assert api_mock.fetch_tickers.call_count == 0
    assert exchange.get_conversion_rate('ETH', 'USDT') == 42
    assert exchange.get_conversion_rate('ETH', 'USDC') is None
    assert exchange.get_conversion_rate('ETH', 'BTC') == 250
    assert exchange.get_conversion_rate('BTC', 'ETH') == 0.004
    assert api_mock.fetch_tickers.call_count == 1
    api_mock.fetch_tickers.reset_mock()
    assert exchange.get_conversion_rate('ADA', 'USDT') == 2.5
    assert api_mock.fetch_tickers.call_count == 1
    if exchange_name == 'binance':
        assert exchange.get_conversion_rate('BNFCR', 'USDT') is None
        assert exchange.get_conversion_rate('BNFCR', 'USDC') == 1
        assert exchange.get_conversion_rate('USDT', 'BNFCR') is None
        assert exchange.get_conversion_rate('USDC', 'BNFCR') == 1

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_ticker(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    tick = {'symbol': 'ETH/BTC', 'bid': 1.098e-05, 'ask': 1.099e-05, 'last': 0.0001}
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    api_mock.markets = {'ETH/BTC': {'active': True}}
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    ticker = exchange.fetch_ticker(pair='ETH/BTC')
    assert ticker['bid'] == 1.098e-05
    assert ticker['ask'] == 1.099e-05
    tick = {'symbol': 'ETH/BTC', 'bid': 0.5, 'ask': 1, 'last': 42}
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    ticker = exchange.fetch_ticker(pair='ETH/BTC')
    assert api_mock.fetch_ticker.call_count == 1
    assert ticker['bid'] == 0.5
    assert ticker['ask'] == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_ticker', 'fetch_ticker', pair='ETH/BTC')
    api_mock.fetch_ticker = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_ticker(pair='ETH/BTC')
    with pytest.raises(DependencyException, match='Pair XRP/ETH not available'):
        exchange.fetch_ticker(pair='XRP/ETH')

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test___now_is_time_to_refresh(default_conf, mocker, exchange_name, time_machine):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair = 'BTC/USDT'
    candle_type = CandleType.SPOT
    start_dt = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    assert (pair, '5m', candle_type) not in exchange._pairs_last_refresh_time
    assert exchange._now_is_time_to_refresh(pair, '5m', candle_type) is True
    last_closed_candle = dt_ts(start_dt - timedelta(minutes=5))
    exchange._pairs_last_refresh_time[pair, '5m', candle_type] = last_closed_candle
    time_machine.move_to(start_dt + timedelta(minutes=4, seconds=59), tick=False)
    assert exchange._now_is_time_to_refresh(pair, '5m', candle_type) is False
    time_machine.move_to(start_dt + timedelta(minutes=5, seconds=0), tick=False)
    assert exchange._now_is_time_to_refresh(pair, '5m', candle_type) is True
    time_machine.move_to(start_dt + timedelta(minutes=5, seconds=1), tick=False)
    assert exchange._now_is_time_to_refresh(pair, '5m', candle_type) is True

@pytest.mark.parametrize('candle_type', ['mark', ''])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_historic_ohlcv(default_conf, mocker, caplog, exchange_name, candle_type):
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair = 'ETH/BTC'
    calls = 0
    now = dt_now()

    async def mock_candle_hist(pair, timeframe, candle_type, since_ms):
        nonlocal calls
        calls += 1
        ohlcv = [[dt_ts(now + timedelta(minutes=5 * (calls + i))), 1, 2, 3, 4, 5] for i in range(2)]
        return (pair, timeframe, candle_type, ohlcv, True)
    exchange._async_get_candle_history = Mock(wraps=mock_candle_hist)
    since = 5 * 60 * exchange.ohlcv_candle_limit('5m', candle_type) * 1.8
    ret = exchange.get_historic_ohlcv(pair, '5m', dt_ts(dt_now() - timedelta(seconds=since)), candle_type=candle_type)
    assert exchange._async_get_candle_history.call_count == 2
    assert len(ret) == 2
    assert log_has_re('Downloaded data for .* from ccxt with length .*\\.', caplog)
    caplog.clear()

    async def mock_get_candle_hist_error(pair, *args, **kwargs):
        raise TimeoutError()
    exchange._async_get_candle_history = MagicMock(side_effect=mock_get_candle_hist_error)
    ret = exchange.get_historic_ohlcv(pair, '5m', dt_ts(dt_now() - timedelta(seconds=2000)), candle_type=candle_type)
    assert log_has_re('Async code raised an exception: .*', caplog)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
async def test__async_get_historic_ohlcv(default_conf, mocker, caplog, exchange_name):
    ohlcv = [[int((datetime.now(timezone.utc).timestamp() - 1000) * 1000), 1, 2, 3, 4, 5]]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    pair = 'ETH/USDT'
    respair, restf, _, res, _ = await exchange._async_get_historic_ohlcv(pair, '5m', 1500000000000, candle_type=CandleType.SPOT)
    assert respair == pair
    assert restf == '5m'
    assert exchange._api_async.fetch_ohlcv.call_count > 200
    assert res[0] == ohlcv[0]
    exchange._api_async.fetch_ohlcv.reset_mock()
    end_ts = 1500500000000
    start_ts = 1500000000000
    respair, restf, _, res, _ = await exchange._async_get_historic_ohlcv(pair, '5m', since_ms=start_ts, candle_type=CandleType.SPOT, until_ms=end_ts)
    candles = (end_ts - start_ts) / 300000
    exp = candles // exchange.ohlcv_candle_limit('5m', CandleType.SPOT, start_ts) + 1
    assert exchange._api_async.fetch_ohlcv.call_count == exp

@pytest.mark.parametrize('candle_type', [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT])
def test_refresh_latest_ohlcv(mocker, default_conf, caplog, candle_type):
    ohlcv = [[dt_ts(dt_now() - timedelta(minutes=5)), 1, 2, 3, 4, 5], [dt_ts(), 3, 1, 4, 6, 5]]
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    pairs = [('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type)]
    assert not exchange._klines
    res = exchange.refresh_latest_ohlcv(pairs, cache=False)
    assert not exchange._klines
    assert len(res) == len(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    exchange._api_async.fetch_ohlcv.reset_mock()
    exchange.required_candle_call_count = 2
    res = exchange.refresh_latest_ohlcv(pairs)
    assert len(res) == len(pairs)
    assert log_has(f'Refreshing candle (OHLCV) data for {len(pairs)} pairs', caplog)
    assert exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 4
    exchange._api_async.fetch_ohlcv.reset_mock()
    for pair in pairs:
        assert isinstance(exchange.klines(pair), DataFrame)
        assert len(exchange.klines(pair)) > 0
        assert exchange.klines(pair) is not exchange.klines(pair)
        assert exchange.klines(pair) is not exchange.klines(pair, copy=True)
        assert exchange.klines(pair, copy=True) is not exchange.klines(pair, copy=True)
        assert exchange.klines(pair, copy=False) is exchange.klines(pair, copy=False)
    res = exchange.refresh_latest_ohlcv([('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type)])
    assert len(res) == 0
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    caplog.clear()
    for pair in pairs:
        trades = [{'timestamp': dt_ts(exchange._klines[pair].iloc[-1].date - timedelta(minutes=5)), 'amount': 16.512, 'cost': 10134.07488, 'fee': None, 'fees': [], 'id': '354669639', 'order': None, 'price': 613.74, 'side': 'sell', 'takerOrMaker': None, 'type': None}]
        trades_df = DataFrame(trades)
        trades_df['date'] = to_datetime(trades_df['timestamp'], unit='ms', utc=True)
        exchange._trades[pair] = trades_df
    res = exchange.refresh_latest_ohlcv([('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type)])
    assert len(res) == 0
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    caplog.clear()
    for pair in pairs:
        trades = [{'timestamp': dt_ts(exchange._klines[pair].iloc[-1].date - timedelta(minutes=5)), 'amount': 16.512, 'cost': 10134.07488, 'fee': None, 'fees': [], 'id': '354669639', 'order': None, 'price': 613.74, 'side': 'sell', 'takerOrMaker': None, 'type': None}]
        trades_df = DataFrame(trades)
        trades_df['date'] = to_datetime(trades_df['timestamp'], unit='ms', utc=True)
        exchange._klines[pair] = trades_df
    res = exchange.refresh_latest_ohlcv([('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type)])
    assert len(res) == len(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 4
    exchange._api_async.fetch_ohlcv.reset_mock()
    exchange.required_candle_call_count = 1
    pairlist = [('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type), ('XRP/ETH', '1d', candle_type)]
    res = exchange.refresh_latest_ohlcv(pairlist, cache=False)
    assert len(res) == 3
    assert exchange._api_async.fetch_ohlcv.call_count == 6
    exchange._api_async.fetch_ohlcv.reset_mock()
    res = exchange.refresh_latest_ohlcv(pairlist, cache=False)
    assert len(res) == 3
    assert exchange._api_async.fetch_ohlcv.call_count == 6
    exchange._api_async.fetch_ohlcv.reset_mock()
    caplog.clear()
    res = exchange.refresh_latest_ohlcv([('IOTA/ETH', '3m', candle_type)], cache=False)
    if candle_type != CandleType.MARK:
        assert not res
        assert len(res) == 0
        assert log_has_re('Cannot download \\(IOTA\\/ETH, 3m\\).*', caplog)
    else:
        assert len(res) == 1

@pytest.mark.parametrize('candle_type', [CandleType.FUTURES, CandleType.SPOT])
def test_refresh_latest_trades(mocker, default_conf, caplog, candle_type, tmp_path, time_machine):
    time_machine.move_to(dt_now(), tick=False)
    trades = [{'timestamp': dt_ts(dt_now() - timedelta(minutes=5)), 'amount': 16.512, 'cost': 10134.07488, 'fee': None, 'fees': [], 'id': '354669639', 'order': None, 'price': 613.74, 'side': 'sell', 'takerOrMaker': None, 'type': None}, {'timestamp': dt_ts(), 'amount': 12.512, 'cost': 1000, 'fee': None, 'fees': [], 'id': '354669640', 'order': None, 'price': 613.84, 'side': 'buy', 'takerOrMaker': None, 'type': None}]
    caplog.set_level(logging.DEBUG)
    use_trades_conf = default_conf
    use_trades_conf['exchange']['use_public_trades'] = True
    use_trades_conf['datadir'] = tmp_path
    use_trades_conf['orderflow'] = {'max_candles': 1500}
    exchange = get_patched_exchange(mocker, use_trades_conf)
    exchange._api_async.fetch_trades = get_mock_coro(trades)
    exchange._ft_has['exchange_has_overrides'] = {'fetchTrades': True}
    pairs = [('IOTA/USDT:USDT', '5m', candle_type), ('XRP/USDT:USDT', '5m', candle_type)]
    assert not exchange._trades
    res = exchange.refresh_latest_trades(pairs, cache=False)
    assert not exchange._trades
    assert len(res) == len(pairs)
    assert exchange._api_async.fetch_trades.call_count == 4
    exchange._api_async.fetch_trades.reset_mock()
    exchange.required_candle_call_count = 2
    res = exchange.refresh_latest_trades(pairs)
    assert len(res) == len(pairs)
    assert log_has(f'Refreshing TRADES data for {len(pairs)} pairs', caplog)
    assert exchange._trades
    assert exchange._api_async.fetch_trades.call_count == 4
    exchange._api_async.fetch_trades.reset_mock()
    for pair in pairs:
        assert isinstance(exchange.trades(pair), DataFrame)
        assert len(exchange.trades(pair)) > 0
        assert exchange.trades(pair) is not exchange.trades(pair)
        assert exchange.trades(pair) is not exchange.trades(pair, copy=True)
        assert exchange.trades(pair, copy=True) is not exchange.trades(pair, copy=True)
        assert exchange.trades(pair, copy=False) is exchange.trades(pair, copy=False)
        ohlcv = [[dt_ts(dt_now() - timedelta(minutes=5)), 1, 2, 3, 4, 5], [dt_ts(), 3, 1, 4, 6, 5]]
        cols = DEFAULT_DATAFRAME_COLUMNS
        trades_df = DataFrame(ohlcv, columns=cols)
        trades_df['date'] = to_datetime(trades_df['date'], unit='ms', utc=True)
        trades_df['date'] = trades_df['date'].apply(lambda date: timeframe_to_prev_date('5m', date))
        exchange._klines[pair] = trades_df
    res = exchange.refresh_latest_trades([('IOTA/USDT:USDT', '5m', candle_type), ('XRP/USDT:USDT', '5m', candle_type)])
    assert len(res) == 0
    assert exchange._api_async.fetch_trades.call_count == 0
    caplog.clear()
    for pair in pairs:
        trades = [{'timestamp': dt_ts(exchange._klines[pair].iloc[-1].date - timedelta(minutes=5)), 'amount': 16.512, 'cost': 10134.07488, 'fee': None, 'fees': [], 'id': '354669639', 'order': None, 'price': 613.74, 'side': 'sell', 'takerOrMaker': None, 'type': None}]
        trades_df = DataFrame(trades)
        trades_df['date'] = to_datetime(trades_df['timestamp'], unit='ms', utc=True)
        exchange._trades[pair] = trades_df
    res = exchange.refresh_latest_trades([('IOTA/USDT:USDT', '5m', candle_type), ('XRP/USDT:USDT', '5m', candle_type)])
    assert len(res) == len(pairs)
    assert exchange._api_async.fetch_trades.call_count == 4
    exchange._api_async.fetch_trades.reset_mock()
    exchange.required_candle_call_count = 1
    pairlist = [('IOTA/ETH', '5m', candle_type), ('XRP/ETH', '5m', candle_type), ('XRP/ETH', '1d', candle_type)]
    res = exchange.refresh_latest_trades(pairlist, cache=False)
    assert len(res) == 3
    assert exchange._api_async.fetch_trades.call_count == 6
    exchange._api_async.fetch_trades.reset_mock()
    res = exchange.refresh_latest_trades(pairlist, cache=False)
    assert len(res) == 3
    assert exchange._api_async.fetch_trades.call_count == 6
    exchange._api_async.fetch_trades.reset_mock()
    caplog.clear()

@pytest.mark.parametrize('candle_type', [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT])
def test_refresh_latest_ohlcv_cache(mocker, default_conf, candle_type, time_machine):
    start = datetime(2021, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    ohlcv = generate_test_data_raw('1h', 100, start.strftime('%Y-%m-%d'))
    time_machine.move_to(start + timedelta(hours=99, minutes=30))
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(f'{EXMS}.ohlcv_candle_limit', return_value=100)
    assert exchange._startup_candle_count == 0
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    pair1 = ('IOTA/ETH', '1h', candle_type)
    pair2 = ('XRP/ETH', '1h', candle_type)
    pairs = [pair1, pair2]
    assert not exchange._klines
    res = exchange.refresh_latest_ohlcv(pairs, cache=False)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(res) == 2
    assert len(res[pair1]) == 99
    assert len(res[pair2]) == 99
    assert not exchange._klines
    exchange._api_async.fetch_ohlcv.reset_mock()
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(res) == 2
    assert len(res[pair1]) == 99
    assert len(res[pair2]) == 99
    assert exchange._klines
    assert exchange._pairs_last_refresh_time[pair1] == ohlcv[-2][0]
    exchange._api_async.fetch_ohlcv.reset_mock()
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    assert len(res) == 2
    assert len(res[pair1]) == 99
    assert len(res[pair2]) == 99
    assert exchange._pairs_last_refresh_time[pair1] == ohlcv[-2][0]
    time_machine.move_to(start + timedelta(hours=101))
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(res) == 2
    assert len(res[pair1]) == 99
    assert len(res[pair2]) == 99
    assert res[pair2].at[0, 'open']
    assert exchange._pairs_last_refresh_time[pair1] == ohlcv[-2][0]
    refresh_pior = exchange._pairs_last_refresh_time[pair1]
    new_startdate = (start + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
    ohlcv = generate_test_data_raw('1h', 100, new_startdate)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(res) == 2
    assert len(res[pair1]) == 100
    assert len(res[pair2]) == 100
    assert res[pair2].at[0, 'open']
    assert refresh_pior != exchange._pairs_last_refresh_time[pair1]
    assert exchange._pairs_last_refresh_time[pair1] == ohlcv[-2][0]
    assert exchange._pairs_last_refresh_time[pair2] == ohlcv[-2][0]
    exchange._api_async.fetch_ohlcv.reset_mock()
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    assert len(res) == 2
    assert len(res[pair1]) == 100
    assert len(res[pair2]) == 100
    assert res[pair2].at[0, 'open']
    time_machine.move_to(start + timedelta(hours=2000))
    ohlcv = generate_test_data_raw('1h', 100, start + timedelta(hours=1900))
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    res = exchange.refresh_latest_ohlcv(pairs)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(res) == 2
    assert len(res[pair1]) == 99
    assert len(res[pair2]) == 99
    assert res[pair2].at[0, 'open']

@pytest.mark.parametrize('exchange_name', EXCHANGES)
async def test__async_get_candle_history(default_conf, mocker, caplog, exchange_name):

    def sort_data(data, key):
        return sorted(data, key=key)
    ohlcv = [[1527833100000, 0.07666, 0.07671, 0.07666, 0.07668, 16.65244264], [1527832800000, 0.07662, 0.07666, 0.07662, 0.07666, 1.30051526], [1527832500000, 0.07656, 0.07661, 0.07656, 0.07661, 12.034778840000001], [1527832200000, 0.07658, 0.07658, 0.07655, 0.07656, 0.59780186], [1527831900000, 0.07658, 0.07658, 0.07658, 0.07658, 1.76278136], [1527831600000, 0.07658, 0.07658, 0.07658, 0.07658, 2.22646521], [1527831300000, 0.07655, 0.07657, 0.07655, 0.07657, 1.1753], [1527831000000, 0.07654, 0.07654, 0.07651, 0.07651, 0.8073060299999999], [1527830700000, 0.07652, 0.07652, 0.07651, 0.07652, 10.04822687], [1527830400000, 0.07649, 0.07651, 0.07649, 0.07651, 2.5734867]]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock = mocker.patch('freqtrade.exchange.exchange.sorted', MagicMock(side_effect=sort_data))
    res = await exchange._async_get_candle_history('ETH/BTC', default_conf['timeframe'], CandleType.SPOT)
    assert res[0] == 'ETH/BTC'
    res_ohlcv = res[3]
    assert sort_mock.call_count == 1
    assert res_ohlcv[0][0] == 1527830400000
    assert res_ohlcv[0][1] == 0.07649
    assert res_ohlcv[0][2] == 0.07651
    assert res_ohlcv[0][3] == 0.07649
    assert res_ohlcv[0][4] == 0.07651
    assert res_ohlcv[0][5] == 2.5734867
    assert res_ohlcv[9][0] == 1527833100000
    assert res_ohlcv[9][1] == 0.07666
    assert res_ohlcv[9][2] == 0.07671
    assert res_ohlcv[9][3] == 0.07666
    assert res_ohlcv[9][4] == 0.07668
    assert res_ohlcv[9][5] == 16.65244264
    ohlcv = [[1527827700000, 0.07659999, 0.0766, 0.07627, 0.07657998, 1.85216924], [1527828000000, 0.07657995, 0.07657995, 0.0763, 0.0763, 26.04051037], [1527828300000, 0.0763, 0.07659998, 0.0763, 0.0764, 10.36434124], [1527828600000, 0.0764, 0.0766, 0.0764, 0.0766, 5.71044773], [1527828900000, 0.0764, 0.07666998, 0.0764, 0.07666998, 47.48888565], [1527829200000, 0.0765, 0.07672999, 0.0765, 0.07672999, 3.37640326], [1527829500000, 0.0766, 0.07675, 0.0765, 0.07675, 8.36203831], [1527829800000, 0.07675, 0.07677999, 0.07620002, 0.076695, 119.22963884], [1527830100000, 0.076695, 0.07671, 0.07624171, 0.07671, 1.80689244], [1527830400000, 0.07671, 0.07674399, 0.07629216, 0.07655213, 2.31452783]]
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock = mocker.patch('freqtrade.exchange.exchange.sorted', MagicMock(side_effect=sort_data))
    res = await exchange._async_get_candle_history('ETH/BTC', default_conf['timeframe'], CandleType.SPOT)
    assert res[0] == 'ETH/BTC'
    assert res[1] == default_conf['timeframe']
    res_ohlcv = res[3]
    assert sort_mock.call_count == 0
    assert res_ohlcv[0][0] == 1527827700000
    assert res_ohlcv[0][1] == 0.07659999
    assert res_ohlcv[0][2] == 0.0766
    assert res_ohlcv[0][3] == 0.07627
    assert res_ohlcv[0][4] == 0.07657998
    assert res_ohlcv[0][5] == 1.85216924
    assert res_ohlcv[9][0] == 1527830400000
    assert res_ohlcv[9][1] == 0.07671
    assert res_ohlcv[9][2] == 0.07674399
    assert res_ohlcv[9][3] == 0.07629216
    assert res_ohlcv[9][4] == 0.07655213
    assert res_ohlcv[9][5] == 2.31452783

@pytest.mark.parametrize('exchange_name', EXCHANGES)
async def test__async_get_trade_history_id(default_conf, mocker, exchange_name, fetch_trades_result):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != 'id':
        exchange.close()
        pytest.skip('Exchange does not support pagination by trade id')
    pagination_arg = exchange._trades_pagination_arg

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if 'since' in kwargs:
            return fetch_trades_result[:-2]
        elif kwargs.get('params', {}).get(pagination_arg) in (fetch_trades_result[-3]['id'], 1565798399752):
            return fetch_trades_result[-3:-1]
        else:
            return fetch_trades_result[-2:]
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_id(pair, since=fetch_trades_result[0]['timestamp'], until=fetch_trades_result[-1]['timestamp'] - 1)
    assert isinstance(ret, tuple)
    assert ret[0] == pair
    assert isinstance(ret[1], list)
    if exchange_name != 'kraken':
        assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 3
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == fetch_trades_result[0]['timestamp']
    assert fetch_trades_cal[1][0][0] == pair
    assert 'params' in fetch_trades_cal[1][1]
    assert exchange._ft_has['trades_pagination_arg'] in fetch_trades_cal[1][1]['params']

@pytest.mark.parametrize('trade_id, expected', [('1234', True), ('170544369512007228', True), ('1705443695120072285', True), ('170544369512007228555', True)])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test__valid_trade_pagination_id(mocker, default_conf_usdt, exchange_name, trade_id, expected):
    if exchange_name == 'kraken':
        pytest.skip('Kraken has a different pagination id format, and an explicit test.')
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange_name)
    assert exchange._valid_trade_pagination_id('XRP/USDT', trade_id) == expected

@pytest.mark.parametrize('exchange_name', EXCHANGES)
async def test__async_get_trade_history_time(default_conf, mocker, caplog, exchange_name, fetch_trades_result):
    caplog.set_level(logging.DEBUG)

    async def mock_get_trade_hist(pair, *args, **kwargs):
        if kwargs['since'] == fetch_trades_result[0]['timestamp']:
            return fetch_trades_result[:-1]
        else:
            return fetch_trades_result[-1:]
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != 'time':
        exchange.close()
        pytest.skip('Exchange does not support pagination by timestamp')
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_time(pair, since=fetch_trades_result[0]['timestamp'], until=fetch_trades_result[-1]['timestamp'] - 1)
    assert isinstance(ret, tuple)
    assert ret[0] == pair
    assert isinstance(ret[1], list)
    assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 2
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == fetch_trades_result[0]['timestamp']
    assert fetch_trades_cal[1][0][0] == pair
    assert fetch_trades_cal[1][1]['since'] == fetch_trades_result[-2]['timestamp']
    assert log_has_re('Stopping because until was reached.*', caplog)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_historic_trades(default_conf, mocker, exchange_name, trades_history):
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair = 'ETH/BTC'
    exchange._async_get_trade_history_id = get_mock_coro((pair, trades_history))
    exchange._async_get_trade_history_time = get_mock_coro((pair, trades_history))
    ret = exchange.get_historic_trades(pair, since=trades_history[0][0], until=trades_history[-1][0])
    assert sum([exchange._async_get_trade_history_id.call_count, exchange._async_get_trade_history_time.call_count]) == 1
    assert len(ret) == 2
    assert ret[0] == pair
    assert len(ret[1]) == len(trades_history)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_historic_trades_notsupported(default_conf, mocker, exchange_name, trades_history):
    mocker.patch(f'{EXMS}.exchange_has', return_value=False)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair = 'ETH/BTC'
    with pytest.raises(OperationalException, match='This exchange does not support downloading Trades.'):
        exchange.get_historic_trades(pair, since=trades_history[0][0], until=trades_history[-1][0])

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_cancel_order_dry_run(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=True)
    assert exchange.cancel_order(order_id='123', pair='TKN/BTC') == {}
    assert exchange.cancel_stoploss_order(order_id='123', pair='TKN/BTC') == {}
    order = exchange.create_order(pair='ETH/BTC', ordertype='limit', side='buy', amount=5, rate=0.55, time_in_force='gtc', leverage=1.0)
    cancel_order = exchange.cancel_order(order_id=order['id'], pair='ETH/BTC')
    assert order['id'] == cancel_order['id']
    assert order['amount'] == cancel_order['amount']
    assert order['symbol'] == cancel_order['symbol']
    assert cancel_order['status'] == 'canceled'

@pytest.mark.parametrize('exchange_name', EXCHANGES)
@pytest.mark.parametrize('order,result', [({'status': 'closed', 'filled': 10}, False), ({'status': 'closed', 'filled': 0.0}, True), ({'status': 'canceled', 'filled': 0.0}, True), ({'status': 'canceled', 'filled': 10.0}, False), ({'result': 'testest123'}, False)])
def test_check_order_canceled_empty(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.check_order_canceled_empty(order) == result

@pytest.mark.parametrize('exchange_name', EXCHANGES)
@pytest.mark.parametrize('order,result', [({'status': 'closed', 'amount': 10, 'fee': {}}, True), ({'status': 'closed', 'amount': 0.0, 'fee': {}}, True), ({'status': 'canceled', 'amount': 0.0, 'fee': {}}, True), ({'status': 'canceled', 'amount': 10.0}, False), ({'amount': 10.0, 'fee': {}}, False), ({'result': 'testest123'}, False), ('hello_world', False), ({'status': 'canceled', 'amount': None, 'fee': None}, False), ({'status': 'canceled', 'filled': None, 'amount': None, 'fee': None}, False)])
def test_is_cancel_order_result_suitable(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.is_cancel_order_result_suitable(order) == result

@pytest.mark.parametrize('exchange_name', EXCHANGES)
@pytest.mark.parametrize('corder,call_corder,call_forder', [({'status': 'closed', 'amount': 10, 'fee': {}}, 1, 0), ({'amount': 10, 'fee': {}}, 1, 1)])
def test_cancel_order_with_result(default_conf, mocker, exchange_name, corder, call_corder, call_forder):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=corder)
    api_mock.fetch_order = MagicMock(return_value={'id': '1234'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order_with_result('1234', 'ETH/BTC', 1234)
    assert isinstance(res, dict)
    assert api_mock.cancel_order.call_count == call_corder
    assert api_mock.fetch_order.call_count == call_forder

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_cancel_order_with_result_error(default_conf, mocker, exchange_name, caplog):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder('Did not find order'))
    api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder('Did not find order'))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order_with_result('1234', 'ETH/BTC', 1541)
    assert isinstance(res, dict)
    assert log_has('Could not cancel order 1234 for ETH/BTC.', caplog)
    assert log_has('Could not fetch cancelled order 1234.', caplog)
    assert res['amount'] == 1541

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_cancel_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={'id': '123'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.cancel_order(order_id='_', pair='TKN/BTC') == {'id': '123'}
    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder('Did not find order'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'cancel_order', 'cancel_order', order_id='_', pair='TKN/BTC')

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_cancel_stoploss_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={'id': '123'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.cancel_stoploss_order(order_id='_', pair='TKN/BTC') == {'id': '123'}
    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder('Did not find order'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_stoploss_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'cancel_stoploss_order', 'cancel_order', order_id='_', pair='TKN/BTC')

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_order(default_conf, mocker, exchange_name, caplog):
    default_conf['dry_run'] = True
    default_conf['exchange']['log_responses'] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = 'TKN/BTC'
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    with patch('freqtrade.exchange.common.time.sleep') as tm:
        with pytest.raises(InvalidOrderException):
            exchange.fetch_order(order_id='_', pair='TKN/BTC')
        assert tm.call_args_list[0][0][0] == 1
        assert tm.call_args_list[1][0][0] == 2
        if API_FETCH_ORDER_RETRY_COUNT > 2:
            assert tm.call_args_list[2][0][0] == 5
        if API_FETCH_ORDER_RETRY_COUNT > 3:
            assert tm.call_args_list[3][0][0] == 10
    assert api_mock.fetch_order.call_count == API_FETCH_ORDER_RETRY_COUNT + 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_order', 'fetch_order', retries=API_FETCH_ORDER_RETRY_COUNT + 1, order_id='_', pair='TKN/BTC')

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_order_emulated(default_conf, mocker, exchange_name, caplog):
    default_conf['dry_run'] = True
    default_conf['exchange']['log_responses'] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = 'TKN/BTC'
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch(f'{EXMS}.exchange_has', return_value=False)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.exchange_has', return_value=False)
    api_mock = MagicMock()
    api_mock.fetch_open_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    api_mock.fetch_closed_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_open_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 0
    caplog.clear()
    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    api_mock.fetch_closed_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_closed_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 1
    caplog.clear()
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_open_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        api_mock.fetch_closed_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_open_order.call_count == 1
    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_order_emulated', 'fetch_open_order', retries=1, order_id='_', pair='TKN/BTC', params={})

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_fetch_stoploss_order(default_conf, mocker, exchange_name):
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_stoploss_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_stoploss_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = {'id': '123', 'symbol': 'TKN/BTC'}
    if exchange_name == 'okx':
        res = {'id': '123', 'symbol': 'TKN/BTC', 'type': 'stoploss'}
    assert exchange.fetch_stoploss_order('X', 'TKN/BTC') == res
    if exchange_name == 'okx':
        return
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_stoploss_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'fetch_stoploss_order', 'fetch_order', retries=API_FETCH_ORDER_RETRY_COUNT + 1, order_id='_', pair='TKN/BTC')

def test_fetch_order_or_stoploss_order(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    fetch_order_mock = MagicMock()
    fetch_stoploss_order_mock = MagicMock()
    mocker.patch.multiple(EXMS, fetch_order=fetch_order_mock, fetch_stoploss_order=fetch_stoploss_order_mock)
    exchange.fetch_order_or_stoploss_order('1234', 'ETH/BTC', False)
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][0][0] == '1234'
    assert fetch_order_mock.call_args_list[0][0][1] == 'ETH/BTC'
    assert fetch_stoploss_order_mock.call_count == 0
    fetch_order_mock.reset_mock()
    fetch_stoploss_order_mock.reset_mock()
    exchange.fetch_order_or_stoploss_order('1234', 'ETH/BTC', True)
    assert fetch_order_mock.call_count == 0
    assert fetch_stoploss_order_mock.call_count == 1
    assert fetch_stoploss_order_mock.call_args_list[0][0][0] == '1234'
    assert fetch_stoploss_order_mock.call_args_list[0][0][1] == 'ETH/BTC'

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_name(default_conf, mocker, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name

@pytest.mark.parametrize('trading_mode,amount', [('spot', 0.2340606), ('futures', 2.340606)])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_trades_for_order(default_conf, mocker, exchange_name, trading_mode, amount):
    order_id = 'ABCD-ABCD'
    since = datetime(2018, 5, 5, 0, 0, 0)
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    api_mock = MagicMock()
    api_mock.fetch_my_trades = MagicMock(return_value=[{'id': 'TTR67E-3PFBD-76IISV', 'order': 'ABCD-ABCD', 'info': {'pair': 'XLTCZBTC', 'time': 1519860024.4388, 'type': 'buy', 'ordertype': 'limit', 'price': '20.00000', 'cost': '38.62000', 'fee': '0.06179', 'vol': '5', 'id': 'ABCD-ABCD'}, 'timestamp': 1519860024438, 'datetime': '2018-02-28T23:20:24.438Z', 'symbol': 'ETH/USDT:USDT', 'type': 'limit', 'side': 'buy', 'price': 165.0, 'amount': 0.2340606, 'fee': {'cost': 0.06179, 'currency': 'BTC'}}])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    orders = exchange.get_trades_for_order(order_id, 'ETH/USDT:USDT', since)
    assert len(orders) == 1
    assert orders[0]['price'] == 165
    assert pytest.approx(orders[0]['amount']) == amount
    assert api_mock.fetch_my_trades.call_count == 1
    assert isinstance(api_mock.fetch_my_trades.call_args[0][1], int)
    assert api_mock.fetch_my_trades.call_args[0][0] == 'ETH/USDT:USDT'
    assert api_mock.fetch_my_trades.call_args[0][1] == 1525478395000
    assert api_mock.fetch_my_trades.call_args[0][1] == int(since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'get_trades_for_order', 'fetch_my_trades', order_id=order_id, pair='ETH/USDT:USDT', since=since)
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=False))
    assert exchange.get_trades_for_order(order_id, 'ETH/USDT:USDT', since) == []

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_fee(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(return_value={'type': 'taker', 'currency': 'BTC', 'rate': 0.025, 'cost': 0.05})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._config.pop('fee', None)
    assert exchange.get_fee('ETH/BTC') == 0.025
    assert api_mock.calculate_fee.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, 'get_fee', 'calculate_fee', symbol='ETH/BTC')
    api_mock.calculate_fee.reset_mock()
    exchange._config['fee'] = 0.001
    assert exchange.get_fee('ETH/BTC') == 0.001
    assert api_mock.calculate_fee.call_count == 0

def test_stoploss_order_unsupported_exchange(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange='bitpanda')
    with pytest.raises(OperationalException, match='stoploss is not implemented .*'):
        exchange.create_stoploss(pair='ETH/BTC', amount=1, stop_price=220, order_types={}, side='sell', leverage=1.0)
    with pytest.raises(OperationalException, match='stoploss is not implemented .*'):
        exchange.stoploss_adjust(1, {}, side='sell')

@pytest.mark.parametrize('side,ratio,expected', [('sell', 0.99, 99.0), ('sell', 0.999, 99.9), ('sell', 1, 100), ('sell', 1.1, InvalidOrderException), ('buy', 0.99, 101.0), ('buy', 0.999, 100.1), ('buy', 1, 100), ('buy', 1.1, InvalidOrderException)])
def test__get_stop_limit_rate(default_conf_usdt, mocker, side, ratio, expected):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange='binance')
    order_types = {'stoploss_on_exchange_limit_ratio': ratio}
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            exchange._get_stop_limit_rate(100, order_types, side)
    else:
        assert exchange._get_stop_limit_rate(100, order_types, side) == expected

def test_merge_ft_has_dict(default_conf, mocker):
    mocker.patch.multiple(EXMS, _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_stakecurrency=MagicMock(), validate_pricing=MagicMock())
    ex = Exchange(default_conf)
    assert ex._ft_has == Exchange._ft_has_default
    ex = Kraken(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex.get_option('trades_pagination') == 'id'
    assert ex.get_option('trades_pagination_arg') == 'since'
    ex = Binance(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex.get_option('stoploss_on_exchange')
    assert ex.get_option('order_time_in_force') == ['GTC', 'FOK', 'IOC', 'PO']
    assert ex.get_option('trades_pagination') == 'id'
    assert ex.get_option('trades_pagination_arg') == 'fromId'
    conf = copy.deepcopy(default_conf)
    conf['exchange']['_ft_has_params'] = {'DeadBeef': 20, 'stoploss_on_exchange': False}
    ex = Binance(conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert not ex._ft_has['stoploss_on_exchange']
    assert ex._ft_has['DeadBeef'] == 20

def test_get_valid_pair_combination(default_conf, mocker, markets):
    mocker.patch.multiple(EXMS, _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_stakecurrency=MagicMock(), markets=PropertyMock(return_value=markets))
    ex = Exchange(default_conf)
    assert next(ex.get_valid_pair_combination('ETH', 'BTC')) == 'ETH/BTC'
    assert next(ex.get_valid_pair_combination('BTC', 'ETH')) == 'ETH/BTC'
    multicombs = list(ex.get_valid_pair_combination('ETH', 'USDT'))
    assert len(multicombs) == 2
    assert 'ETH/USDT' in multicombs
    assert 'ETH/USDT:USDT' in multicombs
    with pytest.raises(ValueError, match='Could not combine.* to get a valid pair.'):
        for x in ex.get_valid_pair_combination('NOPAIR', 'ETH'):
            pass

def test_get_markets(default_conf, mocker, markets_static):
    mocker.patch.multiple(EXMS, _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_pricing=MagicMock(), markets=PropertyMock(return_value=markets_static))
    ex = Exchange(default_conf)
    pairs = ex.get_markets('LTC', 'USDT', True, False)
    assert sorted(pairs.keys()) == sorted(['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XRP/BTC', 'ADA/USDT:USDT', 'ETH/USDT:USDT'])

def test_get_markets_error(default_conf, mocker):
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=None))
    with pytest.raises(OperationalException, match='Markets were not loaded.'):
        ex.get_markets('LTC', 'USDT', True, False)

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_ohlcv_candle_limit(default_conf, mocker, exchange_name):
    if exchange_name == 'okx':
        pytest.skip('Tested separately for okx')
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    timeframes = ('1m', '5m', '1h')
    expected = exchange._ft_has.get('ohlcv_candle_limit', 500)
    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == expected

@pytest.mark.parametrize('market_symbol,base,quote,exchange,spot,margin,futures,trademode,add_dict,expected_result', [('BTC/USDT', 'BTC', 'USDT', 'binance', True, False, False, 'spot', {}, True), ('USDT/BTC', 'USDT', 'BTC', 'binance', True, False, False, 'spot', {}, True), ('BTCUSDT', 'BTC', 'USDT', 'binance', True, False, False, 'spot', {}, True), ('BTCUSDT', None, 'USDT', 'binance', True, False, False, 'spot', {}, False), ('USDT/BTC', 'BTC', None, 'binance', True, False, False, 'spot', {}, False), ('BTCUSDT', 'BTC', None, 'binance', True, False, False, 'spot', {}, False), ('BTC/USDT', 'BTC', 'USDT', 'binance', True, False, False, 'spot', {}, True), ('BTC/USDT', 'BTC', 'USDT', 'binance', True, False, False, 'futures', {}, False), ('BTC/USDT', 'BTC', 'USDT', 'binance', True, False, False, 'margin', {}, False), ('BTC/USDT', 'BTC', 'USDT', 'binance', True, True, True, 'margin', {}, True), ('BTC/UNK', 'BTC', 'UNK', 'binance', False, False, True, 'spot', {}, False), ('BTC/EUR', 'BTC', 'EUR', 'kraken', True, False, False, 'spot', {'darkpool': False}, True), ('EUR/BTC', 'EUR', 'BTC', 'kraken', True, False, False, 'spot', {'darkpool': False}, True), ('BTC/EUR.d', 'BTC', 'EUR', 'kraken', True, False, False, 'spot', {'darkpool': True}, False), ('BTC/USDT:USDT', 'BTC', 'USD', 'okx', False, False, True, 'spot', {}, False), ('BTC/USDT:USDT', 'BTC', 'USD', 'okx', False, False, True, 'margin', {}, False), ('BTC/USDT:USDT', 'BTC', 'USD', 'okx', False, False, True, 'futures', {}, True)])
def test_market_is_tradable(mocker, default_conf, market_symbol, base, quote, spot, margin, futures, trademode, add_dict, exchange, expected_result):
    default_conf['trading_mode'] = trademode
    mocker.patch(f'{EXMS}.validate_trading_mode_and_margin_mode')
    ex = get_patched_exchange(mocker, default_conf, exchange=exchange)
    market = {'symbol': market_symbol, 'type': 'swap', 'base': base, 'quote': quote, 'spot': spot, 'future': futures, 'swap': futures, 'margin': margin, 'linear': True, **add_dict}
    assert ex.market_is_tradable(market) == expected_result

@pytest.mark.parametrize('market,expected_result', [({'symbol': 'ETH/BTC', 'active': True}, True), ({'symbol': 'ETH/BTC', 'active': False}, False), ({'symbol': 'ETH/BTC'}, True)])
def test_market_is_active(market, expected_result):
    assert market_is_active(market) == expected_result

@pytest.mark.parametrize('order,expected', [([{'fee'}], False), ({'fee': None}, False), ({'fee': {'currency': 'ETH/BTC'}}, False), ({'fee': {'currency': 'ETH/BTC', 'cost': None}}, False), ({'fee': {'currency': 'ETH/BTC', 'cost': 0.01}}, True)])
def test_order_has_fee(order, expected):
    assert Exchange.order_has_fee(order) == expected

@pytest.mark.parametrize('order,expected', [({'symbol': 'ETH/BTC', 'fee': {'currency': 'ETH', 'cost': 0.43}}, (0.43, 'ETH', 0.01)), ({'symbol': 'ETH/USDT', 'fee': {'currency': 'USDT', 'cost': 0.01}}, (0.01, 'USDT', 0.01)), ({'symbol': 'BTC/USDT', 'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.01}}, (0.34, 'USDT', 0.01))])
def test_extract_cost_curr_rate(mocker, default_conf, order, expected):
    mocker.patch(f'{EXMS}.calculate_fee_rate', MagicMock(return_value=0.01))
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.extract_cost_curr_rate(order['fee'], order['symbol'], cost=20, amount=1) == expected

@pytest.mark.parametrize('order,unknown_fee_rate,expected', [({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'ETH', 'cost': 0.004, 'rate': None}}, None, 0.1), ({'symbol': 'ETH/BTC', 'amount': 0.05, 'cost': 0.05, 'fee': {'currency': 'ETH', 'cost': 0.004, 'rate': None}}, None, 0.08), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'BTC', 'cost': 0.005}}, None, 0.1), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'BTC', 'cost': 0.002, 'rate': None}}, None, 0.04), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'NEO', 'cost': 0.0012}}, None, 0.001944), ({'symbol': 'ETH/BTC', 'amount': 2.21, 'cost': 0.02992561, 'fee': {'currency': 'NEO', 'cost': 0.00027452}}, None, 0.00074305), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.01}}, None, 0.01), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': 'USDT', 'cost': 0.34, 'rate': 0.005}}, None, 0.005), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0, 'fee': {'currency': 'BTC', 'cost': 0.0, 'rate': None}}, None, None), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0, 'fee': {'currency': 'ETH', 'cost': 0.0, 'rate': None}}, None, 0.0), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.0, 'fee': {'currency': 'NEO', 'cost': 0.0, 'rate': None}}, None, None), ({'symbol': 'POINT/BTC', 'amount': 0.04, 'cost': 0.5, 'fee': {'currency': 'POINT', 'cost': 2.0, 'rate': None}}, None, None), ({'symbol': 'POINT/BTC', 'amount': 0.04, 'cost': 0.5, 'fee': {'currency': 'POINT', 'cost': 2.0, 'rate': None}}, 1, 4.0), ({'symbol': 'POINT/BTC', 'amount': 0.04, 'cost': 0.5, 'fee': {'currency': 'POINT', 'cost': 2.0, 'rate': None}}, 2, 8.0), ({'symbol': 'ETH/BTC', 'amount': 0.04, 'cost': 0.05, 'fee': {'currency': None, 'cost': 0.005}}, None, None)])
def test_calculate_fee_rate(mocker, default_conf, order, expected, unknown_fee_rate):
    mocker.patch(f'{EXMS}.get_tickers', return_value={'NEO/BTC': {'last': 0.081}})
    if unknown_fee_rate:
        default_conf['exchange']['unknown_fee_rate'] = unknown_fee_rate
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.calculate_fee_rate(order['fee'], order['symbol'], cost=order['cost'], amount=order['amount']) == expected

@pytest.mark.parametrize('retrycount,max_retries,expected', [(0, 3, 10), (1, 3, 5), (2, 3, 2), (3, 3, 1), (0, 1, 2), (1, 1, 1), (0, 4, 17), (1, 4, 10), (2, 4, 5), (3, 4, 2), (4, 4, 1), (0, 5, 26), (1, 5, 17), (2, 5, 10), (3, 5, 5), (4, 5, 2), (5, 5, 1)])
def test_calculate_backoff(retrycount, max_retries, expected):
    assert calculate_backoff(retrycount, max_retries) == expected

@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_get_funding_fees(default_conf_usdt, mocker, exchange_name, caplog):
    now = datetime.now(timezone.utc)
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)
    exchange._fetch_and_calculate_funding_fees = MagicMock(side_effect=ExchangeError)
    assert exchange.get_funding_fees('BTC/USDT:USDT', 1, False, now) == 0.0
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1
    assert log_has('Could not update funding fees for BTC/USDT:USDT.', caplog)

@pytest.mark.parametrize('exchange', ['binance', 'kraken'])
@pytest.mark.parametrize('stake_amount,leverage,min_stake_with_lev', [(9.0, 3.0, 3.0), (20.0, 5.0, 4.0), (100.0, 100.0, 1.0)])
def test_get_stake_amount_considering_leverage(exchange, stake_amount, leverage, min_stake_with_lev, mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange)
    assert exchange._get_stake_amount_considering_leverage(stake_amount, leverage) == min_stake_with_lev

@pytest.mark.parametrize('margin_mode', [MarginMode.CROSS, MarginMode.ISOLATED])
def test_set_margin_mode(mocker, default_conf, margin_mode):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'setMarginMode': True})
    default_conf['dry_run'] = False
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'binance', 'set_margin_mode', 'set_margin_mode', pair='XRP/USDT', margin_mode=margin_mode)

@pytest.mark.parametrize('exchange_name,trading_mode,ccxt_config', [('binance', 'spot', {}), ('binance', 'margin', {'options': {'defaultType': 'margin'}}), ('binance', 'futures', {'options': {'defaultType': 'swap'}}), ('bybit', 'spot', {'options': {'defaultType': 'spot'}}), ('bybit', 'futures', {'options': {'defaultType': 'swap'}}), ('gate', 'futures', {'options': {'defaultType': 'swap'}}), ('hitbtc', 'futures', {'options': {'defaultType': 'swap'}}), ('kraken', 'futures', {'options': {'defaultType': 'swap'}}), ('kucoin', 'futures', {'options': {'defaultType': 'swap'}}), ('okx', 'futures', {'options': {'defaultType': 'swap'}})])
def test__ccxt_config(default_conf, mocker, exchange_name, trading_mode, ccxt_config):
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange._ccxt_config == ccxt_config

@pytest.mark.parametrize('pair,nominal_value,max_lev', [('XRP/USDT:USDT', 0.0, 2.0), ('TKN/USDT:USDT', 210.3, 1.0)])
def test_get_max_leverage_from_margin(default_conf, mocker, pair, nominal_value, max_lev):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'margin'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.get_max_leverage(pair, nominal_value) == max_lev

@pytest.mark.parametrize('size,funding_rate,mark_price,time_in_ratio,funding_fee,kraken_fee', [(10, 0.0001, 2.0, 1.0, 0.002, 0.002), (10, 0.0002, 2.0, 0.01, 0.004, 4e-05), (10, 0.0002, nan, None, 0.0, None)])
def test_calculate_funding_fees(default_conf, mocker, size, funding_rate_history_hourly, funding_rate_history_octohourly, rate_start, rate_end, mark_ohlcv, exchange, d1, d2, amount, expected_fees, kraken_fee):
    """
    nominal_value = mark_price * size
    funding_fee = nominal_value * funding_rate
    size: 30
        time: 0, mark: 2.77, nominal_value: 83.1, fundRate: -0.000008, fundFee: -0.0006648
        time: 1, mark: 2.73, nominal_value: 81.9, fundRate: -0.000004, fundFee: -0.0003276
        time: 2, mark: 2.74, nominal_value: 82.2, fundRate: 0.000012, fundFee: 0.0009864
        time: 3, mark: 2.76, nominal_value: 82.8, fundRate: -0.000003, fundFee: -0.0002484
        time: 4, mark: 2.76, nominal_value: 82.8, fundRate: -0.000007, fundFee: -0.0005796
        time: 5, mark: 2.77, nominal_value: 83.1, fundRate: 0.000003, fundFee: 0.0002493
        time: 6, mark: 2.78, nominal_value: 83.4, fundRate: 0.000019, fundFee: 0.0015846
        time: 7, mark: 2.78, nominal_value: 83.4, fundRate: 0.000003, fundFee: 0.0002502
        time: 8, mark: 2.77, nominal_value: 83.1, fundRate: -0.000003, fundFee: -0.0002493
        time: 9, mark: 2.77, nominal_value: 83.1, fundRate: 0, fundFee: 0.0
        time: 10, mark: 2.84, nominal_value: 85.2, fundRate: 0.000013, fundFee: 0.0011076
        time: 11, mark: 2.81, nominal_value: 84.3, fundRate: 0.000077, fundFee: 0.0064911
        time: 12, mark: 2.81, nominal_value: 84.3, fundRate: 0.000072, fundFee: 0.0060696
        time: 13, mark: 2.82, nominal_value: 84.6, fundRate: 0.000097, fundFee: 0.0082062

    size: 50
        time: 0, mark: 2.77, nominal_value: 138.5, fundRate: -0.000008, fundFee: -0.001108
        time: 1, mark: 2.73, nominal_value: 136.5, fundRate: -0.000004, fundFee: -0.000546
        time: 2, mark: 2.74, nominal_value: 137.0, fundRate: 0.000012, fundFee: 0.001644
        time: 3, mark: 2.76, nominal_value: 138.0, fundRate: -0.000003, fundFee: -0.000414
        time: 4, mark: 2.76, nominal_value: 138.0, fundRate: -0.000007, fundFee: -0.000966
        time: 5, mark: 2.77, nominal_value: 138.5, fundRate: 0.000003, fundFee: 0.0004155
        time: 6, mark: 2.78, nominal_value: 139.0, fundRate: 0.000019, fundFee: 0.002641
        time: 7, mark: 2.78, nominal_value: 139.0, fundRate: 0.000003, fundFee: 0.000417
        time: 8, mark: 2.77, nominal_value: 138.5, fundRate: -0.000003, fundFee: -0.0004155
        time: 9, mark: 2.77, nominal_value: 138.5, fundRate: 0, fundFee: 0.0
        time: 10, mark: 2.84, nominal_value: 142.0, fundRate: 0.000013, fundFee: 0.001846
        time: 11, mark: 2.81, nominal_value: 140.5, fundRate: 0.000077, fundFee: 0.0108185
        time: 12, mark: 2.81, nominal_value: 140.5, fundRate: 0.000072, fundFee: 0.010116
        time: 13, mark: 2.82, nominal_value: 141.0, fundRate: 0.000097, fundFee: 0.013677
    """
    d1 = datetime.strptime(f'{d1} +0000', '%Y-%m-%d %H:%M:%S %z')
    d2 = datetime.strptime(f'{d2} +0000', '%Y-%m-%d %H:%M:%S %z')
    funding_rate_history = {'binance': funding_rate_history_octohourly, 'gate': funding_rate_history_octohourly}[exchange][rate_start:rate_end]
    api_mock = MagicMock()
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=funding_rate_history)
    api_mock.fetch_ohlcv = get_mock_coro(return_value=mark_ohlcv)
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    type(api_mock).has = PropertyMock(return_value={'fetchFundingRateHistory': True})
    ex = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    mocker.patch(f'{EXMS}.timeframes', PropertyMock(return_value=['1h', '4h', '8h']))
    funding_fees = ex._fetch_and_calculate_funding_fees(pair='ADA/USDT', amount=amount, is_short=True, open_date=d1, close_date=d2)
    assert pytest.approx(funding_fees) == expected_fees
    funding_fees = ex._fetch_and_calculate_funding_fees(pair='ADA/USDT', amount=amount, is_short=False, open_date=d1, close_date=d2)
    assert pytest.approx(funding_fees) == -expected_fees
    mocker.patch(f'{EXMS}.refresh_latest_ohlcv', return_value={})
    ex = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    with pytest.raises(ExchangeError, match='Could not find funding rates.'):
        ex._fetch_and_calculate_funding_fees(pair='ADA/USDT', amount=amount, is_short=False, open_date=d1, close_date=d2)

@pytest.mark.parametrize('exchange,expected_fees', [('binance', -0.0009140999999999999), ('gate', -0.0009140999999999999)])
def test__fetch_and_calculate_funding_fees_datetime_called(mocker, default_conf, funding_rate_history_octohourly, mark_ohlcv, exchange, time_machine, expected_fees):
    api_mock = MagicMock()
    api_mock.fetch_ohlcv = get_mock_coro(return_value=mark_ohlcv)
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=funding_rate_history_octohourly)
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    type(api_mock).has = PropertyMock(return_value={'fetchFundingRateHistory': True})
    mocker.patch(f'{EXMS}.timeframes', PropertyMock(return_value=['4h', '8h']))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    d1 = datetime.strptime('2021-08-31 23:00:01 +0000', '%Y-%m-%d %H:%M:%S %z')
    time_machine.move_to('2021-09-01 08:00:00 +00:00')
    funding_fees = exchange._fetch_and_calculate_funding_fees('ADA/USDT', 30.0, True, d1)
    assert funding_fees == expected_fees
    funding_fees = exchange._fetch_and_calculate_funding_fees('ADA/USDT', 30.0, False, d1)
    assert funding_fees == 0 - expected_fees

@pytest.mark.parametrize('pair,expected_size,trading_mode', [('XLTCUSDT', 1, 'spot'), ('LTC/USD', 1, 'futures'), ('XLTCUSDT', 0.01, 'futures'), ('ETH/USDT:USDT', 10, 'futures'), ('TORN/USDT:USDT', None, 'futures')])
def test__get_contract_size(mocker, default_conf, pair, expected_size, trading_mode):
    api_mock = MagicMock()
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    mocker.patch(f'{EXMS}.markets', {'LTC/USD': {'symbol': 'LTC/USD', 'contractSize': None}, 'XLTCUSDT': {'symbol': 'XLTCUSDT', 'contractSize': '0.01'}, 'ETH/USDT:USDT': {'symbol': 'ETH/USDT:USDT', 'contractSize': '10'}})
    size = exchange.get_contract_size(pair)
    assert expected_size == size

@pytest.mark.parametrize('pair,param_amount,param_size', [('ADA/USDT:USDT', 40, 4000), ('LTC/ETH', 30, 30), ('LTC/USD', 30, 30), ('ETH/USDT:USDT', 10, 1)])
def test__amount_to_contracts(mocker, default_conf, pair, param_amount, param_size):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'spot'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    mocker.patch(f'{EXMS}.markets', {'LTC/USD': {'symbol': 'LTC/USD', 'contractSize': None}, 'XLTCUSDT': {'symbol': 'XLTCUSDT', 'contractSize': '0.01'}, 'LTC/ETH': {'symbol': 'LTC/ETH'}, 'ETH/USDT:USDT': {'symbol': 'ETH/USDT:USDT', 'contractSize': '10'}})
    result_size = exchange._amount_to_contracts(pair, param_amount)
    assert result_size == param_amount
    result_amount = exchange._contracts_to_amount(pair, param_size)
    assert result_amount == param_size
    default_conf['trading_mode'] = 'futures'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result_size = exchange._amount_to_contracts(pair, param_amount)
    assert result_size == param_size
    result_amount = exchange._contracts_to_amount(pair, param_size)
    assert result_amount == param_amount

@pytest.mark.parametrize('pair,amount,expected_spot,expected_fut', [('ADA/USDT:USDT', 40, 40, 40), ('ADA/USDT:USDT', 10.4445555, 10.4, 10.444), ('LTC/ETH', 30, 30, 30), ('LTC/USD', 30, 30, 30), ('ADA/USDT:USDT', 1.17, 1.1, 1.17), ('ETH/USDT:USDT', 10.111, 10.1, 10), ('ETH/USDT:USDT', 10.188, 10.1, 10), ('ETH/USDT:USDT', 10.988, 10.9, 10)])
def test_amount_to_contract_precision(mocker, default_conf, pair, amount, expected_spot, expected_fut):
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'spot'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result_size = exchange.amount_to_contract_precision(pair, amount)
    assert result_size == expected_spot
    default_conf['trading_mode'] = 'futures'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result_size = exchange.amount_to_contract_precision(pair, amount)
    assert result_size == expected_fut

@pytest.mark.parametrize('exchange_name,open_rate,is_short,trading_mode,margin_mode,leverage,open_rate,amount,expected_liq', [(False, 'spot', 'binance', '', 5.0, 10.0, 1.0, None), (True, 'spot', 'binance', '', 5.0, 10.0, 1.0, None), (False, 'spot', 'gate', '', 5.0, 10.0, 1.0, None), (True, 'spot', 'gate', '', 5.0, 10.0, 1.0, None), (True, 'spot', 'okx', '', 5.0, 10.0, 1.0, None), (True, 'futures', 'binance', 'isolated', 5.0, 10.0, 1.0, 11.89108910891089), (True, 'futures', 'binance', 'isolated', 3.0, 10.0, 1.0, 13.211221122079207), (True, 'futures', 'binance', 'isolated', 5.0, 8.0, 1.0, 9.514851485148514), (True, 'futures', 'binance', 'isolated', 5.0, 10.0, 0.6, 11.897689768976898), (False, 'futures', 'binance', 'isolated', 5, 10, 1.0, 8.070707070707071), (False, 'futures', 'binance', 'isolated', 5, 8, 1.0, 6.454545454545454), (False, 'futures', 'binance', 'isolated', 3, 10, 1.0, 6.723905723905723), (False, 'futures', 'binance', 'isolated', 5, 10, 0.6, 8.063973063973064), (True, 'futures', 'gate', 'isolated', 5, 10, 1.0, 11.87413417771621), (True, 'futures', 'gate', 'isolated', 5, 10, 2.0, 11.87413417771621), (True, 'futures', 'gate', 'isolated', 3, 10, 1.0, 13.193482419684678), (True, 'futures', 'gate', 'isolated', 5, 8, 1.0, 9.499307342172967), (True, 'futures', 'okx', 'isolated', 3, 10, 1.0, 13.193482419684678), (False, 'futures', 'gate', 'isolated', 5.0, 10.0, 1.0, 8.085708510208207), (False, 'futures', 'gate', 'isolated', 3.0, 10.0, 1.0, 6.738090425173506), (False, 'futures', 'okx', 'isolated', 3.0, 10.0, 1.0, 6.738090425173506), (False, 'futures', 'bybit', 'isolated', 1.0, 10.0, 1.0, 0.1), (False, 'futures', 'bybit', 'isolated', 3.0, 10.0, 1.0, 6.7666666), (False, 'futures', 'bybit', 'isolated', 5.0, 10.0, 1.0, 8.1), (False, 'futures', 'bybit', 'isolated', 10.0, 10.0, 1.0, 9.1), (True, 'futures', 'bybit', 'isolated', 1.0, 10.0, 1.0, 19.9), (True, 'futures', 'bybit', 'isolated', 3.0, 10.0, 1.0, 13.233333), (True, 'futures', 'bybit', 'isolated', 5.0, 10.0, 1.0, 11.9), (True, 'futures', 'bybit', 'isolated', 10.0, 10.0, 1.0, 10.9)])
def test_get_liquidation_price(mocker, default_conf_usdt, is_short, trading_mode, exchange_name, margin_mode, leverage, open_rate, amount, expected_liq):
    """
    position = 0.2 * 5
    wb: wallet balance (stake_amount if isolated)
    cum_b: maintenance amount
    side_1: -1 if is_short else 1
    ep1: entry price
    mmr_b: maintenance margin ratio

    Binance, Short
    leverage = 5, open_rate = 10, amount = 1.0
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - ((-1) * 1 * 10)) / ((1 * 0.01) - ((-1) * 1)) = 11.89108910891089
    leverage = 3, open_rate = 10, amount = 1.0
        ((3.3333333333 + 0.01) - ((-1) * 1.0 * 10)) / ((1.0 * 0.01) - ((-1) * 1.0)) = 13.2112211220
    leverage = 5, open_rate = 8, amount = 1.0
        ((1.6 + 0.01) - ((-1) * 1 * 8)) / ((1 * 0.01) - ((-1) * 1)) = 9.514851485148514
    leverage = 5, open_rate = 10, amount = 0.6
        ((1.6 + 0.01) - ((-1) * 0.6 * 10)) / ((0.6 * 0.01) - ((-1) * 0.6)) = 12.557755775577558

    Binance, Long
    leverage = 5, open_rate = 10, amount = 1.0
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - (1 * 1 * 10)) / ((1 * 0.01) - (1 * 1)) = 8.070707070707071
    leverage = 5, open_rate = 8, amount = 1.0
        ((1.6 + 0.01) - (1 * 1 * 8)) / ((1 * 0.01) - (1 * 1)) = 6.454545454545454
    leverage = 3, open_rate = 10, amount = 1.0
        ((2 + 0.01) - (1 * 0.6 * 10)) / ((0.6 * 0.01) - (1 * 0.6)) = 6.717171717171718
    leverage = 5, open_rate = 10, amount = 0.6
        ((1.6 + 0.01) - (1 * 0.6 * 10)) / ((0.6 * 0.01) - (1 * 0.6)) = 7.39057239057239

    Gate/Okx, Short
    leverage = 5, open_rate = 10, amount = 1.0
        (open_rate + (wallet_balance / position)) / (1 + (mm_ratio + taker_fee_rate))
        (10 + (2 / 1.0)) / (1 + (0.01 + 0.0006)) = 11.87413417771621
    leverage = 5, open_rate = 10, amount = 2.0
        (10 + (4 / 2.0)) / (1 + (0.01 + 0.0006)) = 11.87413417771621
    leverage = 3, open_rate = 10, amount = 1.0
        (10 + (3.3333333333333 / 1.0)) / (1 - (0.01 + 0.0006)) = 13.476180850346978
    leverage = 5, open_rate = 8, amount = 1.0
        (8 + (1.6 / 1.0)) / (1 + (0.01 + 0.0006)) = 9.499307342172967

    Gate/Okx, Long
    leverage = 5, open_rate = 10, amount = 1.0
        (open_rate - (wallet_balance / position)) / (1 - (mm_ratio + taker_fee_rate))
        (10 - (2 / 1)) / (1 - (0.01 + 0.0006)) = 8.085708510208207
    leverage = 5, open_rate = 10, amount = 2.0
        (10 - (4 / 2.0)) / (1 + (0.01 + 0.0006)) = 7.916089451810806
    leverage = 3, open_rate = 10, amount = 1.0
        (10 - (3.333333333333333333 / 1.0)) / (1 - (0.01 + 0.0006)) = 6.738090425173506
    leverage = 5, open_rate = 8, amount = 1.0
        (8 - (1.6 / 1.0)) / (1 + (0.01 + 0.0006)) = 6.332871561448645
    """
    default_conf_usdt['liquidation_buffer'] = 0.0
    default_conf_usdt['trading_mode'] = trading_mode
    default_conf_usdt['exchange']['name'] = exchange_name
    default_conf_usdt['margin_mode'] = margin_mode
    mocker.patch('freqtrade.exchange.gate.Gate.validate_ordertypes')
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)
    exchange.get_maintenance_ratio_and_amt = MagicMock(return_value=(0.01, 0.01))
    exchange.name = exchange_name
    liq = exchange.get_liquidation_price(pair='ETH/USDT:USDT', open_rate=open_rate, amount=amount, stake_amount=amount * open_rate / leverage, wallet_balance=amount * open_rate / leverage, leverage=leverage, is_short=is_short, open_trades=[])
    if expected_liq is None:
        assert liq is None
    else:
        buffer_amount = 0.05 * abs(open_rate - expected_liq)
        expected_liq = expected_liq - buffer_amount if is_short else expected_liq + buffer_amount
        assert pytest.approx(expected_liq) == liq

@pytest.mark.parametrize('contract_size,order_amount', [(10, 10), (0.01, 10000)])
def test_stoploss_contract_size(mocker, default_conf, contract_size, order_amount):
    api_mock = MagicMock()
    order_id = f'test_prod_buy_{randint(0, 10 ** 6)}'
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'info': {'foo': 'bar'}, 'amount': order_amount, 'cost': order_amount, 'filled': order_amount, 'remaining': order_amount, 'symbol': 'ETH/BTC'})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_contract_size = MagicMock(return_value=contract_size)
    api_mock.create_order.reset_mock()
    order = exchange.create_stoploss(pair='ETH/BTC', amount=100, stop_price=220, order_types={}, side='buy', leverage=1.0)
    assert api_mock.create_order.call_args_list[0][1]['amount'] == order_amount
    assert order['amount'] == 100
    assert order['cost'] == order_amount
    assert order['filled'] == 100
    assert order['remaining'] == 100

def test_price_to_precision_with_default_conf(default_conf, mocker):
    conf = copy.deepcopy(default_conf)
    patched_ex = get_patched_exchange(mocker, conf)
    prec_price = patched_ex.price_to_precision('XRP/USDT', 1.0000000101)
    assert prec_price == 1.00000001
    assert prec_price == 1.00000001

def test_exchange_features(default_conf, mocker):
    conf = copy.deepcopy(default_conf)
    exchange = get_patched_exchange(mocker, conf)
    exchange._api_async.features = {'spot': {'fetchOHLCV': {'limit': 995}}, 'swap': {'linear': {'fetchOHLCV': {'limit': 997}}}}
    assert exchange.features('spot', 'fetchOHLCV', 'limit', 500) == 995
    assert exchange.features('futures', 'fetchOHLCV', 'limit', 500) == 997
    assert exchange.features('futures', 'fetchOHLCV_else', 'limit', 601) == 601
