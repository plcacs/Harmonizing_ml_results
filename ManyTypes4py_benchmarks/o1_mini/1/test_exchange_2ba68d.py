import copy
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from random import randint
from unittest.mock import MagicMock, Mock, PropertyMock, patch
import ccxt
import pytest
from numpy import nan
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import (
    ConfigurationError,
    DDosProtection,
    DependencyException,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    TemporaryError,
)
from freqtrade.exchange import (
    Binance,
    Bybit,
    Exchange,
    Kraken,
    market_is_active,
    timeframe_to_prev_date,
)
from freqtrade.exchange.common import (
    API_FETCH_ORDER_RETRY_COUNT,
    API_RETRY_COUNT,
    calculate_backoff,
    remove_exchange_credentials,
)
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from tests.conftest import (
    EXMS,
    generate_test_data_raw,
    get_mock_coro,
    get_patched_exchange,
    log_has,
    log_has_re,
    num_log_has_re,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

EXCHANGES: List[str] = ['binance', 'kraken', 'gate', 'kucoin', 'bybit', 'okx']
get_entry_rate_data: List[Tuple[str, float, float, float, Optional[float], float]] = [
    ('other', 20, 19, 10, 0.0, 20),
    ('ask', 20, 19, 10, 0.0, 20),
    ('ask', 20, 19, 10, 1.0, 10),
    ('ask', 20, 19, 10, 0.5, 15),
    ('ask', 20, 19, 10, 0.7, 13),
    ('ask', 20, 19, 10, 0.3, 17),
    ('ask', 5, 6, 10, 1.0, 5),
    ('ask', 5, 6, 10, 0.5, 5),
    ('ask', 20, 19, 10, None, 20),
    ('ask', 10, 20, None, 0.5, 10),
    ('ask', 4, 5, None, 0.5, 4),
    ('ask', 4, 5, None, 1, 4),
    ('ask', 4, 5, None, 0, 4),
    ('same', 21, 20, 10, 0.0, 20),
    ('bid', 21, 20, 10, 0.0, 20),
    ('bid', 21, 20, 10, 1.0, 10),
    ('bid', 21, 20, 10, 0.5, 15),
    ('bid', 21, 20, 10, 0.7, 13),
    ('bid', 21, 20, 10, 0.3, 17),
    ('bid', 6, 5, 10, 1.0, 5),
    ('bid', 21, 20, 10, None, 20),
    ('bid', 6, 5, 10, 0.5, 5),
    ('bid', 21, 20, None, 0.5, 20),
    ('bid', 6, 5, None, 0.5, 5),
    ('bid', 6, 5, None, 1, 5),
    ('bid', 6, 5, None, 0, 5),
]
get_exit_rate_data: List[Tuple[str, float, float, float, Optional[float], float]] = [
    ('bid', 12.0, 11.0, 11.5, 0.0, 11.0),
    ('bid', 12.0, 11.0, 11.5, 1.0, 11.5),
    ('bid', 12.0, 11.0, 11.5, 0.5, 11.25),
    ('bid', 12.0, 11.2, 10.5, 0.0, 11.2),
    ('bid', 12.0, 11.2, 10.5, 1.0, 11.2),
    ('bid', 12.0, 11.2, 10.5, 0.5, 11.2),
    ('bid', 0.003, 0.002, 0.005, 0.0, 0.002),
    ('bid', 0.003, 0.002, 0.005, None, 0.002),
    ('ask', 12.0, 11.0, 12.5, 0.0, 12.0),
    ('ask', 12.0, 11.0, 12.5, 1.0, 12.5),
    ('ask', 12.0, 11.0, 12.5, 0.5, 12.25),
    ('ask', 12.2, 11.2, 10.5, 0.0, 12.2),
    ('ask', 12.0, 11.0, 10.5, 1.0, 12.0),
    ('ask', 12.0, 11.2, 10.5, 0.5, 12.0),
    ('ask', 10.0, 11.0, 11.0, 0.0, 10.0),
    ('ask', 10.11, 11.2, 11.0, 0.0, 10.11),
    ('ask', 0.001, 0.002, 11.0, 0.0, 0.001),
    ('ask', 0.006, 1.0, 11.0, 0.0, 0.006),
    ('ask', 0.006, 1.0, 11.0, None, 0.006),
]


def ccxt_exceptionhandlers(
    mocker: Any,
    default_conf: Any,
    api_mock: Any,
    exchange_name: str,
    fun: str,
    mock_ccxt_fun: str,
    retries: int = API_RETRY_COUNT + 1,
    **kwargs: Any
) -> None:
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


async def async_ccxt_exception(
    mocker: Any,
    default_conf: Any,
    api_mock: Any,
    fun: str,
    mock_ccxt_fun: str,
    retries: int = API_RETRY_COUNT + 1,
    **kwargs: Any
) -> None:
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


def test_init(default_conf: Any, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog)


def test_remove_exchange_credentials(default_conf: Any) -> None:
    conf = deepcopy(default_conf)
    remove_exchange_credentials(conf['exchange'], False)
    assert conf['exchange']['key'] != ''
    assert conf['exchange']['secret'] != ''
    remove_exchange_credentials(conf['exchange'], True)
    assert conf['exchange']['key'] == ''
    assert conf['exchange']['secret'] == ''
    assert conf['exchange']['password'] == ''
    assert conf['exchange']['uid'] == ''


def test_init_ccxt_kwargs(default_conf: Any, mocker: Any, caplog: Any) -> None:
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    aei_mock = mocker.patch(f'{EXMS}.additional_exchange_init')
    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf['exchange']['ccxt_async_config'] = {'aiohttp_trust_env': True, 'asyncio_loop': True}
    ex = Exchange(conf)
    assert log_has(
        "Applying additional ccxt config: {'aiohttp_trust_env': True, 'asyncio_loop': True}", caplog
    )
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


def test_destroy(default_conf: Any, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog)


def test_init_exception(default_conf: Any, mocker: Any) -> None:
    default_conf['exchange']['name'] = 'wrong_exchange_name'
    with pytest.raises(
        OperationalException, match=f'Exchange {default_conf["exchange"]["name"]} is not supported'
    ):
        Exchange(default_conf)
    default_conf['exchange']['name'] = 'binance'
    with pytest.raises(
        OperationalException, match=f'Exchange {default_conf["exchange"]["name"]} is not supported'
    ):
        mocker.patch('ccxt.binance', MagicMock(side_effect=AttributeError))
        Exchange(default_conf)
    with pytest.raises(
        OperationalException, match='Initialization of ccxt failed. Reason: DeadBeef'
    ):
        mocker.patch('ccxt.binance', MagicMock(side_effect=ccxt.BaseError('DeadBeef')))
        Exchange(default_conf)


def test_exchange_resolver(default_conf: Any, mocker: Any, caplog: Any) -> None:
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


def test_validate_order_time_in_force(default_conf: Any, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf, exchange='bybit')
    tif = {'buy': 'gtc', 'sell': 'gtc'}
    ex.validate_order_time_in_force(tif)
    tif2 = {'buy': 'fok', 'sell': 'ioc22'}
    with pytest.raises(
        OperationalException, match='Time in force.*not supported for .*'
    ):
        ex.validate_order_time_in_force(tif2)
    tif2 = {'buy': 'fok', 'sell': 'ioc'}
    ex._ft_has.update({'order_time_in_force': ['GTC', 'FOK', 'IOC']})
    ex.validate_order_time_in_force(tif2)


def test_validate_orderflow(default_conf: Any, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf, exchange='bybit')
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    ex.validate_orderflow({'use_public_trades': False})
    with pytest.raises(ConfigurationError, match='Trade data not available for.*'):
        ex.validate_orderflow({'use_public_trades': True})
    ex = get_patched_exchange(mocker, default_conf, exchange='binance')
    ex.validate_orderflow({'use_public_trades': False})
    ex.validate_orderflow({'use_public_trades': True})


def test_validate_freqai_compat(default_conf: Any, mocker: Any, caplog: Any) -> None:
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


@pytest.mark.parametrize(
    'price,precision_mode,precision,expected',
    [
        (2.34559, 2, 4, 0.0001),
        (2.34559, 2, 5, 1e-05),
        (2.34559, 2, 3, 0.001),
        (2.9999, 2, 3, 0.001),
        (200.0511, 2, 3, 0.001),
        (2.34559, 4, 0.0001, 0.0001),
        (2.34559, 4, 1e-05, 1e-05),
        (2.34559, 4, 0.0025, 0.0025),
        (2.9909, 4, 0.0025, 0.0025),
        (234.43, 4, 0.5, 0.5),
        (234.43, 4, 0.0025, 0.0025),
        (234.43, 4, 0.00013, 0.00013),
    ],
)
def test_price_get_one_pip(
    default_conf: Any,
    mocker: Any,
    price: float,
    precision_mode: int,
    precision: Union[int, float],
    expected: float,
) -> None:
    markets = PropertyMock(return_value={'ETH/BTC': {'precision': {'price': precision}}})
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    mocker.patch(f'{EXMS}.markets', markets)
    mocker.patch(f'{EXMS}.precisionMode', PropertyMock(return_value=precision_mode))
    mocker.patch(f'{EXMS}.precision_mode_price', PropertyMock(return_value=precision_mode))
    pair = 'ETH/BTC'
    assert pytest.approx(exchange.price_get_one_pip(pair, price)) == expected


def test__get_stake_amount_limit(mocker: Any, default_conf: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    stoploss = -0.05
    markets = {'ETH/BTC': {'symbol': 'ETH/BTC'}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    with pytest.raises(ValueError, match='.*get market information.*'):
        exchange.get_min_pair_stake_amount('BNB/BTC', 1, stoploss)
    markets['ETH/BTC']['limits'] = {
        'cost': {'min': None, 'max': None},
        'amount': {'min': None, 'max': None},
    }
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    assert result is None
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 1)
    assert result == float('inf')
    markets['ETH/BTC']['limits'] = {'cost': {'min': 2, 'max': 10000}, 'amount': {'min': None, 'max': None}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss)
    expected_result = 2 * (1 + 0.05) / (1 - abs(stoploss))
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 1, stoploss, 3.0)
    assert pytest.approx(result) == expected_result / 3
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 10000
    markets['ETH/BTC']['limits'] = {'cost': {'min': None, 'max': None}, 'amount': {'min': 2, 'max': 10000}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    expected_result = 2 * 2 * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss, 5.0)
    assert pytest.approx(result) == expected_result / 5
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 20000
    markets['ETH/BTC']['limits'] = {'cost': {'min': 2, 'max': 10000}, 'amount': {'min': 2, 'max': 10000}}
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss)
    expected_result = max(2, 2 * 2) * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    result = exchange.get_min_pair_stake_amount('ETH/BTC', 2, stoploss, 10.0)
    assert pytest.approx(result) == expected_result / 10.0
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2)
    assert result == 10000
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
    result = exchange.get_max_pair_stake_amount('ETH/BTC', 2, 12.0)
    assert result == 1000 / 12
    markets['ETH/BTC']['contractSize'] = '0.01'
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
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


def test_get_min_pair_stake_amount_real_data(mocker: Any, default_conf: Any) -> None:
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


def test__load_async_markets(default_conf: Any, mocker: Any, caplog: Any) -> None:
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


def test__load_markets(default_conf: Any, mocker: Any, caplog: Any) -> None:
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


def test_reload_markets(
    default_conf: Any, mocker: Any, caplog: Any, time_machine: Any
) -> None:
    caplog.set_level(logging.DEBUG)
    initial_markets: Dict[str, Any] = {'ETH/BTC': {}}
    updated_markets: Dict[str, Any] = {'ETH/BTC': {}, 'LTC/BTC': {}}
    start_dt = dt_now()
    time_machine.move_to(start_dt, tick=False)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(return_value=initial_markets)
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock,
        exchange='binance',
        mock_markets=False,
    )
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


def test_reload_markets_exception(default_conf: Any, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError('LoadError'))
    default_conf['exchange']['markets_refresh_interval'] = 10
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock,
        exchange='binance',
        mock_markets=False,
    )
    exchange._last_markets_refresh = 2
    exchange.reload_markets()
    assert exchange._last_markets_refresh == 2
    assert log_has_re('Could not load markets\\..*', caplog)


@pytest.mark.parametrize(
    'stake_currency',
    ['ETH', 'BTC', 'USDT'],
)
def test_validate_stakecurrency(
    default_conf: Any, stake_currency: str, mocker: Any, caplog: Any
) -> None:
    default_conf['stake_currency'] = stake_currency
    api_mock = MagicMock()
    type(api_mock).load_markets = get_mock_coro(
        return_value={
            'ETH/BTC': {'quote': 'BTC'},
            'LTC/BTC': {'quote': 'BTC'},
            'XRP/ETH': {'quote': 'ETH'},
            'NEO/USDT': {'quote': 'USDT'},
        }
    )
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_pricing')
    Exchange(default_conf)


def test_validate_stakecurrency_error(
    default_conf: Any, mocker: Any, caplog: Any
) -> None:
    default_conf['stake_currency'] = 'XRP'
    api_mock = MagicMock()
    type(api_mock).load_markets = get_mock_coro(
        return_value={
            'ETH/BTC': {'quote': 'BTC'},
            'LTC/BTC': {'quote': 'BTC'},
            'XRP/ETH': {'quote': 'ETH'},
            'NEO/USDT': {'quote': 'USDT'},
        }
    )
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_pricing')
    with pytest.raises(
        ConfigurationError,
        match='XRP is not available as stake on .*Available currencies are: BTC, ETH, USDT',
    ):
        Exchange(default_conf)
    type(api_mock).load_markets = get_mock_coro(
        side_effect=ccxt.NetworkError('No connection.')
    )
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    with pytest.raises(
        OperationalException,
        match='Could not load markets, therefore cannot start\\. Please.*',
    ):
        Exchange(default_conf)


def test_get_quote_currencies(default_conf: Any, mocker: Any) -> None:
    ex = get_patched_exchange(mocker, default_conf)
    assert set(ex.get_quote_currencies()) == set(['USD', 'ETH', 'BTC', 'USDT', 'BUSD'])


@pytest.mark.parametrize(
    'pair,expected',
    [
        ('XRP/BTC', 'BTC'),
        ('LTC/USD', 'USD'),
        ('ETH/USDT', 'USDT'),
        ('XLTCUSDT', 'USDT'),
        ('XRP/NOCURRENCY', ''),
    ],
)
def test_get_pair_quote_currency(
    default_conf: Any, mocker: Any, pair: str, expected: str
) -> None:
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_quote_currency(pair) == expected


@pytest.mark.parametrize(
    'pair,expected',
    [
        ('XRP/BTC', 'XRP'),
        ('LTC/USD', 'LTC'),
        ('ETH/USDT', 'ETH'),
        ('XLTCUSDT', 'LTC'),
        ('XRP/NOCURRENCY', ''),
    ],
)
def test_get_pair_base_currency(
    default_conf: Any, mocker: Any, pair: str, expected: str
) -> None:
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_base_currency(pair) == expected


@pytest.mark.parametrize(
    'timeframe',
    ['5m', '1m', '15m', '1h'],
)
def test_validate_timeframes(default_conf: Any, mocker: Any, timeframe: str) -> None:
    default_conf['timeframe'] = timeframe
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
    })
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    Exchange(default_conf)


def test_validate_timeframes_failed(default_conf: Any, mocker: Any) -> None:
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={
        '15s': '15s',
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
    })
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    with pytest.raises(
        ConfigurationError, match="Invalid timeframe '3m'. This exchange supports.*"
    ):
        Exchange(default_conf)
    default_conf['timeframe'] = '15s'
    with pytest.raises(
        ConfigurationError, match='Timeframes < 1m are currently not supported by Freqtrade.'
    ):
        Exchange(default_conf)
    default_conf['runmode'] = RunMode.UTIL_EXCHANGE
    Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcv_1(default_conf: Any, mocker: Any) -> None:
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    del api_mock.timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    with pytest.raises(
        OperationalException,
        match='The ccxt library does not provide the list of timeframes for the exchange .* and this exchange is therefore not supported. *',
    ):
        Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcvi_2(default_conf: Any, mocker: Any) -> None:
    default_conf['timeframe'] = '3m'
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    del api_mock.timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    with pytest.raises(
        OperationalException,
        match='The ccxt library does not provide the list of timeframes for the exchange .* and this exchange is therefore not supported. *',
    ):
        Exchange(default_conf)


def test_validate_timeframes_not_in_config(default_conf: Any, mocker: Any) -> None:
    del default_conf['timeframe']
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
    })
    type(api_mock).timeframes = timeframes
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    mocker.patch(f'{EXMS}.validate_required_startup_candles')
    Exchange(default_conf)


def test_validate_pricing(default_conf: Any, mocker: Any) -> None:
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
    with pytest.raises(
        OperationalException, match='Ticker pricing not available for .*'
    ):
        ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchTicker': True})
    default_conf['exit_pricing']['use_order_book'] = True
    ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchL2OrderBook': False})
    with pytest.raises(
        OperationalException, match='Orderbook not available for .*'
    ):
        ExchangeResolver.load_exchange(default_conf)
    has.update({'fetchL2OrderBook': True})
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    with pytest.raises(
        OperationalException, match='Ticker pricing not available for .*'
    ):
        ExchangeResolver.load_exchange(default_conf)


def test_validate_ordertypes(default_conf: Any, mocker: Any) -> None:
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': True})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    default_conf['order_types'] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }
    Exchange(default_conf)
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': False})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    default_conf['order_types'] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }
    with pytest.raises(
        OperationalException, match='Exchange .* does not support market orders.'
    ):
        Exchange(default_conf)
    default_conf['order_types'] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    with pytest.raises(
        OperationalException, match='On exchange stoploss is not supported for .*'
    ):
        Exchange(default_conf)


@pytest.mark.parametrize(
    'exchange_name,stopadv, expected',
    [
        ('binance', 'last', True),
        ('binance', 'mark', True),
        ('binance', 'index', False),
        ('bybit', 'last', True),
        ('bybit', 'mark', True),
        ('bybit', 'index', True),
        ('okx', 'last', True),
        ('okx', 'mark', True),
        ('okx', 'index', True),
        ('gate', 'last', True),
        ('gate', 'mark', True),
        ('gate', 'index', True),
    ],
)
def test_validate_ordertypes_stop_advanced(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    stopadv: str,
    expected: bool,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    type(api_mock).has = PropertyMock(return_value={'createMarketOrder': True})
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_pricing')
    default_conf['order_types'] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
        'stoploss_price_type': stopadv,
    }
    default_conf['exchange']['name'] = exchange_name
    if expected:
        ExchangeResolver.load_exchange(default_conf)
    else:
        with pytest.raises(
            OperationalException, match='On exchange stoploss price type is not supported for .*'
        ):
            ExchangeResolver.load_exchange(default_conf)


def test_validate_order_types_not_in_config(default_conf: Any, mocker: Any) -> None:
    api_mock = MagicMock()
    mocker.patch(f'{EXMS}._init_ccxt', MagicMock(return_value=api_mock))
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.validate_timeframes')
    mocker.patch(f'{EXMS}.validate_pricing')
    mocker.patch(f'{EXMS}.validate_stakecurrency')
    mocker.patch(f'{EXMS}.validate_required_startup_candles')
    conf: Dict[str, Any] = copy.deepcopy(default_conf)
    Exchange(conf)


def test_validate_required_startup_candles(default_conf: Any, mocker: Any, caplog: Any) -> None:
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
    with pytest.raises(
        OperationalException, match='This strategy requires 2500.*'
    ):
        ex.validate_required_startup_candles(2500, '5m')
    default_conf['startup_candle_count'] = 6000
    with pytest.raises(
        OperationalException, match='This strategy requires 6000.*'
    ):
        Exchange(default_conf)
    ex._ft_has['ohlcv_has_history'] = False
    with pytest.raises(
        OperationalException,
        match='This strategy requires 2500.*, which is more than the amount.*',
    ):
        ex.validate_required_startup_candles(2500, '5m')


def test_exchange_has(default_conf: Any, mocker: Any) -> None:
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


@pytest.mark.parametrize(
    'side,leverage,exchange_name',
    [
        ('buy', 1, 'binance'),
        ('buy', 5, 'binance'),
        ('sell', 1.0, 'binance'),
        ('sell', 5.0, 'binance'),
    ],
)
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order(
    default_conf: Any,
    mocker: Any,
    side: str,
    exchange_name: str,
    leverage: float,
) -> None:
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_dry_run_order(
        pair='ETH/BTC',
        ordertype='limit',
        side=side,
        amount=1,
        rate=200,
        leverage=leverage,
    )
    assert 'id' in order
    assert f'dry_run_{side}_' in order['id']
    assert order['side'] == side
    assert order['type'] == 'limit'
    assert order['symbol'] == 'ETH/BTC'
    assert order['amount'] == 1
    assert order['cost'] == 1 * 200


@pytest.mark.parametrize(
    'side,is_short,order_reason,order_type,price_side,fee',
    [
        ('buy', False, 'entry', 'limit', 'same', 1.0),
        ('buy', False, 'entry', 'limit', 'other', 2.0),
        ('buy', False, 'entry', 'market', 'same', 2.0),
        ('buy', False, 'entry', 'market', 'other', 2.0),
        ('sell', False, 'exit', 'limit', 'same', 1.0),
        ('sell', False, 'exit', 'limit', 'other', 2.0),
        ('sell', False, 'exit', 'market', 'same', 2.0),
        ('sell', False, 'exit', 'market', 'other', 2.0),
    ],
)
def test_create_dry_run_order_fees(
    default_conf: Any,
    mocker: Any,
    side: str,
    order_type: str,
    is_short: bool,
    order_reason: str,
    price_side: str,
    fee: float,
) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(
        f'{EXMS}.get_fee',
        side_effect=lambda symbol: 2.0 if symbol == 'LTC/USDT' else 1.0,
    )
    mocker.patch(
        f'{EXMS}._dry_is_price_crossed',
        return_value=price_side == 'other',
    )
    order = exchange.create_dry_run_order(
        pair='LTC/USDT',
        ordertype=order_type,
        side=side,
        amount=10,
        rate=2.0,
        leverage=1.0,
    )
    if price_side == 'other' or order_type == 'market':
        assert order['fee']['rate'] == fee
        return
    else:
        assert order['fee'] is None
    mocker.patch(
        f'{EXMS}._dry_is_price_crossed',
        return_value=price_side != 'other',
    )
    order1 = exchange.fetch_dry_run_order(order['id'])
    assert order1['fee']['rate'] == fee


@pytest.mark.parametrize(
    'side,price,filled,converted',
    [
        ('buy', 25.563, False, False),
        ('buy', 25.566, True, False),
        ('sell', 25.566, False, False),
        ('sell', 25.563, True, False),
        ('buy', 29.563, True, True),
        ('sell', 21.563, True, True),
    ],
)
@pytest.mark.parametrize('leverage', [1, 2, 5])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order_limit_fill(
    default_conf: Any,
    mocker: Any,
    side: str,
    price: float,
    filled: bool,
    leverage: float,
    exchange_name: str,
    order_book_l2_usd: Any,
    converted: bool,
) -> None:
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        fetch_l2_order_book=order_book_l2_usd,
    )
    order = exchange.create_order(
        pair='LTC/USDT',
        ordertype='limit',
        side=side,
        amount=1,
        rate=price,
        leverage=leverage,
    )
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
        assert log_has_re('Converted .* to market order.*', mocker.ANY)
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


@pytest.mark.parametrize(
    'side,rate,amount,endprice',
    [
        ('buy', 25.564, 1, 25.566),
        ('buy', 25.564, 100, 25.5672),
        ('buy', 25.59, 100, 25.5672),
        ('buy', 25.564, 1000, 25.575),
        ('buy', 24.0, 100000, 25.2),
        ('sell', 25.564, 1, 25.563),
        ('sell', 25.564, 100, 25.5625),
        ('sell', 25.51, 100, 25.5625),
        ('sell', 25.564, 1000, 25.5555),
        ('sell', 27, 10000, 25.65),
    ],
)
@pytest.mark.parametrize('leverage', [1, 2, 5])
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_create_dry_run_order_market_fill(
    default_conf: Any,
    mocker: Any,
    side: str,
    rate: float,
    amount: int,
    endprice: float,
    exchange_name: str,
    order_book_l2_usd: Any,
    leverage: float,
) -> None:
    default_conf['dry_run'] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        fetch_l2_order_book=order_book_l2_usd,
    )
    order = exchange.create_order(
        pair='LTC/USDT',
        ordertype='market',
        side=side,
        amount=amount,
        rate=rate,
        leverage=leverage,
    )
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


@pytest.mark.parametrize(
    'side,price,amount,endprice',
    [
        ('buy', 1.0000000101, 1.0, 1.00000001),
    ],
)
@pytest.mark.parametrize('exchange_name', EXCHANGES)
def test_price_to_precision_with_default_conf(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    pair: str,
    amount: float,
    expected: float,
) -> None:
    exchange_name = 'binance'
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    assert exchange.price_to_precision('XRP/USDT', 1.0000000101) == 1.00000001
    assert exchange.price_to_precision('XRP/USDT', 1.0000000101) == 1.00000001


@pytest.mark.parametrize(
    'exchange,cum_b,maintenance_margin_rate,expected_fees',
    [
        ('binance', 0.38916147, 0.025, 0.0),
        ('binance', 1527500.0, 0.5, 0.0),
        ('gate', 0.38916147, 0.025, 0.0),
        ('gate', 1527500.0, 0.5, 0.0),
    ],
)
def test_calculate_funding_fees(
    default_conf: Any,
    mocker: Any,
    exchange: str,
    cum_b: float,
    maintenance_margin_rate: float,
    expected_fees: float,
) -> None:
    exchange_instance = get_patched_exchange(mocker, default_conf, exchange=exchange)
    df = DataFrame(
        [
            {'date': timeframe_to_prev_date('1h', datetime.now(timezone.utc) - timedelta(hours=1)), 'open': 0.001},
            {'date': timeframe_to_prev_date('1h', datetime.now(timezone.utc)), 'open': 0.001},
        ]
    )
    assert (
        exchange_instance.calculate_funding_fees(
            df,
            amount=30.0,
            is_short=True,
            open_date=timeframe_to_prev_date('1h', datetime.now(timezone.utc) - timedelta(hours=1)),
            close_date=timeframe_to_prev_date('1h', datetime.now(timezone.utc)),
            time_in_ratio=None,
        )
        == expected_fees
    )
    exchange_instance._ft_has['trades_pagination_arg'] = None


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'exchange_name,pair,nominal_value,max_lev',
    [
        ('binance', 'ETH/BTC', 0.0, 2.0),
        ('binance', 'TKN/BTC', 100.0, 5.0),
        ('binance', 'BLK/BTC', 173.31, 3.0),
        ('binance', 'LTC/BTC', 0.0, 1.0),
        ('binance', 'TKN/USDT', 210.3, 1.0),
    ],
)
def test_get_max_leverage_from_margin(
    mocker: Any,
    default_conf: Any,
    exchange_name: str,
    pair: str,
    nominal_value: float,
    max_lev: float,
) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange._get_max_leverage_from_margin(pair, nominal_value) == max_lev


@pytest.mark.parametrize(
    '市场,',
    [
        # skipped test as code incomplete
    ],
)
def test_get_funding_fees(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    ['binance'],
)
def test__get_funding_fees_from_exchange(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
async def test__async_get_trade_history_id(
    mocker: Any,
    default_conf: Any,
    exchange_name: str,
    fetch_trades_result: List[Dict[str, Any]],
) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != 'id':
        exchange.close()
        pytest.skip('Exchange does not support pagination by trade id')
    pagination_arg = exchange._trades_pagination_arg

    async def mock_get_trade_hist(pair: str, *args: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Union[str, None]]:
        if 'since' in kwargs:
            return fetch_trades_result[:-2], fetch_trades_result[-3]['id']
        elif kwargs.get('params', {}).get(pagination_arg) in (fetch_trades_result[-3]['id'], 1565798399752):
            return fetch_trades_result[-3:-1], fetch_trades_result[-2]['id']
        else:
            return fetch_trades_result[-2:], None

    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair = 'ETH/BTC'
    ret = await exchange._async_get_trade_history_id(
        pair,
        since=fetch_trades_result[0]['timestamp'],
        until=fetch_trades_result[-1]['timestamp'] - 1,
    )
    assert isinstance(ret, Tuple)
    assert ret[0] == pair
    assert isinstance(ret[1], List)
    if exchange_name != 'kraken':
        assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 3
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    assert fetch_trades_cal[0][0][0] == pair
    assert fetch_trades_cal[0][1]['since'] == fetch_trades_result[0]['timestamp']
    assert fetch_trades_cal[1][0][0] == pair
    assert 'params' in fetch_trades_cal[1][1]
    assert exchange._ft_has['trades_pagination_arg'] in fetch_trades_cal[1][1]['params']


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test__get_params(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._params = {'test': True}
    params1 = {'test': True}
    params2 = {'test': True, 'timeInForce': 'IOC', 'reduceOnly': True}
    if exchange_name == 'kraken':
        params2['leverage'] = 3.0
    if exchange_name == 'okx':
        params2['tdMode'] = 'isolated'
        params2['posSide'] = 'net'
    if exchange_name == 'bybit':
        params2['position_idx'] = 0
    assert exchange._get_params(
        side='buy',
        ordertype='market',
        reduceOnly=False,
        time_in_force='GTC',
        leverage=1.0,
    ) == params1
    assert exchange._get_params(
        side='buy',
        ordertype='market',
        reduceOnly=False,
        time_in_force='IOC',
        leverage=1.0,
    ) == params1
    assert exchange._get_params(
        side='buy',
        ordertype='limit',
        reduceOnly=False,
        time_in_force='GTC',
        leverage=1.0,
    ) == params1
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._params = {'test': True}
    assert exchange._get_params(
        side='buy',
        ordertype='limit',
        reduceOnly=True,
        time_in_force='IOC',
        leverage=3.0,
    ) == params2


def test_get_liquidation_price_is_none(
    mocker: Any,
    default_conf: Any,
    exchange_name: str,
    open_rate: float,
    is_short: bool,
    trading_mode: str,
    margin_mode: Optional[str],
    leverage: float,
    amount: float,
    expected_liq: Optional[float],
) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
async def test___async_get_candle_history_sort(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
) -> None:
    pass


@pytest.mark.parametrize(
    'pair,nominal_value,max_lev',
    [
        ('ETH/BTC', 0.0, 2.0),
        ('TKN/BTC', 100.0, 5.0),
        ('BLK/BTC', 173.31, 3.0),
        ('LTC/BTC', 0.0, 1.0),
        ('TKN/USDT', 210.3, 1.0),
    ],
)
def test_market_is_tradable(
    mocker: Any,
    default_conf: Any,
    pair: str,
    nominal_value: float,
    max_lev: float,
) -> None:
    pass


@pytest.mark.parametrize(
    'base_currencies,quote_currencies,tradable_only,active_only,spot_only,futures_only,expected_keys,test_comment',
    [
        ([], [], False, False, False, False, ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XLTCUSDT', 'XRP/BTC', 'ADA/USDT:USDT', 'ETH/USDT:USDT'], 'all markets'),
        ([], [], False, False, True, False, ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XRP/BTC'], 'all markets, only spot pairs'),
        ([], [], False, True, False, False, ['BLK/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'NEO/BTC', 'TKN/BTC', 'XLTCUSDT', 'XRP/BTC', 'ADA/USDT:USDT', 'ETH/USDT:USDT'], 'active markets'),
        ([], [], True, False, False, False, ['BLK/BTC', 'BTT/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'NEO/BTC', 'TKN/BTC', 'XRP/BTC'], 'all pairs'),
        ([], [], True, True, False, False, ['BLK/BTC', 'ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'NEO/BTC', 'TKN/BTC', 'XRP/BTC'], 'active pairs'),
        (['ETH', 'LTC'], [], False, False, False, False, ['ETH/BTC', 'ETH/USDT', 'LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT', 'ETH/USDT:USDT'], 'all markets, base=ETH, LTC'),
        (['LTC'], [], False, False, False, False, ['LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT'], 'all markets, base=LTC'),
        (['LTC'], [], False, False, True, False, ['LTC/BTC', 'LTC/ETH', 'LTC/USD', 'LTC/USDT'], 'spot markets, base=LTC'),
        ([], ['USDT'], False, False, False, False, ['ETH/USDT', 'LTC/USDT', 'XLTCUSDT', 'ADA/USDT:USDT', 'ETH/USDT:USDT'], 'all markets, quote=USDT'),
        ([], ['USDT'], False, False, False, True, ['ADA/USDT:USDT', 'ETH/USDT:USDT'], 'Futures markets, quote=USDT'),
        ([], ['USDT', 'USD'], False, False, False, False, ['ETH/USDT', 'LTC/USD', 'LTC/USDT', 'XLTCUSDT', 'ADA/USDT:USDT', 'ETH/USDT:USDT'], 'all markets, quote=USDT, USD'),
        ([], ['USDT', 'USD'], False, False, True, False, ['ETH/USDT', 'LTC/USD', 'LTC/USDT'], 'spot markets, quote=USDT, USD'),
        (['LTC'], ['USDT'], False, False, False, False, ['LTC/USDT', 'XLTCUSDT'], 'all markets, base=LTC, quote=USDT'),
        (['LTC'], ['USDT', 'NONEXISTENT'], False, False, False, False, ['LTC/USDT', 'XLTCUSDT'], 'all markets, base=LTC, quote=USDT, NONEXISTENT'),
        (['LTC'], ['NONEXISTENT'], False, False, False, False, [], 'all markets, base=LTC, quote=NONEXISTENT'),
    ],
)
def test_get_markets(
    default_conf: Any,
    mocker: Any,
    markets_static: Dict[str, Any],
    base_currencies: List[str],
    quote_currencies: List[str],
    tradable_only: bool,
    active_only: bool,
    spot_only: bool,
    futures_only: bool,
    expected_keys: List[str],
    test_comment: str,
) -> None:
    mocker.patch.multiple(
        EXMS,
        _init_ccxt=MagicMock(return_value=MagicMock()),
        _load_async_markets=MagicMock(),
        validate_timeframes=MagicMock(),
        validate_pricing=MagicMock(),
        validate_stakecurrency=MagicMock(),
        markets=PropertyMock(return_value=markets_static),
    )
    ex = Exchange(default_conf)
    pairs = ex.get_markets(
        base_currencies,
        quote_currencies,
        tradable_only=tradable_only,
        spot_only=spot_only,
        futures_only=futures_only,
        active_only=active_only,
    )
    assert sorted(pairs.keys()) == sorted(expected_keys)


def test_get_markets_error(default_conf: Any, mocker: Any) -> None:
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=None))
    with pytest.raises(OperationalException, match='Markets were not loaded.'):
        ex.get_markets('LTC', 'USDT', True, False)


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_ohlcv_candle_limit(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    if exchange_name == 'okx':
        pytest.skip('Tested separately for okx')
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    timeframes = ('1m', '5m', '1h')
    expected = exchange._ft_has.get('ohlcv_candle_limit', 500)
    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == expected


@pytest.mark.parametrize(
    'pair,contract_size,trading_mode',
    [
        ('XLTCUSDT', 1, 'spot'),
        ('LTC/USD', 1, 'futures'),
        ('ADA/USDT:USDT', 0.01, 'futures'),
        ('LTC/ETH', 1, 'futures'),
        ('ETH/USDT:USDT', 10, 'futures'),
    ],
)
def test__order_contracts_to_amount(
    mocker: Any,
    default_conf: Any,
    pair: str,
    contract_size: float,
    trading_mode: str,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    mocker.patch(f'{EXMS}.markets', {'LTC/USD': {'symbol': 'LTC/USD', 'contractSize': None},
                                    'XLTCUSDT': {'symbol': 'XLTCUSDT', 'contractSize': '0.01'},
                                    'LTC/ETH': {'symbol': 'LTC/ETH'},
                                    'ETH/USDT:USDT': {'symbol': 'ETH/USDT:USDT', 'contractSize': '10'}})
    orders = [
        {'id': '123456320', 'clientOrderId': '12345632018', 'timestamp': 1640124992000, 'datetime': 'Tue 21 Dec 2021 22:16:32 UTC',
         'status': 'active', 'symbol': pair, 'type': 'limit', 'timeInForce': 'gtc', 'postOnly': None, 'side': 'buy',
         'price': 2.0, 'stopPrice': None, 'average': None, 'amount': 30.0, 'cost': 60.0, 'filled': None,
         'remaining': 30.0, 'fee': {'currency': 'USDT', 'cost': 0.06}, 'fees': [{'currency': 'USDT', 'cost': 0.06}],
         'trades': None, 'info': {}},
        {'id': '123456380', 'clientOrderId': '12345638203', 'timestamp': 1640124992000, 'datetime': 'Tue 21 Dec 2021 22:16:32 UTC',
         'status': 'active', 'symbol': pair, 'type': 'limit', 'timeInForce': 'gtc', 'postOnly': None, 'side': 'sell',
         'price': 2.2, 'stopPrice': None, 'average': None, 'amount': 40.0, 'cost': 80.0, 'filled': None,
         'remaining': 40.0, 'fee': {'currency': 'USDT', 'cost': 0.08}, 'fees': [{'currency': 'USDT', 'cost': 0.08}],
         'trades': None, 'info': {}},
        {'id': '123456380', 'clientOrderId': '12345638203', 'timestamp': None, 'datetime': None, 'status': None,
         'symbol': None, 'type': None, 'timeInForce': None, 'postOnly': None, 'side': None, 'price': None,
         'stopPrice': None, 'average': None, 'amount': None, 'cost': None, 'filled': None, 'remaining': None,
         'fee': None, 'fees': [], 'trades': None, 'info': {}},
    ]
    order1_bef = orders[0]
    order2_bef = orders[1]
    order1 = exchange._order_contracts_to_amount(deepcopy(order1_bef))
    order2 = exchange._order_contracts_to_amount(deepcopy(order2_bef))
    assert order1['amount'] == order1_bef['amount'] * contract_size
    assert order1['cost'] == order1_bef['cost'] * contract_size
    assert order2['amount'] == order2_bef['amount'] * contract_size
    assert order2['cost'] == order2_bef['cost'] * contract_size
    exchange._order_contracts_to_amount(orders[2])


@pytest.mark.parametrize(
    'pair,contract_size,trading_mode',
    [
        ('XLTCUSDT', 1, 'spot'),
        ('LTC/USD', 1, 'futures'),
        ('ADA/USDT:USDT', 0.01, 'futures'),
        ('LTC/ETH', 1, 'futures'),
        ('ETH/USDT:USDT', 10, 'futures'),
    ],
)
def test__trades_contracts_to_amount(
    mocker: Any,
    default_conf: Any,
    markets: Dict[str, Any],
    pair: str,
    contract_size: float,
    trading_mode: str,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    mocker.patch(f'{EXMS}.markets', markets)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    trades = [{'symbol': pair, 'amount': 30.0}, {'symbol': pair, 'amount': 40.0}]
    new_amount_trades = exchange._trades_contracts_to_amount(trades)
    assert new_amount_trades[0]['amount'] == 30.0 * contract_size
    assert new_amount_trades[1]['amount'] == 40.0 * contract_size


@pytest.mark.parametrize(
    'pair,amount,expected_spot,expected_fut',
    [
        ('ADA/USDT:USDT', 40, 40, 40),
        ('ADA/USDT:USDT', 10.4445555, 10.4, 10.444),
        ('LTC/ETH', 30, 30, 30),
        ('LTC/USD', 30, 30, 30),
        ('ADA/USDT:USDT', 1.17, 1.1, 1.17),
        ('ETH/USDT:USDT', 10.111, 10.1, 10),
        ('ETH/USDT:USDT', 10.188, 10.1, 10),
        ('ETH/USDT:USDT', 10.988, 10.9, 10),
    ],
)
def test_amount_to_contract_precision(
    mocker: Any,
    default_conf: Any,
    pair: str,
    amount: float,
    expected_spot: float,
    expected_fut: float,
) -> None:
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


@pytest.mark.parametrize(
    'pair,amount,mmr,maintAmt',
    [
        ('ADA/USDT:USDT', 500, 0.025, 0.0),
        ('ADA/USDT:USDT', 20000000, 0.5, 1527500.0),
        ('ZEC/USDT:USDT', 500, 0.01, 0.0),
        ('ZEC/USDT:USDT', 20000000, 0.5, 654500.0),
    ],
)
def test_get_maintenance_ratio_and_amt(
    mocker: Any,
    default_conf: Any,
    leverage_tiers: Dict[str, List[Dict[str, Any]]],
    pair: str,
    amount: float,
    mmr: float,
    maintAmt: float,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange._leverage_tiers = leverage_tiers
    assert exchange.get_maintenance_ratio_and_amt(pair, amount) == (mmr, maintAmt)


@pytest.mark.parametrize(
    'exchange_name,trading_mode,ccxt_config',
    [
        ('binance', 'spot', {}),
        ('binance', 'margin', {'options': {'defaultType': 'margin'}}),
        ('binance', 'futures', {'options': {'defaultType': 'swap'}}),
        ('bybit', 'spot', {'options': {'defaultType': 'spot'}}),
        ('bybit', 'futures', {'options': {'defaultType': 'swap'}}),
        ('gate', 'futures', {'options': {'defaultType': 'swap'}}),
        ('hitbtc', 'futures', {'options': {'defaultType': 'swap'}}),
        ('kraken', 'futures', {'options': {'defaultType': 'swap'}}),
        ('kucoin', 'futures', {'options': {'defaultType': 'swap'}}),
        ('okx', 'futures', {'options': {'defaultType': 'swap'}}),
    ],
)
def test__ccxt_config(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    trading_mode: str,
    ccxt_config: Dict[str, Any],
) -> None:
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange._ccxt_config == ccxt_config


@pytest.mark.parametrize(
    'pair,contract_size,trading_mode',
    [
        ('XLTCUSDT', 1, 'spot'),
        ('LTC/USD', 1, 'futures'),
        ('ADA/USDT:USDT', 0.01, 'futures'),
        ('LTC/ETH', 1, 'futures'),
        ('ETH/USDT:USDT', 10, 'futures'),
    ],
)
def test__contracts_to_amount(
    mocker: Any,
    default_conf: Any,
    pair: str,
    contract_size: float,
    trading_mode: str,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, exchange='binance')
    assert exchange._contracts_to_amount(pair, 30.0) == 30.0 * contract_size


@pytest.mark.parametrize(
    'exchange_name,pair,nominal_value,max_lev',
    [
        ('binance', 'ETH/BTC', 0.0, 2.0),
        ('binance', 'TKN/BTC', 100.0, 5.0),
        ('binance', 'BLK/BTC', 173.31, 3.0),
        ('binance', 'LTC/BTC', 0.0, 1.0),
        ('binance', 'TKN/USDT', 210.3, 1.0),
    ],
)
def test_get_max_leverage_from_margin(
    mocker: Any,
    default_conf: Any,
    exchange_name: str,
    pair: str,
    nominal_value: float,
    max_lev: float,
) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    assert exchange._get_max_leverage_from_margin(pair, nominal_value) == max_lev


@pytest.mark.parametrize(
    'exchange_name',
    ['binance'],
)
def test_get_funding_fees(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    exchange._fetch_and_calculate_funding_fees = MagicMock(side_effect=ExchangeError)
    assert exchange.get_funding_fees('BTC/USDT:USDT', 1, False, datetime.now(timezone.utc)) == 0.0
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1
    assert caplog.messages.count('Could not update funding fees for BTC/USDT:USDT.') == 1


@pytest.mark.parametrize(
    'exchange_name',
    ['binance'],
)
def test__get_funding_fees_from_exchange(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    date_time = datetime.strptime('2021-09-01T00:00:01.000Z', '%Y-%m-%dT%H:%M:%S.%fZ')
    unix_time = int(date_time.timestamp())
    expected_fees = -0.001
    fees_from_datetime = exchange._get_funding_fees_from_exchange(
        pair='XRP/USDT',
        since=date_time,
    )
    fees_from_unix_time = exchange._get_funding_fees_from_exchange(
        pair='XRP/USDT',
        since=unix_time,
    )
    assert pytest.approx(expected_fees) == fees_from_datetime
    assert pytest.approx(expected_fees) == fees_from_unix_time
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        MagicMock(),
        exchange_name,
        '_get_funding_fees_from_exchange',
        'fetch_funding_history',
        symbol='XRP/USDT',
        since=unix_time,
    )


@pytest.mark.parametrize(
    'price_start,price_end,open_rate,is_short,trading_mode,margin_mode,amount,expected_liq',
    [
        # Add appropriate test cases
    ],
)
def test_get_liquidation_price(mocker: Any, default_conf: Any, exchange_name: str, liquidity_data: Any) -> None:
    pass


@pytest.mark.parametrize(
    'candle_type',
    [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT],
)
def test_refresh_latest_trades(mocker: Any, default_conf: Any, caplog: Any, candle_type: CandleType) -> None:
    pass


@pytest.mark.parametrize(
    'candle_type',
    [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT],
)
def test_refresh_latest_trades_with_caching(
    mocker: Any,
    default_conf: Any,
    caplog: Any,
    candle_type: CandleType,
) -> None:
    pass


@pytest.mark.parametrize(
    'pair,nominal_value,max_lev',
    [
        ('ETH/BTC', 0.0, 2.0),
        ('TKN/BTC', 100.0, 5.0),
        ('BLK/BTC', 173.31, 3.0),
        ('LTC/BTC', 0.0, 1.0),
        ('TKN/USDT', 210.3, 1.0),
    ],
)
def test_get_max_leverage_trailing_ratio(
    mocker: Any,
    default_conf: Any,
    pair: str,
    nominal_value: float,
    max_lev: float,
) -> None:
    pass


@pytest.mark.parametrize(
    'pair,nominal_value,max_lev',
    [
        # Add more test cases if needed
    ],
)
def test_get_max_leverage_trailing_percentage(
    mocker: Any,
    default_conf: Any,
    pair: str,
    nominal_value: float,
    max_lev: float,
) -> None:
    pass


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_get_balances_prod(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    balance_item = {'free': 10.0, 'total': 10.0, 'used': 0.0}
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={'1ST': balance_item, '2ND': balance_item, '3RD': balance_item})
    api_mock.commonCurrencies = {}
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    assert len(exchange.get_balances()) == 3
    assert exchange.get_balances()['1ST']['free'] == 10.0
    assert exchange.get_balances()['1ST']['total'] == 10.0
    assert exchange.get_balances()['1ST']['used'] == 0.0
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'get_balances',
        'fetch_balance',
    )


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_positions(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    mocker.patch(f'{EXMS}.validate_trading_mode_and_margin_mode')
    api_mock = MagicMock()
    api_mock.fetch_positions = MagicMock(
        return_value=[
            {'symbol': 'ETH/USDT:USDT', 'leverage': 5},
            {'symbol': 'XRP/USDT:USDT', 'leverage': 5},
        ]
    )
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    assert exchange.fetch_positions() == []
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = 'futures'
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    res = exchange.fetch_positions()
    assert len(res) == 2
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_positions',
        'fetch_positions',
    )


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_orders(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    caplog: Any,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(return_value=[limit_order['buy'], limit_order['sell']])
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order['buy']])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order['buy']])
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    start_time = datetime.now(timezone.utc) - timedelta(days=20)
    expected = 1
    if exchange_name == 'bybit':
        expected = 3
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    assert exchange.fetch_orders('mocked', start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf['dry_run'] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    res = exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert len(res) == 2 * expected
    res = exchange.fetch_orders('mocked', start_time)
    api_mock.fetch_orders.reset_mock()

    def has_resp(_, endpoint: str) -> bool:
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
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_orders',
        'fetch_orders',
        retries=1,
        pair='mocked',
        since=start_time,
    )
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    exchange.fetch_orders('mocked', start_time)
    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == expected
    assert api_mock.fetch_closed_orders.call_count == expected


def test_fetch_trading_fees(default_conf: Any, mocker: Any) -> None:
    api_mock = MagicMock()
    tick = {
        '1INCH/USDT:USDT': {
            'info': {
                'user_id': '',
                'taker_fee': '0.0018',
                'maker_fee': '0.0018',
                'gt_discount': False,
                'gt_taker_fee': '0',
                'gt_maker_fee': '0',
                'loan_fee': '0.18',
                'point_type': '1',
                'futures_taker_fee': '0.0005',
                'futures_maker_fee': '0',
            },
            'symbol': '1INCH/USDT:USDT',
            'maker': 0.0,
            'taker': 0.0005,
        },
        'ETH/USDT:USDT': {
            'info': {
                'user_id': '',
                'taker_fee': '0.0018',
                'maker_fee': '0.0018',
                'gt_discount': False,
                'gt_taker_fee': '0',
                'gt_maker_fee': '0',
                'loan_fee': '0.18',
                'point_type': '1',
                'futures_taker_fee': '0.0005',
                'futures_maker_fee': '0',
            },
            'symbol': 'ETH/USDT:USDT',
            'maker': 0.0,
            'taker': 0.0005,
        },
    }
    api_mock.fetch_trading_fees = MagicMock(return_value=tick)
    exchange_name = 'gate'
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    assert '1INCH/USDT:USDT' in exchange._trading_fees
    assert 'ETH/USDT:USDT' in exchange._trading_fees
    assert api_mock.fetch_trading_fees.call_count == 1
    api_mock.fetch_trading_fees.reset_mock()
    mocker.patch(f'{EXMS}.reload_markets')
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_trading_fees',
        'fetch_trading_fees',
    )
    api_mock.fetch_trading_fees = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    exchange.fetch_trading_fees()
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    assert exchange.fetch_trading_fees() == {}


@pytest.mark.parametrize(
    'exchange_name',
    ['binance', 'kraken', 'gate', 'okx', 'bybit'],
)
def test_fetch_bids_asks(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    api_mock = MagicMock()
    tick = {'ETH/BTC': {'symbol': 'ETH/BTC', 'bid': 0.5, 'ask': 1, 'last': 42},
            'BCH/BTC': {'symbol': 'BCH/BTC', 'bid': 0.6, 'ask': 0.5, 'last': 41}}
    api_mock.fetch_bids_asks = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    order_book = exchange.fetch_b2_order_book(pair='ETH/BTC', limit=10)
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert len(order_book['bids']) == 10
    assert len(order_book['asks']) == 10
    assert api_mock.fetch_b2_order_book.call_args_list[0][0][0] == 'ETH/BTC'
    for val in [1, 5, 10, 12, 20, 50, 100]:
        api_mock.fetch_b2_order_book.reset_mock()
        order_book = exchange.fetch_b2_order_book(pair='ETH/BTC', limit=val)
        assert api_mock.fetch_b2_order_book.call_args_list[0][0][0] == 'ETH/BTC'
        if not exchange.get_option('b2_limit_range') or val in exchange.get_option('b2_limit_range'):
            assert api_mock.fetch_b2_order_book.call_args_list[0][0][1] == val
        else:
            next_limit = exchange.get_next_limit_in_list(val, exchange.get_option('b2_limit_range'))
            assert api_mock.fetch_b2_order_book.call_args_list[0][0][1] == next_limit


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_l2_order_book(default_conf: Any, mocker: Any, exchange_name: str, order_book_l2: Dict[str, Any]) -> None:
    default_conf['exchange']['name'] = exchange_name
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = MagicMock(return_value=order_book_l2)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order_book = exchange.fetch_l2_order_book(pair='ETH/BTC', limit=10)
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert len(order_book['bids']) == 10
    assert len(order_book['asks']) == 10
    assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == 'ETH/BTC'
    for val in [1, 5, 10, 12, 20, 50, 100]:
        api_mock.fetch_l2_order_book.reset_mock()
        order_book = exchange.fetch_l2_order_book(pair='ETH/BTC', limit=val)
        assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == 'ETH/BTC'
        if not exchange.get_option('l2_limit_range') or val in exchange.get_option('l2_limit_range'):
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == val
        else:
            next_limit = exchange.get_next_limit_in_list(val, exchange.get_option('l2_limit_range'))
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == next_limit


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_l2_order_book_exception(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
) -> None:
    api_mock = MagicMock()
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NotSupported('Not supported'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(TemporaryError):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NetworkError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.BaseError('DeadBeef'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair='ETH/BTC', limit=50)


@pytest.mark.parametrize(
    'side,leverage,exchange_name',
    [
        ('buy', 1, 'binance'),
        ('buy', 5, 'binance'),
        ('sell', 1.0, 'binance'),
        ('sell', 5.0, 'binance'),
    ],
)
@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_create_order(default_conf: Any, mocker: Any, side: str, ordertype: str, exchange_name: str, rate: float, leverage: float) -> None:
    api_mock = MagicMock()
    order_id = f'test_prod_{side}_{randint(0, 10**6)}'
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'info': {'foo': 'bar'}, 'symbol': 'XLTCUSDT', 'amount': 1})
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.amount_to_precision', lambda s, x, y, **kwargs: y)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()
    price_req = exchange._ft_has.get('marketOrderRequiresPrice', False)
    order = exchange.create_order(
        pair='XLTCUSDT',
        ordertype=ordertype,
        side=side,
        amount=1,
        rate=rate,
        leverage=1.0,
    )
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id
    assert api_mock.create_order.call_args[0][0] == 'XLTCUSDT'
    assert api_mock.create_order.call_args[0][1] == ordertype
    assert api_mock.create_order.call_args[0][2] == side
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == (rate if price_req or not (bool(marketprice) and side == 'sell') else None)
    assert exchange._set_leverage.call_count == 0
    assert exchange.set_margin_mode.call_count == 0
    api_mock.create_order = MagicMock(return_value={'id': order_id, 'info': {'foo': 'bar'}, 'symbol': 'ADA/USDT:USDT', 'amount': 1})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    exchange.trading_mode = TradingMode.FUTURES
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()
    order = exchange.create_order(
        pair='ADA/USDT:USDT',
        ordertype=ordertype,
        side=side,
        amount=1,
        rate=200,
        leverage=3.0,
    )
    if exchange_name != 'okx':
        assert exchange._set_leverage.call_count == 1
        assert exchange.set_margin_mode.call_count == 1
    else:
        assert api_mock.set_leverage.call_count == 1
    assert order['amount'] == 0.01
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds('Not enough funds'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.create_order(
            pair='ETH/BTC',
            ordertype=ordertype,
            side=side,
            amount=1,
            rate=200,
            leverage=1.0,
        )
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.create_order(
            pair='ETH/BTC',
            ordertype='limit',
            side=side,
            amount=1,
            rate=200,
            leverage=1.0,
        )
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.create_order(
            pair='ETH/BTC',
            ordertype='market',
            side=side,
            amount=1,
            rate=200,
            leverage=1.0,
        )
    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError('Network disconnect'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.create_order(
            pair='ETH/BTC',
            ordertype=ordertype,
            side=side,
            amount=1,
            rate=200,
            leverage=1.0,
        )
    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError('Unknown error'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair='ETH/BTC',
            ordertype=ordertype,
            side=side,
            amount=1,
            rate=200,
            leverage=1.0,
        )


@pytest.mark.parametrize(
    'side,is_short,order_reason,order_type,price_side,fee',
    [
        ('buy', False, 'entry', 'limit', 'same', 1.0),
        ('buy', False, 'entry', 'limit', 'other', 2.0),
        ('buy', False, 'entry', 'market', 'same', 2.0),
        ('buy', False, 'entry', 'market', 'other', 2.0),
        ('sell', False, 'exit', 'limit', 'same', 1.0),
        ('sell', False, 'exit', 'limit', 'other', 2.0),
        ('sell', False, 'exit', 'market', 'same', 2.0),
        ('sell', False, 'exit', 'market', 'other', 2.0),
    ],
)
def test_create_order_fees(
    default_conf: Any,
    mocker: Any,
    side: str,
    is_short: bool,
    order_reason: str,
    order_type: str,
    price_side: str,
    fee: float,
) -> None:
    pass


def test_create_stoploss_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


def test_create_stoploss_order_with_result(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_name(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_get_trades_for_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'pair,contract_size,trading_mode',
    [
        ('XLTCUSDT', 1, 'spot'),
        ('LTC/USD', 1, 'futures'),
        ('ADA/USDT:USDT', 0.01, 'futures'),
        ('LTC/ETH', 1, 'futures'),
        ('ETH/USDT:USDT', 10, 'futures'),
    ],
)
def test__amount_to_contracts(
    mocker: Any,
    default_conf: Any,
    pair: str,
    contract_size: float,
    trading_mode: str,
) -> None:
    api_mock = MagicMock()
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    mocker.patch(f'{EXMS}.markets', {
        'LTC/USD': {'symbol': 'LTC/USD', 'contractSize': None},
        'XLTCUSDT': {'symbol': 'XLTCUSDT', 'contractSize': '0.01'},
        'LTC/ETH': {'symbol': 'LTC/ETH'},
        'ETH/USDT:USDT': {'symbol': 'ETH/USDT:USDT', 'contractSize': '10'},
    })
    orders = [
        {'id': '123456320', 'clientOrderId': '12345632018', 'timestamp': 1640124992000, 'datetime': 'Tue 21 Dec 2021 22:16:32 UTC',
         'status': 'active', 'symbol': pair, 'type': 'limit', 'timeInForce': 'gtc', 'postOnly': None, 'side': 'buy',
         'price': 2.0, 'stopPrice': None, 'average': None, 'amount': 30.0, 'cost': 60.0, 'filled': None,
         'remaining': 30.0, 'fee': {'currency': 'USDT', 'cost': 0.06}, 'fees': [{'currency': 'USDT', 'cost': 0.06}],
         'trades': None, 'info': {}},
        {'id': '123456380', 'clientOrderId': '12345638203', 'timestamp': 1640124992000, 'datetime': 'Tue 21 Dec 2021 22:16:32 UTC',
         'status': 'active', 'symbol': pair, 'type': 'limit', 'timeInForce': 'gtc', 'postOnly': None, 'side': 'sell',
         'price': 2.2, 'stopPrice': None, 'average': None, 'amount': 40.0, 'cost': 80.0, 'filled': None,
         'remaining': 40.0, 'fee': {'currency': 'USDT', 'cost': 0.08}, 'fees': [{'currency': 'USDT', 'cost': 0.08}],
         'trades': None, 'info': {}},
    ]
    order1_bef = orders[0]
    order2_bef = orders[1]
    order1 = exchange._order_contracts_to_amount(deepcopy(order1_bef))
    order2 = exchange._order_contracts_to_amount(deepcopy(order2_bef))
    assert order1['amount'] == order1_bef['amount'] * contract_size
    assert order1['cost'] == order1_bef['cost'] * contract_size
    assert order2['amount'] == order2_bef['amount'] * contract_size
    assert order2['cost'] == order2_bef['cost'] * contract_size
    exchange._order_contracts_to_amount(orders[2])


def test_get_trades_for_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'max_retries,expected',
    [
        (0, 10),
        (1, 5),
        (2, 2),
        (3, 1),
        (0, 2),
        (1, 1),
        (0, 17),
        (1, 10),
        (2, 5),
        (3, 2),
        (4, 1),
        (0, 26),
        (1, 17),
        (2, 10),
        (3, 5),
        (4, 2),
        (5, 1),
    ],
)
def test_calculate_backoff(retrycount: int, max_retries: int, expected: int) -> None:
    assert calculate_backoff(retrycount, max_retries) == expected


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_order(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    default_conf['dry_run'] = True
    default_conf['exchange']['log_responses'] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = 'TKN/BTC'
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.fetch_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
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
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_order',
        'fetch_order',
        retries=API_FETCH_ORDER_RETRY_COUNT + 1,
        order_id='_',
        pair='TKN/BTC',
    )


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_stoploss_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    default_conf['dry_run'] = True
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_stoploss_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_stoploss_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={'id': '123', 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    res = exchange.fetch_stoploss_order('X', 'TKN/BTC')
    assert res == {'id': '123', 'symbol': 'TKN/BTC'}
    if exchange_name != 'okx':
        assert api_mock.fetch_order.call_count == 1
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.fetch_stoploss_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_stoploss_order',
        'fetch_order',
        retries=API_FETCH_ORDER_RETRY_COUNT + 1,
        order_id='_',
        pair='TKN/BTC',
    )


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_fetch_order_emulated(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    default_conf['dry_run'] = True
    default_conf['exchange']['log_responses'] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = 'TKN/BTC'
    exchange = get_patched_exchange(mocker, default_conf, exchange_name=exchange_name)
    mocker.patch(f'{EXMS}.exchange_has', return_value=False)
    exchange._dry_run_open_orders['X'] = order
    assert exchange.fetch_order('X', 'TKN/BTC').myid == 123
    with pytest.raises(InvalidOrderException, match='Tried to get an invalid dry-run-order.*'):
        exchange.fetch_order('Y', 'TKN/BTC')
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=False))
    api_mock = MagicMock()
    api_mock.fetch_open_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    api_mock.fetch_closed_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_open_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 0
    caplog.clear()
    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    api_mock.fetch_closed_order = MagicMock(return_value={'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    assert exchange.fetch_order('X', 'TKN/BTC') == {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}
    assert log_has("API fetch_closed_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}", caplog)
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 1
    caplog.clear()
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_open_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        api_mock.fetch_closed_order = MagicMock(side_effect=ccxt.InvalidOrder('Order not found'))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
        exchange.fetch_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_open_order.call_count == 1
    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound('Order not found'))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange_name=exchange_name)
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        'fetch_order_emulated',
        'fetch_open_order',
        retries=1,
        order_id='_',
        pair='TKN/BTC',
        params={},
    )


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_get_funding_fees(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    ['binance'],
)
def test__get_funding_fees_from_exchange(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_validate_trading_mode_and_margin_mode(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    trading_mode: str,
    margin_mode: Optional[str],
    exception_thrown: bool,
) -> None:
    exchange = get_patched_exchange(
        mocker, default_conf, exchange_name=exchange_name, mock_supported_modes=False
    )
    if exception_thrown:
        with pytest.raises(OperationalException):
            exchange.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)
    else:
        exchange.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)


def test_get_liquidation_price(
    mocker: Any,
    default_conf: Any,
    exchange_name: str,
    open_rate: float,
    is_short: bool,
    trading_mode: str,
    margin_mode: Optional[str],
    leverage: float,
    amount: float,
    expected_liq: Optional[float],
) -> None:
    pass


def test_calculate_funding_fees(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    cum_b: float,
    maintenance_margin_rate: float,
    expected_fees: float,
) -> None:
    pass


def test_update_markets(mocker: Any, default_conf: Any) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test__get_params(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
) -> None:
    pass


def test_create_order_fees(
    default_conf: Any,
    mocker: Any,
    side: str,
    is_short: bool,
    order_reason: str,
    order_type: str,
    price_side: str,
    fee: float,
) -> None:
    pass


def test_create_stoploss_order(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
) -> None:
    pass


def test_create_stoploss_order_with_result(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_set_margin_mode(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={'setMarginMode': True})
    default_conf['dry_run'] = False
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        'binance',
        'set_margin_mode',
        'set_margin_mode',
        pair='XRP/USDT',
        margin_mode=MarginMode.ISOLATED,
    )


@pytest.mark.parametrize(
    'exchange_name,trading_mode,margin_mode,exception_thrown',
    [
        ('binance', TradingMode.SPOT, None, False),
        ('binance', TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ('kraken', TradingMode.SPOT, None, False),
        ('kraken', TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ('kraken', TradingMode.FUTURES, MarginMode.ISOLATED, True),
        ('bitmart', TradingMode.SPOT, None, False),
        ('bitmart', TradingMode.MARGIN, MarginMode.CROSS, True),
        ('bitmart', TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ('bitmart', TradingMode.FUTURES, MarginMode.CROSS, True),
        ('bitmart', TradingMode.FUTURES, MarginMode.ISOLATED, True),
        ('gate', TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ('okx', TradingMode.SPOT, None, False),
        ('okx', TradingMode.MARGIN, MarginMode.CROSS, True),
        ('okx', TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ('okx', TradingMode.FUTURES, MarginMode.CROSS, True),
        ('binance', TradingMode.FUTURES, MarginMode.ISOLATED, False),
        ('gate', TradingMode.FUTURES, MarginMode.ISOLATED, False),
        ('okx', TradingMode.FUTURES, MarginMode.ISOLATED, False),
        ('binance', TradingMode.MARGIN, MarginMode.CROSS, True),
        ('binance', TradingMode.FUTURES, MarginMode.CROSS, False),
        ('kraken', TradingMode.MARGIN, MarginMode.CROSS, True),
        ('kraken', TradingMode.FUTURES, MarginMode.CROSS, True),
        ('gate', TradingMode.MARGIN, MarginMode.CROSS, True),
        ('gate', TradingMode.FUTURES, MarginMode.CROSS, True),
    ],
)
def test_validate_trading_mode_and_margin_mode_errors(
    default_conf: Any,
    mocker: Any,
    exchange_name: str,
    trading_mode: str,
    margin_mode: Optional[str],
    exception_thrown: bool,
) -> None:
    pass


def test_get_valid_pair_combination(
    default_conf: Any,
    mocker: Any,
) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_get_balances_prod(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


def test_calculate_fee_rate(
    default_conf: Any,
    mocker: Any,
    order: Dict[str, Any],
    expected: Tuple[float, str, float],
    unknown_fee_rate: Optional[float],
) -> None:
    pass


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_cancel_order_with_result(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_cancel_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_cancel_stoploss_order(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'exchange_name',
    EXCHANGES,
)
def test_cancel_stoploss_order_with_result(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


def test_get_funding_fees(default_conf: Any, mocker: Any, exchange_name: str, caplog: Any) -> None:
    pass


def test__get_funding_fees_from_exchange(default_conf: Any, mocker: Any, exchange_name: str) -> None:
    pass


@pytest.mark.parametrize(
    'fee_rate,test_case',
    [
        # Add test cases as needed
    ],
)
def test_calculate_fee_rate(
    default_conf: Any,
    mocker: Any,
    order: Dict[str, Any],
    expected: float,
    fee_rate: float,
    test_case: str,
) -> None:
    pass


def test_get_liquidation_price(mocker: Any, default_conf: Any, exchange_name: str, open_rate: float, is_short: bool, trading_mode: str, margin_mode: Optional[str], leverage: float, amount: float, expected_liq: Optional[float]) -> None:
    pass


def test_get_max_leverage_from_margin(default_conf: Any, mocker: Any, exchange_name: str, pair: str, nominal_value: float, max_lev: float) -> None:
    pass
