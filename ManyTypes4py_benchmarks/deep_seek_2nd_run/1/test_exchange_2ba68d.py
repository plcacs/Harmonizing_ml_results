import copy
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from random import randint
from unittest.mock import MagicMock, Mock, PropertyMock, patch
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import pytest
from numpy import nan
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import (ConfigurationError, DDosProtection, DependencyException,
                                 ExchangeError, InsufficientFundsError, InvalidOrderException,
                                 OperationalException, PricingError, TemporaryError)
from freqtrade.exchange import Binance, Bybit, Exchange, Kraken, market_is_active, timeframe_to_prev_date
from freqtrade.exchange.common import (API_FETCH_ORDER_RETRY_COUNT, API_RETRY_COUNT,
                                      calculate_backoff, remove_exchange_credentials)
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from tests.conftest import EXMS, generate_test_data_raw, get_mock_coro, get_patched_exchange, log_has, log_has_re, num_log_has_re

EXCHANGES: List[str] = ['binance', 'kraken', 'gate', 'kucoin', 'bybit', 'okx']
get_entry_rate_data: List[Tuple[str, float, float, Optional[float], Optional[float], float]] = [
    ('other', 20, 19, 10, 0.0, 20), ('ask', 20, 19, 10, 0.0, 20),
    ('ask', 20, 19, 10, 1.0, 10), ('ask', 20, 19, 10, 0.5, 15),
    ('ask', 20, 19, 10, 0.7, 13), ('ask', 20, 19, 10, 0.3, 17),
    ('ask', 5, 6, 10, 1.0, 5), ('ask', 5, 6, 10, 0.5, 5),
    ('ask', 20, 19, 10, None, 20), ('ask', 10, 20, None, 0.5, 10),
    ('ask', 4, 5, None, 0.5, 4), ('ask', 4, 5, None, 1, 4),
    ('ask', 4, 5, None, 0, 4), ('same', 21, 20, 10, 0.0, 20),
    ('bid', 21, 20, 10, 0.0, 20), ('bid', 21, 20, 10, 1.0, 10),
    ('bid', 21, 20, 10, 0.5, 15), ('bid', 21, 20, 10, 0.7, 13),
    ('bid', 21, 20, 10, 0.3, 17), ('bid', 6, 5, 10, 1.0, 5),
    ('bid', 21, 20, 10, None, 20), ('bid', 6, 5, 10, 0.5, 5),
    ('bid', 21, 20, None, 0.5, 20), ('bid', 6, 5, None, 0.5, 5),
    ('bid', 6, 5, None, 1, 5), ('bid', 6, 5, None, 0, 5)
]
get_exit_rate_data: List[Tuple[str, float, float, float, Optional[float], float]] = [
    ('bid', 12.0, 11.0, 11.5, 0.0, 11.0), ('bid', 12.0, 11.0, 11.5, 1.0, 11.5),
    ('bid', 12.0, 11.0, 11.5, 0.5, 11.25), ('bid', 12.0, 11.2, 10.5, 0.0, 11.2),
    ('bid', 12.0, 11.2, 10.5, 1.0, 11.2), ('bid', 12.0, 11.2, 10.5, 0.5, 11.2),
    ('bid', 0.003, 0.002, 0.005, 0.0, 0.002), ('bid', 0.003, 0.002, 0.005, None, 0.002),
    ('ask', 12.0, 11.0, 12.5, 0.0, 12.0), ('ask', 12.0, 11.0, 12.5, 1.0, 12.5),
    ('ask', 12.0, 11.0, 12.5, 0.5, 12.25), ('ask', 12.2, 11.2, 10.5, 0.0, 12.2),
    ('ask', 12.0, 11.0, 10.5, 1.0, 12.0), ('ask', 12.0, 11.2, 10.5, 0.5, 12.0),
    ('ask', 10.0, 11.0, 11.0, 0.0, 10.0), ('ask', 10.11, 11.2, 11.0, 0.0, 10.11),
    ('ask', 0.001, 0.002, 11.0, 0.0, 0.001), ('ask', 0.006, 1.0, 11.0, 0.0, 0.006),
    ('ask', 0.006, 1.0, 11.0, None, 0.006)
]

def ccxt_exceptionhandlers(mocker: Any, default_conf: Dict, api_mock: MagicMock,
                          exchange_name: str, fun: str, mock_ccxt_fun: str,
                          retries: int = API_RETRY_COUNT + 1, **kwargs: Any) -> None:
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

async def async_ccxt_exception(mocker: Any, default_conf: Dict, api_mock: MagicMock,
                              fun: str, mock_ccxt_fun: str,
                              retries: int = API_RETRY_COUNT + 1, **kwargs: Any) -> None:
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

def test_init(default_conf: Dict, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Instance is running with dry_run enabled', caplog)

def test_remove_exchange_credentials(default_conf: Dict) -> None:
    conf = deepcopy(default_conf)
    remove_exchange_credentials(conf['exchange'], False)
    assert conf['exchange']['key'] != ''
    assert conf['exchange']['secret'] != ''
    remove_exchange_credentials(conf['exchange'], True)
    assert conf['exchange']['key'] == ''
    assert conf['exchange']['secret'] == ''
    assert conf['exchange']['password'] == ''
    assert conf['exchange']['uid'] == ''

def test_init_ccxt_kwargs(default_conf: Dict, mocker: Any, caplog: Any) -> None:
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

def test_destroy(default_conf: Dict, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has('Exchange object destroyed, closing async loop', caplog)

def test_init_exception(default_conf: Dict, mocker: Any) -> None:
    default_conf['exchange']['name