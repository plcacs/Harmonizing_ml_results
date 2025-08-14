import copy
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from random import randint
from typing import Any, Dict, List, Optional, Tuple, Union
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


# Make sure to always keep one exchange here which is NOT subclassed!!
EXCHANGES: List[str] = ["binance", "kraken", "gate", "kucoin", "bybit", "okx"]

get_entry_rate_data: List[Tuple[str, float, float, float, Optional[float], float]] = [
    ("other", 20, 19, 10, 0.0, 20),  # Full ask side
    ("ask", 20, 19, 10, 0.0, 20),  # Full ask side
    ("ask", 20, 19, 10, 1.0, 10),  # Full last side
    ("ask", 20, 19, 10, 0.5, 15),  # Between ask and last
    ("ask", 20, 19, 10, 0.7, 13),  # Between ask and last
    ("ask", 20, 19, 10, 0.3, 17),  # Between ask and last
    ("ask", 5, 6, 10, 1.0, 5),  # last bigger than ask
    ("ask", 5, 6, 10, 0.5, 5),  # last bigger than ask
    ("ask", 20, 19, 10, None, 20),  # price_last_balance missing
    ("ask", 10, 20, None, 0.5, 10),  # last not available - uses ask
    ("ask", 4, 5, None, 0.5, 4),  # last not available - uses ask
    ("ask", 4, 5, None, 1, 4),  # last not available - uses ask
    ("ask", 4, 5, None, 0, 4),  # last not available - uses ask
    ("same", 21, 20, 10, 0.0, 20),  # Full bid side
    ("bid", 21, 20, 10, 0.0, 20),  # Full bid side
    ("bid", 21, 20, 10, 1.0, 10),  # Full last side
    ("bid", 21, 20, 10, 0.5, 15),  # Between bid and last
    ("bid", 21, 20, 10, 0.7, 13),  # Between bid and last
    ("bid", 21, 20, 10, 0.3, 17),  # Between bid and last
    ("bid", 6, 5, 10, 1.0, 5),  # last bigger than bid
    ("bid", 21, 20, 10, None, 20),  # price_last_balance missing
    ("bid", 6, 5, 10, 0.5, 5),  # last bigger than bid
    ("bid", 21, 20, None, 0.5, 20),  # last not available - uses bid
    ("bid", 6, 5, None, 0.5, 5),  # last not available - uses bid
    ("bid", 6, 5, None, 1, 5),  # last not available - uses bid
    ("bid", 6, 5, None, 0, 5),  # last not available - uses bid
]

get_exit_rate_data: List[Tuple[str, float, float, float, Optional[float], float]] = [
    ("bid", 12.0, 11.0, 11.5, 0.0, 11.0),  # full bid side
    ("bid", 12.0, 11.0, 11.5, 1.0, 11.5),  # full last side
    ("bid", 12.0, 11.0, 11.5, 0.5, 11.25),  # between bid and lat
    ("bid", 12.0, 11.2, 10.5, 0.0, 11.2),  # Last smaller than bid
    ("bid", 12.0, 11.2, 10.5, 1.0, 11.2),  # Last smaller than bid - uses bid
    ("bid", 12.0, 11.2, 10.5, 0.5, 11.2),  # Last smaller than bid - uses bid
    ("bid", 0.003, 0.002, 0.005, 0.0, 0.002),
    ("bid", 0.003, 0.002, 0.005, None, 0.002),
    ("ask", 12.0, 11.0, 12.5, 0.0, 12.0),  # full ask side
    ("ask", 12.0, 11.0, 12.5, 1.0, 12.5),  # full last side
    ("ask", 12.0, 11.0, 12.5, 0.5, 12.25),  # between bid and lat
    ("ask", 12.2, 11.2, 10.5, 0.0, 12.2),  # Last smaller than ask
    ("ask", 12.0, 11.0, 10.5, 1.0, 12.0),  # Last smaller than ask - uses ask
    ("ask", 12.0, 11.2, 10.5, 0.5, 12.0),  # Last smaller than ask - uses ask
    ("ask", 10.0, 11.0, 11.0, 0.0, 10.0),
    ("ask", 10.11, 11.2, 11.0, 0.0, 10.11),
    ("ask", 0.001, 0.002, 11.0, 0.0, 0.001),
    ("ask", 0.006, 1.0, 11.0, 0.0, 0.006),
    ("ask", 0.006, 1.0, 11.0, None, 0.006),
]


def ccxt_exceptionhandlers(
    mocker: Any,
    default_conf: Dict[str, Any],
    api_mock: MagicMock,
    exchange_name: str,
    fun: str,
    mock_ccxt_fun: str,
    retries: int = API_RETRY_COUNT + 1,
    **kwargs: Any,
) -> None:
    with patch("freqtrade.exchange.common.time.sleep"):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("DDos"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
            getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.OperationFailed("DeaDBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


async def async_ccxt_exception(
    mocker: Any,
    default_conf: Dict[str, Any],
    api_mock: MagicMock,
    fun: str,
    mock_ccxt_fun: str,
    retries: int = API_RETRY_COUNT + 1,
    **kwargs: Any,
) -> None:
    with patch("freqtrade.exchange.common.asyncio.sleep", get_mock_coro(None)):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("Dooh"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock)
            await getattr(exchange, fun)(**kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1
    exchange.close()
