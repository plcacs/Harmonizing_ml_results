from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock
from sqlalchemy import Select
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC
import pytest
import unittest.mock

EXMS: unittest.mock.MagicMock

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: Dict[str, Any],
    ticker: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture
) -> None:
    ...

@pytest.mark.parametrize('balance_ratio,result1', [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(
    default_conf: Dict[str, Any],
    ticker: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture,
    balance_ratio: float,
    result1: int
) -> None:
    ...

def test_dca_buying(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture
) -> None:
    ...

def test_dca_short(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture
) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_order_adjust(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    leverage: int,
    fee: Callable,
    mocker: pytest_mock.MockFixture
) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    is_short: bool,
    leverage: int
) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_exiting(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    fee: Callable,
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    leverage: int
) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_handle_similar_open_order(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable,
    is_short: bool,
    leverage: int,
    fee: Callable,
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture
) -> None:
    ...