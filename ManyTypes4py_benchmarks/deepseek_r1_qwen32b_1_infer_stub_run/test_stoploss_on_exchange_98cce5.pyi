from __future__ import annotations
from datetime import timedelta, datetime as datetime_
from typing import Any, Dict, List, Optional, Union

from pytest import LogCaptureFixture, MockerFixture, mark
from sqlalchemy import ScalarResult
from unittest.mock import MagicMock

from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, InvalidOrderException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.util.datetime_helpers import datetime as datetime_

@pytest.mark.parametrize('is_short', [False, True])
def test_add_stoploss_on_exchange(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], limit_order: Dict[str, Any], is_short: bool, fee: MagicMock) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, caplog: LogCaptureFixture, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_emergency(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial_cancel_here(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any], caplog: LogCaptureFixture, time_machine: Any) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_sle_cancel_cant_recreate(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, caplog: LogCaptureFixture, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_invalid_order(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], caplog: LogCaptureFixture, fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_insufficient_funds(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], caplog: LogCaptureFixture, fee: MagicMock, limit_order: Dict[str, Any], is_short: bool) -> None:
    ...

@pytest.mark.parametrize('is_short,bid,ask,stop_price,hang_price', [(False, [float, float], [float, float], [str, float], float), (True, [float, float], [float, float], [str, float], float)])
def test_handle_stoploss_on_exchange_trailing(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, bid: List[float], ask: List[float], limit_order: Dict[str, Any], stop_price: List[Union[str, float]], hang_price: float, time_machine: Any) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_trailing_error(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, caplog: LogCaptureFixture, limit_order: Dict[str, Any], is_short: bool, time_machine: Any) -> None:
    ...

def test_stoploss_on_exchange_price_rounding(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, open_trade_usdt: Trade) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_custom_stop(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    ...

def test_tsl_on_exchange_compatible_with_edge(mocker: MockerFixture, edge_conf: Dict[str, Any], fee: MagicMock, limit_order: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(default_conf_usdt: Dict[str, Any], ticker_usdt: MagicMock, fee: MagicMock, is_short: bool, ticker_usdt_sell_down: MagicMock, ticker_usdt_sell_up: MagicMock, mocker: MockerFixture) -> None:
    ...

def test_execute_trade_exit_sloe_cancel_exception(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], ticker_usdt: MagicMock, fee: MagicMock, caplog: LogCaptureFixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_with_stoploss_on_exchange(default_conf_usdt: Dict[str, Any], ticker_usdt: MagicMock, fee: MagicMock, ticker_usdt_sell_up: MagicMock, is_short: bool, mocker: MockerFixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(default_conf_usdt: Dict[str, Any], ticker_usdt: MagicMock, fee: MagicMock, mocker: MockerFixture, is_short: bool) -> None:
    ...