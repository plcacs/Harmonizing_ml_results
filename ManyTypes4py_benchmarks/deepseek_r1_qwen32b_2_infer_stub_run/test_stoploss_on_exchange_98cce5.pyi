from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from datetime import datetime
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from sqlalchemy import ScalarResult
from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, InvalidOrderException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from tests.conftest import EXMS

def test_add_stoploss_on_exchange(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    limit_order: Dict[str, Any],
    is_short: bool,
    fee: Callable[..., Dict[str, Any]],
) -> None:
    ...

def test_handle_stoploss_on_exchange(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    caplog: Any,
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_handle_stoploss_on_exchange_emergency(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_handle_stoploss_on_exchange_partial(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_handle_stoploss_on_exchange_partial_cancel_here(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    limit_order: Dict[str, Any],
    caplog: Any,
    time_machine: Any,
) -> None:
    ...

def test_handle_sle_cancel_cant_recreate(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    caplog: Any,
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_create_stoploss_order_invalid_order(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    caplog: Any,
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_create_stoploss_order_insufficient_funds(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    caplog: Any,
    fee: Callable[..., Dict[str, Any]],
    limit_order: Dict[str, Any],
    is_short: bool,
) -> None:
    ...

def test_handle_stoploss_on_exchange_trailing(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    bid: List[float],
    ask: List[float],
    limit_order: Dict[str, Any],
    stop_price: List[str],
    hang_price: float,
    time_machine: Any,
) -> None:
    ...

def test_handle_stoploss_on_exchange_trailing_error(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    caplog: Any,
    limit_order: Dict[str, Any],
    is_short: bool,
    time_machine: Any,
) -> None:
    ...

def test_stoploss_on_exchange_price_rounding(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    open_trade_usdt: Trade,
) -> None:
    ...

def test_handle_stoploss_on_exchange_custom_stop(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    limit_order: Dict[str, Any],
) -> None:
    ...

def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable[..., Dict[str, Any]],
    fee: Callable[..., Dict[str, Any]],
    is_short: bool,
    ticker_usdt_sell_down: Callable[..., Dict[str, Any]],
    ticker_usdt_sell_up: Callable[..., Dict[str, Any]],
    mocker: MockerFixture,
) -> None:
    ...

def test_execute_trade_exit_sloe_cancel_exception(
    mocker: MockerFixture,
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable[..., Dict[str, Any]],
    fee: Callable[..., Dict[str, Any]],
    caplog: Any,
) -> None:
    ...

def test_execute_trade_exit_with_stoploss_on_exchange(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable[..., Dict[str, Any]],
    fee: Callable[..., Dict[str, Any]],
    ticker_usdt_sell_up: Callable[..., Dict[str, Any]],
    is_short: bool,
    mocker: MockerFixture,
) -> None:
    ...

def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Callable[..., Dict[str, Any]],
    fee: Callable[..., Dict[str, Any]],
    mocker: MockerFixture,
    is_short: bool,
) -> None:
    ...