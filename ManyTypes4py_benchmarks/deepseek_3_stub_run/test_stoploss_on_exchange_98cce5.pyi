from collections.abc import Sequence
from datetime import datetime
from typing import Any, Optional, Union
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, InvalidOrderException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import EXMS
from tests.conftest_trades import entry_side, exit_side

@pytest.mark.parametrize('is_short', [False, True])
def test_add_stoploss_on_exchange(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    limit_order: dict[str, dict[str, Any]],
    is_short: bool,
    fee: MagicMock
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    caplog: LogCaptureFixture,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_emergency(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial_cancel_here(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]],
    caplog: LogCaptureFixture,
    time_machine: Any
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_sle_cancel_cant_recreate(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    caplog: LogCaptureFixture,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_invalid_order(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    caplog: LogCaptureFixture,
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_insufficient_funds(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    caplog: LogCaptureFixture,
    fee: MagicMock,
    limit_order: dict[str, dict[str, Any]],
    is_short: bool
) -> None: ...

@pytest.mark.parametrize('is_short,bid,ask,stop_price,hang_price', [
    (False, [4.38, 4.16], [4.4, 4.17], ['2.0805', 4.4 * 0.95], 3),
    (True, [1.09, 1.21], [1.1, 1.22], ['2.321', 1.09 * 1.05], 1.5)
])
@pytest.mark.usefixtures('init_persistence')
def test_handle_stoploss_on_exchange_trailing(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    bid: list[float],
    ask: list[float],
    limit_order: dict[str, dict[str, Any]],
    stop_price: list[Union[str, float]],
    hang_price: float,
    time_machine: Any
) -> None: ...

def test_stoploss_on_exchange_price_rounding(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    open_trade_usdt: Trade
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_handle_stoploss_on_exchange_custom_stop(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

def test_tsl_on_exchange_compatible_with_edge(
    mocker: MagicMock,
    edge_conf: dict[str, Any],
    fee: MagicMock,
    limit_order: dict[str, dict[str, Any]]
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    is_short: bool,
    ticker_usdt_sell_down: MagicMock,
    ticker_usdt_sell_up: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_execute_trade_exit_sloe_cancel_exception(
    mocker: MagicMock,
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    caplog: LogCaptureFixture
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_with_stoploss_on_exchange(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    ticker_usdt_sell_up: MagicMock,
    is_short: bool,
    mocker: MagicMock
) -> None: ...

@pytest.mark.parametrize('is_short', [False, True])
def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MagicMock,
    is_short: bool
) -> None: ...