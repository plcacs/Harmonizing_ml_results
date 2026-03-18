```python
from typing import Any
from _pytest.fixtures import FixtureRequest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from datetime import datetime
from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, InvalidOrderException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from pytest import MonkeyPatch
from time_machine import TimeMachineFixture
from unittest.mock import MagicMock

def test_add_stoploss_on_exchange(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    limit_order: dict[str, Any],
    is_short: bool,
    fee: MagicMock
) -> None: ...

def test_handle_stoploss_on_exchange(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    caplog: LogCaptureFixture,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_handle_stoploss_on_exchange_emergency(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_handle_stoploss_on_exchange_partial(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_handle_stoploss_on_exchange_partial_cancel_here(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, Any],
    caplog: LogCaptureFixture,
    time_machine: TimeMachineFixture
) -> None: ...

def test_handle_sle_cancel_cant_recreate(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    caplog: LogCaptureFixture,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_create_stoploss_order_invalid_order(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    caplog: LogCaptureFixture,
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_create_stoploss_order_insufficient_funds(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    caplog: LogCaptureFixture,
    fee: MagicMock,
    limit_order: dict[str, Any],
    is_short: bool
) -> None: ...

def test_handle_stoploss_on_exchange_trailing(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    bid: list[float],
    ask: list[float],
    limit_order: dict[str, Any],
    stop_price: list[Any],
    hang_price: float,
    time_machine: TimeMachineFixture
) -> None: ...

def test_handle_stoploss_on_exchange_trailing_error(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    caplog: LogCaptureFixture,
    limit_order: dict[str, Any],
    is_short: bool,
    time_machine: TimeMachineFixture
) -> None: ...

def test_stoploss_on_exchange_price_rounding(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    open_trade_usdt: Trade
) -> None: ...

def test_handle_stoploss_on_exchange_custom_stop(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    fee: MagicMock,
    is_short: bool,
    limit_order: dict[str, Any]
) -> None: ...

def test_tsl_on_exchange_compatible_with_edge(
    mocker: MonkeyPatch,
    edge_conf: dict[str, Any],
    fee: MagicMock,
    limit_order: dict[str, Any]
) -> None: ...

def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    is_short: bool,
    ticker_usdt_sell_down: MagicMock,
    ticker_usdt_sell_up: MagicMock,
    mocker: MonkeyPatch
) -> None: ...

def test_execute_trade_exit_sloe_cancel_exception(
    mocker: MonkeyPatch,
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    caplog: LogCaptureFixture
) -> None: ...

def test_execute_trade_exit_with_stoploss_on_exchange(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    ticker_usdt_sell_up: MagicMock,
    is_short: bool,
    mocker: MonkeyPatch
) -> None: ...

def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MonkeyPatch,
    is_short: bool
) -> None: ...
```