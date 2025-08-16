# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import time_machine
from pandas import DataFrame
from sqlalchemy import select
from telegram import Chat, Message, ReplyKeyboardMarkup, Update
from telegram.error import BadRequest, NetworkError, TelegramError

from freqtrade import __version__
from freqtrade.constants import CANCEL_REASON
from freqtrade.edge import PairInfo
from freqtrade.enums import (
    ExitType,
    MarketDirection,
    RPCMessageType,
    RunMode,
    SignalDirection,
    State,
)
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.loggers import setup_logging
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc import RPC
from freqtrade.rpc.rpc import RPCException
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    patch_exchange,
    patch_get_signal,
    patch_whitelist,
)


@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker: MagicMock) -> None:
    mocker.patch("freqtrade.exchange.exchange.Exchange._init_async_loop")


@pytest.fixture
def default_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    # Telegram is enabled by default
    default_conf["telegram"]["enabled"] = True
    return default_conf


@pytest.fixture
def update() -> Update:
    message = Message(0, datetime.now(timezone.utc), Chat(1235, 0))
    _update = Update(0, message=message)

    return _update


def patch_eventloop_threading(telegrambot: Telegram) -> None:
    is_init = False

    def thread_fuck() -> None:
        nonlocal is_init
        telegrambot._loop = asyncio.new_event_loop()
        is_init = True
        telegrambot._loop.run_forever()

    x = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


class DummyCls(Telegram):
    """
    Dummy class for testing the Telegram @authorized_only decorator
    """

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        super().__init__(rpc, config)
        self.state: Dict[str, bool] = {"called": False}

    def _init(self) -> None:
        pass

    @authorized_only
    async def dummy_handler(self, *args: Any, **kwargs: Any) -> None:
        """
        Fake method that only change the state of the object
        """
        self.state["called"] = True

    @authorized_only
    async def dummy_exception(self, *args: Any, **kwargs: Any) -> None:
        """
        Fake method that throw an exception
        """
        raise Exception("test")


def get_telegram_testobject(mocker: MagicMock, default_conf: Dict[str, Any], mock: bool = True, ftbot: Optional[FreqtradeBot] = None) -> Tuple[Telegram, FreqtradeBot, MagicMock]:
    msg_mock = AsyncMock()
    if mock:
        mocker.patch.multiple(
            "freqtrade.rpc.telegram.Telegram",
            _init=MagicMock(),
            _send_msg=msg_mock,
            _start_thread=MagicMock(),
        )
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    telegram = Telegram(rpc, default_conf)
    telegram._loop = MagicMock()
    patch_eventloop_threading(telegram)

    return telegram, ftbot, msg_mock


def test_telegram__init__(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


def test_telegram_init(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    app_mock = MagicMock()
    mocker.patch("freqtrade.rpc.telegram.Telegram._start_thread", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init_telegram_app", return_value=app_mock)
    mocker.patch("freqtrade.rpc.telegram.Telegram._startup_telegram", AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._init()
    assert app_mock.call_count == 0

    # number of handles registered
    assert app_mock.add_handler.call_count > 0
    # assert start_polling.start_polling.call_count == 1

    message_str = (
        "rpc.telegram is listening for following commands: [['status'], ['profit'], "
        "['balance'], ['start'], ['stop'], "
        "['forceexit', 'forcesell', 'fx'], ['forcebuy', 'forcelong'], ['forceshort'], "
        "['reload_trade'], ['trades'], ['delete'], ['cancel_open_order', 'coo'], "
        "['performance'], ['buys', 'entries'], ['exits', 'sells'], ['mix_tags'], "
        "['stats'], ['daily'], ['weekly'], ['monthly'], "
        "['count'], ['locks'], ['delete_locks', 'unlock'], "
        "['reload_conf', 'reload_config'], ['show_conf', 'show_config'], "
        "['stopbuy', 'stopentry'], ['whitelist'], ['blacklist'], "
        "['bl_delete', 'blacklist_delete'], "
        "['logs'], ['edge'], ['health'], ['help'], ['version'], ['marketdir'], "
        "['order'], ['list_custom_data'], ['tg_info']]"
    )

    assert log_has(message_str, caplog)


async def test_telegram_startup(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    app_mock = MagicMock()
    app_mock.initialize = AsyncMock()
    app_mock.start = AsyncMock()
    app_mock.updater.start_polling = AsyncMock()
    app_mock.updater.running = False
    sleep_mock = mocker.patch("freqtrade.rpc.telegram.asyncio.sleep", AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    await telegram._startup_telegram()
    assert app_mock.initialize.call_count == 1
    assert app_mock.start.call_count == 1
    assert app_mock.updater.start_polling.call_count == 1
    assert sleep_mock.call_count == 1


async def test_telegram_cleanup(
    default_conf: Dict[str, Any],
    mocker: MagicMock,
) -> None:
    app_mock = MagicMock()
    app_mock.stop = AsyncMock()
    app_mock.initialize = AsyncMock()

    updater_mock = MagicMock()
    updater_mock.stop = AsyncMock()
    app_mock.updater = updater_mock
    # mocker.patch('freqtrade.rpc.telegram.Application', app_mock)

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    telegram._loop = asyncio.get_running_loop()
    telegram._thread = MagicMock()
    telegram.cleanup()
    await asyncio.sleep(0.1)
    assert app_mock.stop.call_count == 1
    assert telegram._thread.join.call_count == 1


async def test_authorized_only(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture, update: Update) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state["called"] is True
    assert log_has("Executing handler: dummy_handler for chat_id: 1235", caplog)
    assert not log_has("Rejected unauthorized message from: 1235", caplog)
    assert not log_has("Exception occurred within Telegram module", caplog)


async def test_authorized_only_unauthorized(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    chat = Chat(0xDEADBEEF, 0)
    message = Message(randint(1, 100), datetime.now(timezone.utc), chat)
    update = Update(randint(1, 100), message=message)

    default_conf["telegram"]["enabled"] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state["called"] is False
    assert not log_has("Executing handler: dummy_handler for chat_id: 3735928559", caplog)
    assert log_has("Rejected unauthorized message from: 3735928559", caplog)
    assert not log_has("Exception occurred within Telegram module", caplog)


async def test_authorized_only_exception(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture, update: Update) -> None:
    patch_exchange(mocker)

    default_conf["telegram"]["enabled"] = False

    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)

    await dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state["called"] is False
    assert not log_has("Executing handler: dummy_handler for chat_id: 0", caplog)
    assert not log_has("Rejected unauthorized message from: 0", caplog)
    assert log_has("Exception occurred within Telegram module", caplog)


async def test_telegram_status(default_conf: Dict[str, Any], update: Update, mocker: MagicMock) -> None:
    default_conf["telegram"]["enabled"] = False

    status_table = MagicMock()
    mocker.patch("freqtrade.rpc.telegram.Telegram._status_table", status_table)

    mocker.patch.multiple(
        "freqtrade.rpc.rpc.RPC",
        _rpc_trade_status=MagicMock(
            return_value=[
                {
                    "trade_id": 1,
                    "pair": "ETH/BTC",
                    "base_currency": "ETH",
                    "quote_currency": "BTC",
                    "open_date": dt_now(),
                    "close_date": None,
                    "open_rate": 1.099e-05,
                    "close_rate": None,
                    "current_rate": 1.098e-05,
                    "amount": 90.99181074,
                    "stake_amount": 90.99181074,
                    "max_stake_amount": 90.99181074,
                    "buy_tag": None,
                    "enter_tag": None,
                    "close_profit_ratio": None,
                    "profit": -0.0059,
                    "profit_ratio": -0.0059,
                    "profit_abs": -0.225,
                    "realized_profit": 0.0,
                    "total_profit_abs": -0.225,
                    "initial_stop_loss_abs": 1.098e-05,
                    "stop_loss_abs": 1.099e-05,
                    "exit_order_status": None,
                    "initial_stop_loss_ratio": -0.0005,
                    "stoploss_current_dist": 1e-08,
                    "stoploss_current_dist_ratio": -0.0002,
                    "stop_loss_ratio": -0.0001,
                    "open_order": "(limit buy rem=0.00000000)",
                    "is_open": True,
                    "is_short": False,
                    "filled_entry_orders": [],
                    "orders": [],
                }
            ]
        ),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1

    context = MagicMock()
    # /status table
    context.args = ["table"]
    await telegram._status(update=update, context=context)
    assert status_table.call_count == 1


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_multi_entry(default_conf: Dict[str, Any], update: Update, mocker: MagicMock, fee: float) -> None:
    default_conf["telegram"]["enabled"] = False
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # Average may be empty on some exchanges
    trade.orders[0].average = 0
    trade.orders.append(
        Order(
            order_id="5412vbb",
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate * 0.95,
            average=0,
            filled=trade.amount,
            remaining=0,
            cost=trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r"Number of Entries.*2", msg)
    assert re.search(r"Number of Exits.*1", msg)
    assert re.search(r"Close Date:", msg) is None
    assert re.search(r"Close Profit:", msg) is None


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_closed_trade(default_conf: Dict[str, Any], update: Update, mocker: MagicMock, fee: float) -> None:
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.is_open.is_(False)]).first()
    context = MagicMock()
    context.args = [str(trade.id)]
    await telegram._status(update=update, context=context)
    assert msg_mock.call_count == 1
    msg = msg_mock.call_args_list[0][0][0]
    assert re.search(r"Close Date:", msg)
    assert re.search(r"Close Profit:", msg)


async def test_order_handle(default_conf: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock) -> None:
    default_conf["max_open_trades"] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        "freqtrade.rpc.telegram.Telegram",
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.RUNNING
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()

    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert "Order List for Trade #*`2`" in msg1

    msg_mock.reset_mock()
    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 50)
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 2

    msg1 = msg_mock.call_args_list[0][0][0]
    msg2 = msg_mock.call_args_list[1][0][0]

    assert "Order List for Trade #*`2`" in msg1
    assert "*Order List for Trade #*`2` - continued" in msg2


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_order_multi_entry(default_conf: Dict[str, Any], update: Update, mocker: MagicMock, fee: float) -> None:
    default_conf["telegram"]["enabled"] = False
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # Average may be empty on some exchanges
    trade.orders[0].average = 0
    trade.orders.append(
        Order(
            order_id="5412vbb",
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate * 0.95,
            average=0,
            filled=trade.amount,
            remaining=0,
            cost=trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._order(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r"from 1st entry rate", msg)
    assert re.search(r"Order Filled", msg)


async def test_status_handle(default_conf: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock) -> None:
    default_conf["max_open_trades"] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        "freqtrade.rpc.telegram.Telegram",
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status is also enabled when stopped
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "no active trade" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "no active trade" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()
    # Trigger status while we have a fulfilled order for the open trade
    await telegram._status(update=update, context=MagicMock())

    # close_rate should not be included in the message as the trade is not closed
    # and no line should be empty
    lines = msg_mock.call_args_list[0][0][0].split("\n")
    assert "" not in lines[:-1]
    assert "Close Rate" not in "".join(lines)
    assert "Close Profit" not in "".join(lines)

    assert msg_mock.call_count == 3
    assert "ETH/BTC" in msg_mock.call_args_list[0][0][0]
    assert "LTC/BTC" in msg_mock.call_args_list[1][0][0]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2", "3"]

    await telegram._status(update=update, context=context)

    lines = msg_mock.call_args_list[0][0][0].split("\n")
    assert "" not in lines[:-1]
    assert "Close Rate" not in "".join(lines)
    assert "Close Profit" not in "".join(lines)

    assert msg_mock.call_count == 2
    assert "LTC/BTC" in msg_mock.call_args_list[0][0][0]

    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._status(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert "Close Rate" not in msg1
    assert "Trade ID:* `2`" in msg1


async def test_status_table_handle(default_conf: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    default_conf["stake_amount"] = 15.0

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # Status table is also enabled when stopped
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "no active trade" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "no active trade" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()

    await telegram._status_table(update=update, context=MagicMock())

    text = re.sub("</?pre>", "", msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub("[ ]+", " ", line[2].strip()).split(" ")

    assert int(fields[0]) == 1
    # assert 'L' in fields[1]
    assert "ETH/BTC" in fields[1]
    assert msg_mock.call_count == 1


async def test_daily_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock, time_machine: Any) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # Move date to within day
    time_machine.move_to("2022-06-11 08:00:00+00:00")
    # Create some test data
    create_mock_trades_usdt(fee)

    # Try valid data
    # /daily 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Daily Profit over the last 2 days</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Day " in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]
    assert "(2)  6.83 USDT  7.51 USD  0.64%" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Daily Profit over the last 7 days</b>:" in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert (
        str((datetime.now(timezone.utc) - timedelta(days=5)).date())
        in msg_mock.call_args_list[0][0][0]
    )
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]
    assert "(1)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()

    # /daily 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._daily(update=update, context=context)
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]


async def test_daily_wrong_input(default_conf: Dict[str, Any], update: Update, ticker: MagicMock, mocker: MagicMock) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily -2
    context = MagicMock()
    context.args = ["-2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "must be an integer greater than 0" in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily today
    context = MagicMock()
    context.args = ["today"]
    await telegram._daily(update=update, context=context)
    assert "Daily Profit over the last 7 days</b>:" in msg_mock.call_args_list[0][0][0]


async def test_weekly_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock, time_machine: Any) -> None:
    default_conf_usdt["max_open_trades"] = 1
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # Move to saturday - so all trades are within that week
    time_machine.move_to("2022-06-11")
    create_mock_trades_usdt(fee)

    # Try valid data
    # /weekly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "Weekly Profit over the last 2 weeks (starting from Monday)</b>:"
        in msg_mock.call_args_list[0][0][0]
    )
    assert "Monday " in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    first_iso_day_of_current_week = today - timedelta(days=today.weekday())
    assert str(first_iso_day_of_current_week) in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "Weekly Profit over the last 8 weeks (starting from Monday)</b>:"
        in msg_mock.call_args_list[0][0][0]
    )
    assert "Weekly" in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "must be an integer greater than 0" in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly this week
    context = MagicMock()
    context.args = ["this week"]
    await telegram._weekly(update=update, context=context)
    assert (
        "Weekly Profit over the last 8 weeks (starting from Monday)</b>:"
        in msg_mock.call_args_list[0][0][0]
    )


async def test_monthly_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock, time_machine: Any) -> None:
    default_conf_usdt["max_open_trades"] = 1
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # Move to day within the month so all mock trades fall into this week.
    time_machine.move_to("2022-06-11")
    create_mock_trades_usdt(fee)

    # Try valid data
    # /monthly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Monthly Profit over the last 2 months</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Month " in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    current_month = f"{today.year}-{today.month:02} "
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    # Default to 6 months
    assert "Monthly Profit over the last 6 months</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Month " in msg_mock.call_args_list[0][0][0]
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # Reset msg_mock
    msg_mock.reset_mock()

    # /monthly 12
    context = MagicMock()
    context.args = ["12"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Monthly Profit over the last 12 months</b>:" in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]

    # The one-digit months should contain a zero, Eg: September 2021 = "2021-09"
    # Since we loaded the last 12 months, any month should appear
    assert "-09" in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "must be an integer greater than 0" in msg_mock.call_args_list[0][0][0]

    # Try invalid data
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly february
    context = MagicMock()
    context.args = ["february"]
    await telegram._monthly(update=update, context=context)
    assert "Monthly Profit over the last 6 months</b>:" in msg_mock.call_args_list[0][0][0]


async def test_telegram_profit_handle(
    default_conf_usdt: Dict[str, Any], update: Update, ticker_usdt: MagicMock, ticker_sell_up: MagicMock, fee: float, limit_sell_order_usdt: Dict[str, Any], mocker: MagicMock
) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    await telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "No trades yet." in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # Create some test data
    freqtradebot.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()

    context = MagicMock()
    # Test with invalid 2nd argument (should silently pass)
    context.args = ["aaa"]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "No closed trade" in msg_mock.call_args_list[-1][0][0]
    assert "*ROI:* All trades" in msg_mock.call_args_list[-1][0][0]
    mocker.patch("freqtrade.wallets.Wallets.get_starting_balance", return_value=1000)
    assert (
        "∙ `0.298 USDT (0.50%) (0.03 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    msg_mock.reset_mock()

    # Update the ticker with a market going up
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_sell_up)
    # Simulate fulfilled LIMIT_SELL order for trade
    trade = Trade.session.scalars(select(Trade)).first()
    oobj = Order.parse_from_ccxt_object(
        limit_sell_order_usdt, limit_sell_order_usdt["symbol"], "sell"
    )
    trade.orders.append(oobj)
    trade.update_trade(oobj)

    trade.close_date = datetime.now(timezone.utc)
    trade.is_open = False
    Trade.commit()

    context.args = [3]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "*ROI:* Closed trades" in msg_mock.call_args_list[-1][0][0]
    assert (
        "∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    assert "∙ `6.253 USD`" in msg_mock.call_args_list[-1][0][0]
    assert "*ROI:* All trades" in msg_mock.call_args_list[-1][0][0]
    assert (
        "∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    assert "∙ `6.253 USD`" in msg_mock.call_args_list[-1][0][0]

    assert "*Best Performing:* `ETH/USDT: 5.685 USDT (9.47%)`" in msg_mock.call_args_list[-1][0][0]
    assert "*Max Drawdown:*" in msg_mock.call_args_list[-1][0][0]
    assert "*Profit factor:*" in msg_mock.call_args_list[-1][0][0]
    assert "*Winrate:*" in msg_mock.call_args_list[-1][0][0]
    assert "*Expectancy (Ratio):*" in msg_mock.call_args_list[-1][0][0]
    assert "*Trading volume:* `126 USDT`" in msg_mock.call_args_list[-1][0][0]


@pytest.mark.parametrize(
    "is_short,regex_pattern",
    [(True, r"now[ ]*XRP\/BTC \(#3\)  -1.00% \("), (False, r"now[ ]*XRP\/BTC \(#3\)  1.00% \(")],
)
async def test_telegram_trades(mocker: MagicMock, update: Update, default_conf: Dict[str, Any], fee: float, is_short: bool, regex_pattern: str) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []

    await telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    context.args = ["hello"]
    await telegram._trades(update=update, context=context)
    assert "<b>0 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._trades(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "2 recent trades</b>:" in msg_mock.call_args_list[0][0][0]
    assert "Profit (" in msg_mock.call_args_list[0][0][0]
    assert "Close Date" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" in msg_mock.call_args_list[0][0][0]
    assert bool(re.search(regex_pattern, msg_mock.call_args_list[0][0][0]))


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_delete_trade(mocker: MagicMock, update: Update, default_conf: Dict[str, Any], fee: float, is_short: bool) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._delete_trade(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [1]
    await telegram._delete_trade(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Deleted trade 1." in msg_mock.call_args_list[0][0][0]
    assert "Please make sure to take care of this asset" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_reload_trade_from_exchange(mocker: MagicMock, update: Update, default_conf: Dict[str, Any], fee: float, is_short: bool) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context.args = [5]

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "Status: `Reloaded from orders from exchange`" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_delete_open_order(mocker: MagicMock, update: Update, default_conf: Dict[str, Any], fee: float, is_short: bool, ticker: MagicMock) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
    )
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._cancel_open_order(update=update, context=context)
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._cancel_open_order(update=update, context=context)
    assert "No open order for trade_id" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f"{EXMS}.fetch_order", return_value=trade.orders[-1].to_ccxt_object())
    context = MagicMock()
    context.args = [6]
    await telegram._cancel_open_order(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Open order canceled." in msg_mock.call_args_list[0][0][0]


async def test_help_handle(default_conf: Dict[str, Any], update: Update, mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*/help:* `This help message`" in msg_mock.call_args_list[0][0][0]


async def test_version_handle(default_conf: Dict[str, Any], update: Update, mocker: MagicMock) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f"*Version:* `{__version__}`" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.strategy.version = lambda: "1.1.1"

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f"*Version:* `{__version__}`" in msg_mock.call_args_list[0][0][0]
    assert "*Strategy version: * `1.1.1`" in msg_mock.call_args_list[0][0][0]


async def test_show_config_handle(default_conf: Dict[str, Any], update: Update, mocker: MagicMock) -> None:
    default_conf["runmode"] = RunMode.DRY_RUN

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*Mode:* `{}`".format("Dry-run") in msg_mock.call_args_list[0][0][0]
    assert "*Exchange:* `binance`" in msg_mock.call_args_list[0][0][0]
    assert f"*Strategy:* `{CURRENT_TEST_STRATEGY}`" in msg_mock.call_args_list[0][0][0]
    assert "*Stoploss:* `-0.1`" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.config["trailing_stop"] = True
    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*Mode:* `{}`".format("Dry-run") in msg_mock.call_args_list[0][0][0]
    assert "*Exchange:* `binance`" in msg_mock.call_args_list[0][0][0]
    assert f"*Strategy:* `{CURRENT_TEST_STRATEGY}`" in msg_mock.call_args_list[0][0][0]
    assert "*Initial Stoploss:* `-0.1`" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize(
    "message_type,enter,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY, "Long", "long_signal_01", None),
        (RPCMessageType.ENTRY, "Long", "long_signal_01", 1.0),
        (RPCMessageType.ENTRY, "Long", "long_signal_01", 5.0),
        (RPCMessageType.ENTRY, "Short", "short_signal_01", 2.0),
    ],
)
def test_send_msg_enter_notification(
    default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture, message_type: RPCMessageType, enter: str, enter_signal: str, leverage: Optional[float]
) -> None:
    default_conf["telegram"]["notification_settings"]["show_candle"] = "ohlc"
    df = DataFrame(
        {
            "open": [1.1],
            "high": [2.2],
            "low": [1.0],
            "close": [1.5],
        }
    )
    mocker.patch(
        "freqtrade.data.dataprovider.DataProvider.get_analyzed_dataframe", return_value=(df, 1)
    )

    msg = {
        "type": message_type,
        "trade_id": 1,
        "enter_tag": enter_signal,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": leverage,
        "open_rate": 1.099e-05,
        "order_type": "limit",
        "direction": enter,
        "stake_amount": 0.01465333,
        "stake_amount_fiat": 0.0,
        "stake_currency": "BTC",
        "quote_currency": "BTC",
        "base_currency": "ETH",
        "fiat_currency": "USD",
        "sub_trade": False,
        "current_rate": 1.099e-05,
        "amount": 1333.3333333333335,
        "analyzed_candle": {"open": 1.1, "high": 2.2, "low": 1.0, "close": 1.5},
        "open_date": dt_now() + timedelta(hours=-1),
    }
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(msg)
    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""

    assert msg_mock.call_args[0][0] == (
        f"\N{LARGE BLUE CIRCLE} *Binance (dry):* New Trade (#1)\n"
        f"*Pair:* `ETH/BTC`\n"
        "*Candle OHLC*: `1.1, 2.2, 1.0, 1.5`\n"
        f"*Enter Tag:* `{enter_signal}`\n"
        "*Amount:* `1333.33333333`\n"
        f"*Direction:* `{enter}"
        f"{leverage_text}`\n"
        "*Open Rate:* `0.00001099 BTC`\n"
        "*Current Rate:* `0.00001099 BTC`\n"
        "*Total:* `0.01465333 BTC / 180.895 USD`"
    )

    freqtradebot.config["telegram"]["notification_settings"] = {"entry": "off"}
    caplog.clear()
    msg_mock.reset_mock()
    telegram.send_msg(msg)
    assert msg_mock.call_count == 0
    assert log_has("Notification 'entry' not sent.", caplog)

    freqtradebot.config["telegram"]["notification_settings"] = {"entry": "silent"}
    caplog.clear()
    msg_mock.reset_mock()

    telegram.send_msg(msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]["disable_notification"] is True


@pytest.mark.parametrize(
    "message_type,enter_signal",
    [
        (RPCMessageType.ENTRY_CANCEL, "long_signal_01"),
        (RPCMessageType.ENTRY_CANCEL, "short_signal_01"),
    ],
)
def test_send_msg_enter_cancel_notification(
    default_conf: Dict[str, Any], mocker: MagicMock, message_type: RPCMessageType, enter_signal: str
) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "enter_tag": enter_signal,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "reason": CANCEL_REASON["TIMEOUT"],
        }
    )
    assert (
        msg_mock.call_args[0][0] == "\N{WARNING SIGN} *Binance (dry):* "
        "Cancelling enter Order for ETH/BTC (#1). "
        "Reason: cancelled due to timeout."
    )


def test_send_msg_protection_notification(default_conf: Dict[str, Any], mocker: MagicMock, time_machine: Any) -> None:
    default_conf["telegram"]["notification_settings"]["protection_trigger"] = "on"

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    time_machine.move_to("2021-09-01 05:00:00 +00:00")
    lock = PairLocks.lock_pair("ETH/BTC", dt_now() + timedelta(minutes=6), "randreason")
    msg = {
        "type": RPCMessageType.PROTECTION_TRIGGER,
    }
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (
        msg_mock.call_args[0][0] == "*Protection* triggered due to randreason. "
        "`ETH/BTC` will be locked until `2021-09-01 05:10:00`."
    )

    msg_mock.reset_mock()
    # Test global protection

    msg = {
        "type": RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
    }
    lock = PairLocks.lock_pair("*", dt_now() + timedelta(minutes=100), "randreason")
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (
        msg_mock.call_args[0][0] == "*Protection* triggered due to randreason. "
        "*All pairs* will be locked until `2021-09-01 06:45:00`."
    )


@pytest.mark.parametrize(
    "message_type,entered,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY_FILL, "Long", "long_signal_01", 1.0),
        (RPCMessageType.ENTRY_FILL, "Long", "long_signal_02", 2.0),
        (RPCMessageType.ENTRY_FILL, "Short", "short_signal_01", 2.0),
    ],
)
def test_send_msg_entry_fill_notification(
    default_conf: Dict[str, Any], mocker: MagicMock, message_type: RPCMessageType, entered: str, enter_signal: str, leverage: float
) -> None:
    default_conf["telegram"]["notification_settings"]["entry_fill"] = "on"
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "trade_id": 1,
            "enter_tag": enter_signal,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "stake_amount": 0.01465333,
            "direction": entered,
            "sub_trade": False,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": "USD",
            "open_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )
    leverage_text = f" ({leverage:.3g}x)" if leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        f"\N{CHECK MARK} *Binance (dry):* New Trade filled (#1)\n"
        f"*Pair:* `ETH/BTC`\n"
        f"*Enter Tag:* `{enter_signal}`\n"
        "*Amount:* `1333.33333333`\n"
        f"*Direction:* `{entered}"
        f"{leverage_text}`\n"
        "*Open Rate:* `0.00001099 BTC`\n"
        "*Total:* `0.01465333 BTC / 180.895 USD`"
    )

    msg_mock.reset_mock()
    telegram.send_msg(
        {
            "type": message_type,
            "trade_id": 1,
            "enter_tag": enter_signal,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "stake_amount": 0.01465333,
            "sub_trade": True,
            "direction": entered,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": "USD",
            "open_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )

    assert msg_mock.call_args[0][0] == (
        f"\N{CHECK MARK} *Binance (dry):* Position increase filled (#1)\n"
        f"*Pair:* `ETH/BTC`\n"
        f"*Enter Tag:* `{enter_signal}`\n"
        "*Amount:* `1333.33333333`\n"
        f"*Direction:* `{entered}"
        f"{leverage_text}`\n"
        "*Open Rate:* `0.00001099 BTC`\n"
        "*New Total:* `0.01465333 BTC / 180.895 USD`"
    )


def test_send_msg_exit_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

        old_convamount = telegram._rpc._fiat_converter.convert_amount
        telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "leverage": 1.0,
                "direction": "Long",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": "USD",
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(hours=1),
                "close_date": dt_now(),
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n"
            "*Unrealized Profit:* `-57.41% (loss: -0.05746 ETH / -24.812 USD)`\n"
            "*Enter Tag:* `buy_signal1`\n"
            "*Exit Reason:* `stop_loss`\n"
            "*Direction:* `Long`\n"
            "*Amount:* `1333.33333333`\n"
            "*Open Rate:* `0.00075 ETH`\n"
            "*Current Rate:* `0.0003201 ETH`\n"
            "*Exit Rate:* `0.0003201 ETH`\n"
            "*Duration:* `1:00:00 (60.0 min)`"
        )

        msg_mock.reset_mock()
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "direction": "Long",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "cumulative_profit": -0.15746268,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": "USD",
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
                "stake_amount": 0.01,
                "sub_trade": True,
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (dry):* Partially exiting KEY/ETH (#1)\n"
            "*Unrealized Sub Profit:* `-57.41% (loss: -0.05746 ETH / -24.812 USD)`\n"
            "*Cumulative Profit:* `-0.15746 ETH / -24.812 USD`\n"
            "*Enter Tag:* `buy_signal1`\n"
            "*Exit Reason:* `stop_loss`\n"
            "*Direction:* `Long`\n"
            "*Amount:* `1333.33333333`\n"
            "*Open Rate:* `0.00075 ETH`\n"
            "*Current Rate:* `0.0003201 ETH`\n"
            "*Exit Rate:* `0.0003201 ETH`\n"
            "*Remaining:* `0.01 ETH / -24.812 USD`"
        )

        msg_mock.reset_mock()
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "direction": "Long",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": None,
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n"
            "*Unrealized Profit:* `-57.41% (loss: -0.05746 ETH)`\n"
            "*Enter Tag:* `buy_signal1`\n"
            "*Exit Reason:* `stop_loss`\n"
            "*Direction:* `Long`\n"
            "*Amount:* `1333.33333333`\n"
            "*Open Rate:* `0.00075 ETH`\n"
            "*Current Rate:* `0.0003201 ETH`\n"
            "*Exit Rate:* `0.0003201 ETH`\n"
            "*Duration:* `1 day, 2:30:00 (1590.0 min)`"
        )
        # Reset singleton function to avoid random breaks
        telegram._rpc._fiat_converter.convert_amount = old_convamount


async def test_send_msg_exit_cancel_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT_CANCEL,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "reason": "Cancelled on exchange",
        }
    )
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance (dry):* Cancelling exit Order for KEY/ETH (#1)."
        " Reason: Cancelled on exchange."
    )

    msg_mock.reset_mock()
    # Test with live mode (no dry appendix)
    telegram._config["dry_run"] = False
    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT_CANCEL,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "reason": "timeout",
        }
    )
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance:* Cancelling exit Order for KEY/ETH (#1). Reason: timeout."
    )
    # Reset singleton function to avoid random breaks
    telegram._rpc._fiat_converter.convert_amount = old_convamount


@pytest.mark.parametrize(
    "direction,enter_signal,leverage",
    [
        ("Long", "long_signal_01", None),
        ("Long", "long_signal_01", 1.0),
        ("Long", "long_signal_01", 5.0),
        ("Short", "short_signal_01", 2.0),
    ],
)
def test_send_msg_exit_fill_notification(
    default_conf: Dict[str, Any], mocker: MagicMock, direction: str, enter_signal: str, leverage: Optional[float]
) -> None:
    default_conf["telegram"]["notification_settings"]["exit_fill"] = "on"
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT_FILL,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "leverage": leverage,
                "direction": direction,
                "gain": "loss",
                "limit": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "close_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": None,
                "enter_tag": enter_signal,
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
            }
        )

        leverage_text = f" ({leverage:.3g}x)`\n" if leverage and leverage != 1.0 else "`\n"
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (dry):* Exited KEY/ETH (#1)\n"
            "*Profit:* `-57.41% (loss: -0.05746 ETH)`\n"
            f"*Enter Tag:* `{enter_signal}`\n"
            "*Exit Reason:* `stop_loss`\n"
            f"*Direction:* `{direction}"
            f"{leverage_text}"
            "*Amount:* `1333.33333333`\n"
            "*Open Rate:* `0.00075 ETH`\n"
            "*Exit Rate:* `0.0003201 ETH`\n"
            "*Duration:* `1 day, 2:30:00 (1590.0 min)`"
        )


def test_send_msg_status_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STATUS, "status": "running"})
    assert msg_mock.call_args[0][0] == "*Status:* `running`"


async def test_warning_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.WARNING, "status": "message"})
    assert msg_mock.call_args[0][0] == "\N{WARNING SIGN} *Warning:* `message`"


def test_startup_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STARTUP, "status": "*Custom:* `Hello World`"})
    assert msg_mock.call_args[0][0] == "*Custom:* `Hello World`"


def test_send_msg_strategy_msg_notification(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STRATEGY_MSG, "msg": "hello world, Test msg"})
    assert msg_mock.call_args[0][0] == "hello world, Test msg"


def test_send_msg_unknown_type(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg(
        {
            "type": None,
        }
    )
    assert msg_mock.call_count == 0


@pytest.mark.parametrize(
    "message_type,enter,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY, "Long", "long_signal_01", None),
        (RPCMessageType.ENTRY, "Long", "long_signal_01", 2.0),
        (RPCMessageType.ENTRY, "Short", "short_signal_01", 2.0),
    ],
)
def test_send_msg_buy_notification_no_fiat(
    default_conf: Dict[str, Any], mocker: MagicMock, message_type: RPCMessageType, enter: str, enter_signal: str, leverage: Optional[float]
) -> None:
    del default_conf["fiat_display_currency"]
    default_conf["dry_run"] = False
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "enter_tag": enter_signal,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "open_rate": 1.099e-05,
            "order_type": "limit",
            "direction": enter,
            "sub_trade": False,
            "stake_amount": 0.01465333,
            "stake_amount_fiat": 0.0,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": None,
            "current_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )

    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        f"\N{LARGE BLUE CIRCLE} *Binance:* New Trade (#1)\n"
        "*Pair:* `ETH/BTC`\n"
        f"*Enter Tag:* `{enter_signal}`\n"
        "*Amount:* `1333.33333333`\n"
        f"*Direction:* `{enter}"
        f"{leverage_text}`\n"
        "*Open Rate:* `0.00001099 BTC`\n"
        "*Current Rate:* `0.00001099 BTC`\n"
        "*Total:* `0.01465333 BTC`"
    )


@pytest.mark.parametrize(
    "direction,enter_signal,leverage",
    [
        ("Long", "long_signal_01", None),
        ("Long", "long_signal_01", 1.0),
        ("Long", "long_signal_01", 5.0),
        ("Short", "short_signal_01", 2.0),
    ],
)
@pytest.mark.parametrize("fiat", ["", None])
def test_send_msg_exit_notification_no_fiat(
    default_conf: Dict[str, Any], mocker: MagicMock, direction: str, enter_signal: str, leverage: Optional[float], time_machine: Any, fiat: Optional[str]
) -> None:
    if fiat is None:
        del default_conf["fiat_display_currency"]
    else:
        default_conf["fiat_display_currency"] = fiat
    time_machine.move_to("2022-05-02 00:00:00 +00:00", tick=False)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "gain": "loss",
            "leverage": leverage,
            "direction": direction,
            "sub_trade": False,
            "order_rate": 3.201e-04,
            "amount": 1333.3333333333335,
            "order_type": "limit",
            "open_rate": 7.5e-04,
            "current_rate": 3.201e-04,
            "profit_amount": -0.05746268,
            "profit_ratio": -0.57405275,
            "stake_currency": "ETH",
            "quote_currency": "ETH",
            "base_currency": "KEY",
            "fiat_currency": "USD",
            "enter_tag": enter_signal,
            "exit_reason": ExitType.STOP_LOSS.value,
            "open_date": dt_now() - timedelta(hours=2, minutes=35, seconds=3),
            "close_date": dt_now(),
        }
    )

    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance (dry):* Exiting KEY/ETH (#1)\n"
        "*Unrealized Profit:* `-57.41% (loss: -0.05746 ETH)`\n"
        f"*Enter Tag:* `{enter_signal}`\n"
        "*Exit Reason:* `stop_loss`\n"
        f"*Direction:* `{direction}"
        f"{leverage_text}`\n"
        "*Amount:* `1333.33333333`\n"
        "*Open Rate:* `0.00075 ETH`\n"
        "*Current Rate:* `0.0003201 ETH`\n"
        "*Exit Rate:* `0.0003201 ETH`\n"
        "*Duration:* `2:35:03 (155.1 min)`"
    )


@pytest.mark.parametrize(
    "msg,expected",
    [
        ({"profit_ratio": 0.201, "exit_reason": "roi"}, "\N{ROCKET}"),
        ({"profit_ratio": 0.051, "exit_reason": "roi"}, "\N{ROCKET}"),
        ({"profit_ratio": 0.0256, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": 0.01, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": 0.0, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": -0.05, "exit_reason": "stop_loss"}, "\N{WARNING SIGN}"),
        ({"profit_ratio": -0.02, "exit_reason": "sell_signal"}, "\N{CROSS MARK}"),
    ],
)
def test__exit_emoji(default_conf: Dict[str, Any], mocker: MagicMock, msg: Dict[str, Any], expected: str) -> None:
    del default_conf["fiat_display_currency"]

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)

    assert telegram._get_exit_emoji(msg) == expected


async def test_telegram__send_msg(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    await telegram._send_msg("test")
    assert len(bot.method_calls) == 1

    # Test update
    query = MagicMock()
    query.edit_message_text = AsyncMock()
    await telegram._send_msg("test", callback_path="DeadBeef", query=query, reload_able=True)
    assert query.edit_message_text.call_count == 1
    assert "Updated: " in query.edit_message_text.call_args_list[0][1]["text"]

    query.edit_message_text = AsyncMock(side_effect=BadRequest("not modified"))
    await telegram._send_msg("test", callback_path="DeadBeef", query=query)
    assert query.edit_message_text.call_count == 1
    assert not log_has_re(r"TelegramError: .*", caplog)

    query.edit_message_text = AsyncMock(side_effect=BadRequest(""))
    await telegram._send_msg("test2", callback_path="DeadBeef", query=query)
    assert query.edit_message_text.call_count == 1
    assert log_has_re(r"TelegramError: .*", caplog)

    query.edit_message_text = AsyncMock(side_effect=TelegramError("DeadBEEF"))
    await telegram._send_msg("test3", callback_path="DeadBeef", query=query)

    assert log_has_re(r"TelegramError: DeadBEEF! Giving up.*", caplog)


async def test__send_msg_network_error(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError("Oh snap"))
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    telegram._config["telegram"]["enabled"] = True
    await telegram._send_msg("test")

    # Bot should've tried to send it twice
    assert len(bot.method_calls) == 2
    assert log_has("Telegram NetworkError: Oh snap! Trying one more time.", caplog)


@pytest.mark.filterwarnings("ignore:.*ChatPermissions")
async def test__send_msg_keyboard(default_conf: Dict[str, Any], mocker: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)

    invalid_keys_list = [["/not_valid", "/profit"], ["/daily"], ["/alsoinvalid"]]
    default_keys_list = [
        ["/daily", "/profit", "/balance"],
        ["/status", "/status table", "/performance"],
        ["/count", "/start", "/stop", "/help"],
    ]
    default_keyboard = ReplyKeyboardMarkup(default_keys_list)

    custom_keys_list = [
        ["/daily", "/stats", "/balance", "/profit", "/profit 5"],
        ["/count", "/start", "/reload_config", "/help"],
    ]
    custom_keyboard = ReplyKeyboardMarkup(custom_keys_list)

    def init_telegram(freqtradebot: FreqtradeBot) -> Telegram:
        telegram = Telegram(rpc, default_conf)
        telegram._app = MagicMock()
        telegram._app.bot = bot
        return telegram

    # no keyboard in config -> default keyboard
    freqtradebot.config["telegram"]["enabled"] = True
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg("test")
    used_keyboard = bot.send_message.call_args[1]["reply_markup"]
    assert used_keyboard == default_keyboard

    # invalid keyboard in config -> default keyboard
    freqtradebot.config["telegram"]["enabled"] = True
    freqtradebot.config["telegram"]["keyboard"] = invalid_keys_list
    err_msg = (
        re.escape(
            "config.telegram.keyboard: Invalid commands for custom "
            "Telegram keyboard: ['/not_valid', '/alsoinvalid']"
            "\nvalid commands are: "
        )
        + r"*"
    )
    with pytest.raises(OperationalException, match=err_msg):
        telegram = init_telegram(freqtradebot)

    # valid keyboard in config -> custom keyboard
    freqtradebot.config["telegram"]["enabled"] = True
    freqtradebot.config["telegram"]["keyboard"] = custom_keys_list
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg("test")
    used_keyboard = bot.send_message.call_args[1]["reply_markup"]
    assert used_keyboard == custom_keyboard
    assert log_has(
        "using custom keyboard from config.json: "
        "[['/daily', '/stats', '/balance', '/profit', '/profit 5'], ['/count', "
        "'/start', '/reload_config', '/help']]",
        caplog,
    )


async def test_change_market_direction(default_conf: Dict[str, Any], mocker: MagicMock, update: Update) -> None:
    telegram, _, _msg_mock = get_telegram_testobject(mocker, default_conf)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.NONE
    context = MagicMock()
    context.args = ["long"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG
    context = MagicMock()
    context.args = ["invalid"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG


async def test_telegram_list_custom_data(default_conf_usdt: Dict[str, Any], update: Update, ticker: MagicMock, fee: float, mocker: MagicMock) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # Create some test data
    create_mock_trades_usdt(fee)
    # No trade id
    context = MagicMock()
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Trade-id not set." in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    #
    context.args = ["1"]
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "Didn't find any custom-data entries for Trade ID: `1`" in msg_mock.call_args_list[0][0][0]
    )
    msg_mock.reset_mock()

    # Add some custom data
    trade1 = Trade.get_trades_proxy()[0]
    trade1.set_custom_data("test_int", 1)
    trade1.set_custom_data("test_dict", {"test": "dict"})
    Trade.commit()
    context.args = [f"{trade1.id}"]
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 3
    assert "Found custom-data entries: " in msg_mock.call_args_list[0][0][0]
    assert (
        "*Key:* `test_int`\n*ID:* `1`\n*Trade ID:* `1`\n*Type:* `int`\n*Value:* `1`\n*Create Date:*"
    ) in msg_mock.call_args_list[1][0][0]
    assert (
        "*Key:* `test_dict`\n*ID:* `2`\n*Trade ID:* `1`\n*Type:* `dict`\n"
        '*Value:* `{"test": "dict"}`\n*Create Date:* `'
    ) in msg_mock.call_args_list[2][0][0]

    msg_mock.reset_mock()


def test_noficiation_settings(default_conf_usdt: Dict[str, Any], mocker: MagicMock) -> None:
    (telegram, _, _) = get_telegram_testobject(mocker, default_conf_usdt)
    telegram._config["telegram"].update(
        {
            "notification_settings": {
                "status": "silent",
                "warning": "on",
                "startup": "off",
                "entry": "silent",
                "entry_fill": "on",
                "entry_cancel": "silent",
                "exit": {
                    "roi": "silent",
                    "emergency_exit": "on",
                    "force_exit": "on",
                    "exit_signal": "silent",
                    "trailing_stop_loss": "on",
                    "stop_loss": "on",
                    "stoploss_on_exchange": "on",
                    "custom_exit": "silent",
                    "partial_exit": "off",
                },
                "exit_fill": {
                    "roi": "silent",
                    "partial_exit": "off",
                    "*": "silent",  # Default to silent
                },
                "exit_cancel": "on",
                "protection_trigger": "off",
                "protection_trigger_global": "on",
                "strategy_msg": "off",
                "show_candle": "off",
            }
        }
    )

    loudness = telegram._message_loudness

    assert loudness({"type": RPCMessageType.ENTRY, "exit_reason": ""}) == "silent"
    assert loudness({"type": RPCMessageType.ENTRY_FILL, "exit_reason": ""}) == "on"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": ""}) == "on"
    # Default to silent due to "*" definition
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": ""}) == "silent"
    assert loudness({"type": RPCMessageType.PROTECTION_TRIGGER, "exit_reason": ""}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "roi"}) == "silent"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "partial_exit"}) == "off"
    # Not given key defaults to on
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "cust_exit112"}) == "on"

    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "roi"}) == "silent"
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "partial_exit"}) == "off"
    # Default to silent due to "*" definition
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "cust_exit112"}) == "silent"

    # Simplified setup for exit
    telegram._config["telegram"].update(
        {
            "notification_settings": {
                "status": "silent",
                "warning": "on",
                "startup": "off",
                "entry": "silent",
                "entry_fill": "on",
                "entry_cancel": "silent",
                "exit": "off",
                "exit_cancel": "on",
                "exit_fill": "on",
                "protection_trigger": "off",
                "protection_trigger_global": "on",
                "strategy_msg": "off",
                "show_candle": "off",
            }
        }
    )

    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "roi"}) == "on"
    # All regular exits are off
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "roi"}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "partial_exit"}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "cust_exit112"}) == "off"


async def test__tg_info(default_conf_usdt: Dict[str, Any], mocker: MagicMock, update: Update) -> None:
    (telegram, _, _) = get_telegram_testobject(mocker, default_conf_usdt)
    context = AsyncMock()

    await telegram._tg_info(update, context)

    assert context.bot.send_message.call_count == 1
    content = context.bot.send_message.call_args[1]["text"]
    assert "Freqtrade Bot Info:\n" in content
    assert '"chat_id": "1235"' in content
