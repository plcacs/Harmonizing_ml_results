import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Union
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
from freqtrade.enums import ExitType, MarketDirection, RPCMessageType, RunMode, SignalDirection, State
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.loggers import setup_logging
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc import RPC
from freqtrade.rpc.rpc import RPCException
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, log_has, log_has_re, patch_exchange, patch_get_signal, patch_whitelist


@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker: Any) -> None:
    mocker.patch('freqtrade.exchange.exchange.Exchange._init_async_loop')


@pytest.fixture
def default_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['telegram']['enabled'] = True
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
        self.state = {'called': False}

    def _init(self) -> None:
        pass

    @authorized_only
    async def dummy_handler(self, *args: Any, **kwargs: Any) -> None:
        """
        Fake method that only change the state of the object
        """
        self.state['called'] = True

    @authorized_only
    async def dummy_exception(self, *args: Any, **kwargs: Any) -> None:
        """
        Fake method that throw an exception
        """
        raise Exception('test')


def get_telegram_testobject(mocker: Any, default_conf: Dict[str, Any], mock: bool = True, ftbot: Optional[FreqtradeBot] = None) -> tuple[Telegram, FreqtradeBot, AsyncMock]:
    msg_mock = AsyncMock()
    if mock:
        mocker.patch.multiple('freqtrade.rpc.telegram.Telegram', _init=MagicMock(), _send_msg=msg_mock, _start_thread=MagicMock())
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    telegram = Telegram(rpc, default_conf)
    telegram._loop = MagicMock()
    patch_eventloop_threading(telegram)
    return (telegram, ftbot, msg_mock)


async def test_telegram__init__(default_conf: Dict[str, Any], mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


async def test_telegram_init(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    app_mock = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._start_thread', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init_telegram_app', return_value=app_mock)
    mocker.patch('freqtrade.rpc.telegram.Telegram._startup_telegram', AsyncMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._init()
    assert app_mock.call_count == 0
    assert app_mock.add_handler.call_count > 0
    message_str = "rpc.telegram is listening for following commands: [['status'], ['profit'], ['balance'], ['start'], ['stop'], ['forceexit', 'forcesell', 'fx'], ['forcebuy', 'forcelong'], ['forceshort'], ['reload_trade'], ['trades'], ['delete'], ['cancel_open_order', 'coo'], ['performance'], ['buys', 'entries'], ['exits', 'sells'], ['mix_tags'], ['stats'], ['daily'], ['weekly'], ['monthly'], ['count'], ['locks'], ['delete_locks', 'unlock'], ['reload_conf', 'reload_config'], ['show_conf', 'show_config'], ['stopbuy', 'stopentry'], ['whitelist'], ['blacklist'], ['bl_delete', 'blacklist_delete'], ['logs'], ['edge'], ['health'], ['help'], ['version'], ['marketdir'], ['order'], ['list_custom_data'], ['tg_info']]"
    assert log_has(message_str, caplog)


async def test_telegram_startup(default_conf: Dict[str, Any], mocker: Any) -> None:
    app_mock = MagicMock()
    app_mock.initialize = AsyncMock()
    app_mock.start = AsyncMock()
    app_mock.updater.start_polling = AsyncMock()
    app_mock.updater.running = False
    sleep_mock = mocker.patch('freqtrade.rpc.telegram.asyncio.sleep', AsyncMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    await telegram._startup_telegram()
    assert app_mock.initialize.call_count == 1
    assert app_mock.start.call_count == 1
    assert app_mock.updater.start_polling.call_count == 1
    assert sleep_mock.call_count == 1


async def test_telegram_cleanup(default_conf: Dict[str, Any], mocker: Any) -> None:
    app_mock = MagicMock()
    app_mock.stop = AsyncMock()
    app_mock.initialize = AsyncMock()
    updater_mock = MagicMock()
    updater_mock.stop = AsyncMock()
    app_mock.updater = updater_mock
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    telegram._loop = asyncio.get_running_loop()
    telegram._thread = MagicMock()
    telegram.cleanup()
    await asyncio.sleep(0.1)
    assert app_mock.stop.call_count == 1
    assert telegram._thread.join.call_count == 1


async def test_authorized_only(default_conf: Dict[str, Any], mocker: Any, caplog: Any, update: Update) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is True
    assert log_has('Executing handler: dummy_handler for chat_id: 1235', caplog)
    assert not log_has('Rejected unauthorized message from: 1235', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


async def test_authorized_only_unauthorized(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    chat = Chat(3735928559, 0)
    message = Message(randint(1, 100), datetime.now(timezone.utc), chat)
    update = Update(randint(1, 100), message=message)
    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 3735928559', caplog)
    assert log_has('Rejected unauthorized message from: 3735928559', caplog)
    assert not log_has('Exception occurred within Telegram module', caplog)


async def test_authorized_only_exception(default_conf: Dict[str, Any], mocker: Any, caplog: Any, update: Update) -> None:
    patch_exchange(mocker)
    default_conf['telegram']['enabled'] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)
    await dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert log_has('Exception occurred within Telegram module', caplog)


async def test_telegram_status(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    default_conf['telegram']['enabled'] = False
    status_table = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._status_table', status_table)
    mocker.patch.multiple('freqtrade.rpc.rpc.RPC', _rpc_trade_status=MagicMock(return_value=[{'trade_id': 1, 'pair': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC', 'open_date': dt_now(), 'close_date': None, 'open_rate': 1.099e-05, 'close_rate': None, 'current_rate': 1.098e-05, 'amount': 90.99181074, 'stake_amount': 90.99181074, 'max_stake_amount': 90.99181074, 'buy_tag': None, 'enter_tag': None, 'close_profit_ratio': None, 'profit': -0.0059, 'profit_ratio': -0.0059, 'profit_abs': -0.225, 'realized_profit': 0.0, 'total_profit_abs': -0.225, 'initial_stop_loss_abs': 1.098e-05, 'stop_loss_abs': 1.099e-05, 'exit_order_status': None, 'initial_stop_loss_ratio': -0.0005, 'stoploss_current_dist': 1e-08, 'stoploss_current_dist_ratio': -0.0002, 'stop_loss_ratio': -0.0001, 'open_order': '(limit buy rem=0.00000000)', 'is_open': True, 'is_short': False, 'filled_entry_orders': [], 'orders': []}]))
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    context = MagicMock()
    context.args = ['table']
    await telegram._status(update=update, context=context)
    assert status_table.call_count == 1


@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_multi_entry(default_conf: Dict[str, Any], update: Update, mocker: Any, fee: Any) -> None:
    default_conf['telegram']['enabled'] = False
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(EXMS, fetch_order=MagicMock(return_value=None), get_rate=MagicMock(return_value=0.22))
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    trade.orders[0].average = 0
    trade.orders.append(Order(order_id='5412vbb', ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, ft_amount=trade.amount, ft_price=trade.open_rate, status='closed', symbol=trade.pair, order_type='market', side='buy', price=trade.open_rate * 0.95, average=0, filled=trade.amount, remaining=0, cost=trade.amount, order_date=trade.open_date, order_filled_date=trade.open_date))
    trade.recalc_trade_from_orders()
    Trade.commit()
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search('Number of Entries.*2', msg)
    assert re.search('Number of Exits.*1', msg)
    assert re.search('Close Date:', msg) is None
    assert re.search('Close Profit:', msg) is None


@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_closed_trade(default_conf: Dict[str, Any], update: Update, mocker: Any, fee: Any) -> None:
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(EXMS, fetch_order=MagicMock(return_value=None), get_rate=MagicMock(return_value=0.22))
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.is_open.is_(False)]).first()
    context = MagicMock()
    context.args = [str(trade.id)]
    await telegram._status(update=update, context=context)
    assert msg_mock.call_count == 1
    msg = msg_mock.call_args_list[0][0][0]
    assert re.search('Close Date:', msg)
    assert re.search('Close Profit:', msg)


async def test_order_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    default_conf['max_open_trades'] = 3
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    status_table = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram.Telegram', _status_table=status_table)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.state = State.RUNNING
    msg_mock.reset_mock()
    freqtradebot.enter_positions()
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 500)
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['2']
    await telegram._order(update=update, context=context)
    assert msg_mock.call_count == 1
    msg1 = msg_mock.call_args_list[0][0][0]
    assert 'Order List for Trade #*`2`' in msg1
    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 50)
    context = MagicMock()
    context.args = ['2']
    await telegram._order(update=update, context=context)
    assert msg_mock.call_count == 2
    msg1 = msg_mock.call_args_list[0][0][0]
    msg2 = msg_mock.call_args_list[1][0][0]
    assert 'Order List for Trade #*`2`' in msg1
    assert '*Order List for Trade #*`2` - continued' in msg2


@pytest.mark.usefixtures('init_persistence')
async def test_telegram_order_multi_entry(default_conf: Dict[str, Any], update: Update, mocker: Any, fee: Any) -> None:
    default_conf['telegram']['enabled'] = False
    default_conf['position_adjustment_enable'] = True
    mocker.patch.multiple(EXMS, fetch_order=MagicMock(return_value=None), get_rate=MagicMock(return_value=0.22))
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    trade.orders[0].average = 0
    trade.orders.append(Order(order_id='541