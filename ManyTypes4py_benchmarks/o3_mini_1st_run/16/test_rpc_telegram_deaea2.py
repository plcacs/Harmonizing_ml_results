#!/usr/bin/env python3
"""
Type‐annotated test module.
"""

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from typing import Any, Dict, Optional, Tuple

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

# Fixture for pytest monkeypatching exchange loops
@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker: Any) -> None:
    mocker.patch('freqtrade.exchange.exchange.Exchange._init_async_loop')


@pytest.fixture
def default_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['telegram']['enabled'] = True
    return default_conf


@pytest.fixture
def update() -> Update:
    message: Message = Message(0, datetime.now(timezone.utc), Chat(1235, 0))
    _update: Update = Update(0, message=message)
    return _update


def patch_eventloop_threading(telegrambot: Telegram) -> None:
    is_init: bool = False

    def thread_fuck() -> None:
        nonlocal is_init
        telegrambot._loop = asyncio.new_event_loop()
        is_init = True
        telegrambot._loop.run_forever()
    x: threading.Thread = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


class DummyCls(Telegram):
    """
    Dummy class for testing the Telegram @authorized_only decorator
    """
    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        super().__init__(rpc, config)
        self.state: Dict[str, bool] = {'called': False}

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


def get_telegram_testobject(
    mocker: Any,
    default_conf: Dict[str, Any],
    mock: bool = True,
    ftbot: Optional[FreqtradeBot] = None
) -> Tuple[Telegram, FreqtradeBot, Any]:
    msg_mock: Any = AsyncMock()
    if mock:
        mocker.patch.multiple('freqtrade.rpc.telegram.Telegram', _init=MagicMock(), _send_msg=msg_mock, _start_thread=MagicMock())
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc: RPC = RPC(ftbot)
    telegram: Telegram = Telegram(rpc, default_conf)
    telegram._loop = MagicMock()
    patch_eventloop_threading(telegram)
    return telegram, ftbot, msg_mock


def test_telegram__init__(default_conf: Dict[str, Any], mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


def test_telegram_init(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    app_mock: Any = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._start_thread', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init_telegram_app', return_value=app_mock)
    mocker.patch('freqtrade.rpc.telegram.Telegram._startup_telegram', AsyncMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._init()
    assert app_mock.call_count == 0
    assert app_mock.add_handler.call_count > 0
    message_str: str = ("rpc.telegram is listening for following commands: [['status'], ['profit'], ['balance'], ['start'], "
                          "['stop'], ['forceexit', 'forcesell', 'fx'], ['forcebuy', 'forcelong'], ['forceshort'], ['reload_trade'], "
                          "['trades'], ['delete'], ['cancel_open_order', 'coo'], ['performance'], ['buys', 'entries'], ['exits', 'sells'], "
                          "['mix_tags'], ['stats'], ['daily'], ['weekly'], ['monthly'], ['count'], ['locks'], ['delete_locks', 'unlock'], "
                          "['reload_conf', 'reload_config'], ['show_conf', 'show_config'], ['stopbuy', 'stopentry'], ['whitelist'], "
                          "['blacklist'], ['bl_delete', 'blacklist_delete'], ['logs'], ['edge'], ['health'], ['help'], ['version'], "
                          "['marketdir'], ['order'], ['list_custom_data'], ['tg_info']]")
    assert log_has(message_str, caplog)


async def test_telegram_startup(default_conf: Dict[str, Any], mocker: Any) -> None:
    app_mock: Any = MagicMock()
    app_mock.initialize = AsyncMock()
    app_mock.start = AsyncMock()
    app_mock.updater.start_polling = AsyncMock()
    app_mock.updater.running = False
    sleep_mock: Any = mocker.patch('freqtrade.rpc.telegram.asyncio.sleep', AsyncMock())
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    await telegram._startup_telegram()
    assert app_mock.initialize.call_count == 1
    assert app_mock.start.call_count == 1
    assert app_mock.updater.start_polling.call_count == 1
    assert sleep_mock.call_count == 1


async def test_telegram_cleanup(default_conf: Dict[str, Any], mocker: Any) -> None:
    app_mock: Any = MagicMock()
    app_mock.stop = AsyncMock()
    app_mock.initialize = AsyncMock()
    updater_mock: Any = MagicMock()
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
    bot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(bot)
    dummy: DummyCls = DummyCls(rpc, default_conf)
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
    update_obj = Update(randint(1, 100), message=message)
    default_conf['telegram']['enabled'] = False
    bot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(bot)
    dummy: DummyCls = DummyCls(rpc, default_conf)
    patch_get_signal(bot)
    await dummy.dummy_handler(update=update_obj, context=MagicMock())
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 3735928559', caplog)
    assert log_has('Rejected unauthorized message from: 3735928559', caplog)


async def test_authorized_only_exception(default_conf: Dict[str, Any], mocker: Any, caplog: Any, update: Update) -> None:
    patch_exchange(mocker)
    default_conf['telegram']['enabled'] = False
    bot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(bot)
    dummy: DummyCls = DummyCls(rpc, default_conf)
    patch_get_signal(bot)
    try:
        await dummy.dummy_exception(update=update, context=MagicMock())
    except Exception:
        pass
    assert dummy.state['called'] is False
    assert not log_has('Executing handler: dummy_handler for chat_id: 0', caplog)
    assert not log_has('Rejected unauthorized message from: 0', caplog)
    assert log_has('Exception occurred within Telegram module', caplog)


async def test_telegram_status(default_conf: Dict[str, Any], update: Update, mocker: Any, ticker: Any, fee: Any) -> None:
    default_conf['telegram']['enabled'] = False
    status_table: Any = MagicMock()
    mocker.patch('freqtrade.rpc.telegram.Telegram._status_table', status_table)
    mocker.patch.multiple('freqtrade.rpc.rpc.RPC', _rpc_trade_status=MagicMock(return_value=[{
        'trade_id': 1, 'pair': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC',
        'open_date': dt_now(), 'close_date': None, 'open_rate': 1.099e-05, 'close_rate': None,
        'current_rate': 1.098e-05, 'amount': 90.99181074, 'stake_amount': 90.99181074,
        'max_stake_amount': 90.99181074, 'buy_tag': None, 'enter_tag': None, 'close_profit_ratio': None,
        'profit': -0.0059, 'profit_ratio': -0.0059, 'profit_abs': -0.225, 'realized_profit': 0.0,
        'total_profit_abs': -0.225, 'initial_stop_loss_abs': 1.098e-05, 'stop_loss_abs': 1.099e-05,
        'exit_order_status': None, 'initial_stop_loss_ratio': -0.0005, 'stoploss_current_dist': 1e-08,
        'stoploss_current_dist_ratio': -0.0002, 'stop_loss_ratio': -0.0001,
        'open_order': '(limit buy rem=0.00000000)', 'is_open': True, 'is_short': False,
        'filled_entry_orders': [], 'orders': []
    }]))
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
    trade.orders.append(Order(
        order_id='5412vbb', ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False,
        ft_amount=trade.amount, ft_price=trade.open_rate, status='closed', symbol=trade.pair,
        order_type='market', side='buy', price=trade.open_rate * 0.95, average=0, filled=trade.amount,
        remaining=0, cost=trade.amount, order_date=trade.open_date, order_filled_date=trade.open_date))
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
    status_table: Any = MagicMock()
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
    trade.orders.append(Order(
        order_id='5412vbb', ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False,
        ft_amount=trade.amount, ft_price=trade.open_rate, status='closed', symbol=trade.pair,
        order_type='market', side='buy', price=trade.open_rate * 0.95, average=0, filled=trade.amount,
        remaining=0, cost=trade.amount, order_date=trade.open_date, order_filled_date=trade.open_date))
    trade.recalc_trade_from_orders()
    Trade.commit()
    await telegram._order(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search('from 1st entry rate', msg)
    assert re.search('Order Filled', msg)


async def test_status_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    default_conf['max_open_trades'] = 3
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    status_table: Any = MagicMock()
    mocker.patch.multiple('freqtrade.rpc.telegram.Telegram', _status_table=status_table)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.state = State.STOPPED
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.enter_positions()
    await telegram._status(update=update, context=MagicMock())
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines[:-1]
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)
    assert msg_mock.call_count == 3
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'LTC/BTC' in msg_mock.call_args_list[1][0][0]
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['2', '3']
    await telegram._status(update=update, context=context)
    lines = msg_mock.call_args_list[0][0][0].split('\n')
    assert '' not in lines[:-1]
    assert 'Close Rate' not in ''.join(lines)
    assert 'Close Profit' not in ''.join(lines)
    assert msg_mock.call_count == 2
    assert 'LTC/BTC' in msg_mock.call_args_list[0][0][0]
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 500)
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['2']
    await telegram._status(update=update, context=context)
    assert msg_mock.call_count == 1
    msg1 = msg_mock.call_args_list[0][0][0]
    assert 'Close Rate' not in msg1
    assert 'Trade ID:* `2`' in msg1


async def test_status_table_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    default_conf['stake_amount'] = 15.0
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.state = State.STOPPED
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'no active trade' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.enter_positions()
    await telegram._status_table(update=update, context=MagicMock())
    text = re.sub('</?pre>', '', msg_mock.call_args_list[-1][0][0])
    line = text.split('\n')
    fields = re.sub('[ ]+', ' ', line[2].strip()).split(' ')
    assert int(fields[0]) == 1
    assert 'ETH/BTC' in fields[1]
    assert msg_mock.call_count == 1


async def test_daily_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any, time_machine: Any) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    time_machine.move_to('2022-06-11 08:00:00+00:00')
    create_mock_trades_usdt(fee)
    context = MagicMock()
    context.args = ['2']
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Daily Profit over the last 2 days</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Day ' in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]
    assert '(2)  6.83 USDT  7.51 USD  0.64%' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = []
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Daily Profit over the last 7 days</b>:' in msg_mock.call_args_list[0][0][0]
    assert str(datetime.now(timezone.utc).date()) in msg_mock.call_args_list[0][0][0]
    assert str((datetime.now(timezone.utc) - timedelta(days=5)).date()) in msg_mock.call_args_list[0][0][0]
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]
    assert '(1)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['1']
    await telegram._daily(update=update, context=context)
    assert '  6.83 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  7.51 USD' in msg_mock.call_args_list[0][0][0]
    assert '(2)' in msg_mock.call_args_list[0][0][0]


async def test_daily_wrong_input(default_conf: Dict[str, Any], update: Update, ticker: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['-2']
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['today']
    await telegram._daily(update=update, context=context)
    assert 'Daily Profit over the last 7 days</b>:' in msg_mock.call_args_list[0][0][0]


async def test_weekly_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any, time_machine: Any) -> None:
    default_conf_usdt['max_open_trades'] = 1
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    time_machine.move_to('2022-06-11')
    create_mock_trades_usdt(fee)
    context = MagicMock()
    context.args = ['2']
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Weekly Profit over the last 2 weeks (starting from Monday)</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Monday ' in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    first_iso_day_of_current_week = today - timedelta(days=today.weekday())
    assert str(first_iso_day_of_current_week) in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = []
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Weekly Profit over the last 8 weeks (starting from Monday)</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Weekly' in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['-3']
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['this week']
    await telegram._weekly(update=update, context=context)
    assert 'Weekly Profit over the last 8 weeks (starting from Monday)</b>:' in msg_mock.call_args_list[0][0][0]


async def test_monthly_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any, time_machine: Any) -> None:
    default_conf_usdt['max_open_trades'] = 1
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    time_machine.move_to('2022-06-11')
    create_mock_trades_usdt(fee)
    context = MagicMock()
    context.args = ['2']
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Monthly Profit over the last 2 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Month ' in msg_mock.call_args_list[0][0][0]
    today = datetime.now(timezone.utc).date()
    current_month = f'{today.year}-{today.month:02} '
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = []
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Monthly Profit over the last 6 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Month ' in msg_mock.call_args_list[0][0][0]
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '(0)' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['12']
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Monthly Profit over the last 12 months</b>:' in msg_mock.call_args_list[0][0][0]
    assert '  2.74 USDT' in msg_mock.call_args_list[0][0][0]
    assert '  3.01 USD' in msg_mock.call_args_list[0][0][0]
    assert '(3)' in msg_mock.call_args_list[0][0][0]
    assert '-09' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['-3']
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'must be an integer greater than 0' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['february']
    await telegram._monthly(update=update, context=context)
    assert 'Monthly Profit over the last 6 months</b>:' in msg_mock.call_args_list[0][0][0]


async def test_telegram_profit_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker_usdt: Any, ticker_sell_up: Any, fee: Any, limit_sell_order_usdt: Any, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    await telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    context = MagicMock()
    context.args = ['aaa']
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'No closed trade' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    mocker.patch('freqtrade.wallets.Wallets.get_starting_balance', return_value=1000)
    assert '∙ `0.298 USDT (0.50%) (0.03 Σ%)`' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()
    mocker.patch(f'{EXMS}.fetch_ticker', ticker_sell_up)
    trade = Trade.session.scalars(select(Trade)).first()
    oobj = Order.parse_from_ccxt_object(limit_sell_order_usdt, limit_sell_order_usdt['symbol'], 'sell')
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    trade.close_date = datetime.now(timezone.utc)
    trade.is_open = False
    Trade.commit()
    context.args = [3]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert '*ROI:* Closed trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `5.685 USDT (9.45%) (0.57 Σ%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `6.253 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*ROI:* All trades' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `5.685 USDT (9.45%) (0.57 Σ%)`' in msg_mock.call_args_list[-1][0][0]
    assert '∙ `6.253 USD`' in msg_mock.call_args_list[-1][0][0]
    assert '*Best Performing:* `ETH/USDT: 5.685 USDT (9.47%)`' in msg_mock.call_args_list[-1][0][0]
    assert '*Max Drawdown:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Profit factor:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Winrate:*' in msg_mock.call_args_list[-1][0][0]
    assert '*Expectancy (Ratio):*' in msg_mock.call_args_list[-1][0][0]
    assert '*Trading volume:* `126 USDT`' in msg_mock.call_args_list[-1][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_stats(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any, is_short: bool) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No trades yet.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)
    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Exit Reason' in msg_mock.call_args_list[-1][0][0]
    assert 'ROI' in msg_mock.call_args_list[-1][0][0]
    assert 'Avg. Duration' in msg_mock.call_args_list[-1][0][0]
    assert '0:19:00' in msg_mock.call_args_list[-1][0][0]
    assert 'N/A' in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()


async def test_telegram_balance_handle(default_conf: Dict[str, Any], update: Update, mocker: Any, rpc_balance: Any, tickers: Any) -> None:
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value=rpc_balance)
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.get_valid_pair_combination', side_effect=lambda a, b: [f'{a}/{b}'])
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._balance(update=update, context=MagicMock())
    context = MagicMock()
    context.args = ['full']
    await telegram._balance(update=update, context=context)
    result = msg_mock.call_args_list[0][0][0]
    result_full = msg_mock.call_args_list[1][0][0]
    assert msg_mock.call_count == 2
    assert '*BTC:*' in result
    assert '*ETH:*' not in result
    assert '*USDT:*' not in result
    assert '*EUR:*' not in result
    assert '*LTC:*' not in result
    assert '*LTC:*' in result_full
    assert '*XRP:*' not in result
    assert 'Balance:' in result
    assert 'Est. BTC:' in result
    assert 'BTC: 11' in result
    assert 'BTC: 12' in result_full
    assert '*3 Other Currencies (< 0.0001 BTC):*' in result
    assert 'BTC: 0.00000309' in result
    assert '*Estimated Value*:' in result_full
    assert '*Estimated Value (Bot managed assets only)*:' in result_full


async def test_telegram_balance_handle_futures(default_conf: Dict[str, Any], update: Update, rpc_balance: Any, mocker: Any, tickers: Any) -> None:
    default_conf.update({'dry_run': False, 'trading_mode': 'futures', 'margin_mode': 'isolated'})
    mock_pos = [
        {'symbol': 'ETH/USDT:USDT', 'timestamp': None, 'datetime': None, 'initialMargin': 0.0, 'initialMarginPercentage': None,
         'maintenanceMargin': 0.0, 'maintenanceMarginPercentage': 0.005, 'entryPrice': 0.0, 'notional': 10.0, 'leverage': 5.0,
         'unrealizedPnl': 0.0, 'contracts': 1.0, 'contractSize': 1, 'marginRatio': None, 'liquidationPrice': 0.0, 'markPrice': 2896.41,
         'collateral': 20, 'marginType': 'isolated', 'side': 'short', 'percentage': None},
        {'symbol': 'XRP/USDT:USDT', 'timestamp': None, 'datetime': None, 'initialMargin': 0.0, 'initialMarginPercentage': None,
         'maintenanceMargin': 0.0, 'maintenanceMarginPercentage': 0.005, 'entryPrice': 0.0, 'notional': 10.0, 'leverage': None,
         'unrealizedPnl': 0.0, 'contracts': 1.0, 'contractSize': 1, 'marginRatio': None, 'liquidationPrice': 0.0, 'markPrice': 2896.41,
         'collateral': 20, 'marginType': 'isolated', 'side': 'short', 'percentage': None}
    ]
    mocker.patch(f'{EXMS}.get_balances', return_value=rpc_balance)
    mocker.patch(f'{EXMS}.fetch_positions', return_value=mock_pos)
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.get_valid_pair_combination', side_effect=lambda a, b: [f'{a}/{b}'])
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'ETH/USDT:USDT' in result
    assert '`short: 10' in result
    assert 'XRP/USDT:USDT' in result


async def test_balance_handle_empty_response(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    default_conf['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value={})
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.config['dry_run'] = False
    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert 'Starting capital: `0 BTC' in result


async def test_balance_handle_empty_response_dry(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch(f'{EXMS}.get_balances', return_value={})
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert '*Warning:* Simulated balances in Dry Mode.' in result
    assert 'Starting capital: `990 BTC`' in result


async def test_balance_handle_too_large_response(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    balances = []
    for i in range(100):
        curr = choice(ascii_uppercase) + choice(ascii_uppercase) + choice(ascii_uppercase)
        balances.append({
            'currency': curr, 'free': 1.0, 'used': 0.5, 'balance': i, 'bot_owned': 0.5,
            'est_stake': 1, 'est_stake_bot': 1, 'stake': 'BTC', 'is_position': False,
            'leverage': 1.0, 'position': 0.0, 'side': 'long', 'is_bot_managed': True
        })
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_balance', return_value={
        'currencies': balances, 'total': 100.0, 'total_bot': 100.0, 'symbol': 100.0,
        'value': 1000.0, 'value_bot': 1000.0, 'starting_capital': 1000, 'starting_capital_fiat': 1000
    })
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._balance(update=update, context=MagicMock())
    # message length less than 4096 but close to minimum size adjustment
    assert msg_mock.call_count > 1
    assert len(msg_mock.call_args_list[0][0][0]) < 4096
    assert len(msg_mock.call_args_list[0][0][0]) > 4096 - 120


async def test_start_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1


async def test_start_handle_already_running(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert 'already running' in msg_mock.call_args_list[0][0][0]


async def test_stop_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'stopping trader' in msg_mock.call_args_list[0][0][0]


async def test_stop_handle_already_stopped(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert 'already stopped' in msg_mock.call_args_list[0][0][0]


async def test_stopbuy_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    assert freqtradebot.config['max_open_trades'] != 0
    await telegram._stopentry(update=update, context=MagicMock())
    assert freqtradebot.config['max_open_trades'] == 0
    assert msg_mock.call_count == 1
    assert 'No more entries will occur from now. Run /reload_config to reset.' in msg_mock.call_args_list[0][0][0]


async def test_reload_config_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._reload_config(update=update, context=MagicMock())
    assert freqtradebot.state == State.RELOAD_CONFIG
    assert msg_mock.call_count == 1
    assert 'Reloading config' in msg_mock.call_args_list[0][0][0]


async def test_telegram_forceexit_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, ticker_sell_up: Any, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock: Any = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    freqtradebot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(freqtradebot)
    telegram: Telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    mocker.patch(f'{EXMS}.fetch_ticker', ticker_sell_up)
    context = MagicMock()
    context.args = ['1']
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 4
    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/BTC', 'gain': 'profit', 'leverage': 1.0,
        'limit': 1.173e-05, 'order_rate': 1.173e-05, 'amount': 91.07468123, 'order_type': 'limit', 'open_rate': 1.098e-05,
        'current_rate': 1.173e-05, 'direction': 'Long', 'profit_amount': 6.314e-05, 'profit_ratio': 0.0629778,
        'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH', 'fiat_currency': 'USD',
        'buy_tag': Any, 'enter_tag': Any, 'exit_reason': ExitType.FORCE_EXIT.value, 'open_date': Any,
        'close_date': Any, 'close_rate': Any, 'stake_amount': 0.0009999999999054, 'sub_trade': False,
        'cumulative_profit': 0.0, 'is_final_exit': False, 'final_profit_ratio': None
    } == last_msg


async def test_telegram_force_exit_down_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, ticker_sell_down: Any, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock: Any = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    freqtradebot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(freqtradebot)
    telegram: Telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.enter_positions()
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_sell_down)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    context = MagicMock()
    context.args = ['1']
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 4
    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/BTC', 'gain': 'loss', 'leverage': 1.0,
        'limit': 1.043e-05, 'order_rate': 1.043e-05, 'amount': 91.07468123, 'order_type': 'limit', 'open_rate': 1.098e-05,
        'current_rate': 1.043e-05, 'direction': 'Long', 'profit_amount': -5.497e-05, 'profit_ratio': -0.05482878,
        'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH', 'fiat_currency': 'USD',
        'buy_tag': Any, 'enter_tag': Any, 'exit_reason': ExitType.FORCE_EXIT.value, 'open_date': Any,
        'close_date': Any, 'close_rate': Any, 'stake_amount': 0.0009999999999054, 'sub_trade': False,
        'cumulative_profit': 0.0, 'is_final_exit': False, 'final_profit_ratio': None
    } == last_msg


async def test_forceexit_all_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    msg_mock: Any = mocker.patch('freqtrade.rpc.telegram.Telegram.send_msg', MagicMock())
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    default_conf['max_open_trades'] = 4
    freqtradebot: FreqtradeBot = FreqtradeBot(default_conf)
    rpc: RPC = RPC(freqtradebot)
    telegram: Telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['all']
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 8
    msg = msg_mock.call_args_list[0][0][0]
    assert {
        'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/BTC', 'gain': 'loss', 'leverage': 1.0,
        'order_rate': 1.099e-05, 'limit': 1.099e-05, 'amount': 91.07468123, 'order_type': 'limit', 'open_rate': 1.098e-05,
        'current_rate': 1.099e-05, 'direction': 'Long', 'profit_amount': -4.09e-06, 'profit_ratio': -0.00408133,
        'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH', 'fiat_currency': 'USD',
        'buy_tag': Any, 'enter_tag': Any, 'exit_reason': ExitType.FORCE_EXIT.value, 'open_date': Any,
        'close_date': Any, 'close_rate': Any, 'stake_amount': 0.0009999999999054, 'sub_trade': False,
        'cumulative_profit': 0.0, 'is_final_exit': False, 'final_profit_ratio': None
    } == msg


async def test_forceexit_handle_invalid(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.state = State.STOPPED
    context = MagicMock()
    context.args = ['1']
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    context = MagicMock()
    context.args = ['123456']
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'invalid argument' in msg_mock.call_args_list[0][0][0]


async def test_force_exit_no_pair(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    default_conf['max_open_trades'] = 4
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    femock: Any = mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_exit')
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    context = MagicMock()
    context.args = []
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_args_list[0][1]['msg'] == 'No open trade found.'
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    await telegram._force_exit(update=update, context=context)
    keyboard = msg_mock.call_args_list[0][1]['keyboard']
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    assert keyboard[-1][0].text == 'Cancel'
    assert keyboard[1][0].callback_data == 'force_exit__2 '
    update_obj: Any = MagicMock()
    update_obj.callback_query = AsyncMock()
    update_obj.callback_query.data = keyboard[1][0].callback_data
    await telegram._force_exit_inline(update_obj, None)
    assert update_obj.callback_query.answer.call_count == 1
    assert update_obj.callback_query.edit_message_text.call_count == 1
    assert femock.call_count == 1
    assert femock.call_args_list[0][0][0] == '2'
    update_obj.callback_query.reset_mock()
    await telegram._force_exit(update=update, context=context)
    update_obj.callback_query.data = keyboard[-1][0].callback_data
    await telegram._force_exit_inline(update_obj, None)
    query = update_obj.callback_query
    assert query.answer.call_count == 1
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]['text'] == 'Force exit canceled.'


async def test_force_enter_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    fbuy_mock: Any = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    telegram, freqtradebot, _ = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    context = MagicMock()
    context.args = ['ETH/BTC']
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)
    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert fbuy_mock.call_args_list[0][0][1] is None
    assert fbuy_mock.call_args_list[0][1]['order_side'] == SignalDirection.LONG
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    context = MagicMock()
    context.args = ['ETH/BTC', '0.055']
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)
    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == 'ETH/BTC'
    assert isinstance(fbuy_mock.call_args_list[0][0][1], float)
    assert fbuy_mock.call_args_list[0][0][1] == 0.055


async def test_force_enter_handle_exception(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._force_enter(update=update, context=MagicMock(), order_side=SignalDirection.LONG)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][0][0] == 'Force_entry not enabled.'


async def test_force_enter_no_pair(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    fbuy_mock: Any = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    context = MagicMock()
    context.args = []
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)
    assert fbuy_mock.call_count == 0
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]['msg'] == 'Which pair?'
    keyboard = msg_mock.call_args_list[0][1]['keyboard']
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    update_obj: Any = MagicMock()
    update_obj.callback_query = AsyncMock()
    update_obj.callback_query.data = 'force_enter__XRP/USDT_||_long'
    await telegram._force_enter_inline(update_obj, None)
    assert fbuy_mock.call_count == 1
    fbuy_mock.reset_mock()
    update_obj.callback_query = AsyncMock()
    update_obj.callback_query.data = 'force_enter__cancel'
    await telegram._force_enter_inline(update_obj, None)
    assert fbuy_mock.call_count == 0
    query = update_obj.callback_query
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]['text'] == 'Force enter canceled.'


async def test_telegram_performance_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)
    await telegram._performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Performance' in msg_mock.call_args_list[0][0][0]
    assert '<code>XRP/USDT\t2.842 USDT (9.47%) (1)</code>' in msg_mock.call_args_list[0][0][0]


async def test_telegram_entry_tag_performance_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    create_mock_trades_usdt(fee)
    context = MagicMock()
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Entry Tag Performance' in msg_mock.call_args_list[0][0][0]
    assert '`TEST1\t3.987 USDT (1.99%) (1)`' in msg_mock.call_args_list[0][0][0]
    context.args = ['XRP/USDT']
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2
    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_enter_tag_performance', side_effect=RPCException('Error'))
    await telegram._enter_tag_performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Error' in msg_mock.call_args_list[0][0][0]


async def test_telegram_exit_reason_performance_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    create_mock_trades_usdt(fee)
    context = MagicMock()
    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Exit Reason Performance' in msg_mock.call_args_list[0][0][0]
    assert '`roi\t2.842 USDT (9.47%) (1)`' in msg_mock.call_args_list[0][0][0]
    context.args = ['XRP/USDT']
    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 2
    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_exit_reason_performance', side_effect=RPCException('Error'))
    await telegram._exit_reason_performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Error' in msg_mock.call_args_list[0][0][0]


async def test_telegram_mix_tag_performance_handle(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    create_mock_trades_usdt(fee)
    context = MagicMock()
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Mix Tag Performance' in msg_mock.call_args_list[0][0][0]
    assert '`TEST3 roi\t2.842 USDT (10.00%) (1)`' in msg_mock.call_args_list[0][0][0]
    context.args = ['XRP/USDT']
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2
    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_mix_tag_performance', side_effect=RPCException('Error'))
    await telegram._mix_tag_performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Error' in msg_mock.call_args_list[0][0][0]


async def test_count_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    freqtradebot.state = State.STOPPED
    await telegram._count(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'not running' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    await telegram._count(update=update, context=MagicMock())
    msg = '<pre>  current    max    total stake\n---------  -----  -------------\n        1      {}          {}</pre>'.format(default_conf['max_open_trades'], default_conf['stake_amount'])
    assert msg in msg_mock.call_args_list[0][0][0]


async def test_telegram_lock_handle(default_conf: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._locks(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'No active locks.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    PairLocks.lock_pair('ETH/BTC', dt_now() + timedelta(minutes=4), 'randreason')
    PairLocks.lock_pair('XRP/BTC', dt_now() + timedelta(minutes=20), 'deadbeef')
    await telegram._locks(update=update, context=MagicMock())
    assert 'Pair' in msg_mock.call_args_list[0][0][0]
    assert 'Until' in msg_mock.call_args_list[0][0][0]
    assert 'Reason\n' in msg_mock.call_args_list[0][0][0]
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'XRP/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'deadbeef' in msg_mock.call_args_list[0][0][0]
    assert 'randreason' in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['XRP/BTC']
    msg_mock.reset_mock()
    await telegram._delete_locks(update=update, context=context)
    assert 'ETH/BTC' in msg_mock.call_args_list[0][0][0]
    assert 'randreason' in msg_mock.call_args_list[0][0][0]
    assert 'XRP/BTC' not in msg_mock.call_args_list[0][0][0]
    assert 'deadbeef' not in msg_mock.call_args_list[0][0][0]


async def test_whitelist_static(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "Using whitelist `['StaticPairList']` with 4 pairs\n`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['StaticPairList']` with 4 pairs\n`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['baseonly']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['StaticPairList']` with 4 pairs\n`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['baseonly', 'sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['StaticPairList']` with 4 pairs\n`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0]


async def test_whitelist_dynamic(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    default_conf['pairlists'] = [{'method': 'VolumePairList', 'number_assets': 4}]
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "Using whitelist `['VolumePairList']` with 4 pairs\n`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['VolumePairList']` with 4 pairs\n`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['baseonly']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['VolumePairList']` with 4 pairs\n`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0]
    context = MagicMock()
    context.args = ['baseonly', 'sorted']
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert "Using whitelist `['VolumePairList']` with 4 pairs\n`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0]


async def test_blacklist_static(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._blacklist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Blacklist contains 2 pairs\n`DOGE/BTC, HOT/BTC`' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['ETH/BTC']
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Blacklist contains 3 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC`' in msg_mock.call_args_list[0][0][0]
    assert freqtradebot.pairlists.blacklist == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC']
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['XRP/.*']
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Blacklist contains 4 pairs\n`DOGE/BTC, HOT/BTC, ETH/BTC, XRP/.*`' in msg_mock.call_args_list[0][0][0]
    assert freqtradebot.pairlists.blacklist == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC', 'XRP/.*']
    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ['DOGE/BTC']
    await telegram._blacklist_delete(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Blacklist contains 3 pairs\n`HOT/BTC, ETH/BTC, XRP/.*`' in msg_mock.call_args_list[0][0][0]


async def test_telegram_logs(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch.multiple('freqtrade.rpc.telegram.Telegram', _init=MagicMock())
    setup_logging(default_conf)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'freqtrade\\.rpc\\.telegram' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = ['1']
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1
    msg_mock.reset_mock()
    mocker.patch('freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH', 200)
    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count >= 2


async def test_edge_disabled(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert 'Edge is not enabled.' in msg_mock.call_args_list[0][0][0]


async def test_edge_enabled(edge_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    mocker.patch('freqtrade.edge.Edge._cached_pairs', new_callable=mocker.PropertyMock, return_value={'E/F': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60)})
    telegram, _, msg_mock = get_telegram_testobject(mocker, edge_conf)
    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>\n<pre>' in msg_mock.call_args_list[0][0][0]
    assert 'Pair      Winrate    Expectancy    Stoploss' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    mocker.patch('freqtrade.edge.Edge._cached_pairs', new_callable=mocker.PropertyMock, return_value={})
    await telegram._edge(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '<b>Edge only validated following pairs:</b>' in msg_mock.call_args_list[0][0][0]
    assert 'Winrate' not in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short,regex_pattern', [(True, 'now[ ]*XRP\\/BTC \\(#3\\)  -1.00% \\('), (False, 'now[ ]*XRP\\/BTC \\(#3\\)  1.00% \\(')])
async def test_telegram_trades(mocker: Any, update: Update, default_conf: Dict[str, Any], fee: Any, is_short: bool, regex_pattern: str) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []
    await telegram._trades(update=update, context=context)
    assert '<b>0 recent trades</b>:' in msg_mock.call_args_list[0][0][0]
    assert '<pre>' not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = ['hello']
    await telegram._trades(update=update, context=context)
    assert '<b>0 recent trades</b>:' in msg_mock.call_args_list[0][0][0]
    assert '<pre>' not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)
    context = MagicMock()
    context.args = [5]
    await telegram._trades(update=update, context=context)
    assert msg_mock.call_count == 1
    assert '2 recent trades</b>:' in msg_mock.call_args_list[0][0][0]
    assert 'Profit (' in msg_mock.call_args_list[0][0][0]
    assert 'Close Date' in msg_mock.call_args_list[0][0][0]
    assert '<pre>' in msg_mock.call_args_list[0][0][0]
    assert bool(re.search(regex_pattern, msg_mock.call_args_list[0][0][0]))


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_trade(mocker: Any, update: Update, default_conf: Dict[str, Any], fee: Any, is_short: bool) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []
    await telegram._delete_trade(update=update, context=context)
    assert 'Trade-id not set.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)
    context = MagicMock()
    context.args = [1]
    await telegram._delete_trade(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Deleted trade 1.' in msg_mock.call_args_list[0][0][0]
    assert 'Please make sure to take care of this asset' in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_reload_trade_from_exchange(mocker: Any, update: Update, default_conf: Dict[str, Any], fee: Any, is_short: bool) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []
    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert 'Trade-id not set.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)
    context.args = [5]
    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert 'Status: `Reloaded from orders from exchange`' in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_open_order(mocker: Any, update: Update, default_conf: Dict[str, Any], fee: Any, is_short: bool, ticker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []
    await telegram._cancel_open_order(update=update, context=context)
    assert 'Trade-id not set.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)
    context = MagicMock()
    context.args = [5]
    await telegram._cancel_open_order(update=update, context=context)
    assert 'No open order for trade_id' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', return_value=trade.orders[-1].to_ccxt_object())
    context = MagicMock()
    context.args = [6]
    await telegram._cancel_open_order(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Open order canceled.' in msg_mock.call_args_list[0][0][0]


async def test_help_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*/help:* `This help message`' in msg_mock.call_args_list[0][0][0]


async def test_version_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f'*Version:* `{__version__}`' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.strategy.version = lambda: '1.1.1'
    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f'*Version:* `{__version__}`' in msg_mock.call_args_list[0][0][0]
    assert '*Strategy version: * `1.1.1`' in msg_mock.call_args_list[0][0][0]


async def test_show_config_handle(default_conf: Dict[str, Any], update: Update, mocker: Any) -> None:
    default_conf['runmode'] = RunMode.DRY_RUN
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert f'*Strategy:* `{CURRENT_TEST_STRATEGY}`' in msg_mock.call_args_list[0][0][0]
    assert '*Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.config['trailing_stop'] = True
    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert '*Mode:* `{}`'.format('Dry-run') in msg_mock.call_args_list[0][0][0]
    assert '*Exchange:* `binance`' in msg_mock.call_args_list[0][0][0]
    assert f'*Strategy:* `{CURRENT_TEST_STRATEGY}`' in msg_mock.call_args_list[0][0][0]
    assert '*Initial Stoploss:* `-0.1`' in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', None),
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 1.0),
    (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 5.0),
    (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)
])
def test_send_msg_enter_notification(default_conf: Dict[str, Any], mocker: Any, caplog: Any, message_type: RPCMessageType, enter: str, enter_signal: str, leverage: Optional[float]) -> None:
    default_conf['telegram']['notification_settings']['show_candle'] = 'ohlc'
    df: DataFrame = DataFrame({'open': [1.1], 'high': [2.2], 'low': [1.0], 'close': [1.5]})
    mocker.patch('freqtrade.data.dataprovider.DataProvider.get_analyzed_dataframe', return_value=(df, 1))
    msg: Dict[str, Any] = {
        'type': message_type, 'trade_id': 1, 'enter_tag': enter_signal, 'exchange': 'Binance',
        'pair': 'ETH/BTC', 'leverage': leverage, 'open_rate': 1.099e-05, 'order_type': 'limit',
        'direction': enter, 'stake_amount': 0.01465333, 'stake_amount_fiat': 0.0,
        'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH', 'fiat_currency': 'USD',
        'sub_trade': False, 'current_rate': 1.099e-05, 'amount': 1333.3333333333335,
        'analyzed_candle': {'open': 1.1, 'high': 2.2, 'low': 1.0, 'close': 1.5},
        'open_date': dt_now() + timedelta(hours=-1)
    }
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg(msg)
    leverage_text: str = f' ({leverage:.3g}x)' if leverage and leverage != 1.0 else ''
    expected: str = (f'🔵 *Binance (dry):* New Trade (#1)\n*Pair:* `ETH/BTC`\n*Candle OHLC*: `1.1, 2.2, 1.0, 1.5`\n'
                     f'*Enter Tag:* `{enter_signal}`\n*Amount:* `1333.33333333`\n*Direction:* `{enter}{leverage_text}`\n'
                     f'*Open Rate:* `0.00001099 BTC`\n*Current Rate:* `0.00001099 BTC`\n'
                     f'*Total:* `0.01465333 BTC / 180.895 USD`')
    assert msg_mock.call_args[0][0] == expected
    freqtradebot.config['telegram']['notification_settings'] = {'entry': 'off'}
    caplog.clear()
    msg_mock.reset_mock()
    telegram.send_msg(msg)
    assert msg_mock.call_count == 0
    assert log_has("Notification 'entry' not sent.", caplog)
    freqtradebot.config['telegram']['notification_settings'] = {'entry': 'silent'}
    caplog.clear()
    msg_mock.reset_mock()
    telegram.send_msg(msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]['disable_notification'] is True


@pytest.mark.parametrize('message_type,enter_signal', [
    (RPCMessageType.ENTRY_CANCEL, 'long_signal_01'),
    (RPCMessageType.ENTRY_CANCEL, 'short_signal_01')
])
def test_send_msg_enter_cancel_notification(default_conf: Dict[str, Any], mocker: Any, message_type: RPCMessageType, enter_signal: str) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': message_type, 'enter_tag': enter_signal, 'trade_id': 1, 'exchange': 'Binance',
        'pair': 'ETH/BTC', 'reason': CANCEL_REASON['TIMEOUT']
    })
    expected: str = '⚠ *Binance (dry):* Cancelling enter Order for ETH/BTC (#1). Reason: cancelled due to timeout.'
    assert msg_mock.call_args[0][0] == expected


def test_send_msg_protection_notification(default_conf: Dict[str, Any], mocker: Any, time_machine: Any) -> None:
    default_conf['telegram']['notification_settings']['protection_trigger'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    time_machine.move_to('2021-09-01 05:00:00 +00:00')
    lock = PairLocks.lock_pair('ETH/BTC', dt_now() + timedelta(minutes=6), 'randreason')
    msg: Dict[str, Any] = {'type': RPCMessageType.PROTECTION_TRIGGER}
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    expected: str = '*Protection* triggered due to randreason. `ETH/BTC` will be locked until `2021-09-01 05:10:00`.'
    assert msg_mock.call_args[0][0] == expected
    msg_mock.reset_mock()
    msg = {'type': RPCMessageType.PROTECTION_TRIGGER_GLOBAL}
    lock = PairLocks.lock_pair('*', dt_now() + timedelta(minutes=100), 'randreason')
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    expected = '*Protection* triggered due to randreason. *All pairs* will be locked until `2021-09-01 06:45:00`.'
    assert msg_mock.call_args[0][0] == expected


@pytest.mark.parametrize('message_type,entered,enter_signal,leverage', [
    (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_01', 1.0),
    (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_02', 2.0),
    (RPCMessageType.ENTRY_FILL, 'Short', 'short_signal_01', 2.0)
])
def test_send_msg_entry_fill_notification(default_conf: Dict[str, Any], mocker: Any, message_type: RPCMessageType, entered: str, enter_signal: str, leverage: float) -> None:
    default_conf['telegram']['notification_settings']['entry_fill'] = 'on'
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': message_type, 'trade_id': 1, 'enter_tag': enter_signal, 'exchange': 'Binance',
        'pair': 'ETH/BTC', 'leverage': leverage, 'stake_amount': 0.01465333, 'direction': entered,
        'sub_trade': False, 'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH',
        'fiat_currency': 'USD', 'open_rate': 1.099e-05, 'amount': 1333.3333333333335,
        'open_date': dt_now() - timedelta(hours=1)
    })
    leverage_text: str = f' ({leverage:.3g}x)' if leverage != 1.0 else ''
    expected: str = (f'✓ *Binance (dry):* New Trade filled (#1)\n*Pair:* `ETH/BTC`\n*Enter Tag:* `{enter_signal}`\n'
                     f'*Amount:* `1333.33333333`\n*Direction:* `{entered}{leverage_text}`\n'
                     f'*Open Rate:* `0.00001099 BTC`\n*Total:* `0.01465333 BTC / 180.895 USD`')
    assert msg_mock.call_args[0][0] == expected
    msg_mock.reset_mock()
    telegram.send_msg({
        'type': message_type, 'trade_id': 1, 'enter_tag': enter_signal, 'exchange': 'Binance',
        'pair': 'ETH/BTC', 'leverage': leverage, 'stake_amount': 0.01465333, 'sub_trade': True, 'direction': entered,
        'stake_currency': 'BTC', 'quote_currency': 'BTC', 'base_currency': 'ETH', 'fiat_currency': 'USD',
        'open_rate': 1.099e-05, 'amount': 1333.3333333333335,
        'open_date': dt_now() - timedelta(hours=1)
    })
    expected = (f'✓ *Binance (dry):* Position increase filled (#1)\n*Pair:* `ETH/BTC`\n*Enter Tag:* `{enter_signal}`\n'
                f'*Amount:* `1333.33333333`\n*Direction:* `{entered}{leverage_text}`\n'
                f'*Open Rate:* `0.00001099 BTC`\n*New Total:* `0.01465333 BTC / 180.895 USD`')
    assert msg_mock.call_args[0][0] == expected


def test_send_msg_exit_notification(default_conf: Dict[str, Any], mocker: Any, time_machine: Any) -> None:
    with time_machine.travel('2022-09-01 05:00:00 +00:00', tick=False):
        telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
        old_convamount = telegram._rpc._fiat_converter.convert_amount
        telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
        telegram.send_msg({
            'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
            'leverage': 1.0, 'direction': 'Long', 'gain': 'loss', 'order_rate': 0.0003201,
            'amount': 1333.3333333333335, 'order_type': 'market', 'open_rate': 0.00075,
            'current_rate': 0.0003201, 'profit_amount': -0.05746268, 'profit_ratio': -0.57405275,
            'stake_currency': 'ETH', 'quote_currency': 'ETH', 'base_currency': 'KEY',
            'fiat_currency': 'USD', 'enter_tag': 'buy_signal1', 'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(hours=1), 'close_date': dt_now()
        })
        expected = ('⚠ *Binance (dry):* Exiting KEY/ETH (#1)\n*Unrealized Profit:* '
                    '`-57.41% (loss: -0.05746 ETH / -24.812 USD)`\n*Enter Tag:* `buy_signal1`\n'
                    '*Exit Reason:* `stop_loss`\n*Direction:* `Long`\n*Amount:* `1333.33333333`\n'
                    '*Open Rate:* `0.00075 ETH`\n*Current Rate:* `0.0003201 ETH`\n'
                    '*Exit Rate:* `0.0003201 ETH`\n*Duration:* `1:00:00 (60.0 min)`')
        assert msg_mock.call_args[0][0] == expected
        msg_mock.reset_mock()
        telegram.send_msg({
            'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
            'direction': 'Long', 'gain': 'loss', 'order_rate': 0.0003201, 'amount': 1333.3333333333335,
            'order_type': 'market', 'open_rate': 0.00075, 'current_rate': 0.0003201,
            'cumulative_profit': -0.15746268, 'profit_amount': -0.05746268, 'profit_ratio': -0.57405275,
            'stake_currency': 'ETH', 'quote_currency': 'ETH', 'base_currency': 'KEY',
            'fiat_currency': 'USD', 'enter_tag': 'buy_signal1', 'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(days=1, hours=2, minutes=30), 'close_date': dt_now(),
            'stake_amount': 0.01, 'sub_trade': True
        })
        expected = ('⚠ *Binance (dry):* Partially exiting KEY/ETH (#1)\n*Unrealized Sub Profit:* '
                    '`-57.41% (loss: -0.05746 ETH / -24.812 USD)`\n*Cumulative Profit:* '
                    '`-0.15746 ETH / -24.812 USD`\n*Enter Tag:* `buy_signal1`\n*Exit Reason:* `stop_loss`\n'
                    '*Direction:* `Long`\n*Amount:* `1333.33333333`\n*Open Rate:* `0.00075 ETH`\n'
                    '*Current Rate:* `0.0003201 ETH`\n*Exit Rate:* `0.0003201 ETH`\n'
                    '*Remaining:* `0.01 ETH / -24.812 USD`')
        assert msg_mock.call_args[0][0] == expected
        msg_mock.reset_mock()
        telegram.send_msg({
            'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
            'direction': 'Long', 'gain': 'loss', 'order_rate': 0.0003201, 'amount': 1333.3333333333335,
            'order_type': 'market', 'open_rate': 0.00075, 'current_rate': 0.0003201,
            'profit_amount': -0.05746268, 'profit_ratio': -0.57405275,
            'stake_currency': 'ETH', 'quote_currency': 'ETH', 'base_currency': 'KEY',
            'fiat_currency': None, 'enter_tag': 'buy_signal1', 'exit_reason': ExitType.STOP_LOSS.value,
            'open_date': dt_now() - timedelta(days=1, hours=2, minutes=30), 'close_date': dt_now()
        })
        expected = ('⚠ *Binance (dry):* Exiting KEY/ETH (#1)\n*Unrealized Profit:* '
                    '`-57.41% (loss: -0.05746 ETH)`\n*Enter Tag:* `buy_signal1`\n*Exit Reason:* '
                    '`stop_loss`\n*Direction:* `Long`\n*Amount:* `1333.33333333`\n*Open Rate:* `0.00075 ETH`\n'
                    '*Current Rate:* `0.0003201 ETH`\n*Exit Rate:* `0.0003201 ETH`\n*Duration:* '
                    '`1 day, 2:30:00 (1590.0 min)`')
        assert msg_mock.call_args[0][0] == expected
        telegram._rpc._fiat_converter.convert_amount = old_convamount


def test_send_msg_exit_cancel_notification(default_conf: Dict[str, Any], mocker: Any) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg({
        'type': RPCMessageType.EXIT_CANCEL, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
        'reason': 'Cancelled on exchange'
    })
    expected = '⚠ *Binance (dry):* Cancelling exit Order for KEY/ETH (#1). Reason: Cancelled on exchange.'
    assert msg_mock.call_args[0][0] == expected
    msg_mock.reset_mock()
    telegram._config['dry_run'] = False
    telegram.send_msg({
        'type': RPCMessageType.EXIT_CANCEL, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
        'reason': 'timeout'
    })
    expected = '⚠ *Binance:* Cancelling exit Order for KEY/ETH (#1). Reason: timeout.'
    assert msg_mock.call_args[0][0] == expected
    telegram._rpc._fiat_converter.convert_amount = old_convamount


@pytest.mark.parametrize('direction,enter_signal,leverage', [
    ('Long', 'long_signal_01', None),
    ('Long', 'long_signal_01', 1.0),
    ('Long', 'long_signal_01', 5.0),
    ('Short', 'short_signal_01', 2.0)
])
@pytest.mark.parametrize('fiat', ['', None])
def test_send_msg_exit_notification_no_fiat(default_conf: Dict[str, Any], mocker: Any, direction: str, enter_signal: str, leverage: Optional[float], time_machine: Any, fiat: Optional[str]) -> None:
    if fiat is None:
        del default_conf['fiat_display_currency']
    else:
        default_conf['fiat_display_currency'] = fiat
    time_machine.move_to('2022-05-02 00:00:00 +00:00', tick=False)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({
        'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'KEY/ETH',
        'gain': 'loss', 'leverage': leverage, 'direction': direction, 'sub_trade': False,
        'order_rate': 0.0003201, 'amount': 1333.3333333333335, 'order_type': 'limit',
        'open_rate': 0.00075, 'current_rate': 0.0003201, 'profit_amount': -0.05746268,
        'profit_ratio': -0.57405275, 'stake_currency': 'ETH', 'quote_currency': 'ETH',
        'base_currency': 'KEY', 'fiat_currency': 'USD', 'enter_tag': enter_signal,
        'exit_reason': ExitType.STOP_LOSS.value, 'open_date': dt_now() - timedelta(hours=2, minutes=35, seconds=3),
        'close_date': dt_now()
    })
    leverage_text: str = f' ({leverage:.3g}x)' if leverage and leverage != 1.0 else ''
    expected = (f'⚠ *Binance (dry):* Exiting KEY/ETH (#1)\n*Unrealized Profit:* '
                f'`-57.41% (loss: -0.05746 ETH)`\n*Enter Tag:* `{enter_signal}`\n'
                f'*Exit Reason:* `stop_loss`\n*Direction:* `{direction}{leverage_text}`\n'
                f'*Amount:* `1333.33333333`\n*Open Rate:* `0.00075 ETH`\n'
                f'*Current Rate:* `0.0003201 ETH`\n*Exit Rate:* `0.0003201 ETH`\n'
                f'*Duration:* `2:35:03 (155.1 min)`')
    assert msg_mock.call_args[0][0] == expected


@pytest.mark.parametrize('msg,expected', [
    ({'profit_ratio': 0.201, 'exit_reason': 'roi'}, '🚀'),
    ({'profit_ratio': 0.051, 'exit_reason': 'roi'}, '🚀'),
    ({'profit_ratio': 0.0256, 'exit_reason': 'roi'}, '✳'),
    ({'profit_ratio': 0.01, 'exit_reason': 'roi'}, '✳'),
    ({'profit_ratio': 0.0, 'exit_reason': 'roi'}, '✳'),
    ({'profit_ratio': -0.05, 'exit_reason': 'stop_loss'}, '⚠'),
    ({'profit_ratio': -0.02, 'exit_reason': 'sell_signal'}, '❌')
])
def test__exit_emoji(default_conf: Dict[str, Any], mocker: Any, msg: Dict[str, Any], expected: str) -> None:
    del default_conf['fiat_display_currency']
    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._get_exit_emoji(msg) == expected


async def test_telegram__send_msg(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot: Any = MagicMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot
    await telegram._send_msg('test')
    assert len(bot.method_calls) == 1
    query: Any = MagicMock()
    query.edit_message_text = AsyncMock()
    await telegram._send_msg('test', callback_path='DeadBeef', query=query, reload_able=True)
    assert query.edit_message_text.call_count == 1
    assert 'Updated: ' in query.edit_message_text.call_args_list[0][1]['text']
    query.edit_message_text = AsyncMock(side_effect=BadRequest('not modified'))
    await telegram._send_msg('test', callback_path='DeadBeef', query=query)
    assert query.edit_message_text.call_count == 1
    assert not log_has_re('TelegramError: .*', caplog)
    query.edit_message_text = AsyncMock(side_effect=BadRequest(''))
    await telegram._send_msg('test2', callback_path='DeadBeef', query=query)
    assert query.edit_message_text.call_count == 1
    assert log_has_re('TelegramError: .*', caplog)
    query.edit_message_text = AsyncMock(side_effect=TelegramError('DeadBEEF'))
    await telegram._send_msg('test3', callback_path='DeadBeef', query=query)
    assert log_has_re('TelegramError: DeadBEEF! Giving up.*', caplog)


async def test__send_msg_network_error(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot: Any = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError('Oh snap'))
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot
    telegram._config['telegram']['enabled'] = True
    await telegram._send_msg('test')
    assert len(bot.method_calls) == 2
    assert log_has('Telegram NetworkError: Oh snap! Trying one more time.', caplog)


@pytest.mark.filterwarnings('ignore:.*ChatPermissions')
async def test__send_msg_keyboard(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram._init', MagicMock())
    bot: Any = MagicMock()
    bot.send_message = AsyncMock()
    freqtradebot: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf)
    rpc: RPC = RPC(freqtradebot)
    invalid_keys_list = [['/not_valid', '/profit'], ['/daily'], ['/alsoinvalid']]
    default_keys_list = [['/daily', '/profit', '/balance'], ['/status', '/status table', '/performance'], ['/count', '/start', '/stop', '/help']]
    default_keyboard = ReplyKeyboardMarkup(default_keys_list)
    custom_keys_list = [['/daily', '/stats', '/balance', '/profit', '/profit 5'], ['/count', '/start', '/reload_config', '/help']]
    custom_keyboard = ReplyKeyboardMarkup(custom_keys_list)

    def init_telegram(freqtradebot: FreqtradeBot) -> Telegram:
        telegram_obj: Telegram = Telegram(rpc, default_conf)
        telegram_obj._app = MagicMock()
        telegram_obj._app.bot = bot
        return telegram_obj

    freqtradebot.config['telegram']['enabled'] = True
    telegram_obj = init_telegram(freqtradebot)
    await telegram_obj._send_msg('test')
    used_keyboard = bot.send_message.call_args[1]['reply_markup']
    assert used_keyboard == default_keyboard
    freqtradebot.config['telegram']['enabled'] = True
    freqtradebot.config['telegram']['keyboard'] = invalid_keys_list
    err_msg = re.escape("config.telegram.keyboard: Invalid commands for custom Telegram keyboard: ['/not_valid', '/alsoinvalid']\nvalid commands are: ") + '.*'
    with pytest.raises(OperationalException, match=err_msg):
        telegram_obj = init_telegram(freqtradebot)
    freqtradebot.config['telegram']['enabled'] = True
    freqtradebot.config['telegram']['keyboard'] = custom_keys_list
    telegram_obj = init_telegram(freqtradebot)
    await telegram_obj._send_msg('test')
    used_keyboard = bot.send_message.call_args[1]['reply_markup']
    assert used_keyboard == custom_keyboard
    assert log_has("using custom keyboard from config.json: [['/daily', '/stats', '/balance', '/profit', '/profit 5'], ['/count', '/start', '/reload_config', '/help']]", caplog)


async def test_change_market_direction(default_conf: Dict[str, Any], mocker: Any, update: Update) -> None:
    telegram, _, _msg_mock = get_telegram_testobject(mocker, default_conf)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.NONE
    context = MagicMock()
    context.args = ['long']
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG
    context = MagicMock()
    context.args = ['invalid']
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG


async def test_telegram_list_custom_data(default_conf_usdt: Dict[str, Any], update: Update, ticker: Any, fee: Any, mocker: Any) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)
    context = MagicMock()
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert 'Trade-id not set.' in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    context.args = ['1']
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "Didn't find any custom-data entries for Trade ID: `1`" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    trade1 = Trade.get_trades_proxy()[0]
    trade1.set_custom_data('test_int', 1)
    trade1.set_custom_data('test_dict', {'test': 'dict'})
    Trade.commit()
    context.args = [f'{trade1.id}']
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 3
    assert 'Found custom-data entries: ' in msg_mock.call_args_list[0][0][0]
    assert '*Key:* `test_int`\n*ID:* `1`\n*Trade ID:* `1`\n*Type:* `int`\n*Value:* `1`\n*Create Date:*' in msg_mock.call_args_list[1][0][0]
    assert '*Key:* `test_dict`\n*ID:* `2`\n*Trade ID:* `1`\n*Type:* `dict`\n*Value:* `{"test": "dict"}`\n*Create Date:* ' in msg_mock.call_args_list[2][0][0]
    msg_mock.reset_mock()


def test_noficiation_settings(default_conf_usdt: Dict[str, Any], mocker: Any) -> None:
    telegram, _, _ = get_telegram_testobject(mocker, default_conf_usdt)
    telegram._config['telegram'].update({
        'notification_settings': {
            'status': 'silent', 'warning': 'on', 'startup': 'off', 'entry': 'silent',
            'entry_fill': 'on', 'entry_cancel': 'silent', 'exit': {'roi': 'silent', 'emergency_exit': 'on',
                                                                   'force_exit': 'on', 'exit_signal': 'silent',
                                                                   'trailing_stop_loss': 'on', 'stop_loss': 'on',
                                                                   'stoploss_on_exchange': 'on', 'custom_exit': 'silent',
                                                                   'partial_exit': 'off'},
            'exit_fill': {'roi': 'silent', 'partial_exit': 'off', '*': 'silent'},
            'exit_cancel': 'on', 'protection_trigger': 'off', 'protection_trigger_global': 'on',
            'strategy_msg': 'off', 'show_candle': 'off'
        }
    })
    loudness = telegram._message_loudness
    assert loudness({'type': RPCMessageType.ENTRY, 'exit_reason': ''}) == 'silent'
    assert loudness({'type': RPCMessageType.ENTRY_FILL, 'exit_reason': ''}) == 'on'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': ''}) == 'on'
    assert loudness({'type': RPCMessageType.EXIT_FILL, 'exit_reason': ''}) == 'silent'
    assert loudness({'type': RPCMessageType.PROTECTION_TRIGGER, 'exit_reason': ''}) == 'off'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'roi'}) == 'silent'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'partial_exit'}) == 'off'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'cust_exit112'}) == 'on'
    assert loudness({'type': RPCMessageType.EXIT_FILL, 'exit_reason': 'roi'}) == 'silent'
    assert loudness({'type': RPCMessageType.EXIT_FILL, 'exit_reason': 'partial_exit'}) == 'off'
    assert loudness({'type': RPCMessageType.EXIT_FILL, 'exit_reason': 'cust_exit112'}) == 'silent'
    telegram._config['telegram'].update({
        'notification_settings': {
            'status': 'silent', 'warning': 'on', 'startup': 'off', 'entry': 'silent',
            'entry_fill': 'on', 'entry_cancel': 'silent', 'exit': 'off', 'exit_cancel': 'on',
            'exit_fill': 'on', 'protection_trigger': 'off', 'protection_trigger_global': 'on',
            'strategy_msg': 'off', 'show_candle': 'off'
        }
    })
    assert loudness({'type': RPCMessageType.EXIT_FILL, 'exit_reason': 'roi'}) == 'on'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'roi'}) == 'off'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'partial_exit'}) == 'off'
    assert loudness({'type': RPCMessageType.EXIT, 'exit_reason': 'cust_exit112'}) == 'off'


async def test__tg_info(default_conf_usdt: Dict[str, Any], mocker: Any, update: Update) -> None:
    telegram, _, _ = get_telegram_testobject(mocker, default_conf_usdt)
    context: Any = AsyncMock()
    await telegram._tg_info(update, context)
    assert context.bot.send_message.call_count == 1
    content = context.bot.send_message.call_args[1]['text']
    assert 'Freqtrade Bot Info:\n' in content
    assert '"chat_id": "1235"' in content


# End of annotated code.
