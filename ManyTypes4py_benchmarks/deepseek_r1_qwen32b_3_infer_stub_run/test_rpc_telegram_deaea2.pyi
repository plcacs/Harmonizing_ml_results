from asyncio import AbstractEventLoop, Event, Future, Queue, QueueEmpty, Task, TimeoutError, get_event_loop, new_event_loop, sleep
from datetime import datetime, timezone
from functools import reduce
from logging import Logger
from re import Pattern
from threading import Thread
from typing import (
    Any,
    AnyStr,
    Callable,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Match,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from unittest.mock import AsyncMock, MagicMock, MockerFixture

import pytest
from aiogram import Bot, Dispatcher, types
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    Update,
)
from aiogram.utils.exceptions import (
    BotBlocked,
    ChatNotFound,
    InvalidQueryID,
    MessageNotModified,
    NetworkError,
    TelegramAPIError,
    TelegramError,
)
from pandas import DataFrame
from sqlalchemy import select
from telegram.error import (
    BadRequest,
    ChatMigrated,
    Conflict,
    InvalidToken,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
    Unauthorized,
)
from freqtrade import __version__
from freqtrade.edge import PairInfo
from freqtrade.enums import ExitType, MarketDirection, RPCMessageType, RunMode, SignalDirection, State
from freqtrade.exceptions import (
    AuthenticationError,
    BacktestError,
    DependencyError,
    ExchangeError,
    OperationalException,
    PermissionError,
    TemporaryError,
    TradeError,
)
from freqtrade.rpc import RPC
from freqtrade.rpc.telegram import (
    Telegram,
    authorized_only,
    get_telegram_testobject,
    patch_eventloop_threading,
)
from freqtrade.util import dt_now

@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker: MockerFixture) -> None:
    ...

@pytest.fixture
def default_conf(default_conf: dict) -> dict:
    ...

@pytest.fixture
def update() -> Update:
    ...

class DummyCls(Telegram):
    def __init__(self, rpc: RPC, config: dict) -> None:
        ...

    @authorized_only
    async def dummy_handler(self, *args: Any, **kwargs: Any) -> None:
        ...

    @authorized_only
    async def dummy_exception(self, *args: Any, **kwargs: Any) -> None:
        ...

def get_telegram_testobject(
    mocker: MockerFixture,
    default_conf: dict,
    mock: bool = True,
    ftbot: Optional[FreqtradeBot] = None,
) -> Tuple[Telegram, FreqtradeBot, AsyncMock]:
    ...

def test_telegram__init__(default_conf: dict, mocker: MockerFixture) -> None:
    ...

def test_telegram_init(default_conf: dict, mocker: MockerFixture, caplog: Any) -> None:
    ...

async def test_telegram_startup(default_conf: dict, mocker: MockerFixture) -> None:
    ...

async def test_telegram_cleanup(default_conf: dict, mocker: MockerFixture) -> None:
    ...

async def test_authorized_only(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
    update: Update,
) -> None:
    ...

async def test_authorized_only_unauthorized(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
) -> None:
    ...

async def test_authorized_only_exception(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
    update: Update,
) -> None:
    ...

async def test_telegram_status(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_multi_entry(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
    fee: float,
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_closed_trade(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
    fee: float,
) -> None:
    ...

async def test_order_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_order_multi_entry(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
    fee: float,
) -> None:
    ...

async def test_status_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_status_table_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_daily_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
    time_machine: Any,
) -> None:
    ...

async def test_daily_wrong_input(
    default_conf: dict,
    update: Update,
    ticker: Any,
    mocker: MockerFixture,
) -> None:
    ...

async def test_weekly_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
    time_machine: Any,
) -> None:
    ...

async def test_monthly_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
    time_machine: Any,
) -> None:
    ...

async def test_telegram_profit_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker_usdt: Any,
    ticker_sell_up: Any,
    fee: float,
    limit_sell_order_usdt: Any,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_stats(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
    is_short: bool,
) -> None:
    ...

async def test_telegram_balance_handle(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
    rpc_balance: Any,
    tickers: Any,
) -> None:
    ...

async def test_telegram_balance_handle_futures(
    default_conf: dict,
    update: Update,
    rpc_balance: Any,
    mocker: MockerFixture,
    tickers: Any,
) -> None:
    ...

async def test_balance_handle_empty_response(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_balance_handle_empty_response_dry(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_balance_handle_too_large_response(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_start_handle(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_start_handle_already_running(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_stop_handle(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_stop_handle_already_stopped(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_stopbuy_handle(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_reload_config_handle(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_forceexit_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    ticker_sell_up: Any,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_force_exit_down_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    ticker_sell_down: Any,
    mocker: MockerFixture,
) -> None:
    ...

async def test_forceexit_all_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_forceexit_handle_invalid(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_force_exit_no_pair(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_force_enter_handle(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_force_enter_handle_exception(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_force_enter_no_pair(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_performance_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_entry_tag_performance_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_exit_reason_performance_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_mix_tag_performance_handle(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_count_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MockerFixture) -> None:
    ...

async def test_telegram_lock_handle(
    default_conf: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

async def test_whitelist_static(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_whitelist_dynamic(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_blacklist_static(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

async def test_telegram_logs(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_edge_disabled(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_edge_enabled(
    edge_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.parametrize('is_short,regex_pattern', [(True, 'now[ ]*XRP\\/BTC \\(#3\\)  -1.00% \\('), (False, 'now[ ]*XRP\\/BTC \\(#3\\)  1.00% \\(')])
async def test_telegram_trades(
    mocker: MockerFixture,
    update: Update,
    default_conf: dict,
    fee: float,
    is_short: bool,
    regex_pattern: str,
) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_trade(
    mocker: MockerFixture,
    update: Update,
    default_conf: dict,
    fee: float,
    is_short: bool,
) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_reload_trade_from_exchange(
    mocker: MockerFixture,
    update: Update,
    default_conf: dict,
    fee: float,
    is_short: bool,
) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_open_order(
    mocker: MockerFixture,
    update: Update,
    default_conf: dict,
    fee: float,
    is_short: bool,
    ticker: Any,
) -> None:
    ...

async def test_help_handle(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_version_handle(default_conf: dict, update: Update, mocker: MockerFixture) -> None:
    ...

async def test_show_config_handle(
    default_conf: dict,
    update: Update,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [(RPCMessageType.ENTRY, 'Long', 'long_signal_01', None), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 1.0), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 5.0), (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_enter_notification(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
    message_type: RPCMessageType,
    enter: str,
    enter_signal: str,
    leverage: Optional[float],
) -> None:
    ...

@pytest.mark.parametrize('message_type,enter_signal', [(RPCMessageType.ENTRY_CANCEL, 'long_signal_01'), (RPCMessageType.ENTRY_CANCEL, 'short_signal_01')])
def test_send_msg_enter_cancel_notification(
    default_conf: dict,
    mocker: MockerFixture,
    message_type: RPCMessageType,
    enter_signal: str,
) -> None:
    ...

def test_send_msg_protection_notification(
    default_conf: dict,
    mocker: MockerFixture,
    time_machine: Any,
) -> None:
    ...

@pytest.mark.parametrize('message_type,entered,enter_signal,leverage', [(RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_01', 1.0), (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_02', 2.0), (RPCMessageType.ENTRY_FILL, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_entry_fill_notification(
    default_conf: dict,
    mocker: MockerFixture,
    message_type: RPCMessageType,
    entered: str,
    enter_signal: str,
    leverage: Optional[float],
) -> None:
    ...

def test_send_msg_exit_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

async def test_send_msg_exit_cancel_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.parametrize('direction,enter_signal,leverage', [('Long', 'long_signal_01', None), ('Long', 'long_signal_01', 1.0), ('Long', 'long_signal_01', 5.0), ('Short', 'short_signal_01', 2.0)])
def test_send_msg_exit_fill_notification(
    default_conf: dict,
    mocker: MockerFixture,
    direction: str,
    enter_signal: str,
    leverage: Optional[float],
) -> None:
    ...

def test_send_msg_status_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

async def test_warning_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

def test_startup_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

def test_send_msg_strategy_msg_notification(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

def test_send_msg_unknown_type(
    default_conf: dict,
    mocker: MockerFixture,
) -> None:
    ...

@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [(RPCMessageType.ENTRY, 'Long', 'long_signal_01', None), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 1.0), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 5.0), (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_buy_notification_no_fiat(
    default_conf: dict,
    mocker: MockerFixture,
    message_type: RPCMessageType,
    enter: str,
    enter_signal: str,
    leverage: Optional[float],
) -> None:
    ...

@pytest.mark.parametrize('direction,enter_signal,leverage', [('Long', 'long_signal_01', None), ('Long', 'long_signal_01', 1.0), ('Long', 'long_signal_01', 5.0), ('Short', 'short_signal_01', 2.0)])
@pytest.mark.parametrize('fiat', ['', None])
def test_send_msg_exit_notification_no_fiat(
    default_conf: dict,
    mocker: MockerFixture,
    direction: str,
    enter_signal: str,
    leverage: Optional[float],
    time_machine: Any,
    fiat: Optional[str],
) -> None:
    ...

@pytest.mark.parametrize('msg,expected', [({'profit_ratio': 0.201, 'exit_reason': 'roi'}, '🚀'), ({'profit_ratio': 0.051, 'exit_reason': 'roi'}, '🚀'), ({'profit_ratio': 0.0256, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': 0.01, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': 0.0, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': -0.05, 'exit_reason': 'stop_loss'}, '⚠'), ({'profit_ratio': -0.02, 'exit_reason': 'sell_signal'}, '❌')])
def test__exit_emoji(
    default_conf: dict,
    mocker: MockerFixture,
    msg: dict,
    expected: str,
) -> None:
    ...

async def test_telegram__send_msg(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
) -> None:
    ...

async def test__send_msg_network_error(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
) -> None:
    ...

@pytest.mark.filterwarnings('ignore:.*ChatPermissions')
async def test__send_msg_keyboard(
    default_conf: dict,
    mocker: MockerFixture,
    caplog: Any,
) -> None:
    ...

async def test_change_market_direction(
    default_conf: dict,
    mocker: MockerFixture,
    update: Update,
) -> None:
    ...

async def test_telegram_list_custom_data(
    default_conf_usdt: dict,
    update: Update,
    ticker: Any,
    fee: float,
    mocker: MockerFixture,
) -> None:
    ...

def test_noficiation_settings(
    default_conf_usdt: dict,
    mocker: MockerFixture,
) -> None:
    ...

async def test__tg_info(
    default_conf_usdt: dict,
    mocker: MockerFixture,
    update: Update,
) -> None:
    ...