from __future__ import annotations
from datetime import datetime, timedelta, timezone
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from unittest.mock import ANY, AsyncMock, MagicMock
from pytest import FixtureType
from time_machine import Machine
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
def mock_exchange_loop(mocker: MagicMock) -> None:
    ...

@pytest.fixture
def default_conf(default_conf: dict) -> dict:
    ...

@pytest.fixture
def update() -> Update:
    ...

def patch_eventloop_threading(telegrambot: Telegram) -> None:
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

def get_telegram_testobject(mocker: MagicMock, default_conf: dict, mock: bool = True, ftbot: FreqtradeBot | None = None) -> tuple[Telegram, FreqtradeBot, AsyncMock]:
    ...

def test_telegram__init__(default_conf: dict, mocker: MagicMock) -> None:
    ...

def test_telegram_init(default_conf: dict, mocker: MagicMock, caplog: Any) -> None:
    ...

async def test_telegram_startup(default_conf: dict, mocker: MagicMock) -> None:
    ...

async def test_telegram_cleanup(default_conf: dict, mocker: MagicMock) -> None:
    ...

async def test_authorized_only(default_conf: dict, mocker: MagicMock, caplog: Any, update: Update) -> None:
    ...

async def test_authorized_only_unauthorized(default_conf: dict, mocker: MagicMock, caplog: Any) -> None:
    ...

async def test_authorized_only_exception(default_conf: dict, mocker: MagicMock, caplog: Any, update: Update) -> None:
    ...

async def test_telegram_status(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_multi_entry(default_conf: dict, update: Update, mocker: MagicMock, fee: float) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_status_closed_trade(default_conf: dict, update: Update, mocker: MagicMock, fee: float) -> None:
    ...

async def test_order_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
async def test_telegram_order_multi_entry(default_conf: dict, update: Update, mocker: MagicMock, fee: float) -> None:
    ...

async def test_status_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_status_table_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_daily_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock, time_machine: Machine) -> None:
    ...

async def test_daily_wrong_input(default_conf: dict, update: Update, ticker: Any, mocker: MagicMock) -> None:
    ...

async def test_weekly_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock, time_machine: Machine) -> None:
    ...

async def test_monthly_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock, time_machine: Machine) -> None:
    ...

async def test_telegram_profit_handle(default_conf_usdt: dict, update: Update, ticker_usdt: Any, ticker_sell_up: Any, fee: float, limit_sell_order_usdt: Any, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_stats(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock, is_short: bool) -> None:
    ...

async def test_telegram_balance_handle(default_conf: dict, update: Update, mocker: MagicMock, rpc_balance: dict, tickers: Any) -> None:
    ...

async def test_telegram_balance_handle_futures(default_conf: dict, update: Update, rpc_balance: dict, mocker: MagicMock, tickers: Any) -> None:
    ...

async def test_balance_handle_empty_response(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_balance_handle_empty_response_dry(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_balance_handle_too_large_response(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_start_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_start_handle_already_running(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_stop_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_stop_handle_already_stopped(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_stopbuy_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_reload_config_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_telegram_forceexit_handle(default_conf: dict, update: Update, ticker: Any, fee: float, ticker_sell_up: Any, mocker: MagicMock) -> None:
    ...

async def test_telegram_force_exit_down_handle(default_conf: dict, update: Update, ticker: Any, fee: float, ticker_sell_down: Any, mocker: MagicMock) -> None:
    ...

async def test_forceexit_all_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_forceexit_handle_invalid(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_force_exit_no_pair(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_force_enter_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_force_enter_handle_exception(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_force_enter_no_pair(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_telegram_performance_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_telegram_entry_tag_performance_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_telegram_exit_reason_performance_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_telegram_mix_tag_performance_handle(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_count_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_telegram_lock_handle(default_conf: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

async def test_whitelist_static(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_whitelist_dynamic(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_blacklist_static(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_telegram_logs(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_edge_disabled(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_edge_enabled(edge_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('is_short,regex_pattern', [(True, 'now[ ]*XRP\\/BTC \\(#3\\)  -1.00% \\('), (False, 'now[ ]*XRP\\/BTC \\(#3\\)  1.00% \\(')])
async def test_telegram_trades(mocker: MagicMock, update: Update, default_conf: dict, fee: float, is_short: bool, regex_pattern: str) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_trade(mocker: MagicMock, update: Update, default_conf: dict, fee: float, is_short: bool) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_reload_trade_from_exchange(mocker: MagicMock, update: Update, default_conf: dict, fee: float, is_short: bool) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
async def test_telegram_delete_open_order(mocker: MagicMock, update: Update, default_conf: dict, fee: float, is_short: bool, ticker: Any) -> None:
    ...

async def test_help_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_version_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

async def test_show_config_handle(default_conf: dict, update: Update, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [(RPCMessageType.ENTRY, 'Long', 'long_signal_01', None), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 1.0), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 5.0), (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_enter_notification(default_conf: dict, mocker: MagicMock, caplog: Any, message_type: RPCMessageType, enter: str, enter_signal: str, leverage: float | None) -> None:
    ...

@pytest.mark.parametrize('message_type,enter_signal', [(RPCMessageType.ENTRY_CANCEL, 'long_signal_01'), (RPCMessageType.ENTRY_CANCEL, 'short_signal_01')])
def test_send_msg_enter_cancel_notification(default_conf: dict, mocker: MagicMock, message_type: RPCMessageType, enter_signal: str) -> None:
    ...

def test_send_msg_protection_notification(default_conf: dict, mocker: MagicMock, time_machine: Machine) -> None:
    ...

@pytest.mark.parametrize('message_type,entered,enter_signal,leverage', [(RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_01', 1.0), (RPCMessageType.ENTRY_FILL, 'Long', 'long_signal_02', 2.0), (RPCMessageType.ENTRY_FILL, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_entry_fill_notification(default_conf: dict, mocker: MagicMock, message_type: RPCMessageType, entered: str, enter_signal: str, leverage: float) -> None:
    ...

def test_send_msg_exit_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

async def test_send_msg_exit_cancel_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('direction,enter_signal,leverage', [('Long', 'long_signal_01', None), ('Long', 'long_signal_01', 1.0), ('Long', 'long_signal_01', 5.0), ('Short', 'short_signal_01', 2.0)])
def test_send_msg_exit_fill_notification(default_conf: dict, mocker: MagicMock, direction: str, enter_signal: str, leverage: float | None) -> None:
    ...

def test_send_msg_status_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

async def test_warning_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

def test_startup_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

def test_send_msg_strategy_msg_notification(default_conf: dict, mocker: MagicMock) -> None:
    ...

def test_send_msg_unknown_type(default_conf: dict, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('message_type,enter,enter_signal,leverage', [(RPCMessageType.ENTRY, 'Long', 'long_signal_01', None), (RPCMessageType.ENTRY, 'Long', 'long_signal_01', 2.0), (RPCMessageType.ENTRY, 'Short', 'short_signal_01', 2.0)])
def test_send_msg_buy_notification_no_fiat(default_conf: dict, mocker: MagicMock, message_type: RPCMessageType, enter: str, enter_signal: str, leverage: float | None) -> None:
    ...

@pytest.mark.parametrize('direction,enter_signal,leverage', [('Long', 'long_signal_01', None), ('Long', 'long_signal_01', 1.0), ('Long', 'long_signal_01', 5.0), ('Short', 'short_signal_01', 2.0)])
@pytest.mark.parametrize('fiat', ['', None])
def test_send_msg_exit_notification_no_fiat(default_conf: dict, mocker: MagicMock, direction: str, enter_signal: str, leverage: float | None, time_machine: Machine, fiat: str | None) -> None:
    ...

@pytest.mark.parametrize('msg,expected', [({'profit_ratio': 0.201, 'exit_reason': 'roi'}, '🚀'), ({'profit_ratio': 0.051, 'exit_reason': 'roi'}, '🚀'), ({'profit_ratio': 0.0256, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': 0.01, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': 0.0, 'exit_reason': 'roi'}, '✳'), ({'profit_ratio': -0.05, 'exit_reason': 'stop_loss'}, '⚠'), ({'profit_ratio': -0.02, 'exit_reason': 'sell_signal'}, '❌')])
def test__exit_emoji(default_conf: dict, mocker: MagicMock, msg: dict, expected: str) -> None:
    ...

async def test_telegram__send_msg(default_conf: dict, mocker: MagicMock, caplog: Any) -> None:
    ...

async def test__send_msg_network_error(default_conf: dict, mocker: MagicMock, caplog: Any) -> None:
    ...

async def test__send_msg_keyboard(default_conf: dict, mocker: MagicMock, caplog: Any) -> None:
    ...

async def test_change_market_direction(default_conf: dict, mocker: MagicMock, update: Update) -> None:
    ...

async def test_telegram_list_custom_data(default_conf_usdt: dict, update: Update, ticker: Any, fee: float, mocker: MagicMock) -> None:
    ...

def test_noficiation_settings(default_conf_usdt: dict, mocker: MagicMock) -> None:
    ...

async def test__tg_info(default_conf_usdt: dict, mocker: MagicMock, update: Update) -> None:
    ...