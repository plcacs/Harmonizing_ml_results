from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock, PropertyMock
from pytest import fixture, raises
from numpy import isnan
from sqlalchemy import select
from freqtrade.enums import SignalDirection, State, TradingMode
from freqtrade.exceptions import ExchangeError, InvalidOrderException, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.rpc import RPC, RPCException
from tests.conftest import EXMS

@fixture
def default_conf():
    ...

@fixture
def ticker():
    ...

@fixture
def fee():
    ...

def test_rpc_trade_status(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_rpc_status_table(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test__rpc_timeunit_profit(default_conf_usdt: dict, ticker: MagicMock, fee: MagicMock, markets: list, mocker: fixture, time_machine: fixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_trade_history(mocker: fixture, default_conf: dict, markets: list, fee: MagicMock, is_short: bool) -> None:
    ...

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_delete_trade(mocker: fixture, default_conf: dict, fee: MagicMock, markets: list, caplog: fixture, is_short: bool) -> None:
    ...

def test_rpc_trade_statistics(default_conf_usdt: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_rpc_balance_handle_error(default_conf: dict, mocker: fixture) -> None:
    ...

@pytest.mark.parametrize('proxy_coin', [None, 'BNFCR'])
@pytest.mark.parametrize('margin_mode', ['isolated', 'cross'])
def test_rpc_balance_handle(default_conf_usdt: dict, mocker: fixture, tickers: MagicMock, proxy_coin: str | None, margin_mode: str) -> None:
    ...

def test_rpc_start(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_stop(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_stopentry(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_force_exit(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_performance_handle(default_conf_usdt: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_enter_tag_performance_handle(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_enter_tag_performance_handle_2(mocker: fixture, default_conf: dict, markets: list, fee: MagicMock) -> None:
    ...

def test_exit_reason_performance_handle(default_conf_usdt: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_exit_reason_performance_handle_2(mocker: fixture, default_conf: dict, markets: list, fee: MagicMock) -> None:
    ...

def test_mix_tag_performance_handle(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: fixture) -> None:
    ...

def test_mix_tag_performance_handle_2(mocker: fixture, default_conf: dict, markets: list, fee: MagicMock) -> None:
    ...

def test_rpc_count(mocker: fixture, default_conf: dict, ticker: MagicMock, fee: MagicMock) -> None:
    ...

def test_rpc_force_entry(mocker: fixture, default_conf: dict, ticker: MagicMock, fee: MagicMock, limit_buy_order_open: dict) -> None:
    ...

def test_rpc_force_entry_stopped(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_force_entry_disabled(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_force_entry_wrong_mode(mocker: fixture, default_conf: dict) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
def test_rpc_add_and_delete_lock(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_whitelist(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_whitelist_dynamic(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_blacklist(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_blacklist_delete(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_edge_disabled(mocker: fixture, default_conf: dict) -> None:
    ...

def test_rpc_edge_enabled(mocker: fixture, edge_conf: dict) -> None:
    ...

def test_rpc_health(mocker: fixture, default_conf: dict) -> None:
    ...