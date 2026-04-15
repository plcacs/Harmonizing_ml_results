from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from numpy import isnan
from sqlalchemy import select
from freqtrade.edge import PairInfo
from freqtrade.enums import SignalDirection, State, TradingMode
from freqtrade.exceptions import ExchangeError, InvalidOrderException, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.key_value_store import set_startup_time
from freqtrade.rpc import RPC, RPCException
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, patch_get_signal

def test_rpc_trade_status(
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_rpc_status_table(
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test__rpc_timeunit_profit(
    default_conf_usdt: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    markets: PropertyMock,
    mocker: MagicMock,
    time_machine: Any
) -> None: ...

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_trade_history(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    markets: PropertyMock,
    fee: MagicMock,
    is_short: bool
) -> None: ...

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_delete_trade(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    fee: MagicMock,
    markets: PropertyMock,
    caplog: Any,
    is_short: bool
) -> None: ...

def test_rpc_trade_statistics(
    default_conf_usdt: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_rpc_balance_handle_error(
    default_conf: Dict[str, Any],
    mocker: MagicMock
) -> None: ...

@pytest.mark.parametrize('proxy_coin', [None, 'BNFCR'])
@pytest.mark.parametrize('margin_mode', ['isolated', 'cross'])
def test_rpc_balance_handle(
    default_conf_usdt: Dict[str, Any],
    mocker: MagicMock,
    tickers: MagicMock,
    proxy_coin: Optional[str],
    margin_mode: str
) -> None: ...

def test_rpc_start(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_stop(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_stopentry(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_force_exit(
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_performance_handle(
    default_conf_usdt: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_enter_tag_performance_handle(
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_enter_tag_performance_handle_2(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    markets: PropertyMock,
    fee: MagicMock
) -> None: ...

def test_exit_reason_performance_handle(
    default_conf_usdt: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_exit_reason_performance_handle_2(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    markets: PropertyMock,
    fee: MagicMock
) -> None: ...

def test_mix_tag_performance_handle(
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_mix_tag_performance_handle_2(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    markets: PropertyMock,
    fee: MagicMock
) -> None: ...

def test_rpc_count(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock
) -> None: ...

def test_rpc_force_entry(
    mocker: MagicMock,
    default_conf: Dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    limit_buy_order_open: Dict[str, Any]
) -> None: ...

def test_rpc_force_entry_stopped(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_force_entry_disabled(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_force_entry_wrong_mode(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_rpc_add_and_delete_lock(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_whitelist(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_whitelist_dynamic(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_blacklist(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_edge_disabled(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...

def test_rpc_edge_enabled(
    mocker: MagicMock,
    edge_conf: Dict[str, Any]
) -> None: ...

def test_rpc_health(
    mocker: MagicMock,
    default_conf: Dict[str, Any]
) -> None: ...