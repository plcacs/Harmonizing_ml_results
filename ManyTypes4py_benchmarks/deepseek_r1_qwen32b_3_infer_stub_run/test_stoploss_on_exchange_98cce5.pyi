from pytest import MonkeyPatch, fixture
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from sqlalchemy import Select
from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.util.datetime_helpers import DateTime

@pytest.mark.parametrize('is_short', [False, True])
def test_add_stoploss_on_exchange(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], limit_order: Tuple[Dict[str, Any]], is_short: bool, fee: fixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, caplog: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_emergency(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial_cancel_here(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]], caplog: fixture, time_machine: fixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_sle_cancel_cant_recreate(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, caplog: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_invalid_order(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], caplog: fixture, fee: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_insufficient_funds(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], caplog: fixture, fee: fixture, limit_order: Tuple[Dict[str, Any]], is_short: bool) -> None:
    ...

@pytest.mark.parametrize('is_short,bid,ask,stop_price,hang_price', [(False, [4.38, 4.16], [4.4, 4.17], ['2.0805', 4.4 * 0.95], 3), (True, [1.09, 1.21], [1.1, 1.22], ['2.321', 1.09 * 1.05], 1.5)])
def test_handle_stoploss_on_exchange_trailing(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, is_short: bool, bid: List[float], ask: List[float], limit_order: Tuple[Dict[str, Any]], stop_price: List[str], hang_price: float, time_machine: fixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_trailing_error(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, caplog: fixture, limit_order: Tuple[Dict[str, Any]], is_short: bool, time_machine: fixture) -> None:
    ...

def test_stoploss_on_exchange_price_rounding(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, open_trade_usdt: Trade) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_custom_stop(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], fee: fixture, is_short: bool, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

def test_tsl_on_exchange_compatible_with_edge(mocker: MonkeyPatch, edge_conf: Dict[str, Any], fee: fixture, limit_order: Tuple[Dict[str, Any]]) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(default_conf_usdt: Dict[str, Any], ticker_usdt: fixture, fee: fixture, is_short: bool, ticker_usdt_sell_down: fixture, ticker_usdt_sell_up: fixture, mocker: MonkeyPatch) -> None:
    ...

def test_execute_trade_exit_sloe_cancel_exception(mocker: MonkeyPatch, default_conf_usdt: Dict[str, Any], ticker_usdt: fixture, fee: fixture, caplog: fixture) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_with_stoploss_on_exchange(default_conf_usdt: Dict[str, Any], ticker_usdt: fixture, fee: fixture, ticker_usdt_sell_up: fixture, is_short: bool, mocker: MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('is_short', [False, True])
def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(default_conf_usdt: Dict[str, Any], ticker_usdt: fixture, fee: fixture, mocker: MonkeyPatch, is_short: bool) -> None:
    ...