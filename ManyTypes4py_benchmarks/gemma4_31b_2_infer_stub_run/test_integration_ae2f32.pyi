import pytest
from typing import Any, Dict, List, Optional, Union, Tuple
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: Dict[str, Any], 
    ticker: Any, 
    fee: Any, 
    mocker: Any
) -> None: ...

def test_forcebuy_last_unlimited(
    default_conf: Dict[str, Any], 
    ticker: Any, 
    fee: Any, 
    mocker: Any, 
    balance_ratio: float, 
    result1: float
) -> None: ...

def test_dca_buying(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    fee: Any, 
    mocker: Any
) -> None: ...

def test_dca_short(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    fee: Any, 
    mocker: Any
) -> None: ...

def test_dca_order_adjust(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    leverage: int, 
    fee: Any, 
    mocker: Any
) -> None: ...

def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    fee: Any, 
    mocker: Any, 
    caplog: Any, 
    is_short: bool, 
    leverage: int
) -> None: ...

def test_dca_exiting(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    fee: Any, 
    mocker: Any, 
    caplog: Any, 
    leverage: int
) -> None: ...

def test_dca_handle_similar_open_order(
    default_conf_usdt: Dict[str, Any], 
    ticker_usdt: Any, 
    is_short: bool, 
    leverage: int, 
    fee: Any, 
    mocker: Any, 
    caplog: Any
) -> None: ...