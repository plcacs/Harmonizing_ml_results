from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import pytest
import time
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC
from tests.conftest import EXMS, get_patched_freqtradebot, log_has_re, patch_get_signal

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: Dict[str, Any], ticker: Dict[str, Any], fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_forcebuy_last_unlimited(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_dca_buying(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], leverage: int, fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_dca_order_adjust(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], leverage: int, fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_dca_exiting(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], fee: Dict[str, Any], mocker: Any
) -> None:
    ...

def test_dca_handle_similar_open_order(
    default_conf_usdt: Dict[str, Any], ticker_usdt: Dict[str, Any], is_short: bool, leverage: int, fee: Dict[str, Any], mocker: Any
) -> None:
    ...
