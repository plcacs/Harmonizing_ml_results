import time
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC
from tests.conftest import EXMS, get_patched_freqtradebot, log_has_re, patch_get_signal

def test_may_execute_exit_stoploss_on_exchange_multi(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('balance_ratio, result1', [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(default_conf: dict, ticker: MagicMock, fee: MagicMock, mocker: MagicMock, balance_ratio: float, result1: int) -> None:
    ...

def test_dca_buying(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    ...

def test_dca_short(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_order_adjust(default_conf_usdt: dict, ticker_usdt: MagicMock, leverage: int, fee: MagicMock, mocker: MagicMock) -> None:
    ...

@pytest.mark.parametrize('leverage, is_short', [(1, False), (2, True)])
def test_dca_order_adjust_entry_replace_fails(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock, caplog: MagicMock, is_short: bool, leverage: int) -> None:
    ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_exiting(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock, caplog: MagicMock, leverage: int) -> None:
    ...

@pytest.mark.parametrize('leverage, is_short', [(1, False), (2, True)])
def test_dca_handle_similar_open_order(default_conf_usdt: dict, ticker_usdt: MagicMock, is_short: bool, leverage: int, fee: MagicMock, mocker: MagicMock, caplog: MagicMock) -> None:
    ...
