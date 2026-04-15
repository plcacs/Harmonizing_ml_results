import pytest
from collections.abc import Sequence
from typing import Any, Optional, Union
from unittest.mock import MagicMock
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

@pytest.mark.parametrize('balance_ratio,result1', [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(
    default_conf: dict[str, Any],
    ticker: MagicMock,
    fee: MagicMock,
    mocker: MagicMock,
    balance_ratio: Union[int, float],
    result1: Union[int, float]
) -> None: ...

def test_dca_buying(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

def test_dca_short(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_order_adjust(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    leverage: int,
    fee: MagicMock,
    mocker: MagicMock
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MagicMock,
    caplog: Any,
    is_short: bool,
    leverage: int
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_exiting(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    fee: MagicMock,
    mocker: MagicMock,
    caplog: Any,
    leverage: int
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_handle_similar_open_order(
    default_conf_usdt: dict[str, Any],
    ticker_usdt: MagicMock,
    is_short: bool,
    leverage: int,
    fee: MagicMock,
    mocker: MagicMock,
    caplog: Any
) -> None: ...