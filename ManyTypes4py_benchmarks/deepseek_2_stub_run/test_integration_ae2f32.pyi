```python
import pytest
from typing import Any
from unittest.mock import MagicMock
from freqtrade.enums import ExitCheckTuple, ExitType
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc import RPC

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: Any,
    ticker: Any,
    fee: Any,
    mocker: Any
) -> None: ...

@pytest.mark.parametrize('balance_ratio,result1', [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(
    default_conf: Any,
    ticker: Any,
    fee: Any,
    mocker: Any,
    balance_ratio: float,
    result1: float
) -> None: ...

def test_dca_buying(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any
) -> None: ...

def test_dca_short(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_order_adjust(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    leverage: int,
    fee: Any,
    mocker: Any
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    caplog: Any,
    is_short: bool,
    leverage: int
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_exiting(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    caplog: Any,
    leverage: int
) -> None: ...

@pytest.mark.parametrize('leverage', [1, 2])
@pytest.mark.parametrize('is_short', [False, True])
def test_dca_handle_similar_open_order(
    default_conf_usdt: Any,
    ticker_usdt: Any,
    is_short: bool,
    leverage: int,
    fee: Any,
    mocker: Any,
    caplog: Any
) -> None: ...
```