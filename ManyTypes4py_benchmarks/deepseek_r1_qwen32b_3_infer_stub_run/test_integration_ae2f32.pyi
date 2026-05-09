from pytest import MockerFixture
from typing import Callable, Dict, List, Optional, Tuple, Union
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence.models import Order, Trade
from sqlalchemy import select

def test_may_execute_exit_stoploss_on_exchange_multi(default_conf: Dict, ticker: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture) -> None:
    ...

def test_forcebuy_last_unlimited(default_conf: Dict, ticker: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture, balance_ratio: float, result1: int) -> None:
    ...

def test_dca_buying(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture) -> None:
    ...

def test_dca_short(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture) -> None:
    ...

def test_dca_order_adjust(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], leverage: int, fee: Callable[[], Dict], mocker: MockerFixture) -> None:
    ...

def test_dca_order_adjust_entry_replace_fails(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture, caplog: object, is_short: bool, leverage: int) -> None:
    ...

def test_dca_exiting(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], fee: Callable[[], Dict], mocker: MockerFixture, caplog: object, leverage: int) -> None:
    ...

def test_dca_handle_similar_open_order(default_conf_usdt: Dict, ticker_usdt: Callable[[], Dict], is_short: bool, leverage: int, fee: Callable[[], Dict], mocker: MockerFixture, caplog: object) -> None:
    ...