import json
import logging
import platform
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)
import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling
from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import (
    CandleType,
    MarginMode,
    RunMode,
    SignalDirection,
    TradingMode,
)
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_seconds
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from freqtrade.worker import Worker
from tests.conftest_trades import (
    leverage_trade,
    mock_trade_1,
    mock_trade_2,
    mock_trade_3,
    mock_trade_4,
    mock_trade_5,
    mock_trade_6,
    short_trade,
)
from tests.conftest_trades_usdt import (
    mock_trade_usdt_1,
    mock_trade_usdt_2,
    mock_trade_usdt_3,
    mock_trade_usdt_4,
    mock_trade_usdt_5,
    mock_trade_usdt_6,
    mock_trade_usdt_7,
)

def pytest_addoption(parser: pytest.argparse.ArgParser) -> None:
    ...

def pytest_configure(config: pytest.config.Config) -> None:
    ...

class FixtureScheduler(LoadScopeScheduling):
    def _split_scope(self, nodeid: str) -> str:
        ...

def pytest_xdist_make_scheduler(config: pytest.config.Config, log: Any) -> FixtureScheduler:
    ...

def log_has(line: str, logs: pytest.LogCaptureFixture) -> bool:
    ...

def log_has_when(line: str, logs: pytest.LogCaptureFixture, when: str) -> bool:
    ...

def log_has_re(line: str, logs: pytest.LogCaptureFixture) -> bool:
    ...

def num_log_has(line: str, logs: pytest.LogCaptureFixture) -> int:
    ...

def num_log_has_re(line: str, logs: pytest.LogCaptureFixture) -> int:
    ...

def get_args(args: List[str]) -> Dict[str, Any]:
    ...

def generate_trades_history(
    n_rows: int, start_date: Optional[datetime] = None, days: int = 5
) -> pd.DataFrame:
    ...

def generate_test_data(
    timeframe: str, size: int, start: str = '2020-07-05', random_seed: int = 42
) -> pd.DataFrame:
    ...

def generate_test_data_raw(
    timeframe: str, size: int, start: str = '2020-07-05', random_seed: int = 42
) -> List[List[float]]:
    ...

def get_mock_coro(
    return_value: Optional[Any] = None, side_effect: Optional[Any] = None
) -> Mock:
    ...

def patched_configuration_load_config_file(
    mocker: pytest_mock.MockFixture, config: Dict[str, Any]
) -> None:
    ...

def patch_exchange(
    mocker: pytest_mock.MockFixture,
    api_mock: Optional[Any] = None,
    exchange: str = 'binance',
    mock_markets: Union[bool, Dict[str, Any]] = True,
    mock_supported_modes: bool = True,
) -> None:
    ...

def get_patched_exchange(
    mocker: pytest_mock.MockFixture,
    config: Dict[str, Any],
    api_mock: Optional[Any] = None,
    exchange: str = 'binance',
    mock_markets: Union[bool, Dict[str, Any]] = True,
    mock_supported_modes: bool = True,
) -> Exchange:
    ...

def patch_wallet(mocker: pytest_mock.MockFixture, free: float = 999.9) -> None:
    ...

def patch_whitelist(mocker: pytest_mock.MockFixture, conf: Dict[str, Any]) -> None:
    ...

def patch_edge(mocker: pytest_mock.MockFixture) -> None:
    ...

def patch_freqtradebot(mocker: pytest_mock.MockFixture, config: Dict[str, Any]) -> None:
    ...

def get_patched_freqtradebot(
    mocker: pytest_mock.MockFixture, config: Dict[str, Any]
) -> FreqtradeBot:
    ...

def get_patched_worker(
    mocker: pytest_mock.MockFixture, config: Dict[str, Any]
) -> Worker:
    ...

def patch_get_signal(
    freqtrade: Any,
    enter_long: bool = True,
    exit_long: bool = False,
    enter_short: bool = False,
    exit_short: bool = False,
    enter_tag: Optional[str] = None,
    exit_tag: Optional[str] = None,
) -> None:
    ...

def create_mock_trades(
    fee: float, is_short: Optional[bool] = None, use_db: bool = True
) -> None:
    ...

def create_mock_trades_with_leverage(fee: float, use_db: bool = True) -> None:
    ...

def create_mock_trades_usdt(
    fee: float, is_short: Optional[bool] = None, use_db: bool = True
) -> None:
    ...

def is_arm() -> bool:
    ...

def is_mac() -> bool:
    ...

def patch_torch_initlogs(mocker: pytest_mock.MockFixture) -> None:
    ...

def user_dir(mocker: pytest_mock.MockFixture, tmp_path: Path) -> Path:
    ...

def patch_coingecko(mocker: pytest_mock.MockFixture) -> None:
    ...

def init_persistence(default_conf: Dict[str, Any]) -> None:
    ...

@pytest.fixture(autouse=True)
def patch_gc(mocker: pytest_mock.MockFixture) -> None:
    ...

@pytest.fixture(autouse=True)
def patch_torch_initlogs(mocker: pytest_mock.MockFixture) -> None:
    ...

@pytest.fixture(autouse=True)
def patch_coingecko(mocker: pytest_mock.MockFixture) -> None:
    ...

@pytest.fixture(scope='function')
def init_persistence(default_conf: Dict[str, Any]) -> None:
    ...

@pytest.fixture(scope='function')
def default_conf(testdatadir: Path) -> Dict[str, Any]:
    ...

@pytest.fixture(scope='function')
def default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    ...

def get_default_conf(testdatadir: Path) -> Dict[str, Any]:
    ...

def get_default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    ...

@pytest.fixture
def fee() -> MagicMock:
    ...

@pytest.fixture
def ticker() -> MagicMock:
    ...

@pytest.fixture
def ticker_sell_up() -> MagicMock:
    ...

@pytest.fixture
def ticker_sell_down() -> MagicMock:
    ...

@pytest.fixture
def ticker_usdt() -> MagicMock:
    ...

@pytest.fixture
def ticker_usdt_sell_up() -> MagicMock:
    ...

@pytest.fixture
def ticker_usdt_sell_down() -> MagicMock:
    ...

@pytest.fixture
def markets() -> Dict[str, Any]:
    ...

@pytest.fixture
def markets_static() -> Dict[str, Any]:
    ...

@pytest.fixture
def shitcoinmarkets(markets_static: Dict[str, Any]) -> Dict[str, Any]:
    ...

@pytest.fixture
def markets_empty() -> MagicMock:
    ...

@pytest.fixture(scope='function')
def limit_buy_order_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_buy_order_old() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_old() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_buy_order_old_partial() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_buy_order_old_partial_canceled(
    limit_buy_order_old_partial: Dict[str, Any]
) -> Dict[str, Any]:
    ...

@pytest.fixture(scope='function')
def limit_buy_order_canceled_empty(
    request: pytest.FixtureRequest
) -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order(limit_sell_order_open: Dict[str, Any]) -> Dict[str, Any]:
    ...

@pytest.fixture
def order_book_l2() -> MagicMock:
    ...

@pytest.fixture
def order_book_l2_usd() -> MagicMock:
    ...

@pytest.fixture
def ohlcv_history_list() -> List[List[float]]:
    ...

@pytest.fixture
def ohlcv_history(ohlcv_history_list: List[List[float]]) -> pd.DataFrame:
    ...

@pytest.fixture
def tickers() -> MagicMock:
    ...

@pytest.fixture
def dataframe_1m(testdatadir: Path) -> pd.DataFrame:
    ...

@pytest.fixture(scope='function')
def trades_for_order() -> List[Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def trades_history() -> List[List[float]]:
    ...

@pytest.fixture(scope='function')
def trades_history_df(trades_history: List[List[float]]) -> pd.DataFrame:
    ...

@pytest.fixture(scope='function')
def fetch_trades_result() -> List[Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def trades_for_order2() -> List[Dict[str, Any]]:
    ...

@pytest.fixture
def buy_order_fee() -> Dict[str, Any]:
    ...

@pytest.fixture
def edge_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    ...

@pytest.fixture
def rpc_balance() -> Dict[str, Dict[str, float]]:
    ...

@pytest.fixture
def testdatadir() -> Path:
    ...

@pytest.fixture(scope='function')
def import_fails() -> None:
    ...

@pytest.fixture(scope='function')
def open_trade() -> Trade:
    ...

@pytest.fixture(scope='function')
def open_trade_usdt() -> Trade:
    ...

@pytest.fixture(scope='function')
def limit_buy_order_usdt_open() -> Dict[str, Any]:
    ...

@pytest.fixture(scope='function')
def limit_buy_order_usdt(limit_buy_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_usdt_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_usdt(limit_sell_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    ...

@pytest.fixture(scope='function')
def market_buy_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture
def market_buy_order_usdt_doublefee(
    market_buy_order_usdt: Dict[str, Any]
) -> Dict[str, Any]:
    ...

@pytest.fixture
def market_sell_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture(scope='function')
def limit_order(
    limit_buy_order_usdt: Dict[str, Any],
    limit_sell_order_usdt: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def limit_order_open(
    limit_buy_order_usdt_open: Dict[str, Any],
    limit_sell_order_usdt_open: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def mark_ohlcv() -> List[List[float]]:
    ...

@pytest.fixture(scope='function')
def funding_rate_history_hourly() -> List[Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def funding_rate_history_octohourly() -> List[Dict[str, Any]]:
    ...

@pytest.fixture(scope='function')
def leverage_tiers() -> Dict[str, List[Dict[str, Any]]]:
    ...