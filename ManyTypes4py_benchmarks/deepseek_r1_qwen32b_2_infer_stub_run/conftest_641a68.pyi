import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
from uuid import UUID
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling
from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import CandleType, MarginMode, RunMode, SignalDirection, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Order, Trade
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from freqtrade.worker import Worker
from tests.conftest_trades import leverage_trade, mock_trade_1, mock_trade_2, mock_trade_3, mock_trade_4, mock_trade_5, mock_trade_6, short_trade
from tests.conftest_trades_usdt import mock_trade_usdt_1, mock_trade_usdt_2, mock_trade_usdt_3, mock_trade_usdt_4, mock_trade_usdt_5, mock_trade_usdt_6, mock_trade_usdt_7

class FixtureScheduler(LoadScopeScheduling):
    def _split_scope(self, nodeid: str) -> str:
        ...

def pytest_addoption(parser: Any) -> None:
    ...

def pytest_configure(config: Any) -> None:
    ...

def log_has(line: str, logs: Any) -> bool:
    ...

def log_has_when(line: str, logs: Any, when: str) -> bool:
    ...

def log_has_re(line: str, logs: Any) -> bool:
    ...

def num_log_has(line: str, logs: Any) -> int:
    ...

def num_log_has_re(line: str, logs: Any) -> int:
    ...

def get_args(args: List[str]) -> Dict[str, Any]:
    ...

def generate_trades_history(n_rows: int, start_date: Optional[datetime] = None, days: int = 5) -> pd.DataFrame:
    ...

def generate_test_data(timeframe: str, size: int, start: str = '2020-07-05', random_seed: int = 42) -> pd.DataFrame:
    ...

def generate_test_data_raw(timeframe: str, size: int, start: str = '2020-07-05', random_seed: int = 42) -> List[List[Any]]:
    ...

def get_mock_coro(return_value: Any = None, side_effect: Any = None) -> Mock:
    ...

def patched_configuration_load_config_file(mocker: Any, config: Dict[str, Any]) -> None:
    ...

def patch_exchange(mocker: Any, api_mock: Any = None, exchange: str = 'binance', mock_markets: Union[bool, Dict[str, Any]] = True, mock_supported_modes: bool = True) -> None:
    ...

def get_patched_exchange(mocker: Any, config: Dict[str, Any], api_mock: Any = None, exchange: str = 'binance', mock_markets: Union[bool, Dict[str, Any]] = True, mock_supported_modes: bool = True) -> Exchange:
    ...

def patch_wallet(mocker: Any, free: float = 999.9) -> None:
    ...

def patch_whitelist(mocker: Any, conf: Dict[str, Any]) -> None:
    ...

def patch_edge(mocker: Any) -> None:
    ...

def patch_freqtradebot(mocker: Any, config: Dict[str, Any]) -> None:
    ...

def get_patched_freqtradebot(mocker: Any, config: Dict[str, Any]) -> FreqtradeBot:
    ...

def get_patched_worker(mocker: Any, config: Dict[str, Any]) -> Worker:
    ...

def patch_get_signal(freqtrade: Any, enter_long: bool = True, exit_long: bool = False, enter_short: bool = False, exit_short: bool = False, enter_tag: Optional[str] = None, exit_tag: Optional[str] = None) -> None:
    ...

def create_mock_trades(fee: float, is_short: Optional[bool] = None, use_db: bool = True) -> None:
    ...

def create_mock_trades_with_leverage(fee: float, use_db: bool = True) -> None:
    ...

def create_mock_trades_usdt(fee: float, is_short: Optional[bool] = None, use_db: bool = True) -> None:
    ...

def is_arm() -> bool:
    ...

def is_mac() -> bool:
    ...

def get_default_conf(testdatadir: Path) -> Dict[str, Any]:
    ...

def get_default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    ...

def markets() -> Dict[str, Dict[str, Any]]:
    ...

def markets_static() -> Dict[str, Dict[str, Any]]:
    ...

def markets_empty() -> MagicMock:
    ...

@pytest.fixture
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
def limit_buy_order_old_partial_canceled() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order() -> Dict[str, Any]:
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
def ohlcv_history() -> pd.DataFrame:
    ...

@pytest.fixture
def tickers() -> MagicMock:
    ...

@pytest.fixture
def dataframe_1m(testdatadir: Path) -> pd.DataFrame:
    ...

@pytest.fixture
def trades_for_order() -> List[Dict[str, Any]]:
    ...

@pytest.fixture
def trades_history() -> List[List[float]]:
    ...

@pytest.fixture
def trades_history_df() -> pd.DataFrame:
    ...

@pytest.fixture
def fetch_trades_result() -> List[Dict[str, Any]]:
    ...

@pytest.fixture
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

@pytest.fixture
def open_trade() -> Trade:
    ...

@pytest.fixture
def open_trade_usdt() -> Trade:
    ...

@pytest.fixture
def limit_buy_order_usdt_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_buy_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_usdt_open() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_sell_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture
def market_buy_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture
def market_buy_order_usdt_doublefee() -> Dict[str, Any]:
    ...

@pytest.fixture
def market_sell_order_usdt() -> Dict[str, Any]:
    ...

@pytest.fixture
def limit_order() -> Dict[str, Dict[str, Any]]:
    ...

@pytest.fixture
def limit_order_open() -> Dict[str, Dict[str, Any]]:
    ...

@pytest.fixture
def mark_ohlcv() -> List[List[float]]:
    ...

@pytest.fixture
def funding_rate_history_hourly() -> List[Dict[str, Any]]:
    ...

@pytest.fixture
def funding_rate_history_octohourly() -> List[Dict[str, Any]]:
    ...

@pytest.fixture
def leverage_tiers() -> Dict[str, List[Dict[str, Any]]]:
    ...