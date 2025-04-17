# pragma pylint: disable=missing-docstring
import json
import logging
import platform
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling

from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import CandleType, MarginMode, RunMode, SignalDirection, TradingMode
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


logging.getLogger("").setLevel(logging.INFO)


# Do not mask numpy errors as warnings that no one read, raise the exсeption
np.seterr(all="raise")

CURRENT_TEST_STRATEGY: str = "StrategyTestV3"
TRADE_SIDES: Tuple[str, str] = ("long", "short")
EXMS: str = "freqtrade.exchange.exchange.Exchange"


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="Enable long-run tests (ccxt compat)",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "longrun: mark test that is running slowly and should not be run regularly"
    )
    if not config.option.longrun:
        config.option.markexpr = "not longrun"


class FixtureScheduler(LoadScopeScheduling):
    # Based on the suggestion in
    # https://github.com/pytest-dev/pytest-xdist/issues/18

    def _split_scope(self, nodeid: str) -> str:
        if "exchange_online" in nodeid:
            try:
                # Extract exchange ID from nodeid
                exchange_id = nodeid.split("[")[1].split("-")[0].rstrip("]")
                return exchange_id
            except Exception as e:
                print(e)
                pass

        return nodeid


def pytest_xdist_make_scheduler(config: Any, log: Any) -> FixtureScheduler:
    return FixtureScheduler(config, log)


def log_has(line: str, logs: Any) -> bool:
    """Check if line is found on some caplog's message."""
    return any(line == message for message in logs.messages)


def log_has_when(line: str, logs: Any, when: str) -> bool:
    """Check if line is found in caplog's messages during a specified stage"""
    return any(line == message.message for message in logs.get_records(when))


def log_has_re(line: str, logs: Any) -> bool:
    """Check if line matches some caplog's message."""
    return any(re.match(line, message) for message in logs.messages)


def num_log_has(line: str, logs: Any) -> int:
    """Check how many times line is found in caplog's messages."""
    return sum(line == message for message in logs.messages)


def num_log_has_re(line: str, logs: Any) -> int:
    """Check how many times line matches caplog's messages."""
    return sum(bool(re.match(line, message)) for message in logs.messages)


def get_args(args: Any) -> Any:
    return Arguments(args).get_parsed_arg()


def generate_trades_history(n_rows: int, start_date: Optional[datetime] = None, days: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    if not start_date:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

        # Generate random data
    end_date = start_date + timedelta(days=days)
    _start_timestamp = start_date.timestamp()
    _end_timestamp = pd.to_datetime(end_date).timestamp()

    random_timestamps_in_seconds = np.random.uniform(_start_timestamp, _end_timestamp, n_rows)
    timestamp = pd.to_datetime(random_timestamps_in_seconds, unit="s")

    trade_id = [
        f"a{np.random.randint(1e6, 1e7 - 1)}cd{np.random.randint(100, 999)}" for _ in range(n_rows)
    ]

    side = np.random.choice(["buy", "sell"], n_rows)

    # Initial price and subsequent changes
    initial_price = 0.019626
    price_changes = np.random.normal(0, initial_price * 0.05, n_rows)
    price = np.cumsum(np.concatenate(([initial_price], price_changes)))[:n_rows]

    amount = np.random.uniform(0.011, 20, n_rows)
    cost = price * amount

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "id": trade_id,
            "type": None,
            "side": side,
            "price": price,
            "amount": amount,
            "cost": cost,
        }
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert list(df.columns) == constants.DEFAULT_TRADES_COLUMNS + ["date"]
    return df


def generate_test_data(timeframe: str, size: int, start: str = "2020-07-05", random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    base = np.random.normal(20, 2, size=size)
    if timeframe == "1y":
        date = pd.date_range(start, periods=size, freq="1YS", tz="UTC")
    elif timeframe == "1M":
        date = pd.date_range(start, periods=size, freq="1MS", tz="UTC")
    elif timeframe == "3M":
        date = pd.date_range(start, periods=size, freq="3MS", tz="UTC")
    elif timeframe == "1w" or timeframe == "7d":
        date = pd.date_range(start, periods=size, freq="1W-MON", tz="UTC")
    else:
        tf_mins = timeframe_to_minutes(timeframe)
        if tf_mins >= 1:
            date = pd.date_range(start, periods=size, freq=f"{tf_mins}min", tz="UTC")
        else:
            tf_secs = timeframe_to_seconds(timeframe)
            date = pd.date_range(start, periods=size, freq=f"{tf_secs}s", tz="UTC")
    df = pd.DataFrame(
        {
            "date": date,
            "open": base,
            "high": base + np.random.normal(2, 1, size=size),
            "low": base - np.random.normal(2, 1, size=size),
            "close": base + np.random.normal(0, 1, size=size),
            "volume": np.random.normal(200, size=size),
        }
    )
    df = df.dropna()
    return df


def generate_test_data_raw(timeframe: str, size: int, start: str = "2020-07-05", random_seed: int = 42) -> List[List[Any]]:
    """Generates data in the ohlcv format used by ccxt"""
    df = generate_test_data(timeframe, size, start, random_seed)
    df["date"] = df.loc[:, "date"].astype(np.int64) // 1000 // 1000
    return list(list(x) for x in zip(*(df[x].values.tolist() for x in df.columns), strict=False))


# Source: https://stackoverflow.com/questions/29881236/how-to-mock-asyncio-coroutines
# TODO: This should be replaced with AsyncMock once support for python 3.7 is dropped.
def get_mock_coro(return_value: Optional[Any] = None, side_effect: Optional[Any] = None) -> Mock:
    async def mock_coro(*args: Any, **kwargs: Any) -> Any:
        if side_effect:
            if isinstance(side_effect, list):
                effect = side_effect.pop(0)
            else:
                effect = side_effect
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            return effect
        else:
            return return_value

    return Mock(wraps=mock_coro)


def patched_configuration_load_config_file(mocker: Any, config: Dict[str, Any]) -> None:
    mocker.patch(
        "freqtrade.configuration.load_config.load_config_file", lambda *args, **kwargs: config
    )


def patch_exchange(
    mocker: Any, api_mock: Optional[Any] = None, exchange: str = "binance", mock_markets: bool = True, mock_supported_modes: bool = True
) -> None:
    mocker.patch(f"{EXMS}.validate_config", MagicMock())
    mocker.patch(f"{EXMS}.validate_timeframes", MagicMock())
    mocker.patch(f"{EXMS}.id", PropertyMock(return_value=exchange))
    mocker.patch(f"{EXMS}.name", PropertyMock(return_value=exchange.title()))
    mocker.patch(f"{EXMS}.precisionMode", PropertyMock(return_value=2))
    mocker.patch(f"{EXMS}.precision_mode_price", PropertyMock(return_value=2))
    # Temporary patch ...
    mocker.patch("freqtrade.exchange.bybit.Bybit.cache_leverage_tiers")

    if mock_markets:
        mocker.patch(f"{EXMS}._load_async_markets", return_value={})
        if isinstance(mock_markets, bool):
            mock_markets = get_markets()
        mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=mock_markets))

    if mock_supported_modes:
        mocker.patch(
            f"freqtrade.exchange.{exchange}.{exchange.capitalize()}"
            "._supported_trading_mode_margin_pairs",
            PropertyMock(
                return_value=[
                    (TradingMode.MARGIN, MarginMode.CROSS),
                    (TradingMode.MARGIN, MarginMode.ISOLATED),
                    (TradingMode.FUTURES, MarginMode.CROSS),
                    (TradingMode.FUTURES, MarginMode.ISOLATED),
                ]
            ),
        )

    if api_mock:
        mocker.patch(f"{EXMS}._init_ccxt", return_value=api_mock)
    else:
        mocker.patch(f"{EXMS}.get_fee", return_value=0.0025)
        mocker.patch(f"{EXMS}._init_ccxt", MagicMock())
        mocker.patch(f"{EXMS}.timeframes", PropertyMock(return_value=["5m", "15m", "1h", "1d"]))


def get_patched_exchange(
    mocker: Any, config: Dict[str, Any], api_mock: Optional[Any] = None, exchange: str = "binance", mock_markets: bool = True, mock_supported_modes: bool = True
) -> Exchange:
    patch_exchange(mocker, api_mock, exchange, mock_markets, mock_supported_modes)
    config["exchange"]["name"] = exchange
    try:
        exchange = ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
    except ImportError:
        exchange = Exchange(config)
    return exchange


def patch_wallet(mocker: Any, free: float = 999.9) -> None:
    mocker.patch("freqtrade.wallets.Wallets.get_free", MagicMock(return_value=free))


def patch_whitelist(mocker: Any, conf: Dict[str, Any]) -> None:
    mocker.patch(
        "freqtrade.freqtradebot.FreqtradeBot._refresh_active_whitelist",
        MagicMock(return_value=conf["exchange"]["pair_whitelist"]),
    )


def patch_edge(mocker: Any) -> None:
    # "ETH/BTC",
    # "LTC/BTC",
    # "XRP/BTC",
    # "NEO/BTC"

    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "NEO/BTC": PairInfo(-0.20, 0.66, 3.71, 0.50, 1.71, 10, 25),
                "LTC/BTC": PairInfo(-0.21, 0.66, 3.71, 0.50, 1.71, 11, 20),
            }
        ),
    )
    mocker.patch("freqtrade.edge.Edge.calculate", MagicMock(return_value=True))


# Functions for recurrent object patching


def patch_freqtradebot(mocker: Any, config: Dict[str, Any]) -> None:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.RPCManager._init", MagicMock())
    mocker.patch("freqtrade.freqtradebot.RPCManager.send_msg", MagicMock())
    patch_whitelist(mocker, config)
    mocker.patch("freqtrade.freqtradebot.ExternalMessageConsumer")
    mocker.patch("freqtrade.configuration.config_validation._validate_consumers")


def get_patched_freqtradebot(mocker: Any, config: Dict[str, Any]) -> FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
    return FreqtradeBot(config)


def get_patched_worker(mocker: Any, config: Dict[str, Any]) -> Worker:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: Worker
    """
    patch_freqtradebot(mocker, config)
    return Worker(args=None, config=config)


def patch_get_signal(
    freqtrade: FreqtradeBot,
    enter_long: bool = True,
    exit_long: bool = False,
    enter_short: bool = False,
    exit_short: bool = False,
    enter_tag: Optional[str] = None,
    exit_tag: Optional[str] = None,
) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :return: None
    """

    # returns (Signal-direction, signaname)
    def patched_get_entry_signal(*args: Any, **kwargs: Any) -> Tuple[Optional[SignalDirection], Optional[str]]:
        direction = None
        if enter_long and not any([exit_long, enter_short]):
            direction = SignalDirection.LONG
        if enter_short and not any([exit_short, enter_long]):
            direction = SignalDirection.SHORT

        return direction, enter_tag

    freqtrade.strategy.get_entry_signal = patched_get_entry_signal

    def patched_get_exit_signal(pair: str, timeframe: str, dataframe: pd.DataFrame, is_short: bool) -> Tuple[bool, bool, Optional[str]]:
        if is_short:
            return enter_short, exit_short, exit_tag
        else:
            return enter_long, exit_long, exit_tag

    # returns (enter, exit)
    freqtrade.strategy.get_exit_signal = patched_get_exit_signal

    freqtrade.exchange.refresh_latest_ohlcv = lambda p: None


def create_mock_trades(fee: float, is_short: Optional[bool] = False, use_db: bool = True) -> None:
    """
    Create some fake trades ...
    :param is_short: Optional bool, None creates a mix of long and short trades.
    """

    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False
    # Simulate dry_run entries
    trade = mock_trade_1(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_2(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_3(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_4(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_5(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_6(fee, is_short1)
    add_trade(trade)

    if use_db:
        Trade.commit()


def create_mock_trades_with_leverage(fee: float, use_db: bool = True) -> None:
    """
    Create some fake trades ...
    """
    if use_db:
        Trade.session.rollback()

    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    # Simulate dry_run entries
    trade = mock_trade_1(fee, False)
    add_trade(trade)

    trade = mock_trade_2(fee, False)
    add_trade(trade)

    trade = mock_trade_3(fee, False)
    add_trade(trade)

    trade = mock_trade_4(fee, False)
    add_trade(trade)

    trade = mock_trade_5(fee, False)
    add_trade(trade)

    trade = mock_trade_6(fee, False)
    add_trade(trade)

    trade = short_trade(fee)
    add_trade(trade)

    trade = leverage_trade(fee)
    add_trade(trade)

    if use_db:
        Trade.session.flush()


def create_mock_trades_usdt(fee: float, is_short: Optional[bool] = False, use_db: bool = True) -> None:
    """
    Create some fake trades ...
    """

    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False

    # Simulate dry_run entries
    trade = mock_trade_usdt_1(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_2(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_3(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_4(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_usdt_5(fee, is_short2)
    add_trade(trade)

    trade = mock_trade_usdt_6(fee, is_short1)
    add_trade(trade)

    trade = mock_trade_usdt_7(fee, is_short1)
    add_trade(trade)
    if use_db:
        Trade.commit()


@pytest.fixture(autouse=True)
def patch_gc(mocker: Any) -> None:
    mocker.patch("freqtrade.main.gc_set_threshold")


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


def is_mac() -> bool:
    machine = platform.system()
    return "Darwin" in machine


@pytest.fixture(autouse=True)
def patch_torch_initlogs(mocker: Any) -> None:
    if is_mac():
        # Mock torch import completely
        import sys
        import types

        module_name = "torch"
        mocked_module = types.ModuleType(module_name)
        sys.modules[module_name] = mocked_module
    else:
        mocker.patch("torch._logging._init_logs")


@pytest.fixture(autouse=True)
def user_dir(mocker: Any, tmp_path: Path) -> Path:
    user_dir = tmp_path / "user_data"
    mocker.patch("freqtrade.configuration.configuration.create_userdata_dir", return_value=user_dir)
    return user_dir


@pytest.fixture(autouse=True)
def patch_coingecko(mocker: Any) -> None:
    """
    Mocker to coingecko to speed up tests
    :param mocker: mocker to patch coingecko class
    :return: None
    """

    tickermock = MagicMock(return_value={"bitcoin": {"usd": 12345.0}, "ethereum": {"usd": 12345.0}})
    listmock = MagicMock(
        return_value=[
            {"id": "bitcoin", "name": "Bitcoin", "symbol": "btc", "website_slug": "bitcoin"},
            {"id": "ethereum", "name": "Ethereum", "symbol": "eth", "website_slug": "ethereum"},
        ]
    )
    mocker.patch.multiple(
        "freqtrade.rpc.fiat_convert.FtCoinGeckoApi",
        get_price=tickermock,
        get_coins_list=listmock,
    )


@pytest.fixture(scope="function")
def init_persistence(default_conf: Dict[str, Any]) -> None:
    init_db(default_conf["db_url"])


@pytest.fixture(scope="function")
def default_conf(testdatadir: Path) -> Dict[str, Any]:
    return get_default_conf(testdatadir)


@pytest.fixture(scope="function")
def default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    return get_default_conf_usdt(testdatadir)


def get_default_conf(testdatadir: Path) -> Dict[str, Any]:
    """Returns validated configuration suitable for most tests"""
    configuration = {
        "max_open_trades": 1,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "timeframe": "5m",
        "dry_run": True,
        "cancel_open_orders_on_exit": False,
        "minimal_roi": {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04},
        "dry_run_wallet": 1000,
        "stoploss": -0.10,
        "unfilledtimeout": {"entry": 10, "exit": 30},
        "entry_pricing": {
            "price_last_balance": 0.0,
            "use_order_book": False,
            "order_book_top": 1,
            "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
        },
        "exit_pricing": {
            "use_order_book": False,
            "order_book_top": 1,
        },
        "exchange": {
            "name": "binance",
            "key": "key",
            "enable_ws": False,
            "secret": "secret",
            "pair_whitelist": ["ETH/BTC", "LTC/BTC", "XRP/BTC", "NEO/BTC"],
            "pair_blacklist": [
                "DOGE/BTC",
                "HOT/BTC",
            ],
        },
        "pairlists": [{"method": "StaticPairList"}],
        "telegram": {
            "enabled": False,
            "token": "token",
            "chat_id": "1235",
            "notification_settings": {},
        },
        "datadir": Path(testdatadir),
        "initial_state": "running",
        "db_url": "sqlite://",
        "user_data_dir": Path("user_data"),
        "verbosity": 3,
        "strategy_path": str(Path(__file__).parent / "strategy" / "strats"),
        "strategy": CURRENT_TEST_STRATEGY,
        "disableparamexport": True,
        "internals": {},
        "export": "none",
        "dataformat_ohlcv": "feather",
        "dataformat_trades": "feather",
        "runmode": "dry_run",
        "trading_mode": "spot",
        "margin_mode": "",
        "candle_type_def": CandleType.SPOT,
    }
    return configuration


def get_default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    configuration = get_default_conf(testdatadir)
    configuration.update(
        {
            "stake_amount": 60.0,
            "stake_currency": "USDT",
            "exchange": {
                "name": "binance",
                "enabled": True,
                "key": "key",
                "enable_ws": False,
                "secret": "secret",
                "pair_whitelist": [
                    "ETH/USDT",
                    "LTC/USDT",
                    "XRP/USDT",
                    "NEO/USDT",
                    "TKN/USDT",
                ],
                "pair_blacklist": [
                    "DOGE/USDT",
                    "HOT/USDT",
                ],
            },
        }
    )
    return configuration


@pytest.fixture
def fee() -> MagicMock:
    return MagicMock(return_value=0.0025)


@pytest.fixture
def ticker() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 0.00001098,
            "ask": 0.00001099,
            "last": 0.00001098,
        }
    )


@pytest.fixture
def ticker_sell_up() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 0.00001172,
            "ask": 0.00001173,
            "last": 0.00001172,
        }
    )


@pytest.fixture
def ticker_sell_down() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 0.00001044,
            "ask": 0.00001043,
            "last": 0.00001044,
        }
    )


@pytest.fixture
def ticker_usdt() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 2.0,
            "ask": 2.02,
            "last": 2.0,
        }
    )


@pytest.fixture
def ticker_usdt_sell_up() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 2.2,
            "ask": 2.3,
            "last": 2.2,
        }
    )


@pytest.fixture
def ticker_usdt_sell_down() -> MagicMock:
    return MagicMock(
        return_value={
            "bid": 2.01,
            "ask": 2.0,
            "last": 2.01,
        }
    )


@pytest.fixture
def markets() -> MagicMock:
    return get_markets()


def get_markets() -> Dict[str, Any]:
    # See get_markets_static() for immutable markets and do not modify them unless absolutely
    # necessary!
    return {
        "ETH/BTC": {
            "id": "ethbtc",
            "symbol": "ETH/BTC",
            "base": "ETH",
            "quote": "BTC",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 100000000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {"min": 1.0, "max": 2.0},
            },
        },
        "TKN/BTC": {
            "id": "tknbtc",
            "symbol": "TKN/BTC",
            "base": "TKN",
            "quote": "BTC",
            # According to ccxt, markets without active item set are also active
            # 'active': True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 100000000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {"min": 1.0, "max": 5.0},
            },
        },
        "BLK/BTC": {
            "id": "blkbtc",
            "symbol": "BLK/BTC",
            "base": "BLK",
            "quote": "BTC",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 1000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {"min": 1.0, "max": 3.0},
            },
        },
        "LTC/BTC": {
            "id": "ltcbtc",
            "symbol": "LTC/BTC",
            "base": "LTC",
            "quote": "BTC",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 100000000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {"min": None, "max": None},
            },
            "info": {},
        },
        "XRP/BTC": {
            "id": "xrpbtc",
            "symbol": "XRP/BTC",
            "base": "XRP",
            "quote": "BTC",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 100000000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {
                    "min": None,
                    "max": None,
                },
            },
            "info": {},
        },
        "NEO/BTC": {
            "id": "neobtc",
            "symbol": "NEO/BTC",
            "base": "NEO",
            "quote": "BTC",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {
                "price": 8,
                "amount": 8,
                "cost": 8,
            },
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {
                    "min": 0.01,
                    "max": 100000000,
                },
                "price": {
                    "min": None,
                    "max": 500000,
                },
                "cost": {
                    "min": 0.0001,
                    "max": 500000,
                },
                "leverage": {
                    "min": None,
                    "max": None,
                },
            },
            "info": {},
        },
        "BTT/BTC": {
            "id": "BTTBTC",
            "symbol": "BTT/BTC",
            "base": "BTT",
            "quote": "BTC",
            "active": False,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "contractSize": None,
            "precision": {"base": 8, "quote": 8, "amount": 0, "price": 8},
            "limits": {
                "amount": {"min": 1.0, "max": 90000000.0},
                "price": {"min": None, "max": None},
                "cost