#!/usr/bin/env python3
from __future__ import annotations
import json
import logging
import pytest
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Union

from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.persistence import Trade, Order

# --------------------
# Helper functions
# --------------------

def get_default_conf(testdatadir: Path) -> Dict[str, Any]:
    """Returns validated configuration suitable for most tests"""
    configuration: Dict[str, Any] = {
        "dry_run": True,
        "stake_currency": "BTC",
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "timeframe": "5m",
        "cancel_open_orders_on_exit": False,
        "entry_pricing": {"use_order_book": False, "order_book_top": 1},
        "exit_pricing": {"use_order_book": False, "order_book_top": 1},
        "exchange": {
            "name": "binance",
            "key": "123",
            "secret": "abc",
            "pair_whitelist": ["BTC/USDT", "ETH/BTC"],
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
        "telegram": {"enabled": False},
        "datadir": testdatadir,
        "initial_state": "running",
        "db_url": "sqlite://",
        "user_data_dir": Path("user_data"),
        "verbosity": 3,
        "strategy": "SampleStrategy",
        "internals": {},
        "export": "none",
        "dataformat_ohlcv": "feather",
        "dataformat_trades": "feather",
        "runmode": "dry_run",
        "trading_mode": "spot",
        "margin_mode": "",
        "candle_type_def": "spot",
    }
    return configuration


def get_default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    configuration: Dict[str, Any] = get_default_conf(testdatadir)
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


def get_markets() -> Dict[str, Dict[str, Any]]:
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
            "precision": {"price": 8, "amount": 8, "cost": 8},
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {"min": 0.01, "max": 100000000},
                "price": {"min": None, "max": 500000},
                "cost": {"min": 0.0001, "max": 500000},
                "leverage": {"min": 1.0, "max": 2.0},
            },
        },
        "TKN/BTC": {
            "id": "tknbtc",
            "symbol": "TKN/BTC",
            "base": "TKN",
            "quote": "BTC",
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
            "precision": {"price": 8, "amount": 8, "cost": 8},
            "lot": 0.00000001,
            "contractSize": None,
            "limits": {
                "amount": {"min": 0.01, "max": 100000000},
                "price": {"min": None, "max": 500000},
                "cost": {"min": 0.0001, "max": 500000},
                "leverage": {"min": 1.0, "max": 5.0},
            },
        },
        # ... More market definitions ...
    }


# --------------------
# Pytest fixtures
# --------------------

@pytest.fixture
def fee() -> Any:
    return pytest.helpers.MagicMock(return_value=0.0025)


@pytest.fixture
def ticker() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 0.00001098,
            "ask": 0.00001099,
            "last": 0.00001098,
        }
    )


@pytest.fixture
def ticker_sell_up() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 0.00001172,
            "ask": 0.00001173,
            "last": 0.00001172,
        }
    )


@pytest.fixture
def ticker_sell_down() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 0.00001044,
            "ask": 0.00001043,
            "last": 0.00001044,
        }
    )


@pytest.fixture
def ticker_usdt() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 2.0,
            "ask": 2.02,
            "last": 2.0,
        }
    )


@pytest.fixture
def ticker_usdt_sell_up() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 2.2,
            "ask": 2.3,
            "last": 2.2,
        }
    )


@pytest.fixture
def ticker_usdt_sell_down() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bid": 2.01,
            "ask": 2.0,
            "last": 2.01,
        }
    )


@pytest.fixture
def markets() -> Dict[str, Dict[str, Any]]:
    return get_markets()


@pytest.fixture
def markets_static() -> Dict[str, Dict[str, Any]]:
    static_markets: List[str] = [
        "BLK/BTC",
        "BTT/BTC",
        "ETH/BTC",
        "ETH/USDT",
        "LTC/BTC",
        "LTC/ETH",
        "LTC/USD",
        "LTC/USDT",
        "NEO/BTC",
        "TKN/BTC",
        "XLTCUSDT",
        "XRP/BTC",
        "ADA/USDT:USDT",
        "ETH/USDT:USDT",
    ]
    all_markets: Dict[str, Dict[str, Any]] = get_markets()
    return {m: all_markets[m] for m in static_markets if m in all_markets}


@pytest.fixture
def shitcoinmarkets(markets_static: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    shitmarkets: Dict[str, Dict[str, Any]] = deepcopy(markets_static)
    shitmarkets.update(
        {
            "HOT/BTC": {
                "id": "HOTBTC",
                "symbol": "HOT/BTC",
                "base": "HOT",
                "quote": "BTC",
                "active": True,
                "spot": True,
                "type": "spot",
                "precision": {"base": 8, "quote": 8, "amount": 0, "price": 8},
                "limits": {
                    "amount": {"min": 1.0, "max": 90000000.0},
                    "price": {"min": None, "max": None},
                    "cost": {"min": 0.001, "max": None},
                },
                "info": {},
            },
            "FUEL/BTC": {
                "id": "FUELBTC",
                "symbol": "FUEL/BTC",
                "base": "FUEL",
                "quote": "BTC",
                "active": True,
                "spot": True,
                "type": "spot",
                "precision": {"base": 8, "quote": 8, "amount": 0, "price": 8},
                "limits": {
                    "amount": {"min": 1.0, "max": 90000000.0},
                    "price": {"min": 1e-08, "max": 1000.0},
                    "cost": {"min": 0.001, "max": None},
                },
                "info": {},
            },
            "NANO/USDT": {
                "percentage": True,
                "tierBased": False,
                "taker": 0.001,
                "maker": 0.001,
                "precision": {"base": 8, "quote": 8, "amount": 2, "price": 4},
                "limits": {
                    "leverage": {"min": None, "max": None},
                    "amount": {"min": None, "max": None},
                    "price": {"min": None, "max": None},
                    "cost": {"min": None, "max": None},
                },
                "id": "NANOUSDT",
                "symbol": "NANO/USDT",
                "base": "NANO",
                "quote": "USDT",
                "baseId": "NANO",
                "quoteId": "USDT",
                "info": {},
                "type": "spot",
                "spot": True,
                "future": False,
                "active": True,
            },
            "ADAHALF/USDT": {
                "percentage": True,
                "tierBased": False,
                "taker": 0.001,
                "maker": 0.001,
                "precision": {"base": 8, "quote": 8, "amount": 2, "price": 4},
                "limits": {
                    "leverage": {"min": None, "max": None},
                    "amount": {"min": None, "max": None},
                    "price": {"min": None, "max": None},
                    "cost": {"min": None, "max": None},
                },
                "id": "ADAHALFUSDT",
                "symbol": "ADAHALF/USDT",
                "base": "ADAHALF",
                "quote": "USDT",
                "baseId": "ADAHALF",
                "quoteId": "USDT",
                "info": {},
                "type": "spot",
                "spot": True,
                "future": False,
                "active": True,
            },
            "ADADOUBLE/USDT": {
                "percentage": True,
                "tierBased": False,
                "taker": 0.001,
                "maker": 0.001,
                "precision": {"base": 8, "quote": 8, "amount": 2, "price": 4},
                "limits": {
                    "leverage": {"min": None, "max": None},
                    "amount": {"min": None, "max": None},
                    "price": {"min": None, "max": None},
                    "cost": {"min": None, "max": None},
                },
                "id": "ADADOUBLEUSDT",
                "symbol": "ADADOUBLE/USDT",
                "base": "ADADOUBLE",
                "quote": "USDT",
                "baseId": "ADADOUBLE",
                "quoteId": "USDT",
                "info": {},
                "type": "spot",
                "spot": True,
                "future": False,
                "active": True,
            },
        }
    )
    return shitmarkets


@pytest.fixture
def markets_empty() -> Any:
    return pytest.helpers.MagicMock(return_value=[])


@pytest.fixture(scope="function")
def limit_buy_order_open() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_buy",
        "type": "limit",
        "side": "buy",
        "symbol": "mocked",
        "timestamp": pd.Timestamp.now().timestamp(),
        "datetime": pd.Timestamp.now().isoformat(),
        "price": 0.00001099,
        "average": 0.00001099,
        "amount": 90.99181073,
        "filled": 0.0,
        "cost": 0.0009999,
        "remaining": 90.99181073,
        "status": "open",
    }


@pytest.fixture
def limit_buy_order_old() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_buy_old",
        "type": "limit",
        "side": "buy",
        "symbol": "mocked",
        "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
        "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
        "price": 0.00001099,
        "amount": 90.99181073,
        "filled": 0.0,
        "remaining": 90.99181073,
        "status": "open",
    }


@pytest.fixture
def limit_sell_order_old() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_sell_old",
        "type": "limit",
        "side": "sell",
        "symbol": "ETH/BTC",
        "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
        "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
        "price": 0.00001099,
        "amount": 90.99181073,
        "filled": 0.0,
        "remaining": 90.99181073,
        "status": "open",
    }


@pytest.fixture
def limit_buy_order_old_partial() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_buy_old_partial",
        "type": "limit",
        "side": "buy",
        "symbol": "ETH/BTC",
        "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
        "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
        "price": 0.00001099,
        "amount": 90.99181073,
        "filled": 23.0,
        "cost": 90.99181073 * 23.0,
        "remaining": 67.99181073,
        "status": "open",
    }


@pytest.fixture
def limit_buy_order_old_partial_canceled(limit_buy_order_old_partial: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = deepcopy(limit_buy_order_old_partial)
    res["status"] = "canceled"
    res["fee"] = {"cost": 0.023, "currency": "ETH"}
    return res


@pytest.fixture(scope="function")
def limit_buy_order_canceled_empty(request: Any) -> Dict[str, Any]:
    exchange_name: str = request.param
    if exchange_name == "kraken":
        return {
            "info": {},
            "id": "AZNPFF-4AC4N-7MKTAT",
            "clientOrderId": None,
            "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
            "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
            "lastTradeTimestamp": None,
            "status": "canceled",
            "symbol": "LTC/USDT",
            "type": "limit",
            "side": "buy",
            "price": 34.3225,
            "cost": 0.0,
            "amount": 0.55,
            "filled": 0.0,
            "average": 0.0,
            "remaining": 0.55,
            "fee": {"cost": 0.0, "rate": None, "currency": "USDT"},
            "trades": [],
        }
    elif exchange_name == "binance":
        return {
            "info": {},
            "id": "1234512345",
            "clientOrderId": "alb1234123",
            "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
            "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
            "lastTradeTimestamp": None,
            "symbol": "LTC/USDT",
            "type": "limit",
            "side": "buy",
            "price": 0.016804,
            "amount": 0.55,
            "cost": 0.0,
            "average": None,
            "filled": 0.0,
            "remaining": 0.55,
            "status": "canceled",
            "fee": None,
            "trades": None,
        }
    else:
        return {
            "info": {},
            "id": "1234512345",
            "clientOrderId": "alb1234123",
            "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
            "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
            "lastTradeTimestamp": None,
            "symbol": "LTC/USDT",
            "type": "limit",
            "side": "buy",
            "price": 0.016804,
            "amount": 0.55,
            "cost": 0.0,
            "average": None,
            "filled": 0.0,
            "remaining": 0.55,
            "status": "canceled",
            "fee": None,
            "trades": None,
        }


@pytest.fixture
def limit_sell_order_open() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_sell",
        "type": "limit",
        "side": "sell",
        "symbol": "mocked",
        "datetime": pd.Timestamp.now().isoformat(),
        "timestamp": pd.Timestamp.now().timestamp(),
        "price": 0.00001173,
        "amount": 90.99181073,
        "filled": 0.0,
        "remaining": 90.99181073,
        "status": "open",
    }


@pytest.fixture
def limit_sell_order(limit_sell_order_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_sell_order_open)
    order["remaining"] = 0.0
    order["filled"] = order["amount"]
    order["status"] = "closed"
    return order


@pytest.fixture
def order_book_l2() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "bids": [
                [0.043936, 10.442],
                [0.043935, 31.865],
                [0.043933, 11.212],
                [0.043928, 0.088],
                [0.043925, 10.0],
                [0.043921, 10.0],
                [0.04392, 37.64],
                [0.043899, 0.066],
                [0.043885, 0.676],
                [0.04387, 22.758],
            ],
            "asks": [
                [0.043949, 0.346],
                [0.04395, 0.608],
                [0.043951, 3.948],
                [0.043954, 0.288],
                [0.043958, 9.277],
                [0.043995, 1.566],
                [0.044, 0.588],
                [0.044002, 0.992],
                [0.044003, 0.095],
                [0.04402, 37.64],
            ],
            "timestamp": None,
            "datetime": None,
            "nonce": 288004540,
        }
    )


@pytest.fixture
def order_book_l2_usd() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "symbol": "LTC/USDT",
            "bids": [
                [25.563, 49.269],
                [25.562, 83.0],
                [25.56, 106.0],
                [25.559, 15.381],
                [25.558, 29.299],
                [25.557, 34.624],
                [25.556, 10.0],
                [25.555, 14.684],
                [25.554, 45.91],
                [25.553, 50.0],
            ],
            "asks": [
                [25.566, 14.27],
                [25.567, 48.484],
                [25.568, 92.349],
                [25.572, 31.48],
                [25.573, 23.0],
                [25.574, 20.0],
                [25.575, 89.606],
                [25.576, 262.016],
                [25.577, 178.557],
                [25.578, 78.614],
            ],
            "timestamp": None,
            "datetime": None,
            "nonce": 2372149736,
        }
    )


@pytest.fixture
def ohlcv_history_list() -> List[List[Union[int, float]]]:
    return [
        [1511686200000, 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869],
        [1511686500000, 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 0.05874751],
        [1511686800000, 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 0.7039405],
    ]


@pytest.fixture
def ohlcv_history(ohlcv_history_list: List[List[Union[int, float]]]) -> pd.DataFrame:
    return ohlcv_to_dataframe(
        ohlcv_history_list, "5m", pair="UNITTEST/BTC", fill_missing=True, drop_incomplete=False
    )


@pytest.fixture
def tickers() -> Any:
    return pytest.helpers.MagicMock(
        return_value={
            "ETH/BTC": {
                "symbol": "ETH/BTC",
                "timestamp": 1522014806207,
                "datetime": "2018-03-25T21:53:26.207Z",
                "high": 0.061697,
                "low": 0.060531,
                "bid": 0.061588,
                "bidVolume": 3.321,
                "ask": 0.061655,
                "askVolume": 0.212,
                "vwap": 0.06105296,
                "open": 0.060809,
                "close": 0.060761,
                "first": None,
                "last": 0.061588,
                "change": 1.281,
                "percentage": None,
                "average": None,
                "baseVolume": 111649.001,
                "quoteVolume": 6816.50176926,
                "info": {},
            },
            "TKN/BTC": {
                "symbol": "TKN/BTC",
                "timestamp": 1522014806169,
                "datetime": "2018-03-25T21:53:26.169Z",
                "high": 0.01885,
                "low": 0.018497,
                "bid": 0.018799,
                "bidVolume": 8.38,
                "ask": 0.018802,
                "askVolume": 15.0,
                "vwap": 0.01869197,
                "open": 0.018585,
                "close": 0.018573,
                "last": 0.018799,
                "baseVolume": 81058.66,
                "quoteVolume": 2247.48374509,
            },
            # ... Additional tickers ...
        }
    )


@pytest.fixture
def dataframe_1m(testdatadir: Path) -> pd.DataFrame:
    with (testdatadir / "UNITTEST_BTC-1m.json").open("r") as data_file:
        return ohlcv_to_dataframe(
            json.load(data_file), "1m", pair="UNITTEST/BTC", fill_missing=True
        )


@pytest.fixture(scope="function")
def trades_for_order() -> List[Dict[str, Any]]:
    return [
        {
            "info": {
                "id": 34567,
                "orderId": 123456,
                "price": "2.0",
                "qty": "8.00000000",
                "commission": "0.00800000",
                "commissionAsset": "LTC",
                "time": 1521663363189,
                "isBuyer": True,
                "isMaker": False,
                "isBestMatch": True,
            },
            "timestamp": 1521663363189,
            "datetime": "2018-03-21T20:16:03.189Z",
            "symbol": "LTC/USDT",
            "id": "34567",
            "order": "123456",
            "type": None,
            "side": "buy",
            "price": 2.0,
            "cost": 16.0,
            "amount": 8.0,
            "fee": {"cost": 0.008, "currency": "LTC"},
        }
    ]


@pytest.fixture(scope="function")
def trades_history() -> List[List[Union[int, float, str, None]]]:
    return [
        [1565798389463, "12618132aa9", None, "buy", 0.019627, 0.04, 0.00078508],
        [1565798399629, "1261813bb30", None, "buy", 0.019627, 0.244, 0.004788987999999999],
        [1565798399752, "1261813cc31", None, "sell", 0.019626, 0.011, 0.00021588599999999999],
        [1565798399862, "126181cc332", None, "sell", 0.019626, 0.011, 0.00021588599999999999],
        [1565798399862, "126181cc333", None, "sell", 0.019626, 0.012, 0.00021588599999999999],
        [1565798399872, "1261aa81334", None, "sell", 0.019626, 0.011, 0.00021588599999999999],
    ]


@pytest.fixture(scope="function")
def trades_history_df(trades_history: List[List[Any]]) -> pd.DataFrame:
    trades: pd.DataFrame = trades_list_to_df(trades_history)
    trades["date"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    return trades


@pytest.fixture(scope="function")
def fetch_trades_result() -> List[Dict[str, Any]]:
    return [
        {
            "info": ["0.01962700", "0.04000000", "1565798399.4631551", "b", "m", "", "126181329"],
            "timestamp": 1565798399463,
            "datetime": "2019-08-14T15:59:59.463Z",
            "symbol": "ETH/BTC",
            "id": "126181329",
            "order": None,
            "type": None,
            "takerOrMaker": None,
            "side": "buy",
            "price": 0.019627,
            "amount": 0.04,
            "cost": 0.00078508,
            "fee": None,
        },
        {
            "info": ["0.01962700", "0.24400000", "1565798399.6291551", "b", "m", "", "126181330"],
            "timestamp": 1565798399629,
            "datetime": "2019-08-14T15:59:59.629Z",
            "symbol": "ETH/BTC",
            "id": "126181330",
            "order": None,
            "type": None,
            "takerOrMaker": None,
            "side": "buy",
            "price": 0.019627,
            "amount": 0.244,
            "cost": 0.004788987999999999,
            "fee": None,
        },
        {
            "info": ["0.01962600", "0.01100000", "1565798399.7521551", "s", "m", "", "126181331"],
            "timestamp": 1565798399752,
            "datetime": "2019-08-14T15:59:59.752Z",
            "symbol": "ETH/BTC",
            "id": "126181331",
            "order": None,
            "type": None,
            "takerOrMaker": None,
            "side": "sell",
            "price": 0.019626,
            "amount": 0.011,
            "cost": 0.00021588599999999999,
            "fee": None,
        },
        {
            "info": ["0.01962600", "0.01100000", "1565798399.8621551", "s", "m", "", "126181332"],
            "timestamp": 1565798399862,
            "datetime": "2019-08-14T15:59:59.862Z",
            "symbol": "ETH/BTC",
            "id": "126181332",
            "order": None,
            "type": None,
            "takerOrMaker": None,
            "side": "sell",
            "price": 0.019626,
            "amount": 0.011,
            "cost": 0.00021588599999999999,
            "fee": None,
        },
        {
            "info": [
                "0.01952600",
                "0.01200000",
                "1565798399.8721551",
                "s",
                "m",
                "",
                "126181333",
                1565798399872512133,
            ],
            "timestamp": 1565798399872,
            "datetime": "2019-08-14T15:59:59.872Z",
            "symbol": "ETH/BTC",
            "id": "126181333",
            "order": None,
            "type": None,
            "takerOrMaker": None,
            "side": "sell",
            "price": 0.019626,
            "amount": 0.011,
            "cost": 0.00021588599999999999,
            "fee": None,
        },
    ]


@pytest.fixture(scope="function")
def trades_for_order2() -> List[Dict[str, Any]]:
    return [
        {
            "info": {},
            "timestamp": 1521663363189,
            "datetime": "2018-03-21T20:16:03.189Z",
            "symbol": "LTC/ETH",
            "id": "34567",
            "order": "123456",
            "type": None,
            "side": "buy",
            "price": 0.245441,
            "cost": 1.963528,
            "amount": 4.0,
            "fee": {"cost": 0.004, "currency": "LTC"},
        },
        {
            "info": {},
            "timestamp": 1521663363189,
            "datetime": "2018-03-21T20:16:03.189Z",
            "symbol": "LTC/ETH",
            "id": "34567",
            "order": "123456",
            "type": None,
            "side": "buy",
            "price": 0.245441,
            "cost": 1.963528,
            "amount": 4.0,
            "fee": {"cost": 0.004, "currency": "LTC"},
        },
    ]


@pytest.fixture
def buy_order_fee() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_buy_old",
        "type": "limit",
        "side": "buy",
        "symbol": "mocked",
        "timestamp": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).timestamp(),
        "datetime": (pd.Timestamp.now() - pd.Timedelta(minutes=601)).isoformat(),
        "price": 0.245441,
        "amount": 8.0,
        "cost": 1.963528,
        "remaining": 90.99181073,
        "status": "closed",
        "fee": None,
    }


@pytest.fixture(scope="function")
def edge_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf["runmode"] = "dry_run"
    conf["max_open_trades"] = -1
    conf["tradable_balance_ratio"] = 0.5
    conf["stake_amount"] = -1  # using a constant to indicate unlimited stake amount
    conf["edge"] = {
        "enabled": True,
        "process_throttle_secs": 1800,
        "calculate_since_number_of_days": 14,
        "allowed_risk": 0.01,
        "stoploss_range_min": -0.01,
        "stoploss_range_max": -0.1,
        "stoploss_range_step": -0.01,
        "maximum_winrate": 0.80,
        "minimum_expectancy": 0.20,
        "min_trade_number": 15,
        "max_trade_duration_minute": 1440,
        "remove_pumps": False,
    }
    return conf


@pytest.fixture
def rpc_balance() -> Dict[str, Dict[str, float]]:
    return {
        "BTC": {"total": 12.0, "free": 12.0, "used": 0.0},
        "ETH": {"total": 0.0, "free": 0.0, "used": 0.0},
        "USDT": {"total": 10000.0, "free": 10000.0, "used": 0.0},
        "LTC": {"total": 10.0, "free": 10.0, "used": 0.0},
        "XRP": {"total": 0.1, "free": 0.01, "used": 0.0},
        "EUR": {"total": 10.0, "free": 10.0, "used": 0.0},
    }


@pytest.fixture
def testdatadir() -> Path:
    """Return the path where testdata files are stored"""
    return (Path(__file__).parent / "testdata").resolve()


@pytest.fixture(scope="function")
def import_fails() -> Generator[None, None, None]:
    import builtins
    realimport = builtins.__import__

    def mockedimport(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ["filelock", "cysystemd.journal", "uvloop"]:
            raise ImportError(f"No module named '{name}'")
        return realimport(name, *args, **kwargs)
    builtins.__import__ = mockedimport
    yield
    builtins.__import__ = realimport


@pytest.fixture(scope="function")
def open_trade() -> Trade:
    trade: Trade = Trade(
        pair="ETH/BTC",
        open_rate=0.00001099,
        exchange="binance",
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=pd.Timestamp.now() - pd.Timedelta(minutes=601),
        is_open=True,
    )
    trade.orders = [
        Order(
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id="123456789",
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    ]
    return trade


@pytest.fixture(scope="function")
def open_trade_usdt() -> Trade:
    trade: Trade = Trade(
        pair="ADA/USDT",
        open_rate=2.0,
        exchange="binance",
        amount=30.0,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=60.0,
        open_date=pd.Timestamp.now() - pd.Timedelta(minutes=601),
        is_open=True,
    )
    trade.orders = [
        Order(
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id="123456789",
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        ),
        Order(
            ft_order_side="exit",
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id="123456789_exit",
            status="open",
            symbol=trade.pair,
            order_type="limit",
            side="sell",
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        ),
    ]
    return trade


@pytest.fixture(scope="function")
def limit_buy_order_usdt_open() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_buy_usdt",
        "type": "limit",
        "side": "buy",
        "symbol": "mocked",
        "datetime": pd.Timestamp.now().isoformat(),
        "timestamp": pd.Timestamp.now().timestamp(),
        "price": 2.00,
        "average": 2.00,
        "amount": 30.0,
        "filled": 0.0,
        "cost": 60.0,
        "remaining": 30.0,
        "status": "open",
    }


@pytest.fixture(scope="function")
def limit_buy_order_usdt(limit_buy_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_buy_order_usdt_open)
    order["status"] = "closed"
    order["filled"] = order["amount"]
    order["remaining"] = 0.0
    return order


@pytest.fixture
def limit_sell_order_usdt_open() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_sell_usdt",
        "type": "limit",
        "side": "sell",
        "symbol": "mocked",
        "datetime": pd.Timestamp.now().isoformat(),
        "timestamp": pd.Timestamp.now().timestamp(),
        "price": 2.20,
        "amount": 30.0,
        "cost": 66.0,
        "filled": 0.0,
        "remaining": 30.0,
        "status": "open",
    }


@pytest.fixture
def limit_sell_order_usdt(limit_sell_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_sell_order_usdt_open)
    order["remaining"] = 0.0
    order["filled"] = order["amount"]
    order["status"] = "closed"
    return order


@pytest.fixture(scope="function")
def market_buy_order_usdt() -> Dict[str, Any]:
    return {
        "id": "mocked_market_buy",
        "type": "market",
        "side": "buy",
        "symbol": "mocked",
        "timestamp": pd.Timestamp.now().timestamp(),
        "datetime": pd.Timestamp.now().isoformat(),
        "price": 2.00,
        "amount": 30.0,
        "filled": 30.0,
        "remaining": 0.0,
        "status": "closed",
    }


@pytest.fixture
def market_buy_order_usdt_doublefee(market_buy_order_usdt: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(market_buy_order_usdt)
    order["fee"] = None
    order["fees"] = [
        {"cost": 0.00025125, "currency": "BNB"},
        {"cost": 0.05030681, "currency": "USDT"},
    ]
    order["trades"] = [
        {
            "timestamp": None,
            "datetime": None,
            "symbol": "ETH/USDT",
            "id": None,
            "order": "123",
            "type": "market",
            "side": "sell",
            "takerOrMaker": None,
            "price": 2.01,
            "amount": 25.0,
            "cost": 50.25,
            "fee": {"cost": 0.00025125, "currency": "BNB"},
        },
        {
            "timestamp": None,
            "datetime": None,
            "symbol": "ETH/USDT",
            "id": None,
            "order": "123",
            "type": "market",
            "side": "sell",
            "takerOrMaker": None,
            "price": 2.0,
            "amount": 5,
            "cost": 10,
            "fee": {"cost": 0.0100306, "currency": "USDT"},
        },
    ]
    return order


@pytest.fixture
def market_sell_order_usdt() -> Dict[str, Any]:
    return {
        "id": "mocked_limit_sell",
        "type": "market",
        "side": "sell",
        "symbol": "mocked",
        "timestamp": pd.Timestamp.now().timestamp(),
        "datetime": pd.Timestamp.now().isoformat(),
        "price": 2.20,
        "amount": 30.0,
        "filled": 30.0,
        "remaining": 0.0,
        "status": "closed",
    }


@pytest.fixture(scope="function")
def limit_order(limit_buy_order_usdt: Dict[str, Any], limit_sell_order_usdt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {"buy": limit_buy_order_usdt, "sell": limit_sell_order_usdt}


@pytest.fixture(scope="function")
def limit_order_open(limit_buy_order_usdt_open: Dict[str, Any], limit_sell_order_usdt_open: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {"buy": limit_buy_order_usdt_open, "sell": limit_sell_order_usdt_open}


@pytest.fixture(scope="function")
def mark_ohlcv() -> List[List[Union[int, float]]]:
    return [
        [1630454400000, 2.77, 2.77, 2.73, 2.73, 0],
        [1630458000000, 2.73, 2.76, 2.72, 2.74, 0],
        [1630461600000, 2.74, 2.76, 2.74, 2.76, 0],
        [1630465200000, 2.76, 2.76, 2.74, 2.76, 0],
        [1630468800000, 2.76, 2.77, 2.75, 2.77, 0],
        [1630472400000, 2.77, 2.79, 2.75, 2.78, 0],
        [1630476000000, 2.78, 2.80, 2.77, 2.77, 0],
        [1630479600000, 2.78, 2.79, 2.77, 2.77, 0],
        [1630483200000, 2.77, 2.79, 2.77, 2.78, 0],
        [1630486800000, 2.77, 2.84, 2.77, 2.84, 0],
        [1630490400000, 2.84, 2.85, 2.81, 2.81, 0],
        [1630494000000, 2.81, 2.83, 2.81, 2.81, 0],
        [1630497600000, 2.81, 2.84, 2.81, 2.82, 0],
        [1630501200000, 2.82, 2.83, 2.81, 2.81, 0],
    ]


@pytest.fixture(scope="function")
def funding_rate_history_hourly() -> List[Dict[str, Union[str, float, int]]]:
    return [
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000008,
            "timestamp": 1630454400000,
            "datetime": "2021-09-01T00:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000004,
            "timestamp": 1630458000000,
            "datetime": "2021-09-01T01:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000012,
            "timestamp": 1630461600000,
            "datetime": "2021-09-01T02:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630465200000,
            "datetime": "2021-09-01T03:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000007,
            "timestamp": 1630468800000,
            "datetime": "2021-09-01T04:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000003,
            "timestamp": 1630472400000,
            "datetime": "2021-09-01T05:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000019,
            "timestamp": 1630476000000,
            "datetime": "2021-09-01T06:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000003,
            "timestamp": 1630479600000,
            "datetime": "2021-09-01T07:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630483200000,
            "datetime": "2021-09-01T08:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0,
            "timestamp": 1630486800000,
            "datetime": "2021-09-01T09:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000013,
            "timestamp": 1630490400000,
            "datetime": "2021-09-01T10:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000077,
            "timestamp": 1630494000000,
            "datetime": "2021-09-01T11:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000072,
            "timestamp": 1630497600000,
            "datetime": "2021-09-01T12:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": 0.000097,
            "timestamp": 1630501200000,
            "datetime": "2021-09-01T13:00:00.000Z",
        },
    ]


@pytest.fixture(scope="function")
def funding_rate_history_octohourly() -> List[Dict[str, Union[str, float, int]]]:
    return [
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000008,
            "timestamp": 1630454400000,
            "datetime": "2021-09-01T00:00:00.000Z",
        },
        {
            "symbol": "ADA/USDT:USDT",
            "fundingRate": -0.000003,
            "timestamp": 1630483200000,
            "datetime": "2021-09-01T08:00:00.000Z",
        },
    ]


@pytest.fixture(scope="function")
def leverage_tiers() -> Dict[str, List[Dict[str, Union[int, float]]]]:
    return {
        "1000SHIB/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 150000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 750.0,
            },
            {
                "minNotional": 150000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 4500.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 17000.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 29500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 154500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 654500.0,
            },
        ],
        "1INCH/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 5000,
                "maintenanceMarginRate": 0.012,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 5000,
                "maxNotional": 25000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 65.0,
            },
            {
                "minNotional": 25000,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 690.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 5690.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 2,
                "maintAmt": 11940.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 100000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 386940.0,
            },
        ],
        "AAVE/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 5000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 5000,
                "maxNotional": 25000,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 25,
                "maintAmt": 75.0,
            },
            {
                "minNotional": 25000,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 700.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 5700.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 2,
                "maintAmt": 11950.0,
            },
            {
                "minNotional": 10000000,
                "maxNotional": 50000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 386950.0,
            },
        ],
        "ADA/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0,
            },
            {
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0,
            },
        ],
        "XRP/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0,
            },
            {
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0,
            },
        ],
        "BNB/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 10000,
                "maintenanceMarginRate": 0.0065,
                "maxLeverage": 75,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 10000,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 35.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 25,
                "maintAmt": 535.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 8035.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 58035.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 108035.0,
            },
            {
                "minNotional": 5000000,
                "maxNotional": 10000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 233035.0,
            },
            {
                "minNotional": 10000000,
                "maxNotional": 20000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 1233035.0,
            },
            {
                "minNotional": 20000000,
                "maxNotional": 50000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 6233035.0,
            },
        ],
        "BTC/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.004,
                "maxLeverage": 125,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.005,
                "maxLeverage": 100,
                "maintAmt": 50.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 1300.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 7500000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 16300.0,
            },
            {
                "minNotional": 7500000,
                "maxNotional": 40000000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 203800.0,
            },
            {
                "minNotional": 40000000,
                "maxNotional": 100000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 2203800.0,
            },
            {
                "minNotional": 100000000,
                "maxNotional": 200000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 4703800.0,
            },
            {
                "minNotional": 200000000,
                "maxNotional": 400000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 9703800.0,
            },
            {
                "minNotional": 400000000,
                "maxNotional": 600000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 4.97038e7,
            },
            {
                "minNotional": 600000000,
                "maxNotional": 1000000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1.997038e8,
            },
        ],
        "ZEC/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 150000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 750.0,
            },
            {
                "minNotional": 150000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 4500.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 17000.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 29500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 154500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 654500.0,
            },
        ],
    }

# End of fixtures and functions.
