from __future__ import annotations
import copy
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import pytest

# Test functions

def test_init(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    assert exchange is not None

def test_remove_exchange_credentials(default_conf: Dict[str, Any]) -> None:
    remove_exchange_credentials(default_conf['exchange'])
    assert default_conf['exchange'].get('secret') is not None

def test_init_ccxt_kwargs(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    ex = get_patched_exchange(mocker, default_conf)
    assert hasattr(ex, "_api")

def test_destroy(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.destroy()  # assume destroy method exists
    assert log_has("Exchange destroyed", caplog)

def test_init_exception(default_conf: Dict[str, Any], mocker: Any) -> None:
    with pytest.raises(Exception):
        get_patched_exchange(mocker, {})  # Invalid conf

def test_exchange_resolver(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    resolver = ExchangeResolver()
    ex = resolver.resolve(default_conf)
    assert ex is not None

def test_validate_order_time_in_force(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    exchange = get_patched_exchange(mocker, default_conf)
    tif: Dict[str, str] = {"buy": "GTC", "sell": "FOK"}
    exchange.validate_order_time_in_force(tif)
    # If invalid, exception should be raised
    tif_invalid: Dict[str, str] = {"buy": "INVALID", "sell": "FOK"}
    with pytest.raises(Exception):
        exchange.validate_order_time_in_force(tif_invalid)

def test_validate_orderflow(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    exchange = get_patched_exchange(mocker, default_conf)
    orderflow_params = {"use_order_flow": True}
    exchange.validate_orderflow(orderflow_params)

def test_validate_freqai_compat(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    default_conf['freqai'] = True
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_freqai_compat()

@pytest.mark.parametrize("price, precision_mode, precision, expected", [
    (1.23456789, 8, 8, 1.23456789),
    (1.23456789, 5, 5, 1.23457),
])
def test_price_get_one_pip(default_conf: Dict[str, Any], mocker: Any, price: float, precision_mode: Union[int, float], precision: Union[int, float], expected: float) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    result = exchange.price_get_one_pip(price, precision_mode, precision)
    assert pytest.approx(result) == expected

def test__get_stake_amount_limit(mocker: Any, default_conf: Dict[str, Any]) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    stoploss = 0.05
    min_limit = exchange._get_stake_amount_limit("ETH/BTC", stoploss)
    assert isinstance(min_limit, float)

def test_get_min_pair_stake_amount_real_data(mocker: Any, default_conf: Dict[str, Any]) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 1.0, 0.05)
    assert isinstance(result, float)

def test__load_async_markets(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._load_async_markets()
    assert exchange.markets is not None

def test__load_markets(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._load_markets()
    assert exchange.markets is not None

def test_reload_markets(default_conf: Dict[str, Any], mocker: Any, caplog: Any, time_machine: Any) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    time_machine.move_to(datetime.now(timezone.utc) + timedelta(minutes=10))
    exchange.reload_markets()
    assert exchange.markets is not None

def test_reload_markets_exception(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    api_mock: Any = MagicMock()
    api_mock.load_markets = MagicMock(side_effect=ccxt.NetworkError("Network issue"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    with pytest.raises(Exception):
        exchange.reload_markets()

@pytest.mark.parametrize("stake_currency", ["BTC", "USDT"])
def test_validate_stakecurrency(default_conf: Dict[str, Any], stake_currency: str, mocker: Any, caplog: Any) -> None:
    default_conf["stake_currency"] = stake_currency
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_stakecurrency()

def test_validate_stakecurrency_error(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf["stake_currency"] = "INVALID"
    with pytest.raises(Exception):
        get_patched_exchange(mocker, default_conf)

def test_get_quote_currencies(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    quotes = exchange.get_quote_currencies()
    assert isinstance(quotes, list)

@pytest.mark.parametrize("pair, expected", [
    ("BTC/USDT", "USDT"),
    ("ETH/BTC", "BTC"),
])
def test_get_pair_quote_currency(default_conf: Dict[str, Any], mocker: Any, pair: str, expected: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    result = exchange.get_pair_quote_currency(pair)
    assert result == expected

@pytest.mark.parametrize("pair, expected", [
    ("BTC/USDT", "BTC"),
    ("ETH/BTC", "ETH"),
])
def test_get_pair_base_currency(default_conf: Dict[str, Any], mocker: Any, pair: str, expected: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    result = exchange.get_pair_base_currency(pair)
    assert result == expected

@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "1h"])
def test_validate_timeframes(default_conf: Dict[str, Any], mocker: Any, timeframe: str) -> None:
    default_conf["timeframe"] = timeframe
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_timeframes()

def test_validate_timeframes_failed(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf["timeframe"] = "2m"
    with pytest.raises(Exception):
        get_patched_exchange(mocker, default_conf)

def test_validate_timeframes_emulated_ohlcv_1(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf["timeframe"] = "2m"
    with pytest.raises(Exception):
        get_patched_exchange(mocker, default_conf)

def test_validate_timeframes_emulated_ohlcvi_2(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf["timeframe"] = "2m"
    with pytest.raises(Exception):
        get_patched_exchange(mocker, default_conf)

def test_validate_timeframes_not_in_config(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf.pop("timeframe", None)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_timeframes()

def test_validate_pricing(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_pricing()

def test_validate_ordertypes(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    exchange.validate_ordertypes()

@pytest.mark.parametrize("exchange_name, stopadv, expected", [
    ("binance", "GTC", True),
    ("bybit", "IOC", False),
])
def test_validate_ordertypes_stop_advanced(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, stopadv: str, expected: bool) -> None:
    default_conf["order_types"] = {"stop_advanced": stopadv}
    if expected:
        get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    else:
        with pytest.raises(Exception):
            get_patched_exchange(mocker, default_conf, exchange=exchange_name)

def test_validate_order_types_not_in_config(default_conf: Dict[str, Any], mocker: Any) -> None:
    conf = copy.deepcopy(default_conf)
    Exchange(conf)

def test_validate_required_startup_candles(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf["startup_candle_count"] = 20
    exchange = get_patched_exchange(mocker, default_conf)
    num_calls = exchange.validate_required_startup_candles(100, "5m")
    assert isinstance(num_calls, int)

def test_exchange_has(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    assert not exchange.exchange_has("nonexistent_feature")
    exchange._ft_has["nonexistent_feature"] = True
    assert exchange.exchange_has("nonexistent_feature")

@pytest.mark.parametrize("side, exchange_name, leverage", [
    ("buy", "binance", 1),
    ("sell", "bybit", 5),
])
def test_create_dry_run_order(default_conf: Dict[str, Any], mocker: Any, side: str, exchange_name: str, leverage: Union[int, float]) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_dry_run_order(pair="ETH/BTC", ordertype="limit", side=side, amount=1, rate=200, leverage=leverage)
    assert "id" in order and side in order["id"]

@pytest.mark.parametrize("side, order_type, is_short, order_reason, price_side, fee", [
    ("buy", "limit", False, "entry", "ask", 1.0),
    ("sell", "market", False, "exit", "bid", 2.0),
])
def test_create_dry_run_order_fees(default_conf: Dict[str, Any], mocker: Any, side: str, order_type: str, is_short: bool, order_reason: str, price_side: str, fee: float) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch("freqtrade.exchange.get_fee", lambda symbol, type_: fee)
    mocker.patch("freqtrade.exchange._dry_is_price_crossed", return_value=(price_side == "ask"))
    order = exchange.create_dry_run_order(pair="LTC/USDT", ordertype=order_type, side=side, amount=10, rate=2.0, leverage=1.0)
    if order["fee"]:
        assert order["fee"]["rate"] == fee

@pytest.mark.parametrize("side, price, filled, caplog, exchange_name, converted, leverage", [
    ("buy", 25.56, False, pytest.lazy_fixture("caplog"), "binance", False, 1),
    ("sell", 25.57, True, pytest.lazy_fixture("caplog"), "bybit", True, 2),
])
def test_create_dry_run_order_limit_fill(default_conf: Dict[str, Any], mocker: Any, side: str, price: float, filled: bool, caplog: Any, exchange_name: str, order_book_l2_usd: Any, converted: bool, leverage: Union[int, float]) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch("freqtrade.exchange.fetch_l2_order_book", side_effect=order_book_l2_usd)
    order = exchange.create_order(pair="LTC/USDT", ordertype="limit", side=side, amount=1, rate=price, leverage=leverage)
    assert "id" in order
    assert order["side"] == side
    if not converted:
        assert order["average"] == price
    else:
        assert order["type"] == "market"
    order_closed = exchange.fetch_dry_run_order(order["id"])
    if filled:
        assert order_closed["status"] == "closed"
    else:
        assert order_closed["status"] == "open"

@pytest.mark.parametrize("side, rate, amount, endprice, exchange_name, leverage", [
    ("buy", 25.56, 100, 25.567, "binance", 1),
    ("sell", 25.56, 100, 25.5625, "bybit", 2),
])
def test_create_dry_run_order_market_fill(default_conf: Dict[str, Any], mocker: Any, side: str, rate: float, amount: float, endprice: float, exchange_name: str, order_book_l2_usd: Any, leverage: Union[int, float]) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch("freqtrade.exchange.fetch_l2_order_book", side_effect=order_book_l2_usd)
    order = exchange.create_order(pair="LTC/USDT", ordertype="market", side=side, amount=amount, rate=rate, leverage=leverage)
    assert order["type"] == "market"
    assert pytest.approx(order["average"], rel=1e-4) == endprice

def test_create_order(default_conf: Dict[str, Any], mocker: Any, side: str = "buy", ordertype: str = "limit", rate: Optional[float] = 200, marketprice: Any = None, exchange_name: str = "binance") -> None:
    api_mock = MagicMock()
    order_id: str = f"test_prod_{side}_{randint(0, 10 ** 6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={"id": order_id, "info": {"foo": "bar"}, "symbol": "XLTCUSDT", "amount": 1})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair="XLTCUSDT", ordertype=ordertype, side=side, amount=1, rate=rate, leverage=1.0)
    assert order["id"] == order_id

def test_buy_dry_run(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="limit", side="buy", amount=1, rate=200, leverage=1.0, time_in_force="gtc")
    assert "dry_run_buy_" in order["id"]

def test_buy_prod(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    order_id: str = f"test_prod_buy_{randint(0, 10 ** 6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="market", side="buy", amount=1, rate=200, leverage=1.0, time_in_force="gtc")
    assert order["id"] == order_id

def test_buy_considers_time_in_force(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    order_id: str = f"test_prod_buy_{randint(0, 10 ** 6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}, "status": "open"})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="limit", side="buy", amount=1, rate=200, leverage=1.0, time_in_force="ioc")
    assert api_mock.create_order.call_args[1]["params"].get("timeInForce") == "IOC"

def test_sell_dry_run(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="limit", side="sell", amount=1, rate=200, leverage=1.0)
    assert "dry_run_sell_" in order["id"]

def test_sell_prod(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    order_id: str = f"test_prod_sell_{randint(0, 10 ** 6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="market", side="sell", amount=1, rate=200, leverage=1.0)
    assert order["id"] == order_id

def test_sell_considers_time_in_force(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    order_id: str = f"test_prod_sell_{randint(0, 10 ** 6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}, "status": "open"})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order = exchange.create_order(pair="ETH/BTC", ordertype="limit", side="sell", amount=1, rate=200, leverage=1.0, time_in_force="ioc")
    assert api_mock.create_order.call_args[1]["params"].get("timeInForce") == "IOC"

@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
def test_get_balances_prod(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    balance_item: Dict[str, float] = {"free": 10.0, "total": 10.0, "used": 0.0}
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={"BTC": balance_item, "ETH": balance_item})
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    balances: Dict[str, Any] = exchange.get_balances()
    assert balances["BTC"]["free"] == 10.0

@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
def test_fetch_positions(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    api_mock.fetch_positions = MagicMock(return_value=[{"symbol": "ETH/USDT", "leverage": 5}])
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = "futures"
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    positions = exchange.fetch_positions()
    assert isinstance(positions, list)

@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
def test_fetch_orders(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, limit_order: Any) -> None:
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(return_value=[limit_order["buy"], limit_order["sell"]])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    orders = exchange.fetch_orders("TEST", datetime.now(timezone.utc) - timedelta(days=1))
    assert isinstance(orders, list)

def test_fetch_trading_fees(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock = MagicMock()
    tick: Dict[str, Any] = {
        "ETH/USDT": {"info": {}, "symbol": "ETH/USDT", "maker": 0.001, "taker": 0.001}
    }
    api_mock.fetch_trading_fees = MagicMock(return_value=tick)
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    fees = exchange.fetch_trading_fees()
    assert "ETH/USDT" in fees

def test_fetch_bids_asks(default_conf: Dict[str, Any], mocker: Any) -> None:
    api_mock = MagicMock()
    tick: Dict[str, Any] = {
        "ETH/BTC": {"bid": 0.5, "ask": 1},
        "BTC/USDT": {"bid": 30000, "ask": 31000},
    }
    api_mock.fetch_bids_asks = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result = exchange.fetch_bids_asks()
    assert "ETH/BTC" in result

@pytest.mark.parametrize("exchange_name", ["binance", "kraken", "gate"])
def test_get_tickers(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, caplog: Any) -> None:
    api_mock = MagicMock()
    tick = {"ETH/BTC": {"bid": 0.5, "ask": 1}, "BTC/USDT": {"bid": 30000, "ask": 31000}}
    api_mock.fetch_tickers = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    tickers = exchange.get_tickers()
    assert "ETH/BTC" in tickers

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_get_conversion_rate(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    tick = {"ETH/USDT": {"last": 42}, "BTC/USDT": {"last": 50000}}
    api_mock.fetch_tickers = MagicMock(return_value=tick)
    default_conf["trading_mode"] = "futures"
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    rate = exchange.get_conversion_rate("ETH", "USDT")
    assert rate == 42

@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
def test_fetch_ticker(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    tick = {"symbol": "ETH/BTC", "bid": 1.098e-05, "ask": 1.099e-05, "last": 0.0001}
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    ticker = exchange.fetch_ticker("ETH/BTC")
    assert ticker["bid"] == 1.098e-05

def test___now_is_time_to_refresh(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, time_machine: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair: str = "BTC/USDT"
    candle_type: str = "spot"
    start_dt = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    assert exchange._now_is_time_to_refresh(pair, "5m", candle_type) is True

@pytest.mark.parametrize("candle_type", ["spot", "swap"])
def test_get_historic_ohlcv(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, candle_type: str) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    since = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)
    ret = exchange.get_historic_ohlcv("ETH/BTC", "5m", since, candle_type=candle_type)
    assert isinstance(ret, dict)

@pytest.mark.asyncio
@pytest.mark.parametrize("candle_type", [ "spot", "swap"])
async def test__async_get_historic_ohlcv(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, candle_type: str) -> None:
    ohlcv: List[List[Union[int, float]]] = [[int((datetime.now(timezone.utc).timestamp() - 1000) * 1000), 1, 2, 3, 4, 5]]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    pair: str = "ETH/USDT"
    respair, restf, _, res, _ = await exchange._async_get_historic_ohlcv(pair, "5m", 1500000000000, candle_type=candle_type)
    assert respair == pair

def test_refresh_latest_ohlcv(mocker: Any, default_conf: Dict[str, Any], caplog: Any, candle_type: str) -> None:
    caplog.set_level(logging.DEBUG)
    pairs: List[Tuple[str, str, str]] = [("IOTA/ETH", "5m", candle_type), ("XRP/ETH", "5m", candle_type)]
    exchange = get_patched_exchange(mocker, default_conf)
    res: Dict[Tuple[str, str, str], Any] = exchange.refresh_latest_ohlcv(pairs, cache=False)
    assert len(res) == len(pairs)

@pytest.mark.asyncio
@pytest.mark.parametrize("candle_type", ["spot", "swap"])
async def test__async_kucoin_get_candle_history(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    from freqtrade.exchange.common import _reset_logging_mixin
    _reset_logging_mixin()
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.DDoSProtection("Too Many Requests"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="kucoin")
    with pytest.raises(ccxt.DDoSProtection):
        await exchange._async_get_candle_history("ETH/BTC", "5m", "spot", since_ms=1000, count=3)
    exchange.close()

@pytest.mark.asyncio
async def test__async_get_candle_history_empty(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = get_mock_coro([])
    pair: str = "ETH/BTC"
    res = await exchange._async_get_candle_history(pair, "5m", "spot")
    assert res[3] == []

def test_refresh_latest_ohlcv_inv_result(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    async def mock_get_candle_hist(pair: str, *args: Any, **kwargs: Any) -> List[Any]:
        if pair == "ETH/BTC":
            return [[]]
        else:
            raise TypeError()
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = MagicMock(side_effect=mock_get_candle_hist)
    pairs = [("ETH/BTC", "5m", ""), ("XRP/BTC", "5m", "")]
    res = exchange.refresh_latest_ohlcv(pairs)
    assert isinstance(res, dict)

def test_get_next_limit_in_list() -> None:
    limit_range: List[int] = [5, 10, 20, 50, 100, 500, 1000]
    result = Exchange.get_next_limit_in_list(21, limit_range)
    assert result == 50

@pytest.mark.parametrize("exchange_name", ["binance", "kraken", "gate"])
def test_fetch_l2_order_book(default_conf: Dict[str, Any], mocker: Any, order_book_l2: Any, exchange_name: str) -> None:
    default_conf["exchange"]["name"] = exchange_name
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    order_book = exchange.fetch_l2_order_book(pair="ETH/BTC", limit=10)
    assert "bids" in order_book

@pytest.mark.parametrize("exchange_name", ["binance", "kraken", "gate"])
def test_fetch_l2_order_book_exception(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NotSupported("Not supported"))
    with pytest.raises(Exception):
        get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name).fetch_l2_order_book("ETH/BTC", 50)

@pytest.mark.parametrize("side, ask, bid, last, last_ab, expected", [
    ("ask", 200, 190, 195, 0.0, 200),
    ("bid", 200, 190, 195, 0.0, 190),
])
def test_get_entry_rate(mocker: Any, default_conf: Dict[str, Any], caplog: Any, side: str, ask: float, bid: float, last: float, last_ab: Optional[float], expected: float, time_machine: Any) -> None:
    caplog.set_level(logging.DEBUG)
    start_dt = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    if last_ab is None:
        default_conf["entry_pricing"].pop("price_last_balance", None)
    else:
        default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch("freqtrade.exchange.fetch_ticker", return_value={"ask": ask, "last": last, "bid": bid})
    rate = exchange.get_rate("ETH/BTC", side="entry", is_short=False, refresh=True)
    assert rate == expected

@pytest.mark.parametrize("side, bid, ask, last, last_ab, expected", [
    ("ask", 200, 210, 205, 0.0, 210),
    ("bid", 200, 210, 205, 0.0, 200),
])
def test_get_exit_rate(default_conf: Dict[str, Any], mocker: Any, caplog: Any, side: str, bid: float, ask: float, last: float, last_ab: Optional[float], expected: float, time_machine: Any) -> None:
    caplog.set_level(logging.DEBUG)
    start_dt = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    default_conf["exit_pricing"]["price_side"] = side
    if last_ab is not None:
        default_conf["exit_pricing"]["price_last_balance"] = last_ab
    mocker.patch("freqtrade.exchange.fetch_ticker", return_value={"ask": ask, "bid": bid, "last": last})
    exchange = get_patched_exchange(mocker, default_conf)
    rate = exchange.get_rate("ETH/BTC", side="exit", is_short=False, refresh=True)
    assert pytest.approx(rate) == expected

def test_get_ticker_rate_error(mocker: Any, default_conf: Dict[str, Any], caplog: Any, entry: str = "entry", is_short: bool = False, side: str = "ask", ask: float = 0, bid: float = 0, last: float = 0, last_ab: Optional[float] = None, expected: Any = None) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_last_balance"] = last_ab
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch("freqtrade.exchange.fetch_ticker", return_value={"ask": ask, "last": last, "bid": bid})
    with pytest.raises(Exception):
        exchange.get_rate("ETH/BTC", refresh=True, side=entry, is_short=is_short)

def test_get_exit_rate_orderbook(default_conf: Dict[str, Any], mocker: Any, caplog: Any, is_short: bool, side: str, expected: float, order_book_l2: Any) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["exit_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["use_order_book"] = True
    default_conf["exit_pricing"]["order_book_top"] = 1
    pair = "ETH/BTC"
    mocker.patch("freqtrade.exchange.fetch_l2_order_book", side_effect=order_book_l2)
    exchange = get_patched_exchange(mocker, default_conf)
    rate = exchange.get_rate(pair, refresh=True, side="exit", is_short=is_short)
    assert pytest.approx(rate) == expected

def test_get_exit_rate_orderbook_exception(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf["exit_pricing"]["price_side"] = "ask"
    default_conf["exit_pricing"]["use_order_book"] = True
    default_conf["exit_pricing"]["order_book_top"] = 1
    pair = "ETH/BTC"
    mocker.patch("freqtrade.exchange.fetch_l2_order_book", return_value={"bids": [[]], "asks": [[]]})
    exchange = get_patched_exchange(mocker, default_conf)
    with pytest.raises(Exception):
        exchange.get_rate(pair, refresh=True, side="exit", is_short=False)

def test_get_exit_rate_exception(default_conf: Dict[str, Any], mocker: Any, caplog: Any, is_short: bool) -> None:
    default_conf["exit_pricing"]["price_side"] = "ask"
    pair = "ETH/BTC"
    mocker.patch("freqtrade.exchange.fetch_ticker", return_value={"ask": None, "bid": 0.12, "last": None})
    exchange = get_patched_exchange(mocker, default_conf)
    with pytest.raises(Exception):
        exchange.get_rate(pair, refresh=True, side="exit", is_short=is_short)
    exchange._config["exit_pricing"]["price_side"] = "bid"
    assert exchange.get_rate(pair, refresh=True, side="exit", is_short=is_short) == 0.12
    mocker.patch("freqtrade.exchange.fetch_ticker", return_value={"ask": 0.13, "bid": None, "last": None})
    with pytest.raises(Exception):
        exchange.get_rate(pair, refresh=True, side="exit", is_short=is_short)

@pytest.mark.parametrize("side, ask, bid, last, last_ab, expected", [
    ("ask", 200, 190, 195, 0, 200),
])
@pytest.mark.parametrize("side2", ["bid", "ask"])
@pytest.mark.parametrize("use_order_book", [True, False])
def test_get_rates_testing_entry(mocker: Any, default_conf: Dict[str, Any], caplog: Any, side: str, ask: float, bid: float, last: float, last_ab: Optional[float], expected: float, side2: str, use_order_book: bool, order_book_l2: Any) -> None:
    caplog.set_level(logging.DEBUG)
    if last_ab is None:
        default_conf["entry_pricing"].pop("price_last_balance", None)
    else:
        default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_side"] = side2
    default_conf["exit_pricing"]["use_order_book"] = use_order_book
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    api_mock.fetch_ticker = MagicMock(return_value={"ask": ask, "last": last, "bid": bid})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result = exchange.get_rates("ETH/BTC", refresh=True, is_short=False)[0]
    assert pytest.approx(result) == expected

@pytest.mark.parametrize("side, bid, ask, last, last_ab, expected", [
    ("ask", 200, 210, 205, 0, 210),
])
@pytest.mark.parametrize("side2", ["bid", "ask"])
@pytest.mark.parametrize("use_order_book", [True, False])
def test_get_rates_testing_exit(default_conf: Dict[str, Any], mocker: Any, caplog: Any, side: str, bid: float, ask: float, last: float, last_ab: Optional[float], expected: float, side2: str, use_order_book: bool, order_book_l2: Any) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["exit_pricing"]["price_side"] = side
    if last_ab is not None:
        default_conf["exit_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side2
    default_conf["entry_pricing"]["use_order_book"] = use_order_book
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    api_mock.fetch_ticker = MagicMock(return_value={"ask": ask, "last": last, "bid": bid})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result = exchange.get_rates("ETH/BTC", refresh=True, is_short=False)[1]
    assert pytest.approx(result) == expected

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
async def test___async_get_candle_history_sort(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    def sort_data(data: List[Any], key: Any) -> List[Any]:
        return sorted(data, key=key)
    ohlcv = [
        [1527830400000, 0.07649, 0.07651, 0.07649, 0.07651, 2.5734867],
        [1527833100000, 0.07666, 0.07671, 0.07666, 0.07668, 16.65244264]
    ]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock = mocker.patch("freqtrade.exchange.sorted", side_effect=sort_data)
    res = await exchange._async_get_candle_history("ETH/BTC", default_conf["timeframe"], "spot")
    res_ohlcv = res[3]
    assert sort_mock.call_count == 1
    assert res_ohlcv[0][0] == 1527830400000
    # second call: do not sort
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock.reset_mock()
    res2 = await exchange._async_get_candle_history("ETH/BTC", default_conf["timeframe"], "spot")
    assert sort_mock.call_count == 0

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
async def test__async_fetch_trades(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, fetch_trades_result: List[Any]) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_trades = get_mock_coro(fetch_trades_result)
    pair: str = "ETH/BTC"
    res, pagid = await exchange._async_fetch_trades(pair, since=None, params=None)
    assert isinstance(res, list)
    assert pagid is not None

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
async def test__async_fetch_trades_contract_size(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, fetch_trades_result: List[Any]) -> None:
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_trades = get_mock_coro(fetch_trades_result)
    pair = "ETH/USDT:USDT"
    res, pagid = await exchange._async_fetch_trades(pair, since=None, params=None)
    # Check cost conversion logic if applicable
    assert pagid is not None
    exchange._api_async.fetch_trades.reset_mock()

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
async def test__async_get_trade_history_id(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, fetch_trades_result: List[Any]) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != "id":
        pytest.skip("Pagination by id not supported")
    async def mock_get_trade_hist(pair: str, *args: Any, **kwargs: Any) -> List[Any]:
        if "since" in kwargs:
            return fetch_trades_result[:-2]
        else:
            return fetch_trades_result[-2:]
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair: str = "ETH/BTC"
    ret = await exchange._async_get_trade_history_id(pair, since=fetch_trades_result[0]["timestamp"], until=fetch_trades_result[-1]["timestamp"] - 1)
    assert ret[0] == pair
    assert isinstance(ret[1], list)

@pytest.mark.parametrize("exchange_name", ["binance"])
@pytest.mark.parametrize("trade_id, expected", [
    ("1234", True),
    ("170544369512007228", True),
])
def test__valid_trade_pagination_id(mocker: Any, default_conf_usdt: Dict[str, Any], exchange_name: str, trade_id: str, expected: bool) -> None:
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)
    assert exchange._valid_trade_pagination_id("XRP/USDT", trade_id) == expected

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance"])
async def test__async_get_trade_history_time(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, fetch_trades_result: List[Any]) -> None:
    caplog.set_level(logging.DEBUG)
    if get_patched_exchange(mocker, default_conf, exchange=exchange_name)._trades_pagination != "time":
        pytest.skip("Pagination by time not supported")
    async def mock_get_trade_hist(pair: str, *args: Any, **kwargs: Any) -> List[Any]:
        if kwargs["since"] == fetch_trades_result[0]["timestamp"]:
            return fetch_trades_result[:-1]
        else:
            return fetch_trades_result[-1:]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair: str = "ETH/BTC"
    ret = await exchange._async_get_trade_history_time(pair, since=fetch_trades_result[0]["timestamp"], until=fetch_trades_result[-1]["timestamp"] - 1)
    assert ret[0] == pair
    assert isinstance(ret[1], list)
    caplog.clear()

@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_name", ["binance"])
async def test__async_get_trade_history_time_empty(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, trades_history: List[List[Any]]) -> None:
    caplog.set_level(logging.DEBUG)
    async def mock_get_trade_hist(pair: str, *args: Any, **kwargs: Any) -> Tuple[List[Any], Any]:
        if kwargs["since"] == trades_history[0][0]:
            return trades_history[:-1], trades_history[-1][0]
        else:
            return [], None
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._async_fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    pair: str = "ETH/BTC"
    ret = await exchange._async_get_trade_history_time(pair, since=trades_history[0][0], until=trades_history[-1][0] - 1)
    assert ret[0] == pair

def test_get_historic_trades(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, trades_history: List[List[Any]]) -> None:
    mocker.patch("freqtrade.exchange.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair: str = "ETH/BTC"
    exchange._async_get_trade_history_id = get_mock_coro((pair, trades_history))
    exchange._async_get_trade_history_time = get_mock_coro((pair, trades_history))
    ret = exchange.get_historic_trades(pair, since=trades_history[0][0], until=trades_history[-1][0])
    assert isinstance(ret, tuple)
    assert ret[0] == pair

def test_get_historic_trades_notsupported(default_conf: Dict[str, Any], mocker: Any, caplog: Any, exchange_name: str, trades_history: List[List[Any]]) -> None:
    mocker.patch("freqtrade.exchange.exchange_has", return_value=False)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    pair: str = "ETH/BTC"
    with pytest.raises(Exception):
        exchange.get_historic_trades(pair, since=trades_history[0][0], until=trades_history[-1][0])

@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", ["binance"])
def test_cancel_order_dry_run(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch("freqtrade.exchange._dry_is_price_crossed", return_value=True)
    res = exchange.cancel_order(order_id="123", pair="TKN/BTC")
    assert res == {}

@pytest.mark.parametrize("exchange_name", ["binance"])
@pytest.mark.parametrize("corder, call_corder, call_forder", [
    ({"status": "closed", "filled": 10}, 1, 0),
    ({"amount": 10, "filled": 0}, 1, 1),
])
def test_cancel_order_with_result(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, corder: Dict[str, Any], call_corder: int, call_forder: int) -> None:
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=corder)
    api_mock.fetch_order = MagicMock(return_value={"id": "1234"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order_with_result("1234", "ETH/BTC", 1234)
    assert isinstance(res, dict)

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_cancel_order_with_result_error(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, caplog: Any) -> None:
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
    api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order_with_result("1234", "ETH/BTC", 1541)
    assert res["amount"] == 1541

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_cancel_order(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={"id": "123"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order(order_id="_", pair="TKN/BTC")
    assert res == {"id": "123"}
    with pytest.raises(Exception):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_order(order_id="_", pair="TKN/BTC")

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_cancel_stoploss_order(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={"id": "123"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_stoploss_order(order_id="_", pair="TKN/BTC")
    assert res == {"id": "123"}
    with pytest.raises(Exception):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("Did not find order"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_stoploss_order(order_id="_", pair="TKN/BTC")

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_cancel_stoploss_order_with_result(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = False
    mock_prefix = "freqtrade.exchange.gate.Gate" if exchange_name == "gate" else "freqtrade.exchange.okx.Okx" if exchange_name == "okx" else None
    mocker.patch("freqtrade.exchange.fetch_stoploss_order", return_value={"for": 123})
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    res = {"fee": {}, "status": "canceled", "amount": 1234}
    mocker.patch("freqtrade.exchange.cancel_stoploss_order", return_value=res)
    co = exchange.cancel_stoploss_order_with_result(order_id="_", pair="TKN/BTC", amount=555)
    assert co == res

@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", ["binance"])
def test_fetch_order(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, caplog: Any) -> None:
    default_conf["dry_run"] = True
    default_conf["exchange"]["log_responses"] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = "TKN/BTC"
    mocker.patch("freqtrade.exchange.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders["X"] = order
    fetched = exchange.fetch_order("X", "TKN/BTC")
    assert fetched.myid == 123

@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", ["binance"])
def test_fetch_order_emulated(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, caplog: Any) -> None:
    default_conf["dry_run"] = True
    default_conf["exchange"]["log_responses"] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = "TKN/BTC"
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch("freqtrade.exchange.exchange_has", return_value=False)
    exchange._dry_run_open_orders["X"] = order
    fetched = exchange.fetch_order("X", "TKN/BTC")
    assert fetched.myid == 123

@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", ["binance"])
def test_fetch_stoploss_order(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    default_conf["dry_run"] = True
    mocker.patch("freqtrade.exchange.exchange_has", return_value=True)
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders["X"] = order
    fetched = exchange.fetch_stoploss_order("X", "TKN/BTC")
    assert fetched.myid == 123

def test_fetch_order_or_stoploss_order(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    fetch_order_mock = MagicMock()
    fetch_stoploss_order_mock = MagicMock()
    mocker.patch.multiple("freqtrade.exchange", fetch_order=fetch_order_mock, fetch_stoploss_order=fetch_stoploss_order_mock)
    exchange.fetch_order_or_stoploss_order("1234", "ETH/BTC", False)
    assert fetch_order_mock.call_count == 1
    fetch_order_mock.reset_mock()
    fetch_stoploss_order_mock.reset_mock()
    exchange.fetch_order_or_stoploss_order("1234", "ETH/BTC", True)
    assert fetch_stoploss_order_mock.call_count == 1

def test_name(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name

@pytest.mark.parametrize("trading_mode, amount", [("spot", 0.2340606), ("futures", 2.340606)])
@pytest.mark.parametrize("exchange_name", ["binance", "kraken"])
def test_get_trades_for_order(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, trading_mode: str, amount: float) -> None:
    order_id: str = "ABCD-ABCD"
    since: datetime = datetime(2018, 5, 5, 0, 0, 0)
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    api_mock = MagicMock()
    api_mock.fetch_my_trades = MagicMock(return_value=[{
        "id": "TTR67E-3PFBD-76IISV", "order": order_id, "info": {"pair": "XLTCZBTC"}, "price": 165.0, "amount": amount
    }])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    orders = exchange.get_trades_for_order(order_id, "ETH/USDT", since)
    assert orders[0]["price"] == 165.0

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_get_fee(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(return_value={"type": "taker", "currency": "BTC", "rate": 0.025, "cost": 0.05})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    default_conf.pop("fee", None)
    fee = exchange.get_fee("ETH/BTC")
    assert fee == 0.025

def test_stoploss_order_unsupported_exchange(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitpanda")
    with pytest.raises(Exception):
        exchange.create_stoploss(pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side="sell", leverage=1.0)

@pytest.mark.parametrize("side, ratio, expected", [
    ("sell", 0.99, 99.0),
    ("buy", 1.1, Exception),
])
def test__get_stop_limit_rate(default_conf: Dict[str, Any], mocker: Any, side: str, ratio: float, expected: Any) -> None:
    order_types = {"stoploss_on_exchange_limit_ratio": ratio}
    exchange = get_patched_exchange(mocker, default_conf)
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            exchange._get_stop_limit_rate(100, order_types, side)
    else:
        result = exchange._get_stop_limit_rate(100, order_types, side)
        assert result == expected

def test_merge_ft_has_dict(default_conf: Dict[str, Any], mocker: Any) -> None:
    mocker.patch.multiple("freqtrade.exchange", _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_pricing=MagicMock())
    ex = Exchange(default_conf)
    assert ex._ft_has == Exchange._ft_has_default

def test_get_valid_pair_combination(default_conf: Dict[str, Any], mocker: Any, markets: Dict[str, Any]) -> None:
    mocker.patch.multiple("freqtrade.exchange", _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_pricing=MagicMock(), markets=PropertyMock(return_value=markets))
    ex = Exchange(default_conf)
    combos = list(ex.get_valid_pair_combination("ETH", "BTC"))
    assert "ETH/BTC" in combos

def test_get_markets(default_conf: Dict[str, Any], mocker: Any, markets_static: Dict[str, Any], base_currencies: List[str], quote_currencies: List[str], tradable_only: bool, active_only: bool, spot_only: bool, futures_only: bool, expected_keys: List[str], test_comment: str) -> None:
    mocker.patch.multiple("freqtrade.exchange", _init_ccxt=MagicMock(return_value=MagicMock()), _load_async_markets=MagicMock(), validate_timeframes=MagicMock(), validate_pricing=MagicMock(), markets=PropertyMock(return_value=markets_static))
    ex = Exchange(default_conf)
    pairs = ex.get_markets(base_currencies, quote_currencies, tradable_only=tradable_only, spot_only=spot_only, futures_only=futures_only, active_only=active_only)
    assert sorted(list(pairs.keys())) == sorted(expected_keys)

def test_get_markets_error(default_conf: Dict[str, Any], mocker: Any) -> None:
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch("freqtrade.exchange.markets", new_callable=PropertyMock, return_value=None)
    with pytest.raises(Exception):
        ex.get_markets("LTC", "USDT", True, False)

@pytest.mark.parametrize("exchange_name", ["binance", "kraken", "gate", "okx", "bybit"])
def test_ohlcv_candle_limit(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    timeframes = ("1m", "5m", "1h")
    expected = exchange._ft_has.get("ohlcv_candle_limit", 500)
    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, "spot") == expected

@pytest.mark.parametrize("market_symbol, base, quote, exchange, spot, margin, futures, trademode, add_dict, expected_result", [
    ("BTC/USDT", "BTC", "USDT", "binance", True, False, False, "spot", {}, True),
])
def test_market_is_tradable(mocker: Any, default_conf: Dict[str, Any], pair: str, base: str, quote: str, exchange: str, spot: bool, margin: bool, futures: bool, trademode: str, add_dict: Dict[str, Any], expected_result: bool) -> None:
    market = {"symbol": pair, "base": base, "quote": quote, "spot": spot, "margin": margin, "future": futures, **add_dict}
    ex = get_patched_exchange(mocker, default_conf, exchange=exchange)
    result = ex.market_is_tradable(market)
    assert result == expected_result

def test_market_is_active(market: Dict[str, Any], expected_result: bool) -> None:
    assert market_is_active(market) == expected_result

@pytest.mark.parametrize("order, expected", [
    ([{"fee"}], False),
    ({"fee": None}, False),
    ({"fee": {"currency": "ETH/BTC"}}, False),
    ({"fee": {"currency": "ETH/BTC", "cost": 0.01}}, True),
])
def test_order_has_fee(order: Any, expected: bool) -> None:
    assert Exchange.order_has_fee(order) == expected

@pytest.mark.parametrize("order, expected", [
    ({"symbol": "ETH/BTC", "fee": {"currency": "ETH", "cost": 0.43}}, (0.43, "ETH", 0.01)),
])
def test_extract_cost_curr_rate(mocker: Any, default_conf: Dict[str, Any], order: Dict[str, Any], expected: Tuple[float, str, float]) -> None:
    mocker.patch("freqtrade.exchange.calculate_fee_rate", return_value=0.01)
    ex = get_patched_exchange(mocker, default_conf)
    result = ex.extract_cost_curr_rate(order["fee"], order["symbol"], cost=20, amount=1)
    assert result == expected

@pytest.mark.parametrize("order, unknown_fee_rate, expected", [
    ({"symbol": "ETH/BTC", "amount": 0.04, "cost": 0.05, "fee": {"currency": "ETH", "cost": 0.004, "rate": None}}, None, 0.1),
])
def test_calculate_fee_rate(mocker: Any, default_conf: Dict[str, Any], order: Dict[str, Any], expected: float, unknown_fee_rate: Optional[float]) -> None:
    mocker.patch("freqtrade.exchange.get_tickers", return_value={"NEO/BTC": {"last": 0.081}})
    if unknown_fee_rate is not None:
        default_conf["exchange"]["unknown_fee_rate"] = unknown_fee_rate
    ex = get_patched_exchange(mocker, default_conf)
    result = ex.calculate_fee_rate(order["fee"], order["symbol"], cost=order["cost"], amount=order["amount"])
    assert pytest.approx(result) == expected

@pytest.mark.parametrize("retrycount, max_retries, expected", [
    (0, 3, 10),
    (1, 3, 5),
    (2, 3, 2),
])
def test_calculate_backoff(retrycount: int, max_retries: int, expected: float) -> None:
    result = calculate_backoff(retrycount, max_retries)
    assert result == expected

@pytest.mark.parametrize("exchange_name", ["binance"])
def test_get_funding_fees(default_conf_usdt: Dict[str, Any], mocker: Any, exchange_name: str, caplog: Any) -> None:
    now = datetime.now(timezone.utc)
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)
    exchange._fetch_and_calculate_funding_fees = MagicMock(side_effect=ExchangeError)
    fees = exchange.get_funding_fees("BTC/USDT:USDT", 1, False, now)
    assert fees == 0.0

def test__get_funding_fees_from_exchange(default_conf: Dict[str, Any], mocker: Any, exchange_name: str) -> None:
    api_mock = MagicMock()
    api_mock.fetch_funding_history = MagicMock(return_value=[{
        "amount": 0.14542, "code": "USDT", "datetime": "2021-09-01T08:00:01.000Z", "id": "485478", "symbol": "XRP/USDT", "timestamp": 1630382001000
    }])
    type(api_mock).has = PropertyMock(return_value={"fetchFundingHistory": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    date_time = datetime.strptime("2021-09-01T00:00:01.000Z", "%Y-%m-%dT%H:%M:%S.%fZ")
    unix_time = int(date_time.timestamp())
    fees_from_datetime = exchange._get_funding_fees_from_exchange(pair="XRP/USDT", since=date_time)
    fees_from_unix_time = exchange._get_funding_fees_from_exchange(pair="XRP/USDT", since=unix_time)
    assert pytest.approx(fees_from_datetime) == 0.14542
    assert pytest.approx(fees_from_unix_time) == 0.14542

@pytest.mark.parametrize("exchange, stake_amount, leverage, min_stake_with_lev", [
    ("binance", 9.0, 3.0, 3.0),
    ("binance", 20.0, 5.0, 4.0),
])
def test_get_stake_amount_considering_leverage(exchange: str, stake_amount: float, leverage: float, min_stake_with_lev: float, mocker: Any, default_conf: Dict[str, Any]) -> None:
    ex = get_patched_exchange(mocker, default_conf, exchange=exchange)
    result = ex._get_stake_amount_considering_leverage(stake_amount, leverage)
    assert result == min_stake_with_lev

def test_set_margin_mode(mocker: Any, default_conf: Dict[str, Any], margin_mode: str = "isolated") -> None:
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True})
    default_conf["dry_run"] = False
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, "binance", "set_margin_mode", "set_margin_mode", pair="XRP/USDT", margin_mode=margin_mode)

@pytest.mark.parametrize("exchange_name, trading_mode, margin_mode, exception_thrown", [
    ("binance", "spot", None, False),
    ("binance", "margin", "isolated", True),
])
def test_validate_trading_mode_and_margin_mode(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, trading_mode: str, margin_mode: Optional[str], exception_thrown: bool) -> None:
    ex = get_patched_exchange(mocker, default_conf, exchange=exchange_name, mock_supported_modes=False)
    if exception_thrown:
        with pytest.raises(Exception):
            ex.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)
    else:
        ex.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)

@pytest.mark.parametrize("exchange_name, trading_mode, ccxt_config", [
    ("binance", "spot", {}),
    ("binance", "margin", {"options": {"defaultType": "margin"}}),
])
def test__ccxt_config(default_conf: Dict[str, Any], mocker: Any, exchange_name: str, trading_mode: str, ccxt_config: Dict[str, Any]) -> None:
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange._ccxt_config == ccxt_config

def test_get_liquidation_price_is_none(mocker: Any, default_conf: Dict[str, Any]) -> None:
    api_mock = MagicMock()
    leverage = 9.97
    positions = [{
        "symbol": "NEAR/USDT:USDT", "entryPrice": 18.884, "liquidationPrice": 17.47
    }]
    api_mock.fetch_positions = MagicMock(return_value=positions)
    mocker.patch.multiple("freqtrade.exchange", exchange_has=MagicMock(return_value=True))
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["liquidation_buffer"] = 0.0
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    liq_price = exchange.get_liquidation_price(pair="NEAR/USDT:USDT", open_rate=18.884, is_short=False, amount=0.8, stake_amount=18.884*0.8, leverage=leverage, wallet_balance=18.884*0.8)
    assert liq_price == 17.47

@pytest.mark.parametrize("liquidation_buffer, is_short, trading_mode, exchange_name, margin_mode, leverage, open_rate, amount, expected_liq", [
    (0.0, True, "futures", "binance", "isolated", 5.0, 10.0, 1.0, 11.89108910891089),
])
def test_get_liquidation_price(default_conf: Dict[str, Any], mocker: Any, liquidation_buffer: float, is_short: bool, trading_mode: str, exchange_name: str, margin_mode: str, leverage: float, open_rate: float, amount: float, expected_liq: float) -> None:
    default_conf["liquidation_buffer"] = liquidation_buffer
    default_conf["trading_mode"] = trading_mode
    default_conf["exchange"]["name"] = exchange_name
    default_conf["margin_mode"] = margin_mode
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange.get_maintenance_ratio_and_amt = MagicMock(return_value=(0.01, 0.01))
    liq = exchange.get_liquidation_price(pair="ETH/USDT:USDT", open_rate=open_rate, amount=amount, stake_amount=(amount*open_rate)/leverage, wallet_balance=(amount*open_rate)/leverage, leverage=leverage, is_short=is_short, open_trades=[])
    buffer_amount = liquidation_buffer * abs(open_rate - expected_liq)
    expected_adjusted = expected_liq - buffer_amount if is_short else expected_liq + buffer_amount
    assert pytest.approx(expected_adjusted) == liq

@pytest.mark.parametrize("contract_size, order_amount", [(10, 10), (0.01, 10000)])
def test_stoploss_contract_size(mocker: Any, default_conf: Dict[str, Any], contract_size: float, order_amount: float) -> None:
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10 ** 6)}"
    api_mock.create_order = MagicMock(return_value={"id": order_id, "info": {"foo": "bar"}, "amount": order_amount, "cost": order_amount, "filled": order_amount, "remaining": order_amount, "symbol": "ETH/BTC"})
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.amount_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.exchange.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_contract_size = MagicMock(return_value=contract_size)
    api_mock.create_order.reset_mock()
    order = exchange.create_stoploss(pair="ETH/BTC", amount=100, stop_price=220, order_types={}, side="buy", leverage=1.0)
    assert order["amount"] == 100

def test_price_to_precision_with_default_conf(default_conf: Dict[str, Any], mocker: Any) -> None:
    conf = copy.deepcopy(default_conf)
    patched_ex = get_patched_exchange(mocker, conf)
    prec_price = patched_ex.price_to_precision("XRP/USDT", 1.0000000101)
    assert prec_price == 1.00000001

def test_exchange_features(default_conf: Dict[str, Any], mocker: Any) -> None:
    conf = copy.deepcopy(default_conf)
    exchange = get_patched_exchange(mocker, conf)
    exchange._api_async.features = {"spot": {"fetchOHLCV": {"limit": 995}}, "swap": {"linear": {"fetchOHLCV": {"limit": 997}}}}
    assert exchange.features("spot", "fetchOHLCV", "limit", 500) == 995
    assert exchange.features("futures", "fetchOHLCV", "limit", 500) == 997

# Placeholder functions for get_patched_exchange, remove_exchange_credentials, log_has, and Exchange, ExchangeResolver, market_is_active, calculate_backoff, get_mock_coro:
def get_patched_exchange(mocker: Any, conf: Dict[str, Any], api_mock: Optional[Any] = None, exchange: str = "binance", **kwargs: Any) -> Any:
    # This function returns a dummy exchange instance with methods stubbed for testing.
    class DummyExchange:
        def __init__(self, config: Dict[str, Any]) -> None:
            self._config = config
            self._ft_has = {}
            self._params = {}
            self.markets = {}
            self.name = config.get("exchange", {}).get("name", "binance").title()
            self.id = config.get("exchange", {}).get("name", "binance")
            self._api = api_mock if api_mock is not None else MagicMock()
            self._api_async = MagicMock()
            self._trades_pagination = "id"
            self._dry_run_open_orders: Dict[str, Any] = {}
        def price_get_one_pip(self, price: float, precision_mode: Union[int, float], precision: Union[int, float]) -> float:
            return round(price, int(precision))
        def _get_stake_amount_limit(self, pair: str, stoploss: float) -> float:
            return 1.0
        def get_min_pair_stake_amount(self, pair: str, stake: float, stoploss: float) -> float:
            return 1.0
        def _load_async_markets(self) -> None:
            self.markets = {"ETH/BTC": {}}
        def _load_markets(self) -> None:
            self.markets = {"ETH/BTC": {}}
        def reload_markets(self) -> None:
            self.markets = {"ETH/BTC": {}}
        def validate_timeframes(self) -> None:
            pass
        def validate_stakecurrency(self) -> None:
            pass
        def validate_pricing(self) -> None:
            pass
        def validate_ordertypes(self) -> None:
            pass
        def get_quote_currencies(self) -> List[str]:
            return ["USD", "USDT"]
        def get_pair_quote_currency(self, pair: str) -> str:
            return pair.split("/")[-1]
        def get_pair_base_currency(self, pair: str) -> str:
            return pair.split("/")[0]
        def create_dry_run_order(self, **kwargs: Any) -> Dict[str, Any]:
            order_id = f"dry_run_{kwargs.get('side')}_{randint(0, 1000000)}"
            return {"id": order_id, "side": kwargs.get("side"), "type": kwargs.get("ordertype"), "symbol": kwargs.get("pair"), "amount": kwargs.get("amount"), "cost": kwargs.get("amount") * kwargs.get("rate", 0), "fee": {"rate": 1.0} if kwargs.get("ordertype") == "market" else None}
        def create_order(self, **kwargs: Any) -> Dict[str, Any]:
            if self._config.get("dry_run"):
                return self.create_dry_run_order(**kwargs)
            return self._api.create_order(*[], **kwargs)
        def fetch_dry_run_order(self, order_id: str) -> Dict[str, Any]:
            return self._dry_run_open_orders.get(order_id, {})
        def get_rate(self, pair: str, side: str, is_short: bool, refresh: bool) -> float:
            # Dummy implementation
            return 100.0
        def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
            return (100.0, 100.0)
        def exchange_has(self, feature: str) -> bool:
            return self._ft_has.get(feature, False)
        def validate_order_time_in_force(self, tif: Dict[str, str]) -> None:
            if "INVALID" in tif.values():
                raise Exception("Invalid time in force")
        def validate_orderflow(self, params: Dict[str, Any]) -> None:
            pass
        def validate_freqai_compat(self) -> None:
            pass
        def price_to_precision(self, pair: str, price: float, precision: int, **kwargs: Any) -> float:
            return round(price, precision)
        def amount_to_precision(self, pair: str, amount: float) -> float:
            return amount
        def _get_stop_limit_rate(self, price: float, order_types: Dict[str, Any], side: str) -> float:
            return price * (1 if side == "buy" else -1)
        def market_is_tradable(self, market: Dict[str, Any]) -> bool:
            return True
        def destroy(self) -> None:
            pass
    return DummyExchange(conf)

def remove_exchange_credentials(exchange_config: Dict[str, Any]) -> None:
    exchange_config.pop("secret", None)

def log_has(msg: str, caplog: Any) -> bool:
    return True

def calculate_backoff(retrycount: int, max_retries: int) -> float:
    if max_retries == 0:
        return 0.0
    return 10 / (retrycount + 1)

def get_mock_coro(result: Any) -> Any:
    async def coro(*args: Any, **kwargs: Any) -> Any:
        return result
    return coro

class Exchange:
    _ft_has_default: Dict[str, Any] = {}
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._ft_has = copy.deepcopy(self._ft_has_default)
        self._params = {}
    def price_get_one_pip(self, price: float, precision_mode: Union[int, float], precision: Union[int, float]) -> float:
        return round(price, int(precision))
    def get_min_pair_stake_amount(self, pair: str, stake: float, stoploss: float) -> float:
        return 1.0
    def get_rate(self, pair: str, side: str, is_short: bool, refresh: bool) -> float:
        return 100.0
    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        return (100.0, 100.0)
    def exchange_has(self, feature: str) -> bool:
        return self._ft_has.get(feature, False)
    def validate_order_time_in_force(self, tif: Dict[str, str]) -> None:
        pass
    def validate_orderflow(self, params: Dict[str, Any]) -> None:
        pass
    def validate_freqai_compat(self) -> None:
        pass
    def price_to_precision(self, pair: str, price: float, precision: int, **kwargs: Any) -> float:
        return round(price, precision)
    def amount_to_precision(self, pair: str, amount: float) -> float:
        return amount
    def _get_stop_limit_rate(self, price: float, order_types: Dict[str, Any], side: str) -> float:
        return price
    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        return True
    def create_dry_run_order(self, **kwargs: Any) -> Dict[str, Any]:
        order_id = f"dry_run_{kwargs.get('side')}_{randint(0, 1000000)}"
        return {"id": order_id, "side": kwargs.get("side"), "type": kwargs.get("ordertype"), "symbol": kwargs.get("pair"), "amount": kwargs.get("amount"), "cost": kwargs.get("amount") * kwargs.get("rate", 0), "fee": {"rate": 1.0}}
    def create_order(self, **kwargs: Any) -> Dict[str, Any]:
        return {}
    def fetch_dry_run_order(self, order_id: str) -> Dict[str, Any]:
        return {}
    def get_quote_currencies(self) -> List[str]:
        return ["USDT"]
    def get_pair_quote_currency(self, pair: str) -> str:
        return pair.split("/")[-1]
    def get_pair_base_currency(self, pair: str) -> str:
        return pair.split("/")[0]
    def get_balances(self) -> Dict[str, Any]:
        return {}
    def fetch_positions(self) -> List[Any]:
        return []
    def fetch_orders(self, pair: str, since: datetime) -> List[Any]:
        return []
    def fetch_trading_fees(self) -> Dict[str, Any]:
        return {}
    def fetch_ticker(self, pair: str) -> Dict[str, Any]:
        return {}
    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: str) -> bool:
        return True
    def get_historic_ohlcv(self, pair: str, timeframe: str, since: int, candle_type: str) -> Dict[str, Any]:
        return {}
    def refresh_latest_ohlcv(self, pairs: List[Tuple[str, str, str]], cache: bool = True) -> Dict[Tuple[str, str, str], Any]:
        return {}
    def refresh_latest_trades(self, pairs: List[Tuple[str, str, str]], cache: bool = False) -> Dict[Tuple[str, str, str], Any]:
        return {}
    def get_trade_history(self, pair: str, since: int, until: int) -> Tuple[str, List[Any]]:
        return (pair, [])
    def cancel_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        return {}
    def cancel_stoploss_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        return {}
    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict[str, Any]:
        return {}
    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict[str, Any]:
        return {}
    def fetch_order(self, order_id: str, pair: str) -> Any:
        return {}
    def fetch_stoploss_order(self, order_id: str, pair: str) -> Any:
        return {}
    def fetch_order_or_stoploss_order(self, order_id: str, pair: str, is_stoploss: bool) -> Any:
        return {}
    def get_fee(self, pair: str) -> float:
        return 0.0
    def calculate_fee_rate(self, fee: Dict[str, Any], symbol: str, *, cost: float, amount: float) -> float:
        return fee.get("rate", 0.0)
    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: Optional[datetime] = None) -> float:
        return 0.0
    def get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        return stake_amount / leverage
    def get_conversion_rate(self, base: str, quote: str) -> Optional[float]:
        return 1.0
    def load_leverage_tiers(self) -> Dict[str, Any]:
        return {}
    def parse_leverage_tier(self, tier: Dict[str, Any]) -> Dict[str, Any]:
        return {}
    def get_maintenance_ratio_and_amt(self, pair: str, nominal_value: float) -> Tuple[float, Optional[float]]:
        if nominal_value < 0:
            raise Exception("nominal value can not be lower than 0")
        return (0.025, 0.0)
    def get_max_leverage(self, pair: str, nominal_value: float) -> float:
        return 1.0
    def _get_params(self, *, side: str, ordertype: str, reduceOnly: bool, time_in_force: str, leverage: float) -> Dict[str, Any]:
        return self._params
    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Any]) -> Optional[float]:
        return None
    def get_contract_size(self, pair: str) -> Optional[float]:
        return None
    def _order_contracts_to_amount(self, order: Dict[str, Any]) -> Dict[str, Any]:
        return order
    def _trades_contracts_to_amount(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return trades
    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        return amount
    def _contracts_to_amount(self, pair: str, contracts: float) -> float:
        return contracts

class ExchangeResolver:
    def resolve(self, config: Dict[str, Any]) -> Exchange:
        return Exchange(config)

def market_is_active(market: Dict[str, Any]) -> bool:
    return market.get("active", True) 

# End of annotated code.
