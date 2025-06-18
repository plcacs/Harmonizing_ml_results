from datetime import datetime, timedelta, timezone
from types import FunctionType
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from sqlalchemy import select

from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT
from freqtrade.enums import TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange.exchange_utils import TICK_SIZE
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.util import dt_now
from tests.conftest import (
    create_mock_trades,
    create_mock_trades_usdt,
    create_mock_trades_with_leverage,
    log_has,
    log_has_re,
)

spot: TradingMode = TradingMode.SPOT
margin: TradingMode = TradingMode.MARGIN
futures: TradingMode = TradingMode.FUTURES

@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_enter_exit_side(fee: Any, is_short: bool) -> None:
    entry_side: str = "sell" if is_short else "buy"
    exit_side: str = "buy" if is_short else "sell"
    trade: Trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=0.001,
        open_rate=0.01,
        amount=5,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        leverage=2.0,
        trading_mode=margin,
    )
    assert trade.entry_side == entry_side
    assert trade.exit_side == exit_side
    assert trade.trade_direction == "short" if is_short else "long"

@pytest.mark.usefixtures("init_persistence")
def test_set_stop_loss_liquidation(fee: Any) -> None:
    trade: Trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=False,
        leverage=2.0,
        trading_mode=margin,
    )
    trade.set_liquidation_price(0.09)
    assert trade.liquidation_price == 0.09
    assert trade.stop_loss is None
    assert trade.initial_stop_loss is None

    trade.adjust_stop_loss(2.0, 0.2, True)
    assert trade.liquidation_price == 0.09
    assert trade.stop_loss == 1.8
    assert trade.initial_stop_loss == 1.8

    trade.set_liquidation_price(0.08)
    assert trade.liquidation_price == 0.08
    assert trade.stop_loss == 1.8
    assert trade.initial_stop_loss == 1.8

    trade.set_liquidation_price(0.11)
    trade.adjust_stop_loss(2.0, 0.2)
    assert trade.liquidation_price == 0.11
    assert trade.stop_loss == 1.8
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 1.8

    trade.adjust_stop_loss(1.8, 0.2)
    assert trade.liquidation_price == 0.11
    assert trade.stop_loss == 1.8
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 1.8

    trade.adjust_stop_loss(1.8, 0.22, allow_refresh=True)
    assert trade.liquidation_price == 0.11
    assert trade.stop_loss == 1.602
    assert trade.stop_loss_pct == -0.22
    assert trade.initial_stop_loss == 1.8

    trade.adjust_stop_loss(2.1, 0.1)
    assert trade.liquidation_price == 0.11
    assert pytest.approx(trade.stop_loss) == 1.994999
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.8
    assert trade.stoploss_or_liquidation == trade.stop_loss

    trade.stop_loss = None
    trade.liquidation_price = None
    trade.initial_stop_loss = None
    trade.initial_stop_loss_pct = None

    trade.adjust_stop_loss(2.0, 0.1, True)
    assert trade.liquidation_price is None
    assert trade.stop_loss == 1.9
    assert trade.initial_stop_loss == 1.9
    assert trade.stoploss_or_liquidation == 1.9

    trade.is_short = True
    trade.recalc_open_trade_value()
    trade.stop_loss = None
    trade.initial_stop_loss = None
    trade.initial_stop_loss_pct = None

    trade.set_liquidation_price(3.09)
    assert trade.liquidation_price == 3.09
    assert trade.stop_loss is None
    assert trade.initial_stop_loss is None

    trade.adjust_stop_loss(2.0, 0.2)
    assert trade.liquidation_price == 3.09
    assert trade.stop_loss == 2.2
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 2.2

    trade.set_liquidation_price(3.1)
    assert trade.liquidation_price == 3.1
    assert trade.stop_loss == 2.2
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 2.2

    trade.set_liquidation_price(3.8)
    assert trade.liquidation_price == 3.8
    assert trade.stop_loss == 2.2
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 2.2

    trade.adjust_stop_loss(2.0, 0.3)
    assert trade.liquidation_price == 3.8
    assert trade.stop_loss == 2.2
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 2.2

    trade.adjust_stop_loss(2.0, 0.3, allow_refresh=True)
    assert trade.liquidation_price == 3.8
    assert trade.stop_loss == 2.3
    assert trade.stop_loss_pct == -0.3
    assert trade.initial_stop_loss == 2.2

    trade.set_liquidation_price(1.5)
    trade.adjust_stop_loss(1.8, 0.1)
    assert trade.liquidation_price == 1.5
    assert pytest.approx(trade.stop_loss) == 1.89
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 1.5

@pytest.mark.parametrize(
    "exchange,is_short,lev,minutes,rate,interest,trading_mode",
    [
        ("binance", False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin),
        ("binance", True, 3, 10, 0.0005, 0.000625, margin),
        ("binance", False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin),
        ("binance", True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin),
        ("binance", False, 3, 295, 0.00025, round(0.0020833333333333333, 8), margin),
        ("binance", True, 3, 295, 0.00025, round(0.0015624999999999999, 8), margin),
        ("binance", False, 5, 295, 0.0005, 0.005, margin),
        ("binance", True, 5, 295, 0.0005, round(0.0031249999999999997, 8), margin),
        ("binance", False, 1, 295, 0.0005, 0.0, spot),
        ("binance", True, 1, 295, 0.0005, 0.003125, margin),
        ("binance", False, 3, 10, 0.0005, 0.0, futures),
        ("binance", True, 3, 295, 0.0005, 0.0, futures),
        ("binance", False, 5, 295, 0.0005, 0.0, futures),
        ("binance", True, 5, 295, 0.0005, 0.0, futures),
        ("binance", False, 1, 295, 0.0005, 0.0, futures),
        ("binance", True, 1, 295, 0.0005, 0.0, futures),
        ("kraken", False, 3, 10, 0.0005, 0.040, margin),
        ("kraken", True, 3, 10, 0.0005, 0.030, margin),
        ("kraken", False, 3, 295, 0.0005, 0.06, margin),
        ("kraken", True, 3, 295, 0.0005, 0.045, margin),
        ("kraken", False, 3, 295, 0.00025, 0.03, margin),
        ("kraken", True, 3, 295, 0.00025, 0.0225, margin),
        ("kraken", False, 5, 295, 0.0005, round(0.07200000000000001, 8), margin),
        ("kraken", True, 5, 295, 0.0005, 0.045, margin),
        ("kraken", False, 1, 295, 0.0005, 0.0, spot),
        ("kraken", True, 1, 295, 0.0005, 0.045, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_interest(
    fee: Any,
    exchange: str,
    is_short: bool,
    lev: float,
    minutes: int,
    rate: float,
    interest: float,
    trading_mode: TradingMode,
) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=20.0,
        amount=30.0,
        open_rate=2.0,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=minutes),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange=exchange,
        leverage=lev,
        interest_rate=rate,
        is_short=is_short,
        trading_mode=trading_mode,
    )
    assert round(float(trade.calculate_interest()), 8) == interest

@pytest.mark.parametrize(
    "is_short,lev,borrowed,trading_mode",
    [
        (False, 1.0, 0.0, spot),
        (True, 1.0, 30.0, margin),
        (False, 3.0, 40.0, margin),
        (True, 3.0, 30.0, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_borrowed(
    fee: Any,
    is_short: bool,
    lev: float,
    borrowed: float,
    trading_mode: TradingMode,
) -> None:
    trade: Trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
    )
    assert trade.borrowed == borrowed

@pytest.mark.parametrize(
    "is_short,open_rate,close_rate,lev,profit,trading_mode",
    [
        (False, 2.0, 2.2, 1.0, 0.09451372, spot),
        (True, 2.2, 2.0, 3.0, 0.25894253, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_update_limit_order(
    fee: Any,
    caplog: Any,
    limit_buy_order_usdt: Dict[str, Any],
    limit_sell_order_usdt: Dict[str, Any],
    time_machine: Any,
    is_short: bool,
    open_rate: float,
    close_rate: float,
    lev: float,
    profit: float,
    trading_mode: TradingMode,
) -> None:
    time_machine.move_to("2022-03-31 20:45:00 +00:00")

    enter_order: Dict[str, Any] = limit_sell_order_usdt if is_short else limit_buy_order_usdt
    exit_order: Dict[str, Any] = limit_buy_order_usdt if is_short else limit_sell_order_usdt
    entry_side: str = "sell" if is_short else "buy"
    exit_side: str = "buy" if is_short else "sell"

    trade: Trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=open_rate,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        interest_rate=0.0005,
        leverage=lev,
        trading_mode=trading_mode,
    )
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None

    oobj: Order = Order.parse_from_ccxt_object(enter_order, "ADA/USDT", entry_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == open_rate
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        f"LIMIT_{entry_side.upper()} has been fulfilled for "
        r"Trade\(id=2, pair=ADA/USDT, amount=30.00000000, "
        f"is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, "
        r"open_since=.*\).",
        caplog,
    )

    caplog.clear()
    time_machine.move_to("2022-03-31 21:45:05 +00:00")
    oobj = Order.parse_from_ccxt_object(exit_order, "ADA/USDT", exit_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)

    assert not trade.has_open_orders
    assert trade.close_rate == close_rate
    assert pytest.approx(trade.close_profit) == profit
    assert trade.close_date is not None
    assert log_has_re(
        f"LIMIT_{exit_side.upper()} has been fulfilled for "
        r"Trade\(id=2, pair=ADA/USDT, amount=30.00000000, "
        f"is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, "
        r"open_since=.*\).",
        caplog,
    )
    caplog.clear()

@pytest.mark.usefixtures("init_persistence")
def test_update_market_order(
    market_buy_order_usdt: Dict[str, Any],
    market_sell_order_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
) -> None:
    trade: Trade = Trade(
        id=1,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now(),
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )

    oobj: Order = Order.parse_from_ccxt_object(market_buy_order_usdt, "ADA/USDT", "buy")
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == 2.0
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        r"MARKET_BUY has been fulfilled for Trade\(id=1, "
        r"pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, "
        r"open_rate=2.00000000, open_since=.*\).",
        caplog,
    )

    caplog.clear()
    trade.is_open = True
    oobj = Order.parse_from_ccxt_object(market_sell_order_usdt, "ADA/USDT", "sell")
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == 2.2
    assert pytest.approx(trade.close_profit) == 0.094513715710723
    assert trade.close_date is not None
    assert log_has_re(
        r"MARKET_SELL has been fulfilled for Trade\(id=1, "
        r"pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, "
        r"open_rate=2.00000000, open_since=.*\).",
        caplog,
    )

@pytest.mark.parametrize(
    "exchange,is_short,lev,open_value,close_value,profit,profit_ratio,trading_mode,funding_fees",
    [
        ("binance", False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
        ("binance", True, 1, 65.835, 60.151253125, 5.68374687, 0.08633321, margin, 0.0),
        ("binance", False, 3, 60.15, 65.83416667, 5.68416667, 0.28349958, margin, 0.0),
        ("binance", True, 3, 65.835, 60.151253125, 5.68374687, 0.25899963, margin, 0.0),
        ("kraken", False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
        ("kraken", True, 1, 65.835, 60.21015, 5.62485, 0.0854386, margin, 0.0),
        ("kraken", False, 3, 60.15, 65.795, 5.645, 0.28154613, margin, 0.0),
        ("kraken", True, 3, 65.835, 60.21015, 5.62485, 0.25631579, margin, 0.0),
        ("binance", False, 1, 60.15, 65.835, 5.685, 0.09451371, futures, 0.0),
        ("binance", False, 1, 60.15, 66.835, 6.685, 0.11113881, futures, 1.0),
        ("binance", True, 1, 65.835, 60.15, 5.685, 0.08635224, futures, 0.0),
        ("binance", True, 1, 65.835, 61.15, 4.685, 0.07116276, futures, -1.0),
        ("binance", True, 3, 65.835, 59.15, 6.685, 0.3046252, futures, 1.0),
        ("binance", False, 3, 60.15, 64.835, 4.685, 0.23366583, futures, -1.0),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price(
    limit_order: Dict[str, Any],
    fee: Any,
    exchange: str,
    is_short: bool,
    lev: float,
    open_value: float,
    close_value: float,
    profit: float,
    profit_ratio: float,
    trading_mode: TradingMode,
    funding_fees: float,
) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        interest_rate=0.0005,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange=exchange,
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
    )
    entry_order: Dict[str, Any] = limit_order[trade.entry_side]
    exit_order: Dict[str, Any] = limit_order[trade.exit_side]

    oobj: Order = Order.parse_from_ccxt_object(entry_order, "ADA/USDT", trade.entry_side)
    oobj._trade_live = trade
    oobj.update_from_ccxt_object(entry_order)
    trade.update_trade(oobj)

    trade.funding_fee_running = funding_fees

    oobj = Order.parse_from_ccxt_object(exit_order, "ADA/USDT", trade.exit_side)
    oobj._trade_live = trade
    oobj.update_from_ccxt_object(exit_order)
    trade.update_trade(oobj)

    assert trade.is_open is False
    assert trade.funding_fees == funding_fees
    assert trade.orders[-1].funding_fee == funding_fees

    assert pytest.approx(trade._calc_open_trade_value(trade.amount, trade.open_rate)) == open_value
    assert pytest.approx(trade.calc_close_trade_value(trade.close_rate)) == close_value
    assert pytest.approx(trade.close_profit_abs) == profit
    assert pytest.approx(trade.close_profit) == profit_ratio

@pytest.mark.usefixtures("init_persistence")
def test_trade_close(fee: Any, time_machine: Any) -> None:
    time_machine.move_to("2022-09-01 05:00:00 +00:00", tick=False)

    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now() - timedelta(minutes=10),
        interest_rate=0.0005,
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )
    trade.orders.append(
        Order(
            ft_order_side=trade.entry_side,
            order_id=f"{trade.pair}-{trade.entry_side}-{trade.open_date}",
            ft_is_open=False,
            ft_pair=trade.pair,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=trade.open_rate,
            average=trade.open_rate,
            status="closed",
            order_type="limit",
            side=trade.entry_side,
            order_filled_date=trade.open_date,
        )
    )
    trade.orders.append(
        Order(
            ft_order_side=trade.exit_side,
            order_id=f"{trade.pair}-{trade.exit_side}-{trade.open_date}",
            ft_is_open=False,
            ft_pair=trade.pair,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=2.2,
            average=2.2,
            status="closed",
            order_type="limit",
            side=trade.exit_side,
            order_filled_date=dt_now(),
        )
    )
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(2.2)
    assert trade.is_open is False
    assert pytest.approx(trade.close_profit) == 0.094513715
    assert trade.close_date is not None
    assert trade.close_date_utc == dt_now()

    new_date: datetime = dt_now() + timedelta(minutes=5)
    assert trade.close_date_utc != new_date
    trade.close_date = new_date
    trade.close(2.2)
    assert trade.close_date_utc == new_date

@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception(limit_buy_order_usdt: Dict[str, Any], fee: Any) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )

    oobj: Order = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "buy")
    trade.update_trade(oobj)
    assert trade.calc_close_trade_value(trade.close_rate) == 0.0

@pytest.mark.usefixtures("init_persistence")
def test_update_open_order(limit_buy_order_usdt: Dict[str, Any]) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange="binance",
        trading_mode=margin,
    )

    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None

    limit_buy_order_usdt["status"] = "open"
    oobj: Order = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "buy")
    trade.update_trade(oobj)

    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None

@pytest.mark.usefixtures("init_persistence")
def test_update_invalid_order(limit_buy_order_usdt: Dict[str, Any]) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange="binance",
        trading_mode=margin,
    )
    limit_buy_order_usdt["type"] = "invalid"
    oobj: Order = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "meep")
    with pytest.raises(ValueError, match=r"Unknown order type"):
        trade.update_trade(oobj)

@pytest.mark.parametrize(
    "exchange,is_short,lev,open_rate,close_rate,fee_rate,result,trading_mode,funding_fees",
    [
        ("binance", False, 1, 2.0, 2.5, 0.0025, 74.8125, spot, 0),
        ("binance", False, 1, 2.0, 2.5, 0.003, 74.775, spot, 0),
        ("binance", False, 1, 2.0, 2.2, 0.005, 65.67, margin, 0),
        ("binance", False, 3, 2.0, 2.5, 0.0025, 74.81166667, margin, 0),
        ("binance", False, 3, 2.0, 2.5, 0.003, 74.77416667, margin, 0),
        ("binance", True, 3, 2.2, 2.5, 0.0025, 75.18906641, margin, 0),
        ("binance", True, 3, 2.2, 2.5, 0.003, 75.22656719, margin, 0),
        ("binance", True, 1, 2.2, 2.5, 0.0025, 75.18906641, margin, 0),
        ("binance", True, 1, 2.2, 2.5, 0.003, 75.22656719, margin, 0),
        ("kraken", False, 3, 2.0, 2.5, 0.0025, 74.7725, margin, 0),
        ("kraken", False, 3, 2.0, 2.5, 0.003, 74.735, margin, 0),
        ("kraken", True, 3, 2.2, 2.5, 0.0025, 75.2626875, margin, 0),
        ("kraken", True, 3, 2.2, 2.5, 0.003, 75.300225, margin, 0),
        ("kraken", True, 1, 2.2, 2.5, 0.0025, 75.2626875, margin, 0),
        ("kraken", True, 1, 2.2, 2.5, 0.003, 75.300225, margin, 0),
        ("binance", False, 1, 2.0, 2.5, 0.0025, 75.8125, futures, 1),
        ("binance", False, 3, 2.0, 2.5, 0.0025, 73.8125, futures, -1),
        ("binance", True, 3, 2.0, 2.5, 0.0025, 74.1875, futures, 1),
        ("binance", True, 1, 2.0, 2.5, 0.0025, 76.1875, futures, -1),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price(
    open_rate: float,
    exchange: str,
    is_short: bool,
    lev: float,
    close_rate: float,
    fee_rate: float,
    result: float,
    trading_mode: TradingMode,
    funding_fees: float,
) -> None:
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=open_rate,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        fee_open=fee_rate,
        fee_close=fee_rate,
        exchange=exchange,
        interest_rate=0.0005,
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
        funding_fees=funding_fees,
    )
    assert round(trade.calc_close_trade_value(rate=close_rate), 8) == result

@pytest.mark.parametrize(
    "exchange,is_short,lev,close_rate,fee_close,profit,profit_ratio,trading_mode,funding_fees",
    [
        ("binance", False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
        ("binance", False, 3, 2.1, 0.0025, 2.69166667, 0.134247714, margin, 0),
        ("binance", True, 1, 2.1, 0.0025, -3.3088157, -0.055285142, margin, 0),
        ("binance", True, 3, 2.1, 0.0025, -3.3088157, -0.16585542, margin, 0),
        ("binance", False, 1, 1.9, 0.0025, -3.2925, -0.054738154, margin, 0),
        ("binance", False, 3, 1.9, 0.0025, -3.29333333, -0.164256026, margin, 0),
        ("binance", True, 1, 1.9, 0.0025, 2.70630953, 0.0452182043, margin, 0),
        ("binance", True, 3, 1.9, 0.0025, 2.70630953, 0.135654613, margin, 0),
        ("binance", False, 1, 2.2, 0.0025, 5.685, 0.09451371, margin, 0),
        ("binance", False, 3, 2.2, 0.0025, 5.68416667, 0.28349958, margin, 0),
        ("binance", True, 1, 2.2, 0.0025, -6.3163784, -0.10553681, margin, 0),
        ("binance", True, 3, 2.2, 0.0025, -6.3163784, -0.31661044, margin, 0),
        ("kraken", False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
        ("kraken", False, 3, 2.1, 0.0025, 2.6525, 0.132294264, margin, 0),
        ("kraken", True, 1, 2.1, 0.0025, -3