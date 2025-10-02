#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from types import FunctionType
from typing import Any, Dict, List, Optional
import pytest
from sqlalchemy import select
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT
from freqtrade.enums import TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange.exchange_utils import TICK_SIZE
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.util import dt_now
from tests.conftest import create_mock_trades, create_mock_trades_usdt, create_mock_trades_with_leverage, log_has, log_has_re

spot: TradingMode = TradingMode.SPOT
margin: TradingMode = TradingMode.MARGIN
futures: TradingMode = TradingMode.FUTURES


@pytest.mark.parametrize('is_short', [False, True])
def test_enter_exit_side(fee: Any, is_short: bool) -> None:
    trade = Trade(id=1, pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0,
                  open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange='binance', is_short=is_short, leverage=1.0, trading_mode=margin)
    if is_short:
        assert trade.entry_side == 'sell'
        assert trade.exit_side == 'buy'
    else:
        assert trade.entry_side == 'buy'
        assert trade.exit_side == 'sell'


@pytest.mark.usefixtures('init_persistence')
def test_set_stop_loss_liquidation(fee: Any) -> None:
    trade = Trade(id=1, pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0,
                  open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange='binance', is_short=False, leverage=1.0, trading_mode=margin)
    trade.set_liquidation_price(1.8)
    assert trade.liquidation_price == 1.8


@pytest.mark.parametrize('exchange,is_short,lev,minutes,rate,interest,trading_mode',
                         [('binance', False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin),
                          ('binance', True, 3, 10, 0.0005, 0.000625, margin),
                          ('binance', False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin),
                          ('binance', True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin),
                          ('binance', False, 3, 10, 0.00025, round(0.0004166666666666667, 8), margin),
                          ('binance', True, 3, 10, 0.00025, round(0.0003124999999999999, 8), margin),
                          ('kraken', False, 3, 10, 0.0005, 0.04, margin),
                          ('kraken', True, 3, 10, 0.0005, 0.03, margin),
                          ('kraken', False, 3, 295, 0.0005, 0.06, margin),
                          ('kraken', True, 3, 295, 0.0005, 0.045, margin),
                          ('kraken', False, 3, 10, 0.00025, 0.02, margin),
                          ('kraken', True, 3, 10, 0.00025, 0.015, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_interest(fee: Any, exchange: str, is_short: bool, lev: float, minutes: int, rate: float, 
                  interest: float, trading_mode: TradingMode) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0,
                  open_date=datetime.now(timezone.utc) - timedelta(minutes=minutes),
                  fee_open=fee.return_value, fee_close=fee.return_value, exchange=exchange,
                  leverage=lev, interest_rate=rate, is_short=is_short, trading_mode=trading_mode)
    assert round(float(trade.calculate_interest()), 8) == interest


@pytest.mark.parametrize('is_short,lev,borrowed,trading_mode',
                         [(False, 1.0, 0.0, spot),
                          (True, 1.0, 30.0, margin),
                          (False, 3.0, 40.0, margin),
                          (True, 3.0, 30.0, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_borrowed(fee: Any, is_short: bool, lev: float, borrowed: float, trading_mode: TradingMode) -> None:
    trade = Trade(id=1, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
                  open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange='binance', is_short=is_short, leverage=lev, trading_mode=trading_mode)
    assert trade.borrowed == borrowed


@pytest.mark.parametrize('is_short,open_rate,close_rate,lev,profit,trading_mode',
                         [(False, 2.0, 2.2, 1.0, 0.09451372, spot),
                          (True, 2.2, 2.0, 3.0, 0.25894253, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_update_limit_order(fee: Any, caplog: Any, limit_buy_order_usdt: Dict[str, Any],
                            limit_sell_order_usdt: Dict[str, Any], time_machine: Any, is_short: bool,
                            open_rate: float, close_rate: float, lev: float, profit: float,
                            trading_mode: TradingMode) -> None:
    time_machine.move_to('2022-03-31 20:45:00 +00:00')
    enter_order: Dict[str, Any] = limit_sell_order_usdt if is_short else limit_buy_order_usdt
    exit_order: Dict[str, Any] = limit_buy_order_usdt if is_short else limit_sell_order_usdt
    trade = Trade(id=1, pair='ADA/USDT', stake_amount=60.0, open_rate=open_rate, amount=30.0, is_open=True,
                  open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance',
                  is_short=is_short, interest_rate=0.0005, leverage=lev, trading_mode=trading_mode)
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None
    oobj = Order.parse_from_ccxt_object(enter_order, 'ADA/USDT', trade.entry_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == open_rate
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        f'LIMIT_{trade.entry_side.upper()} has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, '
        f'is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, open_since=.*\\).', caplog)
    caplog.clear()
    time_machine.move_to('2022-03-31 21:45:05 +00:00')
    oobj = Order.parse_from_ccxt_object(exit_order, 'ADA/USDT', trade.exit_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == close_rate
    assert pytest.approx(trade.close_profit) == profit
    assert trade.close_date is not None
    assert log_has_re(
        f'LIMIT_{trade.exit_side.upper()} has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, '
        f'is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, open_since=.*\\).', caplog)
    caplog.clear()


@pytest.mark.usefixtures('init_persistence')
def test_update_market_order(market_buy_order_usdt: Dict[str, Any], market_sell_order_usdt: Dict[str, Any],
                             fee: Any, caplog: Any) -> None:
    trade = Trade(id=1, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
                  fee_open=fee.return_value, fee_close=fee.return_value, open_date=dt_now(), exchange='binance',
                  trading_mode=margin, leverage=1.0)
    oobj = Order.parse_from_ccxt_object(market_buy_order_usdt, 'ADA/USDT', 'buy')
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == 2.0
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re('MARKET_BUY has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, '
                      'is_short=False, leverage=1.0, open_rate=2.00000000, open_since=.*\\).', caplog)
    caplog.clear()
    trade.is_open = True
    oobj = Order.parse_from_ccxt_object(market_sell_order_usdt, 'ADA/USDT', 'sell')
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == 2.2
    assert pytest.approx(trade.close_profit) == 0.094513715710723
    assert trade.close_date is not None
    assert log_has_re('MARKET_SELL has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, '
                      'is_short=False, leverage=1.0, open_rate=2.00000000, open_since=.*\\).', caplog)


@pytest.mark.parametrize('exchange,is_short,lev,open_value,close_value,profit,profit_ratio,trading_mode,funding_fees',
                         [('binance', False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
                          ('binance', True, 1, 65.835, 60.151253125, 5.68374687, 0.08635224, margin, 0.0),
                          ('binance', False, 3, 60.15, 65.83416667, 5.68416667, 0.28349958, margin, 0.0),
                          ('binance', True, 3, 65.835, 60.151253125, 5.68374687, 0.25899963, margin, 0.0),
                          ('kraken', False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
                          ('kraken', True, 1, 65.835, 60.21015, 5.62485, 0.0854386, margin, 0.0),
                          ('kraken', False, 3, 60.15, 65.795, 5.645, 0.28154613, margin, 0.0),
                          ('kraken', True, 3, 65.835, 60.21015, 5.62485, 0.25631579, margin, 0.0),
                          ('binance', False, 1, 60.15, 65.835, 5.685, 0.09451371, futures, 0.0),
                          ('binance', False, 1, 60.15, 66.835, 6.685, 0.11113881, futures, 1.0),
                          ('binance', True, 1, 65.835, 60.15, 5.685, 0.08635224, futures, 0.0),
                          ('binance', True, 1, 65.835, 61.15, 4.685, 0.07116276, futures, -1.0),
                          ('binance', True, 3, 65.835, 59.15, 6.685, 0.3046252, futures, 1.0),
                          ('binance', False, 3, 60.15, 64.835, 4.685, 0.23366583, futures, -1.0)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_open_close_trade_price(limit_order: Dict[str, Dict[str, Any]], fee: Any, exchange: str,
                                     is_short: bool, lev: float, open_value: float, close_value: float,
                                     profit: float, profit_ratio: float, trading_mode: TradingMode,
                                     funding_fees: float) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0,
                  open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
                  interest_rate=0.0005, fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange=exchange, is_short=is_short, leverage=lev, trading_mode=trading_mode)
    entry_order: Dict[str, Any] = limit_order[trade.entry_side]
    exit_order: Dict[str, Any] = limit_order[trade.exit_side]
    oobj = Order.parse_from_ccxt_object(entry_order, 'ADA/USDT', trade.entry_side)
    oobj._trade_live = trade
    oobj.update_from_ccxt_object(entry_order)
    trade.update_trade(oobj)
    trade.funding_fee_running = funding_fees
    oobj = Order.parse_from_ccxt_object(exit_order, 'ADA/USDT', trade.exit_side)
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


@pytest.mark.usefixtures('init_persistence')
def test_trade_close(fee: Any, time_machine: Any) -> None:
    time_machine.move_to('2022-09-01 05:00:00 +00:00', tick=False)
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
                  fee_open=fee.return_value, fee_close=fee.return_value,
                  open_date=dt_now() - timedelta(minutes=10), interest_rate=0.0005, exchange='binance',
                  trading_mode=margin, leverage=1.0)
    trade.orders.append(Order(ft_order_side=trade.entry_side, order_id=f'{trade.pair}-{trade.entry_side}-{trade.open_date}',
                                ft_is_open=False, ft_pair=trade.pair, amount=trade.amount, filled=trade.amount,
                                remaining=0, price=trade.open_rate, average=trade.open_rate, status='closed',
                                order_type='limit', side=trade.entry_side, order_filled_date=trade.open_date))
    trade.orders.append(Order(ft_order_side=trade.exit_side, order_id=f'{trade.pair}-{trade.exit_side}-{trade.open_date}',
                                ft_is_open=False, ft_pair=trade.pair, amount=trade.amount, filled=trade.amount,
                                remaining=0, price=2.2, average=2.2, status='closed', order_type='limit',
                                side=trade.exit_side, order_filled_date=dt_now()))
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(2.2)
    assert trade.is_open is False
    assert pytest.approx(trade.close_profit) == 0.094513715
    assert trade.close_date is not None
    assert trade.close_date_utc == dt_now()
    new_date = dt_now() + timedelta(minutes=5)
    assert trade.close_date_utc != new_date
    assert trade.is_open is False
    trade.close_date = new_date
    trade.close(2.2)
    assert trade.close_date_utc == new_date


@pytest.mark.usefixtures('init_persistence')
def test_calc_close_trade_price_exception(limit_buy_order_usdt: Dict[str, Any], fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0,
                  fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance',
                  trading_mode=margin, leverage=1.0)
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, 'ADA/USDT', 'buy')
    trade.update_trade(oobj)
    assert trade.calc_close_trade_value(trade.close_rate) == 0.0


@pytest.mark.usefixtures('init_persistence')
def test_update_open_order(limit_buy_order_usdt: Dict[str, Any]) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0,
                  fee_open=0.1, fee_close=0.1, exchange='binance', trading_mode=margin)
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None
    limit_buy_order_usdt['status'] = 'open'
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, 'ADA/USDT', 'buy')
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None


@pytest.mark.usefixtures('init_persistence')
def test_update_invalid_order(limit_buy_order_usdt: Dict[str, Any]) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, amount=30.0, open_rate=2.0,
                  fee_open=0.1, fee_close=0.1, exchange='binance', trading_mode=margin)
    limit_buy_order_usdt['type'] = 'invalid'
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, 'ADA/USDT', 'meep')
    with pytest.raises(ValueError, match='Unknown order type'):
        trade.update_trade(oobj)


@pytest.mark.parametrize('exchange', ['binance', 'kraken'])
@pytest.mark.parametrize('trading_mode', [spot, margin, futures])
@pytest.mark.parametrize('lev', [1, 3])
@pytest.mark.parametrize('is_short,fee_rate,result',
                         [(False, 0.003, 60.18), (False, 0.0025, 60.15),
                          (False, 0.003, 60.18), (False, 0.0025, 60.15),
                          (True, 0.003, 59.82), (True, 0.0025, 59.85),
                          (True, 0.003, 59.82), (True, 0.0025, 59.85)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_open_trade_value(limit_buy_order_usdt: Dict[str, Any], exchange: str, lev: int, is_short: bool,
                               fee_rate: float, result: float, trading_mode: TradingMode) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, amount=30.0, open_rate=2.0,
                  open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
                  fee_open=fee_rate, fee_close=fee_rate, exchange=exchange, leverage=lev, is_short=is_short,
                  trading_mode=trading_mode)
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, 'ADA/USDT', 'sell' if is_short else 'buy')
    trade.update_trade(oobj)
    assert trade._calc_open_trade_value(trade.amount, trade.open_rate) == result


@pytest.mark.parametrize('exchange,is_short,lev,open_rate,close_rate,fee_rate,result,trading_mode,funding_fees',
                         [('binance', False, 1, 2.0, 2.5, 0.0025, 75.8125, futures, 1),
                          ('binance', False, 3, 2.0, 2.5, 0.0025, 73.8125, futures, -1),
                          ('binance', True, 3, 2.2, 2.5, 0.0025, 75.18906641, margin, 0),
                          ('binance', True, 1, 2.2, 2.5, 0.0025, 75.18906641, margin, 0)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_close_trade_price(open_rate: float, exchange: str, is_short: bool, lev: int, close_rate: float,
                                fee_rate: float, result: float, trading_mode: TradingMode, funding_fees: float) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, amount=30.0, open_rate=open_rate,
                  open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
                  fee_open=fee_rate, fee_close=fee_rate, exchange=exchange, interest_rate=0.0005,
                  is_short=is_short, leverage=lev, trading_mode=trading_mode, funding_fees=funding_fees)
    assert round(trade.calc_close_trade_value(rate=close_rate), 8) == result


@pytest.mark.parametrize('exchange,is_short,lev,close_rate,fee_close,profit,profit_ratio,trading_mode,funding_fees',
                         [('binance', False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
                          ('binance', False, 3, 2.1, 0.0025, 2.69166667, 0.134247714, margin, 0),
                          ('binance', True, 1, 2.1, 0.0025, -3.3088157, -0.055285142, margin, 0),
                          ('binance', True, 3, 2.1, 0.0025, -3.3088157, -0.16585542, margin, 0),
                          ('kraken', False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
                          ('kraken', False, 3, 2.1, 0.0025, 2.6525, 0.132294264, margin, 0),
                          ('kraken', True, 1, 2.1, 0.0025, -3.3706575, -0.056318421, margin, 0),
                          ('kraken', True, 3, 2.1, 0.0025, -3.3706575, -0.168955263, margin, 0)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_profit(exchange: str, is_short: bool, lev: int, close_rate: float, fee_close: float,
                     profit: float, profit_ratio: float, trading_mode: TradingMode, funding_fees: float) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=60.0, amount=30.0, open_rate=2.0,
                  open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
                  interest_rate=0.0005, exchange=exchange, is_short=is_short, leverage=lev, fee_open=0.0025,
                  fee_close=fee_close, max_stake_amount=60.0, trading_mode=trading_mode, funding_fees=funding_fees)
    profit_res = trade.calculate_profit(close_rate)
    assert pytest.approx(profit_res.profit_abs) == round(profit, 8)
    assert pytest.approx(profit_res.profit_ratio) == round(profit_ratio, 8)
    val = trade.open_trade_value * profit_res.profit_ratio / lev
    assert pytest.approx(val) == profit_res.profit_abs
    assert pytest.approx(profit_res.total_profit) == round(profit, 8)
    assert pytest.approx(trade.calc_profit(rate=close_rate)) == round(profit, 8)
    assert pytest.approx(trade.calc_profit_ratio(rate=close_rate)) == round(profit_ratio, 8)
    profit_res2 = trade.calculate_profit(close_rate, trade.amount, trade.open_rate)
    assert pytest.approx(profit_res2.profit_abs) == round(profit, 8)
    assert pytest.approx(profit_res2.profit_ratio) == round(profit_ratio, 8)
    assert pytest.approx(profit_res2.total_profit) == round(profit, 8)
    assert pytest.approx(trade.calc_profit(close_rate, trade.amount, trade.open_rate)) == round(profit, 8)
    assert pytest.approx(trade.calc_profit_ratio(close_rate, trade.amount, trade.open_rate)) == round(profit_ratio, 8)


def test_adjust_stop_loss(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, amount=30, fee_open=fee.return_value,
                  fee_close=fee.return_value, exchange='binance', open_rate=1, max_rate=1)
    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.96, 0.05)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.3, -0.1)
    assert pytest.approx(trade.stop_loss) == 1.17
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.2, 0.1)
    assert pytest.approx(trade.stop_loss) == 1.17
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.4, 0.1)
    assert pytest.approx(trade.stop_loss) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.7, 0.1, True)
    assert pytest.approx(trade.stop_loss) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    assert trade.stop_loss_pct == -0.1


def test_adjust_stop_loss_short(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=0.001, amount=5, fee_open=fee.return_value,
                  fee_close=fee.return_value, exchange='binance', open_rate=1, max_rate=1, is_short=True)
    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.04, 0.05)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.7, 0.1)
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.8, -0.1)
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.6, -0.1)
    assert round(trade.stop_loss, 8) == 0.66
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.3, -0.1, True)
    assert round(trade.stop_loss, 8) == 0.66
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    assert trade.stop_loss_pct == -0.1
    trade.set_liquidation_price(0.63)
    trade.adjust_stop_loss(0.59, -0.1)
    assert trade.stop_loss == 0.649
    assert trade.liquidation_price == 0.63


def test_adjust_min_max_rates(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, amount=30.0,
                  fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance',
                  open_rate=1)
    trade.adjust_min_max_rates(trade.open_rate, trade.open_rate)
    assert trade.max_rate == 1
    assert trade.min_rate == 1
    trade.adjust_min_max_rates(0.96, 0.96)
    assert trade.max_rate == 1
    assert trade.min_rate == 0.96
    trade.adjust_min_max_rates(1.05, 1.05)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96
    trade.adjust_min_max_rates(1.03, 1.03)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96
    trade.adjust_min_max_rates(1.1, 0.91)
    assert trade.max_rate == 1.1
    assert trade.min_rate == 0.91


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('use_db', [True, False])
@pytest.mark.parametrize('is_short', [True, False])
def test_get_open(fee: Any, is_short: bool, use_db: bool) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades(fee, is_short, use_db)
    assert len(Trade.get_open_trades()) == 4
    assert Trade.get_open_trade_count() == 4
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('use_db', [True, False])
def test_get_open_lev(fee: Any, use_db: bool) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades_with_leverage(fee, use_db)
    assert len(Trade.get_open_trades()) == 5
    assert Trade.get_open_trade_count() == 5
    Trade.use_db = True


@pytest.mark.parametrize('is_short', [True, False])
@pytest.mark.parametrize('use_db', [True, False])
@pytest.mark.usefixtures('init_persistence')
def test_get_open_orders(fee: Any, is_short: bool, use_db: bool) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades_usdt(fee, is_short, use_db)
    trade = Trade.get_trades_proxy(pair='XRP/USDT')[0]
    assert len(trade.orders) == 2
    assert len(trade.open_orders) == 0
    assert not trade.has_open_orders
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
def test_to_json(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=0.001, amount=123.0, amount_requested=123.0,
                  fee_open=fee.return_value, fee_close=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), open_rate=0.123, exchange='binance',
                  enter_tag=None, precision_mode=1, precision_mode_price=1, amount_precision=8.0,
                  price_precision=7.0, contract_size=1)
    result: Dict[str, Any] = trade.to_json()
    assert isinstance(result, dict)
    assert result == {'trade_id': None,
                      'pair': 'ADA/USDT',
                      'base_currency': 'ADA',
                      'quote_currency': 'USDT',
                      'is_open': None,
                      'open_date': trade.open_date.strftime(DATETIME_PRINT_FORMAT),
                      'open_timestamp': int(trade.open_date.timestamp() * 1000),
                      'open_fill_date': None,
                      'open_fill_timestamp': None,
                      'close_date': None,
                      'close_timestamp': None,
                      'open_rate': 0.123,
                      'open_rate_requested': None,
                      'open_trade_value': 15.1668225,
                      'fee_close': 0.0025,
                      'fee_close_cost': None,
                      'fee_close_currency': None,
                      'fee_open': 0.0025,
                      'fee_open_cost': None,
                      'fee_open_currency': None,
                      'close_rate': None,
                      'close_rate_requested': None,
                      'amount': 123.0,
                      'amount_requested': 123.0,
                      'stake_amount': 0.001,
                      'max_stake_amount': None,
                      'trade_duration': None,
                      'trade_duration_s': None,
                      'realized_profit': 0.0,
                      'realized_profit_ratio': None,
                      'close_profit': None,
                      'close_profit_pct': None,
                      'close_profit_abs': None,
                      'profit_ratio': None,
                      'profit_pct': None,
                      'profit_abs': None,
                      'exit_reason': None,
                      'exit_order_status': None,
                      'stop_loss_abs': None,
                      'stop_loss_ratio': None,
                      'stop_loss_pct': None,
                      'stoploss_last_update': None,
                      'stoploss_last_update_timestamp': None,
                      'initial_stop_loss_abs': None,
                      'initial_stop_loss_pct': None,
                      'initial_stop_loss_ratio': None,
                      'min_rate': None,
                      'max_rate': None,
                      'strategy': None,
                      'enter_tag': None,
                      'timeframe': None,
                      'exchange': 'binance',
                      'leverage': None,
                      'interest_rate': None,
                      'liquidation_price': None,
                      'is_short': None,
                      'trading_mode': None,
                      'funding_fees': None,
                      'amount_precision': 8.0,
                      'price_precision': 7.0,
                      'precision_mode': 1,
                      'precision_mode_price': 1,
                      'contract_size': 1,
                      'orders': [],
                      'has_open_orders': False}
    trade = Trade(pair='XRP/BTC', stake_amount=0.001, amount=100.0, amount_requested=101.0,
                  fee_open=fee.return_value, fee_close=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), close_date=dt_now() - timedelta(hours=1),
                  open_rate=0.123, close_rate=0.125, enter_tag='buys_signal_001',
                  exchange='binance', precision_mode=2, precision_mode_price=1,
                  amount_precision=7.0, price_precision=8.0, contract_size=1)
    result = trade.to_json()
    assert isinstance(result, dict)
    assert result == {'trade_id': None,
                      'pair': 'XRP/BTC',
                      'base_currency': 'XRP',
                      'quote_currency': 'BTC',
                      'open_date': trade.open_date.strftime(DATETIME_PRINT_FORMAT),
                      'open_timestamp': int(trade.open_date.timestamp() * 1000),
                      'open_fill_date': None,
                      'open_fill_timestamp': None,
                      'close_date': trade.close_date.strftime(DATETIME_PRINT_FORMAT),
                      'close_timestamp': int(trade.close_date.timestamp() * 1000),
                      'open_rate': 0.123,
                      'close_rate': 0.125,
                      'amount': 100.0,
                      'amount_requested': 101.0,
                      'stake_amount': 0.001,
                      'max_stake_amount': None,
                      'trade_duration': 60,
                      'trade_duration_s': 3600,
                      'stop_loss_abs': None,
                      'stop_loss_pct': None,
                      'stop_loss_ratio': None,
                      'stoploss_last_update': None,
                      'stoploss_last_update_timestamp': None,
                      'initial_stop_loss_abs': None,
                      'initial_stop_loss_pct': None,
                      'initial_stop_loss_ratio': None,
                      'realized_profit': 0.0,
                      'realized_profit_ratio': None,
                      'close_profit': None,
                      'close_profit_pct': None,
                      'close_profit_abs': None,
                      'profit_ratio': None,
                      'profit_pct': None,
                      'profit_abs': None,
                      'close_rate_requested': None,
                      'fee_close': 0.0025,
                      'fee_close_cost': None,
                      'fee_close_currency': None,
                      'fee_open': 0.0025,
                      'fee_open_cost': None,
                      'fee_open_currency': None,
                      'is_open': None,
                      'max_rate': None,
                      'min_rate': None,
                      'open_rate_requested': None,
                      'open_trade_value': 12.33075,
                      'exit_reason': None,
                      'exit_order_status': None,
                      'strategy': None,
                      'enter_tag': 'buys_signal_001',
                      'timeframe': None,
                      'exchange': 'binance',
                      'leverage': None,
                      'interest_rate': None,
                      'liquidation_price': None,
                      'is_short': None,
                      'trading_mode': None,
                      'funding_fees': None,
                      'amount_precision': 7.0,
                      'price_precision': 8.0,
                      'precision_mode': 2,
                      'precision_mode_price': 1,
                      'contract_size': 1,
                      'orders': [],
                      'has_open_orders': False}


def test_stoploss_reinitialization(default_conf: Dict[str, Any], fee: Any) -> None:
    init_db(default_conf['db_url'])
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, fee_open=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), amount=30.0, fee_close=fee.return_value,
                  exchange='binance', open_rate=1, max_rate=1)
    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    Trade.session.add(trade)
    Trade.commit()
    Trade.stoploss_reinitialization(0.06)
    trades: List[Trade] = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.94
    assert trade_adj.stop_loss_pct == -0.06
    assert trade_adj.initial_stop_loss == 0.94
    assert trade_adj.initial_stop_loss_pct == -0.06
    Trade.stoploss_reinitialization(0.04)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.96
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04
    trade.adjust_stop_loss(1.02, 0.04)
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.initial_stop_loss == 0.96
    Trade.stoploss_reinitialization(0.04)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04


def test_stoploss_reinitialization_leverage(default_conf: Dict[str, Any], fee: Any) -> None:
    init_db(default_conf['db_url'])
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, fee_open=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), amount=30.0, fee_close=fee.return_value,
                  exchange='binance', open_rate=1, max_rate=1, leverage=5.0)
    trade.adjust_stop_loss(trade.open_rate, 0.1, True)
    assert trade.stop_loss == 0.98
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 0.98
    assert trade.initial_stop_loss_pct == -0.1
    Trade.session.add(trade)
    Trade.commit()
    Trade.stoploss_reinitialization(0.15)
    trades: List[Trade] = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.97
    assert trade_adj.stop_loss_pct == -0.15
    assert trade_adj.initial_stop_loss == 0.97
    assert trade_adj.initial_stop_loss_pct == -0.15
    Trade.stoploss_reinitialization(0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.99
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 0.99
    assert trade_adj.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(1.02, 0.05)
    assert trade_adj.stop_loss == 1.0098
    assert trade_adj.initial_stop_loss == 0.99
    Trade.stoploss_reinitialization(0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.0098
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 0.99
    assert trade_adj.initial_stop_loss_pct == -0.05


def test_stoploss_reinitialization_short(default_conf: Dict[str, Any], fee: Any) -> None:
    init_db(default_conf['db_url'])
    trade = Trade(pair='ADA/USDT', stake_amount=0.001, fee_open=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), amount=10, fee_close=fee.return_value,
                  exchange='binance', open_rate=1, max_rate=1, is_short=True, leverage=5.0)
    trade.adjust_stop_loss(trade.open_rate, -0.1, True)
    assert trade.stop_loss == 1.02
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.02
    assert trade.initial_stop_loss_pct == -0.1
    Trade.session.add(trade)
    Trade.commit()
    Trade.stoploss_reinitialization(-0.15)
    trades: List[Trade] = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.03
    assert trade_adj.stop_loss_pct == -0.15
    assert trade_adj.initial_stop_loss == 1.03
    assert trade_adj.initial_stop_loss_pct == -0.15
    Trade.stoploss_reinitialization(-0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.01
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 1.01
    assert trade_adj.initial_stop_loss_pct == -0.05
    trade.adjust_stop_loss(0.98, -0.05)
    assert trade_adj.stop_loss == 0.9898
    assert trade_adj.initial_stop_loss == 1.01
    Trade.stoploss_reinitialization(-0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.9898
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 1.01
    assert trade_adj.initial_stop_loss_pct == -0.05
    trade_adj.set_liquidation_price(0.985)
    trade.adjust_stop_loss(0.9799, -0.05)
    assert trade_adj.stop_loss == 0.989699
    assert trade_adj.liquidation_price == 0.985


def test_update_fee(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, fee_open=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), amount=30.0,
                  fee_close=fee.return_value, exchange='binance', open_rate=1, max_rate=1)
    fee_cost: float = 0.15
    fee_currency: str = 'BTC'
    fee_rate: float = 0.0075
    assert trade.fee_open_currency is None
    assert not trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    trade.update_fee(fee_cost, fee_currency, fee_rate, 'buy')
    assert trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert trade.fee_open_currency == fee_currency
    assert trade.fee_open_cost == fee_cost
    assert trade.fee_open == fee_rate
    assert trade.fee_close == fee_rate
    assert trade.fee_close_currency is None
    assert trade.fee_close_cost is None
    fee_rate = 0.0076
    trade.update_fee(fee_cost, fee_currency, fee_rate, 'sell')
    assert trade.fee_updated('buy')
    assert trade.fee_updated('sell')
    assert trade.fee_close == 0.0076
    assert trade.fee_close_cost == fee_cost
    assert trade.fee_close == fee_rate


def test_fee_updated(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=30.0, fee_open=fee.return_value,
                  open_date=dt_now() - timedelta(hours=2), amount=30.0,
                  fee_close=fee.return_value, exchange='binance', open_rate=1, max_rate=1)
    assert trade.fee_open_currency is None
    assert not trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert not trade.fee_updated('asdf')
    trade.update_fee(0.15, 'BTC', 0.0075, 'buy')
    assert trade.fee_updated('buy')
    assert not trade.fee_updated('sell')
    assert trade.fee_open_currency is not None
    assert trade.fee_close_currency is None
    trade.update_fee(0.15, 'ABC', 0.0075, 'sell')
    assert trade.fee_updated('buy')
    assert trade.fee_updated('sell')
    assert not trade.fee_updated('asfd')


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short,result', [(True, -0.006739127), (False, 0.000739127), (None, -0.005429127)])
@pytest.mark.parametrize('use_db', [True, False])
def test_total_open_trades_stakes(fee: Any, is_short: Optional[bool], use_db: bool) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    res: float = Trade.total_open_trades_stakes()
    assert res == 0
    create_mock_trades(fee, is_short, use_db)
    res = Trade.total_open_trades_stakes()
    assert res == 0.004
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short,result', [(True, -0.006739127), (False, 0.000739127), (None, -0.005429127)])
@pytest.mark.parametrize('use_db', [True, False])
def test_get_total_closed_profit(fee: Any, use_db: bool, is_short: Optional[bool],
                                 result: float) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    res: float = Trade.get_total_closed_profit()
    assert res == 0
    create_mock_trades(fee, is_short, use_db)
    res = Trade.get_total_closed_profit()
    assert pytest.approx(res) == result
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [True, False])
def test_get_trades_proxy(fee: Any, use_db: bool, is_short: bool) -> None:
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades(fee, is_short, use_db)
    trades: List[Trade] = Trade.get_trades_proxy()
    assert len(trades) == 6
    assert isinstance(trades[0], Trade)
    trades = Trade.get_trades_proxy(is_open=True)
    assert len(trades) == 4
    assert trades[0].is_open
    trades = Trade.get_trades_proxy(is_open=False)
    assert len(trades) == 2
    opendate: datetime = datetime.now(tz=timezone.utc) - timedelta(minutes=15)
    assert len(Trade.get_trades_proxy(open_date=opendate)) == 3
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [True, False])
def test_get_trades__query(fee: Any, is_short: bool) -> None:
    query = Trade.get_trades_query([])
    query1 = Trade.get_trades_query([], include_orders=False)
    assert query._with_options == ()
    assert query1._with_options != ()
    create_mock_trades(fee, is_short)
    query = Trade.get_trades_query([])
    query1 = Trade.get_trades_query([], include_orders=False)
    assert query._with_options == ()
    assert query1._with_options != ()


def test_get_trades_backtest() -> None:
    Trade.use_db = False
    with pytest.raises(NotImplementedError, match='`Trade.get_trades\\(\\)` not .*'):
        Trade.get_trades([])
    Trade.use_db = True


@pytest.mark.usefixtures('init_persistence')
def test_get_overall_performance(fee: Any) -> None:
    create_mock_trades(fee, False)
    res: List[Dict[str, Any]] = Trade.get_overall_performance()
    assert len(res) == 2
    assert 'pair' in res[0]
    assert 'profit' in res[0]
    assert 'count' in res[0]


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short,pair,profit',
                         [(True, 'XRP/BTC', -0.00018780487), (False, 'ETC/BTC', 3.860975e-05), (None, 'XRP/BTC', 2.5203252e-05)])
def test_get_best_pair(fee: Any, is_short: Optional[bool], pair: str, profit: float) -> None:
    res: Optional[Any] = Trade.get_best_pair()
    assert res is None
    create_mock_trades(fee, is_short)
    res = Trade.get_best_pair()
    assert len(res) == 4
    assert res[0] == pair
    assert pytest.approx(res[1]) == profit


@pytest.mark.usefixtures('init_persistence')
def test_get_best_pair_lev(fee: Any) -> None:
    res = Trade.get_best_pair()
    assert res is None
    create_mock_trades_with_leverage(fee)
    res = Trade.get_best_pair()
    assert len(res) == 4
    assert res[0] == 'ETC/BTC'
    assert pytest.approx(res[1]) == 3.860975e-05


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [True, False])
def test_get_canceled_exit_order_count(fee: Any, is_short: bool) -> None:
    create_mock_trades(fee, is_short=is_short)
    trade = Trade.get_trades([Trade.pair == 'ETC/BTC']).first()
    assert trade.get_canceled_exit_order_count() == 0
    assert trade.canceled_exit_order_count == 0
    trade.orders[-1].status = 'canceled'
    assert trade.get_canceled_exit_order_count() == 1
    assert trade.canceled_exit_order_count == 1


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [True, False])
def test_fully_canceled_entry_order_count(fee: Any, is_short: bool) -> None:
    create_mock_trades(fee, is_short=is_short)
    trade = Trade.get_trades([Trade.pair == 'ETC/BTC']).first()
    assert trade.fully_canceled_entry_order_count == 0
    trade.orders[0].status = 'canceled'
    trade.orders[0].filled = 0
    assert trade.fully_canceled_entry_order_count == 1


@pytest.mark.usefixtures('init_persistence')
def test_update_order_from_ccxt(caplog: Any, time_machine: Any) -> None:
    start: datetime = datetime(2023, 1, 1, 4, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)
    o = Order.parse_from_ccxt_object({'id': '1234'}, 'ADA/USDT', 'buy', 20.01, 1234.6)
    assert isinstance(o, Order)
    assert o.ft_pair == 'ADA/USDT'
    assert o.ft_order_side == 'buy'
    assert o.order_id == '1234'
    assert o.ft_price == 1234.6
    assert o.ft_amount == 20.01
    assert o.ft_is_open
    ccxt_order: Dict[str, Any] = {'id': '1234', 'side': 'buy', 'symbol': 'ADA/USDT', 'type': 'limit',
                                  'price': 1234.5, 'amount': 20.0, 'filled': 9, 'remaining': 11, 'status': 'open',
                                  'timestamp': 1599394315123}
    o = Order.parse_from_ccxt_object(ccxt_order, 'ADA/USDT', 'buy', 20.01, 1234.6)
    assert isinstance(o, Order)
    assert o.ft_pair == 'ADA/USDT'
    assert o.ft_order_side == 'buy'
    assert o.order_id == '1234'
    assert o.order_type == 'limit'
    assert o.price == 1234.5
    assert o.ft_price == 1234.6
    assert o.ft_amount == 20.01
    assert o.filled == 9
    assert o.remaining == 11
    assert o.order_date is not None
    assert o.ft_is_open
    assert o.order_filled_date is None
    ccxt_order.update({'filled': None, 'remaining': 20.0, 'status': 'canceled'})
    o.update_from_ccxt_object(ccxt_order)
    ccxt_order.update({'filled': 20.0, 'remaining': 0.0, 'status': 'closed'})
    o.update_from_ccxt_object(ccxt_order)
    assert o.filled == 20.0
    assert o.remaining == 0.0
    assert not o.ft_is_open
    assert o.order_filled_date == start
    time_machine.move_to(start + timedelta(hours=1), tick=False)
    ccxt_order.update({'id': 'somethingelse'})
    with pytest.raises(DependencyException, match="Order-id's don't match"):
        o.update_from_ccxt_object(ccxt_order)
    message: str = 'aaaa is not a valid response object.'
    assert not log_has(message, caplog)
    Order.update_orders([o], 'aaaa')
    assert log_has(message, caplog)
    Order.update_orders([o], {'id': '1234'})
    assert o.order_filled_date == start
    ccxt_order.update({'id': '1234'})
    Order.update_orders([o], ccxt_order)
    assert o.order_filled_date == start


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [True, False])
def test_select_order(fee: Any, is_short: bool) -> None:
    create_mock_trades(fee, is_short)
    trades: List[Trade] = Trade.get_trades().all()
    order = trades[0].select_order(trades[0].entry_side, True)
    assert order is not None
    order = trades[0].select_order(trades[0].entry_side, False)
    assert order is None
    order = trades[0].select_order(trades[0].exit_side, None)
    assert order is None
    order = trades[1].select_order(trades[1].entry_side, True)
    assert order is None
    order = trades[1].select_order(trades[1].entry_side, False)
    assert order is not None
    order = trades[1].select_order(trades[1].entry_side, None)
    assert order is not None
    order = trades[1].select_order(trades[1].exit_side, True)
    assert order is None
    order = trades[1].select_order(trades[1].exit_side, False)
    assert order is not None
    order = trades[3].select_order(trades[3].entry_side, True)
    assert order is not None
    order = trades[3].select_order(trades[3].entry_side, False)
    assert order is None
    order = trades[4].select_order(trades[4].entry_side, True)
    assert order is None
    order = trades[4].select_order(trades[4].entry_side, False)
    assert order is not None
    trades[4].orders[1].ft_order_side = trades[4].exit_side
    order = trades[4].select_order(trades[4].exit_side, True)
    assert order is not None
    trades[4].orders[1].ft_order_side = 'stoploss'
    order = trades[4].select_order('stoploss', None)
    assert order is not None
    assert order.ft_order_side == 'stoploss'


def test_Trade_object_idem() -> None:
    assert issubclass(Trade, LocalTrade)
    trade_attrs = vars(Trade)
    localtrade_attrs = vars(LocalTrade)
    excludes = ('delete', 'session', 'commit', 'rollback', 'query', 'open_date', 'get_best_pair',
                'get_overall_performance', 'get_total_closed_profit', 'total_open_trades_stakes',
                'get_closed_trades_without_assigned_fees', 'get_open_trades_without_assigned_fees',
                'get_trades', 'get_trades_query', 'get_exit_reason_performance', 'get_enter_tag_performance',
                'get_mix_tag_performance', 'get_trading_volume', 'validate_string_len', 'custom_data')
    EXCLUDES2 = ('bt_trades', 'bt_trades_open', 'bt_trades_open_pp', 'bt_open_open_trade_count', 'bt_total_profit', 'from_json')
    for item in trade_attrs:
        if not item.startswith('_') and item not in excludes:
            assert item in localtrade_attrs
    for item in localtrade_attrs:
        if not item.startswith('__') and item not in EXCLUDES2 and (type(getattr(LocalTrade, item)) not in (property, FunctionType)):
            assert item in trade_attrs


@pytest.mark.usefixtures('init_persistence')
def test_trade_truncates_string_fields(fee: Any) -> None:
    trade = Trade(pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0,
                  open_date=datetime.now(timezone.utc) - timedelta(minutes=20), fee_open=0.001,
                  fee_close=0.001, exchange='binance', leverage=1.0, trading_mode='futures',
                  enter_tag='a' * CUSTOM_TAG_MAX_LENGTH * 2, exit_reason='b' * CUSTOM_TAG_MAX_LENGTH * 2)
    Trade.session.add(trade)
    Trade.commit()
    trade1 = Trade.session.scalars(select(Trade)).first()
    assert trade1.enter_tag == 'a' * CUSTOM_TAG_MAX_LENGTH
    assert trade1.exit_reason == 'b' * CUSTOM_TAG_MAX_LENGTH


def test_recalc_trade_from_orders(fee: Any) -> None:
    o1_amount: float = 100
    o1_rate: float = 1
    o1_cost: float = o1_amount * o1_rate
    o1_fee_cost: float = o1_cost * fee.return_value
    o1_trade_val: float = o1_cost + o1_fee_cost
    trade = Trade(pair='ADA/USDT', stake_amount=o1_cost, open_date=dt_now() - timedelta(hours=2),
                  amount=o1_amount, fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange='binance', open_rate=o1_rate, max_rate=o1_rate, leverage=1)
    assert fee.return_value == 0.0025
    assert trade._calc_open_trade_value(trade.amount, trade.open_rate) == o1_trade_val
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_cost
    assert trade.open_rate == o1_rate
    assert trade.open_trade_value == o1_trade_val
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_cost
    assert trade.open_rate == o1_rate
    assert trade.open_trade_value == o1_trade_val
    trade.update_fee(o1_fee_cost, 'BNB', fee.return_value, 'buy')
    assert len(trade.orders) == 0
    order1 = Order(ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='buy', price=o1_rate, average=o1_rate,
                   filled=o1_amount, remaining=0, cost=o1_amount, order_date=trade.open_date,
                   order_filled_date=trade.open_date)
    trade.orders.append(order1)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    o2_amount: float = 125
    o2_rate: float = 0.9
    o2_cost: float = o2_amount * o2_rate
    o2_fee_cost: float = o2_cost * fee.return_value
    o2_trade_val: float = o2_cost + o2_fee_cost
    order2 = Order(ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='buy', price=o2_rate, average=o2_rate,
                   filled=o2_amount, remaining=0, cost=o2_cost, order_date=dt_now() - timedelta(hours=1),
                   order_filled_date=dt_now() - timedelta(hours=1))
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()
    avg_price: float = (o1_cost + o2_cost) / (o1_amount + o2_amount)
    assert trade.amount == o1_amount + o2_amount
    assert trade.stake_amount == o1_amount + o2_cost
    assert trade.open_rate == avg_price
    assert trade.fee_open_cost == o1_fee_cost + o2_fee_cost
    assert trade.open_trade_value == o1_trade_val + o2_trade_val
    o3_amount: float = 150
    o3_rate: float = 0.85
    o3_cost: float = o3_amount * o3_rate
    o3_fee_cost: float = o3_cost * fee.return_value
    o3_trade_val: float = o3_cost + o3_fee_cost
    order3 = Order(ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='buy', price=o3_rate, average=o3_rate,
                   filled=o3_amount, remaining=0, cost=o3_cost, order_date=dt_now() - timedelta(hours=1),
                   order_filled_date=dt_now() - timedelta(hours=1))
    trade.orders.append(order3)
    trade.recalc_trade_from_orders()
    avg_price = (o1_cost + o2_cost + o3_cost) / (o1_amount + o2_amount + o3_amount)
    assert trade.amount == o1_amount + o2_amount + o3_amount
    assert trade.stake_amount == o1_cost + o2_cost + o3_cost
    assert trade.open_rate == avg_price
    assert pytest.approx(trade.fee_open_cost) == o1_fee_cost + o2_fee_cost + o3_fee_cost
    assert pytest.approx(trade.open_trade_value) == o1_trade_val + o2_trade_val + o3_trade_val
    sell1 = Order(ft_order_side='sell', ft_pair=trade.pair, ft_is_open=False, status='closed',
                  symbol=trade.pair, order_type='market', side='sell', price=avg_price + 0.95,
                  average=avg_price + 0.95, filled=o1_amount + o2_amount + o3_amount, remaining=0,
                  cost=o1_cost + o2_cost + o3_cost, order_date=trade.open_date, order_filled_date=trade.open_date)
    trade.orders.append(sell1)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount + o2_amount + o3_amount
    assert trade.stake_amount == o1_cost + o2_cost + o3_cost
    assert trade.open_rate == avg_price
    assert pytest.approx(trade.fee_open_cost) == o1_fee_cost + o2_fee_cost + o3_fee_cost
    assert pytest.approx(trade.open_trade_value) == o1_trade_val + o2_trade_val + o3_trade_val


@pytest.mark.usefixtures('init_persistence')
def test_recalc_trade_from_orders_kucoin() -> None:
    o1_amount: float = 11511963.86344489
    o2_amount: float = 11750101.774393778
    o3_amount: float = 23262065.63783866
    res: float = o1_amount + o2_amount - o3_amount
    assert res > 0.0
    assert res < 0.1
    o1_rate: float = 2.9901e-05
    o2_rate: float = 2.9295e-05
    o3_rate: float = 2.9822e-05
    o1_cost: float = o1_amount * o1_rate
    trade = Trade(pair='FLOKI/USDT', stake_amount=o1_cost, open_date=dt_now() - timedelta(hours=2),
                  amount=o1_amount, fee_open=0.001, fee_close=0.001, exchange='binance', open_rate=o1_rate,
                  max_rate=o1_rate, leverage=1)
    order1 = Order(ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='buy', price=o1_rate, average=o1_rate,
                   filled=o1_amount, remaining=0, cost=o1_cost, order_date=trade.open_date,
                   order_filled_date=trade.open_date)
    trade.orders.append(order1)
    order2 = Order(ft_order_side='buy', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='buy', price=o2_rate, average=o2_rate,
                   filled=o2_amount, remaining=0, cost=o2_amount * o2_rate, order_date=trade.open_date,
                   order_filled_date=trade.open_date)
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount + o2_amount
    profit = trade.calculate_profit(o3_rate)
    assert profit.profit_abs == pytest.approx(3.90069871)
    assert profit.profit_ratio == pytest.approx(0.00566035)
    order3 = Order(ft_order_side='sell', ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side='sell', price=o3_rate, average=o3_rate,
                   filled=o3_amount, remaining=0, cost=o2_amount * o2_rate, order_date=trade.open_date,
                   order_filled_date=trade.open_date)
    trade.orders.append(order3)
    trade.update_trade(order3)
    assert trade.is_open is False
    assert trade.amount == 8e-09
    assert pytest.approx(trade.close_profit_abs) == 3.90069871
    assert pytest.approx(trade.close_profit) == 0.00566035


@pytest.mark.parametrize('is_short', [True, False])
def test_recalc_trade_from_orders_ignores_bad_orders(fee: Any, is_short: bool) -> None:
    o1_amount: float = 100
    o1_rate: float = 1
    o1_cost: float = o1_amount * o1_rate
    o1_fee_cost: float = o1_cost * fee.return_value
    o1_trade_val: float = o1_cost - o1_fee_cost if is_short else o1_cost + o1_fee_cost
    entry_side: str = 'sell' if is_short else 'buy'
    exit_side: str = 'buy' if is_short else 'sell'
    trade = Trade(pair='ADA/USDT', stake_amount=o1_cost, open_date=dt_now() - timedelta(hours=2),
                  amount=o1_amount, fee_open=fee.return_value, fee_close=fee.return_value,
                  exchange='binance', open_rate=o1_rate, max_rate=o1_rate, is_short=is_short, leverage=1.0)
    trade.update_fee(o1_fee_cost, 'BNB', fee.return_value, entry_side)
    order1 = Order(ft_order_side=entry_side, ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side=entry_side, price=o1_rate, average=o1_rate,
                   filled=o1_amount, remaining=0, cost=o1_amount, order_date=trade.open_date,
                   order_filled_date=trade.open_date)
    trade.orders.append(order1)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1
    order2 = Order(ft_order_side=entry_side, ft_pair=trade.pair, ft_is_open=True, status='open',
                   symbol=trade.pair, order_type='market', side=entry_side, price=o1_rate, average=o1_rate,
                   filled=o1_amount, remaining=0, cost=o1_cost, order_date=dt_now() - timedelta(hours=1),
                   order_filled_date=dt_now() - timedelta(hours=1))
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1
    order3 = Order(ft_order_side=entry_side, ft_pair=trade.pair, ft_is_open=False, status='cancelled',
                   symbol=trade.pair, order_type='market', side=entry_side, price=1, average=2,
                   filled=0, remaining=4, cost=5, order_date=dt_now() - timedelta(hours=1),
                   order_filled_date=dt_now() - timedelta(hours=1))
    trade.orders.append(order3)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1
    order4 = Order(ft_order_side=entry_side, ft_pair=trade.pair, ft_is_open=False, status='closed',
                   symbol=trade.pair, order_type='market', side=entry_side, price=o1_rate, average=o1_rate,
                   filled=o1_amount, remaining=0, cost=o1_cost, order_date=dt_now() - timedelta(hours=1),
                   order_filled_date=dt_now() - timedelta(hours=1))
    trade.orders.append(order4)
    trade.recalc_trade_from_orders()
    assert trade.amount == 2 * o1_amount
    assert trade.stake_amount == 2 * o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == 2 * o1_trade_val
    assert trade.nr_of_successful_entries == 2
    sell1 = Order(ft_order_side=exit_side, ft_pair=trade.pair, ft_is_open=False, status='closed',
                  symbol=trade.pair, order_type='market', side=exit_side, price=4, average=3,
                  filled=o1_amount, remaining=1, cost=5, order_date=trade.open_date, order_filled_date=trade.open_date)
    trade.orders.append(sell1)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 2
    order_noavg = Order(ft_order_side=entry_side, ft_pair=trade.pair, ft_is_open=False, status='closed',
                        symbol=trade.pair, order_type='market', side=entry_side, price=o1_rate, average=None,
                        filled=o1_amount, remaining=0, cost=o1_amount, order_date=trade.open_date,
                        order_filled_date=trade.open_date)
    trade.orders.append(order_noavg)
    trade.recalc_trade_from_orders()
    assert trade.amount == 2 * o1_amount
    assert trade.stake_amount == 2 * o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == 2 * o1_trade_val
    assert trade.nr_of_successful_entries == 3


@pytest.mark.usefixtures('init_persistence')
def test_select_filled_orders(fee: Any) -> None:
    create_mock_trades(fee)
    trades: List[Trade] = Trade.get_trades().all()
    orders = trades[0].select_filled_orders('buy')
    assert isinstance(orders, list)
    assert len(orders) == 0
    orders = trades[0].select_filled_orders('sell')
    assert orders is not None
    assert len(orders) == 0
    orders = trades[1].select_filled_orders('buy')
    assert isinstance(orders, list)
    assert len(orders) == 1
    order = orders[0]
    assert order.amount > 0
    assert order.filled > 0
    assert order.side == 'buy'
    assert order.ft_order_side == 'buy'
    assert order.status == 'closed'
    orders = trades[1].select_filled_orders('sell')
    assert isinstance(orders, list)
    assert len(orders) == 1


@pytest.mark.usefixtures('init_persistence')
def test_select_filled_orders_usdt(fee: Any) -> None:
    create_mock_trades_usdt(fee)
    trades: List[Trade] = Trade.get_trades().all()
    orders = trades[0].select_filled_orders('buy')
    assert isinstance(orders, list)
    assert len(orders) == 1
    assert orders[0].amount == 2.0
    assert orders[0].filled == 2.0
    assert orders[0].side == 'buy'
    assert orders[0].price == 10.0
    assert orders[0].stake_amount == 20
    assert orders[0].stake_amount_filled == 20
    orders = trades[3].select_filled_orders('buy')
    assert isinstance(orders, list)
    assert len(orders) == 0
    orders = trades[3].select_filled_or_open_orders()
    assert isinstance(orders, list)
    assert len(orders) == 1
    assert orders[0].price == 2.0
    assert orders[0].amount == 10
    assert orders[0].filled == 0
    assert orders[0].stake_amount == 20
    assert orders[0].stake_amount_filled == 0


@pytest.mark.usefixtures('init_persistence')
def test_order_to_ccxt(limit_buy_order_open: Dict[str, Any], limit_sell_order_usdt_open: Dict[str, Any]) -> None:
    order = Order.parse_from_ccxt_object(limit_buy_order_open, 'mocked', 'buy')
    order.ft_trade_id = 1
    order.session.add(order)
    Order.session.commit()
    order_resp = Order.order_by_id(limit_buy_order_open['id'])
    assert order_resp
    raw_order = order_resp.to_ccxt_object()
    del raw_order['fee']
    del raw_order['datetime']
    del raw_order['info']
    assert raw_order.get('stopPrice') is None
    raw_order.pop('stopPrice', None)
    del limit_buy_order_open['datetime']
    assert raw_order == limit_buy_order_open
    order1 = Order.parse_from_ccxt_object(limit_sell_order_usdt_open, 'mocked', 'sell')
    order1.ft_order_side = 'stoploss'
    order1.stop_price = order1.price * 0.9
    order1.ft_trade_id = 1
    order1.session.add(order1)
    Order.session.commit()
    order_resp1 = Order.order_by_id(limit_sell_order_usdt_open['id'])
    raw_order1 = order_resp1.to_ccxt_object()
    assert raw_order1.get('stopPrice') is not None


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('data', [
    {'orders': [(('buy', 100, 10), (100.0, 10.0, 1000.0, 0.0, None, None)),
                (('buy', 100, 15), (200.0, 12.5, 2500.0, 0.0, None, None)),
                (('sell', 50, 12), (150.0, 12.5, 1875.0, -25.0, -25.0, -0.01)),
                (('sell', 100, 20), (50.0, 12.5, 625.0, 725.0, 750.0, 0.29)),
                (('sell', 50, 5), (50.0, 12.5, 625.0, 350.0, -375.0, 0.14))],
     'end_profit': 350.0, 'end_profit_ratio': 0.14, 'fee': 0.0},
    {'orders': [(('buy', 100, 10), (100.0, 10.0, 1000.0, 0.0, None, None)),
                (('buy', 100, 15), (200.0, 12.5, 2500.0, 0.0, None, None)),
                (('sell', 50, 12), (150.0, 12.5, 1875.0, -28.0625, -28.0625, -0.011197)),
                (('sell', 100, 20), (50.0, 12.5, 625.0, 713.8125, 741.875, 0.2848129)),
                (('sell', 50, 5), (50.0, 12.5, 625.0, 336.625, -377.1875, 0.1343142))],
     'end_profit': 336.625, 'end_profit_ratio': 0.1343142, 'fee': 0.0025},
    {'orders': [(('buy', 100, 3), (100.0, 3.0, 300.0, 0.0, None, None)),
                (('buy', 100, 7), (200.0, 5.0, 1000.0, 0.0, None, None)),
                (('sell', 100, 11), (100.0, 5.0, 500.0, 596.0, 596.0, 0.5945137)),
                (('buy', 150, 15), (250.0, 11.0, 2750.0, 596.0, 596.0, 0.5945137)),
                (('sell', 100, 19), (150.0, 11.0, 1650.0, 1388.5, 792.5, 0.4261653)),
                (('sell', 150, 23), (150.0, 11.0, 1650.0, 3175.75, 1787.25, 0.974717))],
     'end_profit': 3175.75, 'end_profit_ratio': 0.974717, 'fee': 0.0025},
    {'orders': [(('buy', 100, 3), (100.0, 3.0, 300.0, 0.0, None, None)),
                (('buy', 100, 7), (200.0, 5.0, 1000.0, 0.0, None, None)),
                (('sell', 100, 11), (100.0, 5.0, 500.0, 600.0, 600.0, 0.6)),
                (('buy', 150, 15), (250.0, 11.0, 2750.0, 600.0, 600.0, 0.6)),
                (('sell', 100, 19), (150.0, 11.0, 1650.0, 1400.0, 800.0, 0.43076923)),
                (('sell', 150, 23), (150.0, 11.0, 1650.0, 3200.0, 1800.0, 0.98461538))],
     'end_profit': 3200.0, 'end_profit_ratio': 0.98461538, 'fee': 0.0},
    {'orders': [(('buy', 100, 8), (100.0, 8.0, 800.0, 0.0, None, None)),
                (('buy', 100, 9), (200.0, 8.5, 1700.0, 0.0, None, None)),
                (('sell', 100, 10), (100.0, 8.5, 850.0, 150.0, 150.0, 0.08823529)),
                (('buy', 150, 11), (250.0, 10, 2500.0, 150.0, 150.0, 0.08823529)),
                (('sell', 100, 12), (150.0, 10.0, 1500.0, 350.0, 200.0, 0.1044776)),
                (('sell', 150, 14), (150.0, 10.0, 1500.0, 950.0, 600.0, 0.283582))],
     'end_profit': 950.0, 'end_profit_ratio': 0.283582, 'fee': 0.0}
])
def test_recalc_trade_from_orders_dca(data: Dict[str, Any]) -> None:
    pair: str = 'ETH/USDT'
    trade = Trade(id=2, pair=pair, stake_amount=1000, open_rate=data['orders'][0][0][2],
                  amount=data['orders'][0][0][1], is_open=True, open_date=dt_now(),
                  fee_open=data['fee'], fee_close=data['fee'], exchange='binance', is_short=False,
                  leverage=1.0, trading_mode=TradingMode.SPOT, price_precision=0.001, precision_mode_price=TICK_SIZE)
    Trade.session.add(trade)
    for idx, (order, result) in enumerate(data['orders']):
        amount = order[1]
        price = order[2]
        order_obj = Order(ft_order_side=order[0], ft_pair=trade.pair, order_id=f'order_{order[0]}_{idx}',
                          ft_is_open=False, ft_amount=amount, ft_price=price, status='closed', symbol=trade.pair,
                          order_type='market', side=order[0], price=price, average=price, filled=amount,
                          remaining=0, cost=amount * price, order_date=dt_now() - timedelta(hours=10 + idx),
                          order_filled_date=dt_now() - timedelta(hours=10 + idx))
        trade.orders.append(order_obj)
        trade.recalc_trade_from_orders()
        Trade.commit()
        orders1: List[Order] = list(Order.session.scalars(select(Order)).all())
        assert orders1
        assert len(orders1) == idx + 1
        trade = Trade.session.scalars(select(Trade)).first()
        assert trade
        assert len(trade.orders) == idx + 1
        if idx < len(data['orders']) - 1:
            assert trade.is_open is True
        assert not trade.has_open_orders
        assert trade.amount == result[0]
        assert trade.open_rate == result[1]
        assert trade.stake_amount == result[2]
        assert pytest.approx(trade.realized_profit) == result[3]
        assert pytest.approx(trade.close_profit_abs) == result[4]
        assert pytest.approx(trade.close_profit) == result[5]
    trade.close(price)
    assert pytest.approx(trade.close_profit_abs) == data['end_profit']
    assert pytest.approx(trade.close_profit) == data['end_profit_ratio']
    assert not trade.is_open
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
