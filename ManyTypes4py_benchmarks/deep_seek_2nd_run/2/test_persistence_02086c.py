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
from tests.conftest import create_mock_trades, create_mock_trades_usdt, create_mock_trades_with_leverage, log_has, log_has_re

spot: TradingMode = TradingMode.SPOT
margin: TradingMode = TradingMode.MARGIN
futures: TradingMode = TradingMode.FUTURES

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_enter_exit_side(fee: Any, is_short: bool) -> None:
    entry_side: str = 'sell' if is_short else 'buy'
    exit_side: str = 'buy' if is_short else 'sell'
    trade: Trade = Trade(
        id=2, pair='ADA/USDT', stake_amount=0.001, open_rate=0.01, amount=5, is_open=True,
        open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
        exchange='binance', is_short=is_short, leverage=2.0, trading_mode=margin
    )
    assert trade.entry_side == entry_side
    assert trade.exit_side == exit_side
    assert trade.trade_direction == 'short' if is_short else 'long'

@pytest.mark.usefixtures('init_persistence')
def test_set_stop_loss_liquidation(fee: Any) -> None:
    trade: Trade = Trade(
        id=2, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
        open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
        exchange='binance', is_short=False, leverage=2.0, trading_mode=margin
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

@pytest.mark.parametrize('exchange,is_short,lev,minutes,rate,interest,trading_mode', [
    ('binance', False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin),
    ('binance', True, 3, 10, 0.0005, 0.000625, margin),
    ('binance', False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin),
    ('binance', True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin),
    ('binance', False, 3, 295, 0.00025, round(0.0020833333333333333, 8), margin),
    ('binance', True, 3, 295, 0.00025, round(0.0015624999999999999, 8), margin),
    ('binance', False, 5, 295, 0.0005, 0.005, margin),
    ('binance', True, 5, 295, 0.0005, round(0.0031249999999999997, 8), margin),
    ('binance', False, 1, 295, 0.0005, 0.0, spot),
    ('binance', True, 1, 295, 0.0005, 0.003125, margin),
    ('binance', False, 3, 10, 0.0005, 0.0, futures),
    ('binance', True, 3, 295, 0.0005, 0.0, futures),
    ('binance', False, 5, 295, 0.0005, 0.0, futures),
    ('binance', True, 5, 295, 0.0005, 0.0, futures),
    ('binance', False, 1, 295, 0.0005, 0.0, futures),
    ('binance', True, 1, 295, 0.0005, 0.0, futures),
    ('kraken', False, 3, 10, 0.0005, 0.04, margin),
    ('kraken', True, 3, 10, 0.0005, 0.03, margin),
    ('kraken', False, 3, 295, 0.0005, 0.06, margin),
    ('kraken', True, 3, 295, 0.0005, 0.045, margin),
    ('kraken', False, 3, 295, 0.00025, 0.03, margin),
    ('kraken', True, 3, 295, 0.00025, 0.0225, margin),
    ('kraken', False, 5, 295, 0.0005, round(0.07200000000000001, 8), margin),
    ('kraken', True, 5, 295, 0.0005, 0.045, margin),
    ('kraken', False, 1, 295, 0.0005, 0.0, spot),
    ('kraken', True, 1, 295, 0.0005, 0.045, margin)
])
@pytest.mark.usefixtures('init_persistence')
def test_interest(fee: Any, exchange: str, is_short: bool, lev: float, minutes: int, rate: float, interest: float, trading_mode: TradingMode) -> None:
    trade: Trade = Trade(
        pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=minutes),
        fee_open=fee.return_value, fee_close=fee.return_value, exchange=exchange,
        leverage=lev, interest_rate=rate, is_short=is_short, trading_mode=trading_mode
    )
    assert round(float(trade.calculate_interest()), 8) == interest

@pytest.mark.parametrize('is_short,lev,borrowed,trading_mode', [
    (False, 1.0, 0.0, spot),
    (True, 1.0, 30.0, margin),
    (False, 3.0, 40.0, margin),
    (True, 3.0, 30.0, margin)
])
@pytest.mark.usefixtures('init_persistence')
def test_borrowed(fee: Any, is_short: bool, lev: float, borrowed: float, trading_mode: TradingMode) -> None:
    trade: Trade = Trade(
        id=2, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
        open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
        exchange='binance', is_short=is_short, leverage=lev, trading_mode=trading_mode
    )
    assert trade.borrowed == borrowed

@pytest.mark.parametrize('is_short,open_rate,close_rate,lev,profit,trading_mode', [
    (False, 2.0, 2.2, 1.0, 0.09451372, spot),
    (True, 2.2, 2.0, 3.0, 0.25894253, margin)
])
@pytest.mark.usefixtures('init_persistence')
def test_update_limit_order(fee: Any, caplog: Any, limit_buy_order_usdt: Dict, limit_sell_order_usdt: Dict, time_machine: Any, is_short: bool, open_rate: float, close_rate: float, lev: float, profit: float, trading_mode: TradingMode) -> None:
    time_machine.move_to('2022-03-31 20:45:00 +00:00', tick=False)
    enter_order: Dict = limit_sell_order_usdt if is_short else limit_buy_order_usdt
    exit_order: Dict = limit_buy_order_usdt if is_short else limit_sell_order_usdt
    entry_side: str = 'sell' if is_short else 'buy'
    exit_side: str = 'buy' if is_short else 'sell'
    trade: Trade = Trade(
        id=2, pair='ADA/USDT', stake_amount=60.0, open_rate=open_rate, amount=30.0, is_open=True,
        open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value,
        exchange='binance', is_short=is_short, interest_rate=0.0005, leverage=lev,
        trading_mode=trading_mode
    )
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None
    oobj: Order = Order.parse_from_ccxt_object(enter_order, 'ADA/USDT', entry_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == open_rate
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        f'LIMIT_{entry_side.upper()} has been fulfilled for Trade\\(id=2, pair=ADA/USDT, amount=30.00000000, is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, open_since=.*\\).',
        caplog
    )
    caplog.clear()
    time_machine.move_to('2022-03-31 21:45:05 +00:00', tick=False)
    oobj = Order.parse_from_ccxt_object(exit_order, 'ADA/USDT', exit_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == close_rate
    assert pytest.approx(trade.close_profit) == profit
    assert trade.close_date is not None
    assert log_has_re(
        f'LIMIT_{exit_side.upper()} has been fulfilled for Trade\\(id=2, pair=ADA/USDT, amount=30.00000000, is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, open_since=.*\\).',
        caplog
    )
    caplog.clear()

@pytest.mark.usefixtures('init_persistence')
def test_update_market_order(market_buy_order_usdt: Dict, market_sell_order_usdt: Dict, fee: Any, caplog: Any) -> None:
    trade: Trade = Trade(
        id=1, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True,
        fee_open=fee.return_value, fee_close=fee.return_value, open_date=dt_now(),
        exchange='binance', trading_mode=margin, leverage=1.0
    )
    oobj: Order = Order.parse_from_ccxt_object(market_buy_order_usdt, 'ADA/USDT', 'buy')
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == 2.0
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        'MARKET_BUY has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, open_rate=2.00000000, open_since=.*\\).',
        caplog
    )
    caplog.clear()
    trade.is_open = True
    oobj = Order.parse_from_ccxt_object(market_sell_order_usdt, 'ADA/USDT', 'sell')
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == 2.2
    assert pytest.approx(trade.close_profit) == 0.094513715710723
    assert trade.close_date is not None
    assert log_has_re(
        'MARKET_SELL has been fulfilled for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, open_rate=2.00000000, open_since=.*\\).',
        caplog
    )

@pytest.mark.parametrize('exchange,is_short,lev,open_value,close_value,profit,profit_ratio,trading_mode,funding_fees', [
   