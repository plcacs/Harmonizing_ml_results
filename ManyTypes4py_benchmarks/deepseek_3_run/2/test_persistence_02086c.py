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
spot, margin, futures = (TradingMode.SPOT, TradingMode.MARGIN, TradingMode.FUTURES)

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_enter_exit_side(fee: Any, is_short: bool) -> None:
    entry_side, exit_side = ('sell', 'buy') if is_short else ('buy', 'sell')
    trade = Trade(id=2, pair='ADA/USDT', stake_amount=0.001, open_rate=0.01, amount=5, is_open=True, open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance', is_short=is_short, leverage=2.0, trading_mode=margin)
    assert trade.entry_side == entry_side
    assert trade.exit_side == exit_side
    assert trade.trade_direction == 'short' if is_short else 'long'

@pytest.mark.usefixtures('init_persistence')
def test_set_stop_loss_liquidation(fee: Any) -> None:
    trade = Trade(id=2, pair='ADA/USDT', stake_amount=60.0, open_rate=2.0, amount=30.0, is_open=True, open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance', is_short=False, leverage=2.0, trading_mode=margin)
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

@pytest.mark.parametrize('exchange,is_short,lev,minutes,rate,interest,trading_mode', [('binance', False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin), ('binance', True, 3, 10, 0.0005, 0.000625, margin), ('binance', False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin), ('binance', True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin), ('binance', False, 3, 295, 0.00025, round(0.0020833333333333333, 8), margin), ('binance', True, 3, 295, 0.00025, round(0.0015624999999999999, 8), margin), ('binance', False, 5, 295, 0.0005, 0.005, margin), ('binance', True, 5, 295, 0.0005, round(0.0031249999999999997, 8), margin), ('binance', False, 1, 295, 0.0005, 0.0, spot), ('binance', True, 1, 295, 0.0005, 0.003125, margin), ('binance', False, 3, 10, 0.0005, 0.0, futures), ('binance', True, 3, 295, 0.0005, 0.0, futures), ('binance', False, 5, 295, 0.0005, 0.0, futures), ('binance', True, 5, 295, 0.0005, 0.0, futures), ('binance', False, 1, 295, 0.0005, 0.0, futures), ('binance', True, 1, 295, 0.0005, 0.0, futures), ('kraken', False, 3, 10, 0.0005, 0.04, margin), ('kraken', True, 3, 10, 0.0005, 0.03, margin), ('kraken', False, 3, 295, 0.0005, 0.06, margin), ('kraken', True, 3, 295, 0.0005, 0.045, margin), ('kraken', False, 3, 295, 0.00025, 0.03, margin), ('kraken', True, 3, 295, 0.00025, 0.0225, margin), ('kraken', False, 5, 295, 0.0005, round(0.07200000000000001, 8), margin), ('kraken', True, 5, 295, 0.0005, 0.045, margin), ('kraken', False, 1, 295, 0.0005, 0.0, spot), ('kraken', True, 1, 295, 0.0005, 0.045, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_interest(fee: Any, exchange: str, is_short: bool, lev: float, minutes: int, rate: float, interest: float, trading_mode: TradingMode) -> None:
    """
    10min, 5hr limit trade on Binance/Kraken at 3x,5x leverage
    fee: 0.25 % quote
    interest_rate: 0.05 % per 4 hrs
    open_rate: 2.00 quote
    close_rate: 2.20 quote
    amount: = 30.0 crypto
    stake_amount
        3x, -3x: 20.0  quote
        5x, -5x: 12.0  quote
    borrowed
      10min
         3x: 40 quote
        -3x: 30 crypto
         5x: 48 quote
        -5x: 30 crypto
         1x: 0
        -1x: 30 crypto
    hours: 1/6 (10 minutes)
    time-periods:
        10min
            kraken: (1 + 1) 4hr_periods = 2 4hr_periods
            binance: 1/24 24hr_periods
        4.95hr
            kraken: ceil(1 + 4.95/4) 4hr_periods = 3 4hr_periods
            binance: ceil(4.95)/24 24hr_periods = 5/24 24hr_periods
    interest: borrowed * interest_rate * time-periods
      10min
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 quote
        kraken      3x: 40 * 0.0005 * 2    = 0.040 quote
        binace     -3x: 30 * 0.0005 * 1/24 = 0.000625 crypto
        kraken     -3x: 30 * 0.0005 * 2    = 0.030 crypto
      5hr
        binance     3x: 40 * 0.0005 * 5/24 = 0.004166666666666667 quote
        kraken      3x: 40 * 0.0005 * 3    = 0.06 quote
        binace     -3x: 30 * 0.0005 * 5/24 = 0.0031249999999999997 crypto
        kraken     -3x: 30 * 0.0005 * 3    = 0.045 crypto
      0.00025 interest
        binance     3x: 40 * 0.00025 * 5/24 = 0.0020833333333333333 quote
        kraken      3x: 40 * 0.00025 * 3    = 0.03 quote
        binace     -3x: 30 * 0.00025 * 5/24 = 0.0015624999999999999 crypto
        kraken     -3x: 30 * 0.00025 * 3    = 0.0225 crypto
      5x leverage, 0.0005 interest, 5hr
        binance     5x: 48 * 0.0005 * 5/24 = 0.005 quote
        kraken      5x: 48 * 0.0005 * 3    = 0.07200000000000001 quote
        binace     -5x: 30 * 0.0005 * 5/24 = 0.0031249999999999997 crypto
        kraken     -5x: 30 * 0.0005 * 3    = 0.045 crypto
      1x leverage, 0.0005 interest, 5hr
        binance,kraken 1x: 0.0 quote
        binace        -1x: 30 * 0.0005 * 5/24 = 0.003125 crypto
        kraken        -1x: 30 * 0.0005 * 3    = 0.045 crypto
    """
    trade = Trade(pair='ADA/USDT', stake_amount=20.0, amount=30.0, open_rate=2.0, open_date=datetime.now(timezone.utc) - timedelta(minutes=minutes), fee_open=fee.return_value, fee_close=fee.return_value, exchange=exchange, leverage=lev, interest_rate=rate, is_short=is_short, trading_mode=trading_mode)
    assert round(float(trade.calculate_interest()), 8) == interest

@pytest.mark.parametrize('is_short,lev,borrowed,trading_mode', [(False, 1.0, 0.0, spot), (True, 1.0, 30.0, margin), (False, 3.0, 40.0, margin), (True, 3.0, 30.0, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_borrowed(fee: Any, is_short: bool, lev: float, borrowed: float, trading_mode: TradingMode) -> None:
    """
    10 minute limit trade on Binance/Kraken at 1x, 3x leverage
    fee: 0.25% quote
    interest_rate: 0.05% per 4 hrs
    open_rate: 2.00 quote
    close_rate: 2.20 quote
    amount: = 30.0 crypto
    stake_amount
        1x,-1x: 60.0  quote
        3x,-3x: 20.0  quote
    borrowed
         1x:  0 quote
         3x: 40 quote
        -1x: 30 crypto
        -3x: 30 crypto
    hours: 1/6 (10 minutes)
    time-periods:
        kraken: (1 + 1) 4hr_periods = 2 4hr_periods
        binance: 1/24 24hr_periods
    interest: borrowed * interest_rate * time-periods
        1x            :  /
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 quote
        kraken      3x: 40 * 0.0005 * 2 = 0.040 quote
        binace -1x,-3x: 30 * 0.0005 * 1/24 = 0.000625 crypto
        kraken -1x,-3x: 30 * 0.0005 * 2 = 0.030 crypto
    open_value: (amount * open_rate) ± (amount * open_rate * fee)
         1x, 3x: 30 * 2 + 30 * 2 * 0.0025 = 60.15 quote
        -1x,-3x: 30 * 2 - 30 * 2 * 0.0025 = 59.850 quote
    amount_closed:
        1x, 3x         : amount
        -1x, -3x       : amount + interest
        binance -1x,-3x: 30 + 0.000625 = 30.000625 crypto
        kraken  -1x,-3x: 30 + 0.03 = 30.03 crypto
    close_value:
         1x, 3x: (amount_closed * close_rate) - (amount_closed * close_rate * fee) - interest
        -