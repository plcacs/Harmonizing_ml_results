from datetime import datetime, timedelta
from typing import Any, Optional, Union, List, Dict, Tuple, Sequence
from types import FunctionType
import pytest
from sqlalchemy import select
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT
from freqtrade.enums import TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange.exchange_utils import TICK_SIZE
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.util import dt_now
from tests.conftest import create_mock_trades, create_mock_trades_usdt, create_mock_trades_with_leverage, log_has, log_has_re

spot: TradingMode = ...
margin: TradingMode = ...
futures: TradingMode = ...

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_enter_exit_side(fee: Any, is_short: bool) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_set_stop_loss_liquidation(fee: Any) -> None: ...

@pytest.mark.parametrize('exchange,is_short,lev,minutes,rate,interest,trading_mode', [('binance', False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin), ('binance', True, 3, 10, 0.0005, 0.000625, margin), ('binance', False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin), ('binance', True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin), ('binance', False, 3, 295, 0.00025, round(0.0020833333333333333, 8), margin), ('binance', True, 3, 295, 0.00025, round(0.0015624999999999999, 8), margin), ('binance', False, 5, 295, 0.0005, 0.005, margin), ('binance', True, 5, 295, 0.0005, round(0.0031249999999999997, 8), margin), ('binance', False, 1, 295, 0.0005, 0.0, spot), ('binance', True, 1, 295, 0.0005, 0.003125, margin), ('binance', False, 3, 10, 0.0005, 0.0, futures), ('binance', True, 3, 295, 0.0005, 0.0, futures), ('binance', False, 5, 295, 0.0005, 0.0, futures), ('binance', True, 5, 295, 0.0005, 0.0, futures), ('binance', False, 1, 295, 0.0005, 0.0, futures), ('binance', True, 1, 295, 0.0005, 0.0, futures), ('kraken', False, 3, 10, 0.0005, 0.04, margin), ('kraken', True, 3, 10, 0.0005, 0.03, margin), ('kraken', False, 3, 295, 0.0005, 0.06, margin), ('kraken', True, 3, 295, 0.0005, 0.045, margin), ('kraken', False, 3, 295, 0.00025, 0.03, margin), ('kraken', True, 3, 295, 0.00025, 0.0225, margin), ('kraken', False, 5, 295, 0.0005, round(0.07200000000000001, 8), margin), ('kraken', True, 5, 295, 0.0005, 0.045, margin), ('kraken', False, 1, 295, 0.0005, 0.0, spot), ('kraken', True, 1, 295, 0.0005, 0.045, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_interest(fee: Any, exchange: str, is_short: bool, lev: float, minutes: int, rate: float, interest: float, trading_mode: TradingMode) -> None: ...

@pytest.mark.parametrize('is_short,lev,borrowed,trading_mode', [(False, 1.0, 0.0, spot), (True, 1.0, 30.0, margin), (False, 3.0, 40.0, margin), (True, 3.0, 30.0, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_borrowed(fee: Any, is_short: bool, lev: float, borrowed: float, trading_mode: TradingMode) -> None: ...

@pytest.mark.parametrize('is_short,open_rate,close_rate,lev,profit,trading_mode', [(False, 2.0, 2.2, 1.0, 0.09451372, spot), (True, 2.2, 2.0, 3.0, 0.25894253, margin)])
@pytest.mark.usefixtures('init_persistence')
def test_update_limit_order(fee: Any, caplog: Any, limit_buy_order_usdt: Dict[str, Any], limit_sell_order_usdt: Dict[str, Any], time_machine: Any, is_short: bool, open_rate: float, close_rate: float, lev: float, profit: float, trading_mode: TradingMode) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_update_market_order(market_buy_order_usdt: Dict[str, Any], market_sell_order_usdt: Dict[str, Any], fee: Any, caplog: Any) -> None: ...

@pytest.mark.parametrize('exchange,is_short,lev,open_value,close_value,profit,profit_ratio,trading_mode,funding_fees', [('binance', False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0), ('binance', True, 1, 65.835, 60.151253125, 5.68374687, 0.08633321, margin, 0.0), ('binance', False, 3, 60.15, 65.83416667, 5.68416667, 0.28349958, margin, 0.0), ('binance', True, 3, 65.835, 60.151253125, 5.68374687, 0.25899963, margin, 0.0), ('kraken', False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0), ('kraken', True, 1, 65.835, 60.21015, 5.62485, 0.0854386, margin, 0.0), ('kraken', False, 3, 60.15, 65.795, 5.645, 0.28154613, margin, 0.0), ('kraken', True, 3, 65.835, 60.21015, 5.62485, 0.25631579, margin, 0.0), ('binance', False, 1, 60.15, 65.835, 5.685, 0.09451371, futures, 0.0), ('binance', False, 1, 60.15, 66.835, 6.685, 0.11113881, futures, 1.0), ('binance', True, 1, 65.835, 60.15, 5.685, 0.08635224, futures, 0.0), ('binance', True, 1, 65.835, 61.15, 4.685, 0.07116276, futures, -1.0), ('binance', True, 3, 65.835, 59.15, 6.685, 0.3046252, futures, 1.0), ('binance', False, 3, 60.15, 64.835, 4.685, 0.23366583, futures, -1.0)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_open_close_trade_price(limit_order: Dict[str, Dict[str, Any]], fee: Any, exchange: str, is_short: bool, lev: float, open_value: float, close_value: float, profit: float, profit_ratio: float, trading_mode: TradingMode, funding_fees: float) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_trade_close(fee: Any, time_machine: Any) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_calc_close_trade_price_exception(limit_buy_order_usdt: Dict[str, Any], fee: Any) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_update_open_order(limit_buy_order_usdt: Dict[str, Any]) -> None: ...

@pytest.mark.usefixtures('init_persistence')
def test_update_invalid_order(limit_buy_order_usdt: Dict[str, Any]) -> None: ...

@pytest.mark.parametrize('exchange', ['binance', 'kraken'])
@pytest.mark.parametrize('trading_mode', [spot, margin, futures])
@pytest.mark.parametrize('lev', [1, 3])
@pytest.mark.parametrize('is_short,fee_rate,result', [(False, 0.003, 60.18), (False, 0.0025, 60.15), (False, 0.003, 60.18), (False, 0.0025, 60.15), (True, 0.003, 59.82), (True, 0.0025, 59.85), (True, 0.003, 59.82), (True, 0.0025, 59.85)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_open_trade_value(limit_buy_order_usdt: Dict[str, Any], exchange: str, lev: int, is_short: bool, fee_rate: float, result: float, trading_mode: TradingMode) -> None: ...

@pytest.mark.parametrize('exchange,is_short,lev,open_rate,close_rate,fee_rate,result,trading_mode,funding_fees', [('binance', False, 1, 2.0, 2.5, 0.0025, 74.8125, spot, 0), ('binance', False, 1, 2.0, 2.5, 0.003, 74.775, spot, 0), ('binance', False, 1, 2.0, 2.2, 0.005, 65.67, margin, 0), ('binance', False, 3, 2.0, 2.5, 0.0025, 74.81166667, margin, 0), ('binance', False, 3, 2.0, 2.5, 0.003, 74.77416667, margin, 0), ('binance', True, 3, 2.2, 2.5, 0.0025, 75.18906641, margin, 0), ('binance', True, 3, 2.2, 2.5, 0.003, 75.22656719, margin, 0), ('binance', True, 1, 2.2, 2.5, 0.0025, 75.18906641, margin, 0), ('binance', True, 1, 2.2, 2.5, 0.003, 75.22656719, margin, 0), ('kraken', False, 3, 2.0, 2.5, 0.0025, 74.7725, margin, 0), ('kraken', False, 3, 2.0, 2.5, 0.003, 74.735, margin, 0), ('kraken', True, 3, 2.2, 2.5, 0.0025, 75.2626875, margin, 0), ('kraken', True, 3, 2.2, 2.5, 0.003, 75.300225, margin, 0), ('kraken', True, 1, 2.2, 2.5, 0.0025, 75.2626875, margin, 0), ('kraken', True, 1, 2.2, 2.5, 0.003, 75.300225, margin, 0), ('binance', False, 1, 2.0, 2.5, 0.0025, 75.8125, futures, 1), ('binance', False, 3, 2.0, 2.5, 0.0025, 73.8125, futures, -1), ('binance', True, 3, 2.0, 2.5, 0.0025, 74.1875, futures, 1), ('binance', True, 1, 2.0, 2.5, 0.0025, 76.1875, futures, -1)])
@pytest.mark.usefixtures('init_persistence')
def test_calc_close_trade_price(open_rate: float, exchange: str, is_short: bool, lev: int, close_rate: float, fee_rate: float, result: float, trading_mode: TradingMode, funding_fees: float) -> None: ...

@pytest.mark.parametrize('exchange,is_short,lev,close_rate,fee_close,profit,profit_ratio,trading_mode,funding_fees', [('binance', False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0), ('binance', False, 3, 2.1, 0.0025, 2.69166667, 0.134247714, margin, 0), ('binance', True, 1, 2.1, 0.0025, -3.3088157, -0.055285142, margin, 0), ('binance', True, 3, 2.1, 0.0025, -3.3088157, -0.16585542, margin, 0), ('binance', False, 1, 1.9, 0.0025, -3.2925, -0.054738154, margin, 0), ('binance', False, 3, 1.9, 0.0025, -3.29333333, -0.164256026, margin, 0), ('binance', True, 1, 1.9, 0.0025, 2.70630953, 0.0452182043, margin, 0), ('binance', True, 3, 1.9, 0.0025, 2.70630953, 0.135654613, margin, 0), ('binance', False, 1, 2.2, 0.0025, 5.685, 0.09451371, margin, 0), ('binance', False, 3, 2.2, 0.0025, 5.68416667, 0.28349958, margin, 0), ('binance', True, 1, 2.2, 0.0025, -6.3163784, -0.10553681, margin, 0), ('binance', True, 3, 2.2, 0.0025, -6.3163784, -0.31661044, margin, 0), ('kraken', False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0), ('kraken', False, 3, 2.1, 0.0025, 2.6525, 0.132294264, margin, 0), ('kraken', True, 1, 2.1, 0.0025, -3.3706575, -0.056318421, margin, 0), ('kraken', True, 3, 2.1, 0.0025, -3.3706575, -0.168955263, margin, 0), ('kraken', False, 1, 1.9, 0.0025, -3.2925, -0.054738154, margin, 0), ('kraken', False, 3, 1.9, 0.0025, -3.3325, -0.166209476, margin, 0), ('kraken', True, 1, 1.9, 0.0025, 2.6503575, 0.044283333, margin, 0), ('kraken', True, 3, 1.9, 0.0025, 2.6503575,