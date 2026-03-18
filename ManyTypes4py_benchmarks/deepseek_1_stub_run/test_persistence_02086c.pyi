```python
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from _pytest.logging import LogCaptureFixture
from pytest import approx
from sqlalchemy.sql.selectable import Select
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT
from freqtrade.enums import TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import LocalTrade, Order, Trade
from freqtrade.util.datetime_helpers import dt_now

spot: TradingMode = ...
margin: TradingMode = ...
futures: TradingMode = ...

def test_enter_exit_side(fee: Any, is_short: bool) -> None: ...

def test_set_stop_loss_liquidation(fee: Any) -> None: ...

def test_interest(
    fee: Any,
    exchange: str,
    is_short: bool,
    lev: float,
    minutes: int,
    rate: float,
    interest: float,
    trading_mode: TradingMode
) -> None: ...

def test_borrowed(
    fee: Any,
    is_short: bool,
    lev: float,
    borrowed: float,
    trading_mode: TradingMode
) -> None: ...

def test_update_limit_order(
    fee: Any,
    caplog: LogCaptureFixture,
    limit_buy_order_usdt: Any,
    limit_sell_order_usdt: Any,
    time_machine: Any,
    is_short: bool,
    open_rate: float,
    close_rate: float,
    lev: float,
    profit: float,
    trading_mode: TradingMode
) -> None: ...

def test_update_market_order(
    market_buy_order_usdt: Any,
    market_sell_order_usdt: Any,
    fee: Any,
    caplog: LogCaptureFixture
) -> None: ...

def test_calc_open_close_trade_price(
    limit_order: Any,
    fee: Any,
    exchange: str,
    is_short: bool,
    lev: float,
    open_value: float,
    close_value: float,
    profit: float,
    profit_ratio: float,
    trading_mode: TradingMode,
    funding_fees: float
) -> None: ...

def test_trade_close(fee: Any, time_machine: Any) -> None: ...

def test_calc_close_trade_price_exception(
    limit_buy_order_usdt: Any,
    fee: Any
) -> None: ...

def test_update_open_order(limit_buy_order_usdt: Any) -> None: ...

def test_update_invalid_order(limit_buy_order_usdt: Any) -> None: ...

def test_calc_open_trade_value(
    limit_buy_order_usdt: Any,
    exchange: str,
    lev: int,
    is_short: bool,
    fee_rate: float,
    result: float,
    trading_mode: TradingMode
) -> None: ...

def test_calc_close_trade_price(
    open_rate: float,
    exchange: str,
    is_short: bool,
    lev: float,
    close_rate: float,
    fee_rate: float,
    result: float,
    trading_mode: TradingMode,
    funding_fees: float
) -> None: ...

def test_calc_profit(
    exchange: str,
    is_short: bool,
    lev: float,
    close_rate: float,
    fee_close: float,
    profit: float,
    profit_ratio: float,
    trading_mode: TradingMode,
    funding_fees: float
) -> None: ...

def test_adjust_stop_loss(fee: Any) -> None: ...

def test_adjust_stop_loss_short(fee: Any) -> None: ...

def test_adjust_min_max_rates(fee: Any) -> None: ...

def test_get_open(fee: Any, is_short: bool, use_db: bool) -> None: ...

def test_get_open_lev(fee: Any, use_db: bool) -> None: ...

def test_get_open_orders(
    fee: Any,
    is_short: bool,
    use_db: bool
) -> None: ...

def test_to_json(fee: Any) -> None: ...

def test_stoploss_reinitialization(default_conf: Any, fee: Any) -> None: ...

def test_stoploss_reinitialization_leverage(
    default_conf: Any,
    fee: Any
) -> None: ...

def test_stoploss_reinitialization_short(
    default_conf: Any,
    fee: Any
) -> None: ...

def test_update_fee(fee: Any) -> None: ...

def test_fee_updated(fee: Any) -> None: ...

def test_total_open_trades_stakes(
    fee: Any,
    is_short: bool,
    use_db: bool
) -> None: ...

def test_get_total_closed_profit(
    fee: Any,
    use_db: bool,
    is_short: Optional[bool],
    result: float
) -> None: ...

def test_get_trades_proxy(
    fee: Any,
    use_db: bool,
    is_short: bool
) -> None: ...

def test_get_trades__query(fee: Any, is_short: bool) -> None: ...

def test_get_trades_backtest() -> None: ...

def test_get_overall_performance(fee: Any) -> None: ...

def test_get_best_pair(
    fee: Any,
    is_short: Optional[bool],
    pair: str,
    profit: float
) -> None: ...

def test_get_best_pair_lev(fee: Any) -> None: ...

def test_get_canceled_exit_order_count(
    fee: Any,
    is_short: bool
) -> None: ...

def test_fully_canceled_entry_order_count(
    fee: Any,
    is_short: bool
) -> None: ...

def test_update_order_from_ccxt(
    caplog: LogCaptureFixture,
    time_machine: Any
) -> None: ...

def test_select_order(fee: Any, is_short: bool) -> None: ...

def test_Trade_object_idem() -> None: ...

def test_trade_truncates_string_fields() -> None: ...

def test_recalc_trade_from_orders(fee: Any) -> None: ...

def test_recalc_trade_from_orders_kucoin() -> None: ...

def test_recalc_trade_from_orders_ignores_bad_orders(
    fee: Any,
    is_short: bool
) -> None: ...

def test_select_filled_orders(fee: Any) -> None: ...

def test_select_filled_orders_usdt(fee: Any) -> None: ...

def test_order_to_ccxt(
    limit_buy_order_open: Any,
    limit_sell_order_usdt_open: Any
) -> None: ...

def test_recalc_trade_from_orders_dca(data: Any) -> None: ...
```