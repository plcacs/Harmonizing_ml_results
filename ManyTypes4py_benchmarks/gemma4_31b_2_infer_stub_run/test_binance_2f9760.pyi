from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator
import pandas as pd
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.persistence import Trade

def test__get_params_binance(
    default_conf: Dict[str, Any],
    mocker: Any,
    side: str,
    order_type: str,
    time_in_force: str,
    expected: Dict[str, str]
) -> None: ...

def test_create_stoploss_order_binance(
    default_conf: Dict[str, Any],
    mocker: Any,
    limitratio: Optional[float],
    expected: float,
    side: str,
    trademode: TradingMode
) -> None: ...

def test_create_stoploss_order_dry_run_binance(
    default_conf: Dict[str, Any],
    mocker: Any
) -> None: ...

def test_stoploss_adjust_binance(
    mocker: Any,
    default_conf: Dict[str, Any],
    sl1: float,
    sl2: float,
    sl3: float,
    side: str
) -> None: ...

def test_liquidation_price_binance(
    mocker: Any,
    default_conf: Dict[str, Any],
    pair: str,
    is_short: bool,
    trading_mode: str,
    margin_mode: str,
    wallet_balance: float,
    maintenance_amt: float,
    amount: float,
    open_rate: float,
    open_trades: List[Dict[str, Any]],
    mm_ratio: float,
    expected: float
) -> None: ...

def test_fill_leverage_tiers_binance(
    default_conf: Dict[str, Any],
    mocker: Any
) -> None: ...

def test_fill_leverage_tiers_binance_dryrun(
    default_conf: Dict[str, Any],
    mocker: Any,
    leverage_tiers: Dict[str, List[Dict[str, Any]]]
) -> None: ...

def test_additional_exchange_init_binance(
    default_conf: Dict[str, Any],
    mocker: Any
) -> None: ...

def test__set_leverage_binance(
    mocker: Any,
    default_conf: Dict[str, Any]
) -> None: ...

def patch_binance_vision_ohlcv(
    mocker: Any,
    start: datetime,
    archive_end: datetime,
    api_end: datetime,
    timeframe: str
) -> Tuple[Any, Any, Any]: ...

def test_get_historic_ohlcv_binance(
    mocker: Any,
    default_conf: Dict[str, Any],
    timeframe: str,
    is_new_pair: bool,
    since: datetime,
    until: datetime,
    first_date: Optional[datetime],
    last_date: Optional[datetime],
    candle_called: bool,
    archive_called: bool,
    api_called: bool
) -> None: ...

def test_get_maintenance_ratio_and_amt_binance(
    default_conf: Dict[str, Any],
    mocker: Any,
    leverage_tiers: Dict[str, List[Dict[str, Any]]],
    pair: str,
    notional_value: float,
    mm_ratio: float,
    amt: float
) -> None: ...

async def test__async_get_trade_history_id_binance(
    default_conf_usdt: Dict[str, Any],
    mocker: Any,
    fetch_trades_result: List[Dict[str, Any]]
) -> None: ...