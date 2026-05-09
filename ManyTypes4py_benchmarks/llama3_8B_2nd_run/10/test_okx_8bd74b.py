from datetime import datetime, timedelta
from typing import List, Dict, Any

def test_okx_ohlcv_candle_limit(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    timeframe_to_minutes: Any
) -> None:
    ...

def test_get_maintenance_ratio_and_amt_okx(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    leverage_tiers: List[Dict[str, Any]]
) -> None:
    ...

def test_get_max_pair_stake_amount_okx(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    leverage_tiers: List[Dict[str, Any]]
) -> None:
    ...

def test__get_posSide(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    mode: str, 
    side: str, 
    reduceonly: bool, 
    result: str
) -> None:
    ...

def test_additional_exchange_init_okx(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    markets: List[Dict[str, Any]], 
    tmp_path: Any, 
    caplog: Any, 
    time_machine: Any
) -> None:
    ...

def test_load_leverage_tiers_okx(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    markets: List[Dict[str, Any]], 
    tmp_path: Any, 
    caplog: Any, 
    time_machine: Any
) -> None:
    ...

def test__set_leverage_okx(
    mocker: Any, 
    default_conf: Dict[str, Any]
) -> None:
    ...

def test_fetch_stoploss_order_okx(
    default_conf: Dict[str, Any], 
    mocker: Any
) -> None:
    ...

def test_fetch_stoploss_order_okx_exceptions(
    default_conf_usdt: Dict[str, Any], 
    mocker: Any
) -> None:
    ...

def test_stoploss_adjust_okx(
    mocker: Any, 
    default_conf: Dict[str, Any], 
    sl1: float, 
    sl2: float, 
    sl3: float, 
    side: str
) -> None:
    ...

def test_stoploss_cancel_okx(
    mocker: Any, 
    default_conf: Dict[str, Any]
) -> None:
    ...

def test__get_stop_params_okx(
    mocker: Any, 
    default_conf: Dict[str, Any]
) -> None:
    ...

def test_fetch_orders_okx(
    default_conf: Dict[str, Any], 
    mocker: Any, 
    limit_order: Dict[str, Any]
) -> None:
    ...
