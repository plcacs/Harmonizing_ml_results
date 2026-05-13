from datetime import datetime
from random import randint
from unittest.mock import MagicMock, PropertyMock
from typing import Any, Dict, List, Optional, Tuple
import ccxt
import pandas as pd
import pytest
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers

@pytest.mark.parametrize('side,order_type,time_in_force,expected', [('buy', 'limit', 'gtc', {'timeInForce': 'GTC'}), ('buy', 'limit', 'IOC', {'timeInForce': 'IOC'}), ('buy', 'market', 'IOC', {}), ('buy', 'limit', 'PO', {'timeInForce': 'PO'}), ('sell', 'limit', 'PO', {'timeInForce': 'PO'}), ('sell', 'market', 'PO', {})])
def test__get_params_binance(default_conf: Dict[str, Any], mocker: MagicMock, side: str, order_type: str, time_in_force: str, expected: Dict[str, str]) -> None: ...

@pytest.mark.parametrize('trademode', [TradingMode.FUTURES, TradingMode.SPOT])
@pytest.mark.parametrize('limitratio,expected,side', [(None, 220 * 0.99, 'sell'), (0.99, 220 * 0.99, 'sell'), (0.98, 220 * 0.98, 'sell'), (None, 220 * 1.01, 'buy'), (0.99, 220 * 1.01, 'buy'), (0.98, 220 * 1.02, 'buy')])
def test_create_stoploss_order_binance(default_conf: Dict[str, Any], mocker: MagicMock, limitratio: Optional[float], expected: float, side: str, trademode: TradingMode) -> None: ...

def test_create_stoploss_order_dry_run_binance(default_conf: Dict[str, Any], mocker: MagicMock) -> None: ...

@pytest.mark.parametrize('sl1,sl2,sl3,side', [(1501, 1499, 1501, 'sell'), (1499, 1501, 1499, 'buy')])
def test_stoploss_adjust_binance(mocker: MagicMock, default_conf: Dict[str, Any], sl1: float, sl2: float, sl3: float, side: str) -> None: ...

@pytest.mark.parametrize('pair, is_short, trading_mode, margin_mode, wallet_balance, maintenance_amt, amount, open_rate, open_trades,mm_ratio, expected', [('ETH/USDT:USDT', False, 'futures', 'isolated', 1535443.01, 135365.0, 3683.979, 1456.84, [], 0.1, 1114.78), ('ETH/USDT:USDT', False, 'futures', 'isolated', 1535443.01, 16300.0, 109.488, 32481.98, [], 0.025, 18778.73), ('ETH/USDT:USDT', False, 'futures', 'cross', 1535443.01, 135365.0, 3683.979, 1456.84, [{'pair': 'BTC/USDT:USDT', 'open_rate': 32481.98, 'amount': 109.488, 'stake_amount': 3556387.02624, 'mark_price': 31967.27, 'mm_ratio': 0.025, 'maintenance_amt': 16300.0}, {'pair': 'ETH/USDT:USDT', 'open_rate': 1456.84, 'amount': 3683.979, 'stake_amount': 5366967.96, 'mark_price': 1335.18, 'mm_ratio': 0.1, 'maintenance_amt': 135365.0}], 0.1, 1153.26), ('BTC/USDT:USDT', False, 'futures', 'cross', 1535443.01, 16300.0, 109.488, 32481.98, [{'pair': 'BTC/USDT:USDT', 'open_rate': 32481.98, 'amount': 109.488, 'stake_amount': 3556387.02624, 'mark_price': 31967.27, 'mm_ratio': 0.025, 'maintenance_amt': 16300.0}, {'pair': 'ETH/USDT:USDT', 'open_rate': 1456.84, 'amount': 3683.979, 'stake_amount': 5366967.96, 'mark_price': 1335.18, 'mm_ratio': 0.1, 'maintenance_amt': 135365.0}], 0.025, 26316.89)])
def test_liquidation_price_binance(mocker: MagicMock, default_conf: Dict[str, Any], pair: str, is_short: bool, trading_mode: str, margin_mode: str, wallet_balance: float, maintenance_amt: float, amount: float, open_rate: float, open_trades: List[Dict[str, Any]], mm_ratio: float, expected: float) -> None: ...

def test_fill_leverage_tiers_binance(default_conf: Dict[str, Any], mocker: MagicMock) -> None: ...

def test_fill_leverage_tiers_binance_dryrun(default_conf: Dict[str, Any], mocker: MagicMock, leverage_tiers: Dict[str, List[Dict[str, Any]]]) -> None: ...

def test_additional_exchange_init_binance(default_conf: Dict[str, Any], mocker: MagicMock) -> None: ...

def test__set_leverage_binance(mocker: MagicMock, default_conf: Dict[str, Any]) -> None: ...

def patch_binance_vision_ohlcv(mocker: MagicMock, start: datetime, archive_end: datetime, api_end: datetime, timeframe: str) -> Tuple[MagicMock, MagicMock, MagicMock]: ...

@pytest.mark.parametrize('timeframe,is_new_pair,since,until,first_date,last_date,candle_called,archive_called,api_called', [('1m', True, dt_utc(2020, 1, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 23, 59), True, True, False), ('1m', True, dt_utc(2020, 1, 1), dt_utc(2020, 1, 3), dt_utc(2020, 1, 1), dt_utc(2020, 1, 2, 23, 59), True, True, True), ('1m', True, dt_utc(2020, 1, 2), dt_utc(2020, 1, 2, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 2, 0, 59), True, False, True), ('1m', False, dt_utc(2020, 1, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 23, 59), False, True, False), ('1m', True, dt_utc(2019, 1, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 23, 59), True, True, False), ('1m', False, dt_utc(2019, 1, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 23, 59), False, True, False), ('1m', False, dt_utc(2019, 1, 1), dt_utc(2019, 1, 2), None, None, False, True, True), ('1m', True, dt_utc(2019, 1, 1), dt_utc(2019, 1, 2), None, None, True, False, False), ('1m', False, dt_utc(2021, 1, 1), dt_utc(2021, 1, 2), None, None, False, False, False), ('1m', True, dt_utc(2021, 1, 1), dt_utc(2021, 1, 2), None, None, True, False, False), ('1h', False, dt_utc(2020, 1, 1), dt_utc(2020, 1, 2), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 23), False, False, True), ('1m', False, dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 3, 50, 30), dt_utc(2020, 1, 1), dt_utc(2020, 1, 1, 3, 50), False, True, False)])
def test_get_historic_ohlcv_binance(mocker: MagicMock, default_conf: Dict[str, Any], timeframe: str, is_new_pair: bool, since: datetime, until: datetime, first_date: Optional[datetime], last_date: Optional[datetime], candle_called: bool, archive_called: bool, api_called: bool) -> None: ...

@pytest.mark.parametrize('pair,notional_value,mm_ratio,amt', [('XRP/USDT:USDT', 0.0, 0.025, 0), ('BNB/USDT:USDT', 100.0, 0.0065, 0), ('BTC/USDT:USDT', 170.3, 0.004, 0), ('XRP/USDT:USDT', 999999.9, 0.1, 27500.0), ('BNB/USDT:USDT', 5000000.0, 0.15, 233035.0), ('BTC/USDT:USDT', 600000000, 0.5, 199703800.0)])
def test_get_maintenance_ratio_and_amt_binance(default_conf: Dict[str, Any], mocker: MagicMock, leverage_tiers: Dict[str, List[Dict[str, Any]]], pair: str, notional_value: float, mm_ratio: float, amt: float) -> None: ...

async def test__async_get_trade_history_id_binance(default_conf_usdt: Dict[str, Any], mocker: MagicMock, fetch_trades_result: List[Dict[str, Any]]) -> None: ...