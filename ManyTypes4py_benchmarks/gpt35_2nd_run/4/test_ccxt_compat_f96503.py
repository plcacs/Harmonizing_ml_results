from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
import pytest
from freqtrade.enums import CandleType
from freqtrade.exchange import Exchange
from freqtrade.util import dt_floor_day, dt_now, dt_ts
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE, EXCHANGES

class TestCCXTExchange:

    def test_load_markets(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_has_validations(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ohlcv_limit(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ohlcv_limit_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_load_markets_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_order_parse(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_my_trades_parse(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_balances_parse(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_tickers(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_tickers_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_ticker(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_l2_orderbook(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_ohlcv(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_ohlcv_startdate(self, exchange: Tuple[Exchange, str]) -> None:
    
    def ccxt__async_get_candle_history(self, exchange: Tuple[Exchange, str], exchangename: str, pair: str, timeframe: str, candle_type: CandleType, factor: float = 0.9) -> None:
    
    def test_ccxt__async_get_candle_history(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt__async_get_candle_history_futures(self, exchange_futures: Tuple[Exchange, str], candle_type: CandleType) -> None:
    
    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_fetch_mark_price_history(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt__calculate_funding_fees(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt__async_get_trade_history(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_get_fee(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_get_max_leverage_spot(self, exchange: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_get_max_leverage_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_get_contract_size(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_load_leverage_tiers(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_dry_run_liquidation_price(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_ccxt_get_max_pair_stake_amount(self, exchange_futures: Tuple[Exchange, str]) -> None:
    
    def test_private_method_presence(self, exchange: Tuple[Exchange, str]) -> None:
