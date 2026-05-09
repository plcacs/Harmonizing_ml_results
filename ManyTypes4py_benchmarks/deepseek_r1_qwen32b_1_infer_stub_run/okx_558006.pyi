import logging
from datetime import timedelta
from typing import Any, Optional, Tuple, List, Dict, TypedDict, Union

from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)

class _FTHas(TypedDict):
    ohlcv_candle_limit: int
    mark_ohlcv_timeframe: str
    funding_fee_timeframe: str
    stoploss_order_types: Dict[str, str]
    stoploss_on_exchange: bool
    trades_has_history: bool
    ws_enabled: bool

class _FTHasFutures(TypedDict):
    tickers_have_quoteVolume: bool
    stop_price_type_field: str
    stop_price_type_value_mapping: Dict[PriceType, str]
    ws_enabled: bool

class Okx(Exchange):
    _ft_has: _FTHas
    _ft_has_futures: _FTHasFutures
    _supported_trading_mode_margin_pairs: Tuple[Tuple[TradingMode, MarginMode], ...]
    net_only: bool
    _ccxt_params: Dict[str, Dict[str, str]]

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int]) -> int:
        ...

    @retrier
    def additional_exchange_init(self) -> None:
        ...

    def _get_posSide(self, side: str, reduceOnly: bool) -> str:
        ...

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict[str, Any]:
        ...

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: str) -> bool:
        ...

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        ...

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        ...

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> Dict[str, Any]:
        ...

    def _convert_stop_order(self, pair: str, order: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @retrier
    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> Dict[str, Any]:
        ...

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        ...

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> None:
        ...

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Dict[str, Any]]:
        ...