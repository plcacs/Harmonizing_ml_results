from typing import List, Dict, Any, Union
from datetime import datetime
import ccxt
from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, RetryableOrderError, TemporaryError
from freqtrade.exchange import Exchange, date_minus_candles
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.misc import safe_value_fallback2

class Okx(Exchange):
    _ft_has: Dict[str, Union[int, str, bool]] = {'ohlcv_candle_limit': 100, 'mark_ohlcv_timeframe': '4h', 'funding_fee_timeframe': '8h', 'stoploss_order_types': {'limit': 'limit'}, 'stoploss_on_exchange': True, 'trades_has_history': False, 'ws_enabled': True}
    _ft_has_futures: Dict[str, Union[bool, Dict[str, str]]] = {'tickers_have_quoteVolume': False, 'stop_price_type_field': 'slTriggerPxType', 'stop_price_type_value_mapping': {PriceType.LAST: 'last', PriceType.MARK: 'index', PriceType.INDEX: 'mark'}, 'ws_enabled': True}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [(TradingMode.FUTURES, MarginMode.ISOLATED)]
    net_only: bool = True
    _ccxt_params: Dict[str, Dict[str, str]] = {'options': {'brokerId': 'ffb5405ad327SUDE'}}

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: int = None) -> int:
        ...

    @retrier
    def additional_exchange_init(self) -> None:
        ...

    def _get_posSide(self, side: str, reduceOnly: bool) -> str:
        ...

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict[str, Union[str, float]]:
        ...

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: str) -> bool:
        ...

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        ...

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        ...

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> Dict[str, Union[str, float]]:
        ...

    def _convert_stop_order(self, pair: str, order_id: str, order: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(self, order_id: str, pair: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        ...

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> Dict[str, Any]:
        ...

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        ...

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Dict[str, Any] = None) -> None:
        ...

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Dict[str, Any]]:
        ...
