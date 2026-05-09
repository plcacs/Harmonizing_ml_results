from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import timedelta
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exchange import Exchange

class Okx(Exchange):
    _ft_has: Dict[str, Union[int, str, bool, Dict[str, str]]]
    _ft_has_futures: Dict[str, Union[bool, str, Dict[str, Union[str, Dict[PriceType, str]]]]]
    _supported_trading_mode_margin_pairs: Tuple[Tuple[TradingMode, MarginMode]]
    net_only: bool
    _ccxt_params: Dict[str, Dict[str, str]]

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        ...

    def additional_exchange_init(self) -> None:
        ...

    def _get_posSide(self, side: str, reduceOnly: bool) -> str:
        ...

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict:
        ...

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: str) -> bool:
        ...

    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        ...

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        ...

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> Dict:
        ...

    def _convert_stop_order(self, pair: str, order_id: str, order: Dict) -> Dict:
        ...

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict] = None) -> Dict:
        ...

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> Dict:
        ...

    def get_order_id_conditional(self, order: Dict) -> str:
        ...

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict] = None) -> Dict:
        ...

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Dict]:
        ...