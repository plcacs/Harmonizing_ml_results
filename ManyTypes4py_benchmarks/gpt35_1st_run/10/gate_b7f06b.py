from typing import Any, Dict, List, Optional

class Gate(Exchange):
    unified_account: bool
    _ft_has: Dict[str, Any]
    _ft_has_futures: Dict[str, Any]
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]]

    def additional_exchange_init(self) -> None:
        ...

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict[str, Any]:
        ...

    def get_trades_for_order(self, order_id: str, pair: str, since: Optional[datetime], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ...

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        ...

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...
