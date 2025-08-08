from typing import Dict, Any, List, Union

class Hyperliquid(Exchange):
    _ft_has: Dict[str, Union[bool, List[int], Dict[str, bool], bool]] = {'ohlcv_has_history': False, 'l2_limit_range': [20], 'trades_has_history': False, 'tickers_have_bid_ask': False, 'stoploss_on_exchange': False, 'exchange_has_overrides': {'fetchTrades': False}, 'marketOrderRequiresPrice': True}
    _ft_has_futures: Dict[str, Union[bool, str, int]] = {'stoploss_on_exchange': True, 'stoploss_order_types': {'limit': 'limit'}, 'stop_price_prop': 'stopPrice', 'funding_fee_timeframe': '1h', 'funding_fee_candle_limit': 500}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [(TradingMode.FUTURES, MarginMode.ISOLATED)]

    @property
    def _ccxt_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({'options': {'defaultType': 'spot'}})
        config.update(super()._ccxt_config)
        return config

    def get_max_leverage(self, pair: str, stake_amount: float) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]['limits']['leverage']['max']
        else:
            return 1.0

    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != TradingMode.SPOT:
            leverage = int(leverage)
            self.set_margin_mode(pair, self.margin_mode, params={'leverage': leverage})

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Any]) -> float:
        ...

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        ...

    def _adjust_hyperliquid_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def fetch_order(self, order_id: str, pair: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        ...

    def fetch_orders(self, pair: str, since: datetime, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        ...
