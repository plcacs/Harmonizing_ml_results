from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

class Exchange:
    # ... (rest of the code remains the same)

    _params: Dict[str, Any]
    _ccxt_params: Dict[str, Any]
    _ft_has_default: Dict[str, Any]
    _ft_has: Dict[str, Any]
    _ft_has_futures: Dict[str, Any]
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]]

    def __init__(self, config: Config, *, exchange_config: Optional[Dict[str, Any]] = None, validate: bool = True, load_leverage_tiers: bool = False) -> None:
        # ... (rest of the code remains the same)

    def validate_config(self, config: Config) -> None:
        # ... (rest of the code remains the same)

    def close(self) -> None:
        # ... (rest of the code remains the same)

    def _init_async_loop(self) -> asyncio.AbstractEventLoop:
        # ... (rest of the code remains the same)

    def _init_ccxt(self, exchange_config: Dict[str, Any], sync: bool, ccxt_kwargs: Optional[Dict[str, Any]]) -> ccxt.Exchange:
        # ... (rest of the code remains the same)

    def get_option(self, param: str, default: Optional[Any] = None) -> Any:
        # ... (rest of the code remains the same)

    def exchange_has(self, endpoint: str) -> bool:
        # ... (rest of the code remains the same)

    def features(self, market_type: str, endpoint: str, attribute: str, default: Any) -> Any:
        # ... (rest of the code remains the same)

    def get_precision_amount(self, pair: str) -> Optional[int]:
        # ... (rest of the code remains the same)

    def get_precision_price(self, pair: str) -> Optional[int]:
        # ... (rest of the code remains the same)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        # ... (rest of the code remains the same)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode: int = 0) -> float:
        # ... (rest of the code remains the same)

    def price_get_one_pip(self, pair: str, price: float) -> float:
        # ... (rest of the code remains the same)

    def get_min_pair_stake_amount(self, pair: str, price: float, stoploss: float, leverage: float = 1.0) -> float:
        # ... (rest of the code remains the same)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> Optional[float]:
        # ... (rest of the code remains the same)

    def _get_stake_amount_limit(self, pair: str, price: float, stoploss: float, limit: str, leverage: float = 1.0) -> Union[float, None]:
        # ... (rest of the code remains the same)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        # ... (rest of the code remains the same)

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float, params: Optional[Dict[str, Any]] = None, stop_loss: bool = False) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def create_order(self, *, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float, reduceOnly: bool = False, time_in_force: str = 'GTC') -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def create_stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict[str, str], side: str, leverage: float) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def fetch_order_emulated(self, order_id: str, pair: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def fetch_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def cancel_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def get_balances(self) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def fetch_positions(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        # ... (rest of the code remains the same)

    def fetch_orders(self, pair: str, since: datetime, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # ... (rest of the code remains the same)

    def fetch_trading_fees(self) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def fetch_bids_asks(self, symbols: Optional[List[str]] = None, *, cached: bool = False) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def get_tickers(self, symbols: Optional[List[str]] = None, *, cached: bool = False, market_type: Optional[TradingMode] = None) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def get_proxy_coin(self) -> str:
        # ... (rest of the code remains the same)

    def get_conversion_rate(self, coin: str, currency: str) -> Optional[float]:
        # ... (rest of the code remains the same)

    def fetch_ticker(self, pair: str) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> Dict[str, Any]:
        # ... (rest of the code remains the same)

    def get_rate(self, pair: str, refresh: bool, side: str, is_short: bool, order_book: Optional[Dict[str, Any]] = None, ticker: Optional[Dict[str, Any]] = None) -> float:
        # ... (rest of the code remains the same)

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        # ... (rest of the code remains the same)

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: str, is_new_pair: bool = False, until_ms: Optional[int] = None) -> pd.DataFrame:
        # ... (rest of the code remains the same)

    def refresh_latest_ohlcv(self, pair_list: List[Tuple[str, str, str]], *, since_ms: Optional[int] = None, cache: bool = True, drop_incomplete: Optional[bool] = None) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        # ... (rest of the code remains the same)

    def refresh_ohlcv_with_cache(self, pairs: List[Tuple[str, str, str]], since_ms: int) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        # ... (rest of the code remains the same)

    def get_historic_trades(self, pair: str, since: int, until: Optional[int] = None, from_id: Optional[str] = None) -> List[Dict[str, Any]]:
        # ... (rest of the code remains the same)

    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: Optional[datetime] = None) -> float:
        # ... (rest of the code remains the same)

    def calculate_funding_fees(self, df: pd.DataFrame, amount: float, is_short: bool, open_date: datetime, close_date: datetime, time_in_ratio: Optional[float] = None) -> float:
        # ... (rest of the code remains the same)

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        # ... (rest of the code remains the same)

    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: Optional[float], leverage: float, wallet_balance: float, open_trades: Optional[List[Dict[str, Any]]] = None) -> Optional[float]:
        # ... (rest of the code remains the same)

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Dict[str, Any]]) -> float:
        # ... (rest of the code remains the same)

    def get_maintenance_ratio_and_amt(self, pair: str, notional_value: float) -> Tuple[float, float]:
        # ... (rest of the code remains the same)
