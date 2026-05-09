class Binance(Exchange):
    _ft_has: FtHas
    _ft_has_futures: FtHasFutures
    _supported_trading_mode_margin_pairs: Tuple[Tuple[TradingMode, MarginMode], ...]

    def get_proxy_coin(self) -> str:
        ...

    def get_tickers(self, symbols: Optional[List[str]], *, cached: bool, market_type: Optional[MarketType]) -> Tickers:
        ...

    @retrier
    def additional_exchange_init(self) -> None:
        ...

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool, until_ms: Optional[int]) -> DataFrame:
        ...

    def get_historic_ohlcv_fast(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool, until_ms: Optional[int]) -> DataFrame:
        ...

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        ...

    def fetch_funding_rates(self, symbols: Optional[List[str]]) -> Dict[str, float]:
        ...

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Trade]) -> float:
        ...

    def load_leverage_tiers(self) -> Dict[str, float]:
        ...

    async def _async_get_trade_history_id_startup(self, pair: str, since: int) -> Tuple[List[Trade], str]:
        ...
