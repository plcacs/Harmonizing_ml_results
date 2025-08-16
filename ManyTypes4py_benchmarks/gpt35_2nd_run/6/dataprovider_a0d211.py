    def __init__(self, config: Config, exchange: Exchange, pairlists: ListPairsWithTimeframes = None, rpc: RPCManager = None) -> None:
    def _set_dataframe_max_index(self, limit_index: int) -> None:
    def _set_dataframe_max_date(self, limit_date: Timestamp) -> None:
    def _set_cached_df(self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType) -> None:
    def _set_producer_pairs(self, pairlist: List[str], producer_name: str = 'default') -> None:
    def get_producer_pairs(self, producer_name: str = 'default') -> List[str]:
    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None:
    def _replace_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Timestamp, timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> None:
    def _add_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Timestamp, timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> Tuple[bool, int]:
    def get_producer_df(self, pair: str, timeframe: str = None, candle_type: CandleType = None, producer_name: str = 'default') -> Tuple[DataFrame, Timestamp]:
    def add_pairlisthandler(self, pairlists: ListPairsWithTimeframes) -> None:
    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = '') -> DataFrame:
    def get_required_startup(self, timeframe: str) -> int:
    def get_pair_dataframe(self, pair: str, timeframe: str = None, candle_type: str = '') -> DataFrame:
    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, Timestamp]:
    @property
    def runmode(self) -> RunMode:
    def current_whitelist(self) -> List[str]:
    def refresh(self, pairlist: List[str], helping_pairs: List[str] = None) -> None:
    def refresh_latest_trades(self, pairlist: List[str]) -> None:
    @property
    def available_pairs(self) -> List[PairWithTimeframe]:
    def ohlcv(self, pair: str, timeframe: str = None, copy: bool = True, candle_type: str = '') -> DataFrame:
    def trades(self, pair: str, timeframe: str = None, copy: bool = True, candle_type: str = '') -> DataFrame:
    def market(self, pair: str) -> Any:
    def ticker(self, pair: str) -> Any:
    def orderbook(self, pair: str, maximum: int) -> Any:
    def send_msg(self, message: str, *, always_send: bool = False) -> None:
