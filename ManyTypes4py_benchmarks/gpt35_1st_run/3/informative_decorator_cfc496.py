def informative(timeframe: str, asset: str = '', fmt: str = None, *, candle_type: str = None, ffill: bool = True) -> Callable:
    def decorator(fn: PopulateIndicators) -> PopulateIndicators:
def _create_and_merge_informative_pair(strategy: Any, dataframe: DataFrame, metadata: dict, inf_data: InformativeData, populate_indicators: PopulateIndicators) -> DataFrame:
def __get_pair_formats(market: dict) -> dict:
def _format_pair_name(config: dict, pair: str, market: dict = None) -> str:
