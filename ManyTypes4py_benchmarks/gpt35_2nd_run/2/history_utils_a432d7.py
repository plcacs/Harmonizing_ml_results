from typing import Dict, List, Tuple
from freqtrade.configuration import TimeRange
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import Exchange

def load_pair_history(pair: str, timeframe: str, datadir: str, *, timerange: TimeRange = None, fill_up_missing: bool = True, drop_incomplete: bool = False, startup_candles: int = 0, data_format: str = None, data_handler = None, candle_type: CandleType = CandleType.SPOT) -> DataFrame:
    ...

def load_data(datadir: str, timeframe: str, pairs: List[str], *, timerange: TimeRange = None, fill_up_missing: bool = True, startup_candles: int = 0, fail_without_data: bool = False, data_format: str = 'feather', candle_type: CandleType = CandleType.SPOT, user_futures_funding_rate = None) -> Dict[str, DataFrame]:
    ...

def refresh_data(*, datadir: str, timeframe: str, pairs: List[str], exchange: Exchange, data_format: str = None, timerange: TimeRange = None, candle_type: CandleType) -> None:
    ...

def _load_cached_data_for_updating(pair: str, timeframe: str, timerange: TimeRange, data_handler, candle_type: CandleType, prepend: bool = False) -> Tuple[DataFrame, int, int]:
    ...

def _download_pair_history(pair: str, *, datadir: str, exchange: Exchange, timeframe: str = '5m', new_pairs_days: int = 30, data_handler = None, timerange: TimeRange = None, candle_type: CandleType, erase: bool = False, prepend: bool = False) -> bool:
    ...

def refresh_backtest_ohlcv_data(exchange: Exchange, pairs: List[str], timeframes: List[str], datadir: str, trading_mode: str, timerange: TimeRange = None, new_pairs_days: int = 30, erase: bool = False, data_format: str = None, prepend: bool = False, progress_tracker = None) -> List[str]:
    ...

def _download_trades_history(exchange: Exchange, pair: str, *, new_pairs_days: int = 30, timerange: TimeRange = None, data_handler, trading_mode: str) -> bool:
    ...

def refresh_backtest_trades_data(exchange: Exchange, pairs: List[str], datadir: str, timerange: TimeRange, trading_mode: str, new_pairs_days: int = 30, erase: bool = False, data_format: str = 'feather', progress_tracker = None) -> List[str]:
    ...

def get_timerange(data: Dict[str, DataFrame]) -> Tuple[datetime, datetime]:
    ...

def validate_backtest_data(data: DataFrame, pair: str, min_date: datetime, max_date: datetime, timeframe_min: int) -> bool:
    ...

def download_data_main(config: Dict) -> None:
    ...

def download_data(config: Dict, exchange: Exchange, *, progress_tracker = None) -> None:
    ...
