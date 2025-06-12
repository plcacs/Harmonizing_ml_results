"""
Abstract datahandler interface.
It's subclasses handle and storing data from disk.

"""
import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pandas import DataFrame, to_datetime
from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS, ListPairsWithTimeframes
from freqtrade.data.converter import clean_ohlcv_dataframe, trades_convert_types, trades_df_remove_duplicates, trim_dataframe
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds

logger = logging.getLogger(__name__)
T = TypeVar('T', bound='IDataHandler')

class IDataHandler(ABC):
    _OHLCV_REGEX: str = '^([a-zA-Z_\\d-]+)\\-(\\d+[a-zA-Z]{1,2})\\-?([a-zA-Z_]*)?(?=\\.)'
    _TRADES_REGEX: str = '^([a-zA-Z_\\d-]+)\\-(trades)?(?=\\.)'

    def __init__(self, datadir: Path) -> None:
        self._datadir: Path = datadir

    @classmethod
    @abstractmethod
    def _get_file_extension(cls) -> str:
        """
        Get file extension for this particular datahandler
        """
        raise NotImplementedError()

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> List[Tuple[str, str, CandleType]]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :param trading_mode: trading-mode to be used
        :return: List of Tuples of (pair, timeframe, CandleType)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [re.search(cls._OHLCV_REGEX, p.name) for p in datadir.glob(f'*.{cls._get_file_extension()}')]
        return [(cls.rebuild_pair_from_filename(match[1]), cls.rebuild_timeframe_from_filename(match[2]), CandleType.from_string(match[3])) for match in _tmp if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: List of Pairs
        """
        candle = ''
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f'-{candle_type}'
        ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-' + timeframe + candle + f'.{ext})', p.name) for p in datadir.glob(f'*{timeframe}{candle}.{ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        """
        Store ohlcv data.
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: None
        """

    def ohlcv_data_min_max(self, pair: str, timeframe: str, candle_type: CandleType) -> Tuple[datetime, datetime, int]:
        """
        Returns the min and max timestamp for the given pair and timeframe.
        :param pair: Pair to get min/max for
        :param timeframe: Timeframe to get min/max for
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: (min, max, len)
        """
        df = self._ohlcv_load(pair, timeframe, None, candle_type)
        if df.empty:
            return (datetime.fromtimestamp(0, tz=timezone.utc), datetime.fromtimestamp(0, tz=timezone.utc), 0)
        return (df.iloc[0]['date'].to_pydatetime(), df.iloc[-1]['date'].to_pydatetime(), len(df))

    @abstractmethod
    def _ohlcv_load(self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param timeframe: Timeframe (e.g. "5m")
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @abstractmethod
    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """

    @classmethod
    def trades_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :param trading_mode: trading-mode to be used
        :return: List of Tuples of (pair, timeframe, CandleType)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [re.search(cls._TRADES_REGEX, p.name) for p in datadir.glob(f'*.{cls._get_file_extension()}')]
        return [cls.rebuild_pair_from_filename(match[1]) for match in _tmp if match and len(match.groups()) > 1]

    def trades_data_min_max(self, pair: str, trading_mode: TradingMode) -> Tuple[datetime, datetime, int]:
        """
        Returns the min and max timestamp for the given pair's trades data.
        :param pair: Pair to get min/max for
        :param trading_mode: Trading mode to use (used to determine the filename)
        :return: (min, max, len)
        """
        df = self._trades_load(pair, trading_mode)
        if df.empty:
            return (datetime.fromtimestamp(0, tz=timezone.utc), datetime.fromtimestamp(0, tz=timezone.utc), 0)
        return (to_datetime(df.iloc[0]['timestamp'], unit='ms', utc=True).to_pydatetime(), to_datetime(df.iloc[-1]['timestamp'], unit='ms', utc=True).to_pydatetime(), len(df))

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs for which trade data is available in this
        :param datadir: Directory to search for ohlcv files
        :return: List of Pairs
        """
        _ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-trades.' + _ext + ')', p.name) for p in datadir.glob(f'*trades.{_ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """

    @abstractmethod
    def trades_append(self, pair: str, data: DataFrame) -> None:
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """

    @abstractmethod
    def _trades_load(self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: Dataframe containing trades
        """

    def trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        self._trades_store(pair, data[DEFAULT_TRADES_COLUMNS], trading_mode)

    def trades_purge(self, pair: str, trading_mode: TradingMode) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param trading_mode: Trading mode to use (used to determine the filename)
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def trades_load(self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        Removes duplicates in the process.
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        try:
            trades = self._trades_load(pair, trading_mode, timerange=timerange)
        except Exception:
            logger.exception(f'Error loading trades for {pair}')
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)
        trades = trades_df_remove_duplicates(trades)
        trades = trades_convert_types(trades)
        return trades

    @classmethod
    def create_dir_if_needed(cls, datadir: Path) -> None:
        """
        Creates datadir if necessary
        should only create directories for "futures" mode at the moment.
        """
        if not datadir.parent.is_dir():
            datadir.parent.mkdir()

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str, candle_type: CandleType, no_timeframe_modify: bool = False) -> Path:
        pair_s = misc.pair_to_filename(pair)
        candle = ''
        if not no_timeframe_modify:
            timeframe = cls.timeframe_to_file(timeframe)
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f'-{candle_type}'
        filename = datadir.joinpath(f'{pair_s}-{timeframe}{candle}.{cls._get_file_extension()}')
        return filename

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str, trading_mode: TradingMode) -> Path:
        pair_s = misc.pair_to_filename(pair)
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        filename = datadir.joinpath(f'{pair_s}-trades.{cls._get_file_extension()}')
        return filename

    @staticmethod
    def timeframe_to_file(timeframe: str) -> str:
        return timeframe.replace('M', 'Mo')

    @staticmethod
    def rebuild_timeframe_from_filename(timeframe: str) -> str:
        """
        converts timeframe from disk to file
        Replaces mo with M (to avoid problems on case-insensitive filesystems)
        """
        return re.sub('1mo', '1M', timeframe, flags=re.IGNORECASE)

    @staticmethod
    def rebuild_pair_from_filename(pair: str) -> str:
        """
        Rebuild pair name from filename
        Assumes a asset name of max. 7 length to also support BTC-PERP and BTC-PERP:USD names.
        """
        res = re.sub('^(([A-Za-z\\d]{1,10})|^([A-Za-z\\-]{1,6}))(_)', '\\g<1>/', pair, count=1)
        res = re.sub('_', ':', res, count=1)
        return res

    def ohlcv_load(self, pair: str, timeframe: str, candle_type: CandleType, *, timerange: Optional[TimeRange] = None, fill_missing: bool = True, drop_incomplete: bool = False, startup_candles: int = 0, warn_no_data: bool = True) -> DataFrame:
        """
        Load cached candle (OHLCV) data for the given pair.

        :param pair: Pair to load data for
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange
        :param fill_missing: Fill missing values with "No action"-candles
        :param drop_incomplete: Drop last candle assuming it may be incomplete.
        :param startup_candles: Additional candles to load at the start of the period
        :param warn_no_data: Log a warning message when no data is found
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)
        pairdf = self._ohlcv_load(pair, timeframe, timerange=timerange_startup, candle_type=candle_type)
        if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data):
            return pairdf
        else:
            enddate = pairdf.iloc[-1]['date']
            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timeframe, candle_type, timerange_startup)
                pairdf = trim_dataframe(pairdf, timerange_startup)
                if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data, True):
                    return pairdf
            pairdf = clean_ohlcv_dataframe(pairdf, timeframe, pair=pair, fill_missing=fill_missing, drop_incomplete=drop_incomplete and enddate == pairdf.iloc[-1]['date'])
            self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data)
            return pairdf

    def _check_empty_df(self, pairdf: DataFrame, pair: str, timeframe: str, candle_type: CandleType, warn_no_data: bool, warn_price: bool = False) -> bool:
        """
        Warn on empty dataframe
        """
        if pairdf.empty:
            if warn_no_data:
                logger.warning(f'No history for {pair}, {candle_type}, {timeframe} found. Use `freqtrade download-data` to download the data')
            return True
        elif warn_price:
            candle_price_gap = 0.0
            if candle_type in (CandleType.SPOT, CandleType.FUTURES) and (not pairdf.empty) and ('close' in pairdf.columns) and ('open' in pairdf.columns):
                gaps = (pairdf['open'] - pairdf['close'].shift(1)) / pairdf['close'].shift(1)
                gaps = gaps.dropna()
                if len(gaps):
                    candle_price_gap = max(abs(gaps))
            if candle_price_gap > 0.1:
                logger.info(f'Price jump in {pair}, {timeframe}, {candle_type} between two candles of {candle_price_gap:.2%} detected.')
        return False

    def