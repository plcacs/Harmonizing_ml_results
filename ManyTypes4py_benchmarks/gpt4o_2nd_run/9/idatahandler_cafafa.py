import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from pandas import DataFrame, to_datetime
from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS, ListPairsWithTimeframes
from freqtrade.data.converter import clean_ohlcv_dataframe, trades_convert_types, trades_df_remove_duplicates, trim_dataframe
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class IDataHandler(ABC):
    _OHLCV_REGEX = '^([a-zA-Z_\\d-]+)\\-(\\d+[a-zA-Z]{1,2})\\-?([a-zA-Z_]*)?(?=\\.)'
    _TRADES_REGEX = '^([a-zA-Z_\\d-]+)\\-(trades)?(?=\\.)'

    def __init__(self, datadir: Path) -> None:
        self._datadir = datadir

    @classmethod
    def _get_file_extension(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> List[Tuple[str, str, CandleType]]:
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [re.search(cls._OHLCV_REGEX, p.name) for p in datadir.glob(f'*.{cls._get_file_extension()}')]
        return [(cls.rebuild_pair_from_filename(match[1]), cls.rebuild_timeframe_from_filename(match[2]), CandleType.from_string(match[3])) for match in _tmp if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> List[str]:
        candle = ''
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f'-{candle_type}'
        ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-' + timeframe + candle + f'.{ext})', p.name) for p in datadir.glob(f'*{timeframe}{candle}.{ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        pass

    def ohlcv_data_min_max(self, pair: str, timeframe: str, candle_type: CandleType) -> Tuple[datetime, datetime, int]:
        df = self._ohlcv_load(pair, timeframe, None, candle_type)
        if df.empty:
            return (datetime.fromtimestamp(0, tz=timezone.utc), datetime.fromtimestamp(0, tz=timezone.utc), 0)
        return (df.iloc[0]['date'].to_pydatetime(), df.iloc[-1]['date'].to_pydatetime(), len(df))

    @abstractmethod
    def _ohlcv_load(self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType) -> DataFrame:
        pass

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @abstractmethod
    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        pass

    @classmethod
    def trades_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> List[str]:
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [re.search(cls._TRADES_REGEX, p.name) for p in datadir.glob(f'*.{cls._get_file_extension()}')]
        return [cls.rebuild_pair_from_filename(match[1]) for match in _tmp if match and len(match.groups()) > 1]

    def trades_data_min_max(self, pair: str, trading_mode: TradingMode) -> Tuple[datetime, datetime, int]:
        df = self._trades_load(pair, trading_mode)
        if df.empty:
            return (datetime.fromtimestamp(0, tz=timezone.utc), datetime.fromtimestamp(0, tz=timezone.utc), 0)
        return (to_datetime(df.iloc[0]['timestamp'], unit='ms', utc=True).to_pydatetime(), to_datetime(df.iloc[-1]['timestamp'], unit='ms', utc=True).to_pydatetime(), len(df))

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        _ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-trades.' + _ext + ')', p.name) for p in datadir.glob(f'*trades.{_ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        pass

    @abstractmethod
    def trades_append(self, pair: str, data: DataFrame) -> None:
        pass

    @abstractmethod
    def _trades_load(self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None) -> DataFrame:
        pass

    def trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        self._trades_store(pair, data[DEFAULT_TRADES_COLUMNS], trading_mode)

    def trades_purge(self, pair: str, trading_mode: TradingMode) -> bool:
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def trades_load(self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None) -> DataFrame:
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
        return re.sub('1mo', '1M', timeframe, flags=re.IGNORECASE)

    @staticmethod
    def rebuild_pair_from_filename(pair: str) -> str:
        res = re.sub('^(([A-Za-z\\d]{1,10})|^([A-Za-z\\-]{1,6}))(_)', '\\g<1>/', pair, count=1)
        res = re.sub('_', ':', res, count=1)
        return res

    def ohlcv_load(self, pair: str, timeframe: str, candle_type: CandleType, *, timerange: Optional[TimeRange] = None, fill_missing: bool = True, drop_incomplete: bool = False, startup_candles: int = 0, warn_no_data: bool = True) -> DataFrame:
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
        if pairdf.empty:
            if warn_no_data:
                logger.warning(f'No history for {pair}, {candle_type}, {timeframe} found. Use `freqtrade download-data` to download the data')
            return True
        elif warn_price:
            candle_price_gap = 0
            if candle_type in (CandleType.SPOT, CandleType.FUTURES) and (not pairdf.empty) and ('close' in pairdf.columns) and ('open' in pairdf.columns):
                gaps = (pairdf['open'] - pairdf['close'].shift(1)) / pairdf['close'].shift(1)
                gaps = gaps.dropna()
                if len(gaps):
                    candle_price_gap = max(abs(gaps))
            if candle_price_gap > 0.1:
                logger.info(f'Price jump in {pair}, {timeframe}, {candle_type} between two candles of {candle_price_gap:.2%} detected.')
        return False

    def _validate_pairdata(self, pair: str, pairdata: DataFrame, timeframe: str, candle_type: CandleType, timerange: TimeRange) -> None:
        if timerange.starttype == 'date':
            if pairdata.iloc[0]['date'] > timerange.startdt:
                logger.warning(f'{pair}, {candle_type}, {timeframe}, data starts at {pairdata.iloc[0]["date"]:%Y-%m-%d %H:%M:%S}')
        if timerange.stoptype == 'date':
            if pairdata.iloc[-1]['date'] < timerange.stopdt:
                logger.warning(f'{pair}, {candle_type}, {timeframe}, data ends at {pairdata.iloc[-1]["date"]:%Y-%m-%d %H:%M:%S}')

    def rename_futures_data(self, pair: str, new_pair: str, timeframe: str, candle_type: CandleType) -> None:
        file_old = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        file_new = self._pair_data_filename(self._datadir, new_pair, timeframe, candle_type)
        if file_new.exists():
            logger.warning(f"{file_new} exists already, can't migrate {pair}.")
            return
        file_old.rename(file_new)

    def fix_funding_fee_timeframe(self, ff_timeframe: str) -> None:
        paircombs = self.ohlcv_get_available_data(self._datadir, TradingMode.FUTURES)
        funding_rate_combs = [f for f in paircombs if f[2] == CandleType.FUNDING_RATE and f[1] != ff_timeframe]
        if funding_rate_combs:
            logger.warning(f'Migrating {len(funding_rate_combs)} funding fees to correct timeframe.')
        for pair, timeframe, candletype in funding_rate_combs:
            old_name = self._pair_data_filename(self._datadir, pair, timeframe, candletype)
            new_name = self._pair_data_filename(self._datadir, pair, ff_timeframe, candletype)
            if not Path(old_name).exists():
                logger.warning(f'{old_name} does not exist, skipping.')
                continue
            if Path(new_name).exists():
                logger.warning(f'{new_name} already exists, Removing.')
                Path(new_name).unlink()
            Path(old_name).rename(new_name)

def get_datahandlerclass(datatype: str):
    if datatype == 'json':
        from .jsondatahandler import JsonDataHandler
        return JsonDataHandler
    elif datatype == 'jsongz':
        from .jsondatahandler import JsonGzDataHandler
        return JsonGzDataHandler
    elif datatype == 'hdf5':
        raise OperationalException('DEPRECATED: The hdf5 dataformat is deprecated and has been removed in 2025.1. Please downgrade to 2024.12 and use the convert-data command to convert your data to a supported format.We recommend using the feather format, as it is faster and is more space-efficient.')
    elif datatype == 'feather':
        from .featherdatahandler import FeatherDataHandler
        return FeatherDataHandler
    elif datatype == 'parquet':
        from .parquetdatahandler import ParquetDataHandler
        return ParquetDataHandler
    else:
        raise ValueError(f'No datahandler for datatype {datatype} available.')

def get_datahandler(datadir: Path, data_format: Optional[str] = None, data_handler: Optional[IDataHandler] = None) -> IDataHandler:
    if not data_handler:
        HandlerClass = get_datahandlerclass(data_format or 'feather')
        data_handler = HandlerClass(datadir)
    return data_handler
