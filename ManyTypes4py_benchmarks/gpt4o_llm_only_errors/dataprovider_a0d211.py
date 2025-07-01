import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Timedelta, Timestamp, to_timedelta
from freqtrade.configuration import TimeRange
from freqtrade.constants import FULL_DATAFRAME_THRESHOLD, Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.data.history import get_datahandler, load_pair_history
from freqtrade.enums import CandleType, RPCMessageType, RunMode, TradingMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.misc import append_candles_to_dataframe
from freqtrade.rpc import RPCManager
from freqtrade.rpc.rpc_types import RPCAnalyzedDFMsg
from freqtrade.util import PeriodicCache

logger = logging.getLogger(__name__)
NO_EXCHANGE_EXCEPTION = 'Exchange is not available to DataProvider.'
MAX_DATAFRAME_CANDLES = 1000

class DataProvider:

    def __init__(self, config: Config, exchange: Exchange, pairlists: Optional[ListPairsWithTimeframes] = None, rpc: Optional[RPCManager] = None) -> None:
        self._config = config
        self._exchange = exchange
        self._pairlists = pairlists
        self.__rpc = rpc
        self.__cached_pairs: Dict[Tuple[str, str, CandleType], Tuple[DataFrame, datetime]] = {}
        self.__slice_index: Optional[int] = None
        self.__slice_date: Optional[datetime] = None
        self.__cached_pairs_backtesting: Dict[Tuple[str, str, CandleType], DataFrame] = {}
        self.__producer_pairs_df: Dict[str, Dict[Tuple[str, str, CandleType], Tuple[DataFrame, datetime]]] = {}
        self.__producer_pairs: Dict[str, List[str]] = {}
        self._msg_queue: deque[str] = deque()
        self._default_candle_type: CandleType = self._config.get('candle_type_def', CandleType.SPOT)
        self._default_timeframe: str = self._config.get('timeframe', '1h')
        self.__msg_cache: PeriodicCache[str, bool] = PeriodicCache(maxsize=1000, ttl=timeframe_to_seconds(self._default_timeframe))
        self.producers: List[str] = self._config.get('external_message_consumer', {}).get('producers', [])
        self.external_data_enabled: bool = len(self.producers) > 0

    def _set_dataframe_max_index(self, limit_index: int) -> None:
        self.__slice_index = limit_index

    def _set_dataframe_max_date(self, limit_date: datetime) -> None:
        self.__slice_date = limit_date

    def _set_cached_df(self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType) -> None:
        pair_key = (pair, timeframe, candle_type)
        self.__cached_pairs[pair_key] = (dataframe, datetime.now(timezone.utc))

    def _set_producer_pairs(self, pairlist: List[str], producer_name: str = 'default') -> None:
        self.__producer_pairs[producer_name] = pairlist

    def get_producer_pairs(self, producer_name: str = 'default') -> List[str]:
        return self.__producer_pairs.get(producer_name, []).copy()

    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None:
        if self.__rpc:
            msg = {'type': RPCMessageType.ANALYZED_DF, 'data': {'key': pair_key, 'df': dataframe.tail(1), 'la': datetime.now(timezone.utc)}}
            self.__rpc.send_msg(msg)
            if new_candle:
                self.__rpc.send_msg({'type': RPCMessageType.NEW_CANDLE, 'data': pair_key})

    def _replace_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Optional[datetime], timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> None:
        pair_key = (pair, timeframe, candle_type)
        if producer_name not in self.__producer_pairs_df:
            self.__producer_pairs_df[producer_name] = {}
        _last_analyzed = datetime.now(timezone.utc) if not last_analyzed else last_analyzed
        self.__producer_pairs_df[producer_name][pair_key] = (dataframe, _last_analyzed)
        logger.debug(f'External DataFrame for {pair_key} from {producer_name} added.')

    def _add_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Optional[datetime], timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> Tuple[bool, int]:
        pair_key = (pair, timeframe, candle_type)
        if dataframe.empty:
            return (False, 0)
        if len(dataframe) >= FULL_DATAFRAME_THRESHOLD:
            self._replace_external_df(pair, dataframe, last_analyzed=last_analyzed, timeframe=timeframe, candle_type=candle_type, producer_name=producer_name)
            return (True, 0)
        if producer_name not in self.__producer_pairs_df or pair_key not in self.__producer_pairs_df[producer_name]:
            return (False, 1000)
        existing_df, _ = self.__producer_pairs_df[producer_name][pair_key]
        timeframe_delta = to_timedelta(timeframe)
        local_last = existing_df.iloc[-1]['date']
        incoming_first = dataframe.iloc[0]['date']
        existing_df1 = existing_df[existing_df['date'] < incoming_first]
        candle_difference = (incoming_first - local_last) / timeframe_delta
        if candle_difference > 1:
            return (False, int(candle_difference))
        if existing_df1.empty:
            appended_df = dataframe
        else:
            appended_df = append_candles_to_dataframe(existing_df1, dataframe)
        self._replace_external_df(pair, appended_df, last_analyzed=last_analyzed, timeframe=timeframe, candle_type=candle_type, producer_name=producer_name)
        return (True, 0)

    def get_producer_df(self, pair: str, timeframe: Optional[str] = None, candle_type: Optional[CandleType] = None, producer_name: str = 'default') -> Tuple[DataFrame, datetime]:
        _timeframe = self._default_timeframe if not timeframe else timeframe
        _candle_type = self._default_candle_type if not candle_type else candle_type
        pair_key = (pair, _timeframe, _candle_type)
        if producer_name not in self.__producer_pairs_df:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))
        if pair_key not in self.__producer_pairs_df[producer_name]:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))
        df, la = self.__producer_pairs_df[producer_name][pair_key]
        return (df.copy(), la)

    def add_pairlisthandler(self, pairlists: ListPairsWithTimeframes) -> None:
        self._pairlists = pairlists

    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = '') -> DataFrame:
        _candle_type = CandleType.from_string(candle_type) if candle_type != '' else self._config['candle_type_def']
        saved_pair = (pair, str(timeframe), _candle_type)
        if saved_pair not in self.__cached_pairs_backtesting:
            timerange = TimeRange.parse_timerange(None if self._config.get('timerange') is None else str(self._config.get('timerange')))
            startup_candles = self.get_required_startup(str(timeframe))
            tf_seconds = timeframe_to_seconds(str(timeframe))
            timerange.subtract_start(tf_seconds * startup_candles)
            logger.info(f'Loading data for {pair} {timeframe} from {timerange.start_fmt} to {timerange.stop_fmt}')
            self.__cached_pairs_backtesting[saved_pair] = load_pair_history(pair=pair, timeframe=timeframe, datadir=self._config['datadir'], timerange=timerange, data_format=self._config['dataformat_ohlcv'], candle_type=_candle_type)
        return self.__cached_pairs_backtesting[saved_pair].copy()

    def get_required_startup(self, timeframe: str) -> int:
        freqai_config = self._config.get('freqai', {})
        if not freqai_config.get('enabled', False):
            return self._config.get('startup_candle_count', 0)
        else:
            startup_candles = self._config.get('startup_candle_count', 0)
            indicator_periods = freqai_config['feature_parameters']['indicator_periods_candles']
            self._config['startup_candle_count'] = max(startup_candles, max(indicator_periods))
            tf_seconds = timeframe_to_seconds(timeframe)
            train_candles = freqai_config['train_period_days'] * 86400 / tf_seconds
            total_candles = int(self._config['startup_candle_count'] + train_candles)
            logger.info(f'Increasing startup_candle_count for freqai on {timeframe} to {total_candles}')
        return total_candles

    def get_pair_dataframe(self, pair: str, timeframe: Optional[str] = None, candle_type: str = '') -> DataFrame:
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            data = self.ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
        else:
            timeframe = timeframe or self._config['timeframe']
            data = self.historic_ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
            if self.__slice_date:
                cutoff_date = timeframe_to_prev_date(timeframe, self.__slice_date)
                data = data.loc[data['date'] < cutoff_date]
        if len(data) == 0:
            logger.warning(f'No data found for ({pair}, {timeframe}, {candle_type}).')
        return data

    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, datetime]:
        pair_key = (pair, timeframe, self._config.get('candle_type_def', CandleType.SPOT))
        if pair_key in self.__cached_pairs:
            if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
                df, date = self.__cached_pairs[pair_key]
            else:
                df, date = self.__cached_pairs[pair_key]
                if self.__slice_index is not None:
                    max_index = self.__slice_index
                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES):max_index]
            return (df, date)
        else:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

    @property
    def runmode(self) -> RunMode:
        return RunMode(self._config.get('runmode', RunMode.OTHER))

    def current_whitelist(self) -> List[str]:
        if self._pairlists:
            return self._pairlists.whitelist.copy()
        else:
            raise OperationalException('Dataprovider was not initialized with a pairlist provider.')

    def clear_cache(self) -> None:
        self.__cached_pairs = {}
        self.__slice_index = 0

    def refresh(self, pairlist: List[str], helping_pairs: Optional[List[str]] = None) -> None:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        final_pairs = pairlist + helping_pairs if helping_pairs else pairlist
        self._exchange.refresh_latest_ohlcv(final_pairs)
        self.refresh_latest_trades(pairlist)

    def refresh_latest_trades(self, pairlist: List[str]) -> None:
        use_public_trades = self._config.get('exchange', {}).get('use_public_trades', False)
        if use_public_trades:
            if self._exchange:
                self._exchange.refresh_latest_trades(pairlist)

    @property
    def available_pairs(self) -> List[Tuple[str, str]]:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return list(self._exchange._klines.keys())

    def ohlcv(self, pair: str, timeframe: Optional[str] = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            _candle_type = CandleType.from_string(candle_type) if candle_type != '' else self._config['candle_type_def']
            return self._exchange.klines((pair, timeframe or self._config['timeframe'], _candle_type), copy=copy)
        else:
            return DataFrame()

    def trades(self, pair: str, timeframe: Optional[str] = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            if self._exchange is None:
                raise OperationalException(NO_EXCHANGE_EXCEPTION)
            _candle_type = CandleType.from_string(candle_type) if candle_type != '' else self._config['candle_type_def']
            return self._exchange.trades((pair, timeframe or self._config['timeframe'], _candle_type), copy=copy)
        else:
            data_handler = get_datahandler(self._config['datadir'], data_format=self._config['dataformat_trades'])
            trades_df = data_handler.trades_load(pair, self._config.get('trading_mode', TradingMode.SPOT))
            return trades_df

    def market(self, pair: str) -> Optional[Dict[str, Any]]:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.markets.get(pair)

    def ticker(self, pair: str) -> Dict[str, Any]:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        try:
            return self._exchange.fetch_ticker(pair)
        except ExchangeError:
            return {}

    def orderbook(self, pair: str, maximum: int) -> OrderBook:
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.fetch_l2_order_book(pair, maximum)

    def send_msg(self, message: str, *, always_send: bool = False) -> None:
        if self.runmode not in (RunMode.DRY_RUN, RunMode.LIVE):
            return
        if always_send or message not in self.__msg_cache:
            self._msg_queue.append(message)
        self.__msg_cache[message] = True
