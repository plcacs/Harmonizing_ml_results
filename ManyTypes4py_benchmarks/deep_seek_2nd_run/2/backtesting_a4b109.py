"""
This module contains the backtesting logic
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Generator, Union, DefaultDict
from numpy import nan
from pandas import DataFrame
from freqtrade import constants
from freqtrade.configuration import TimeRange, validate_config_consistency
from freqtrade.constants import DATETIME_PRINT_FORMAT, Config, IntOrInf, LongShort
from freqtrade.data import history
from freqtrade.data.btanalysis import find_existing_backtest_stats, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframe, trim_dataframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.metrics import combined_dataframes_with_rel_mean
from freqtrade.enums import BacktestState, CandleType, ExitCheckTuple, ExitType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import amount_to_contract_precision, price_to_precision, timeframe_to_seconds
from freqtrade.exchange.exchange import Exchange
from freqtrade.ft_types import BacktestResultType, get_BacktestResultType_default
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.backtest_caching import get_strategy_run_id
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import generate_backtest_stats, generate_rejected_signals, generate_trade_signal_candles, show_backtest_results, store_backtest_results
from freqtrade.persistence import CustomDataWrapper, LocalTrade, Order, PairLocks, Trade, disable_database_use, enable_database_use
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import FtPrecise, dt_now
from freqtrade.util.migrations import migrate_data
from freqtrade.wallets import Wallets

logger = logging.getLogger(__name__)

DATE_IDX = 0
OPEN_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
CLOSE_IDX = 4
LONG_IDX = 5
ELONG_IDX = 6
SHORT_IDX = 7
ESHORT_IDX = 8
ENTER_TAG_IDX = 9
EXIT_TAG_IDX = 10
HEADERS = ['date', 'open', 'high', 'low', 'close', 'enter_long', 'exit_long', 'enter_short', 'exit_short', 'enter_tag', 'exit_tag']

class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Config, exchange: Optional[Exchange] = None) -> None:
        LoggingMixin.show_output = False
        self.config: Config = config
        self.results: BacktestResultType = get_BacktestResultType_default()
        self.trade_id_counter: int = 0
        self.order_id_counter: int = 0
        config['dry_run'] = True
        self.run_ids: Dict[str, str] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {'signals': {}, 'rejected': {}, 'exited': {}}
        self.rejected_dict: Dict[str, List] = {}
        self._exchange_name: str = self.config['exchange']['name']
        if not exchange:
            exchange = ExchangeResolver.load_exchange(self.config, load_leverage_tiers=True)
        self.exchange: Exchange = exchange
        self.dataprovider: DataProvider = DataProvider(self.config, self.exchange)
        if self.config.get('strategy_list'):
            if self.config.get('freqai', {}).get('enabled', False):
                logger.warning('Using --strategy-list with FreqAI REQUIRES all strategies to have identical feature_engineering_* functions.')
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)
        else:
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)
        if 'timeframe' not in self.config:
            raise OperationalException('Timeframe needs to be set in either configuration or as cli argument `--timeframe 5m`')
        self.timeframe: str = str(self.config.get('timeframe'))
        self.timeframe_secs: int = timeframe_to_seconds(self.timeframe)
        self.timeframe_min: int = self.timeframe_secs // 60
        self.timeframe_td: timedelta = timedelta(seconds=self.timeframe_secs)
        self.disable_database_use()
        self.init_backtest_detail()
        self.pairlists: PairListManager = PairListManager(self.exchange, self.config, self.dataprovider)
        self._validate_pairlists_for_backtesting()
        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()
        if len(self.pairlists.whitelist) == 0:
            raise OperationalException('No pair in whitelist.')
        if config.get('fee', None) is not None:
            self.fee: float = config['fee']
            logger.info(f'Using fee {self.fee:.4%} from config.')
        else:
            fees = [self.exchange.get_fee(symbol=self.pairlists.whitelist[0], taker_or_maker=mt) for mt in ('taker', 'maker')]
            self.fee = max((fee for fee in fees if fee is not None))
            logger.info(f'Using fee {self.fee:.4%} - worst case fee from exchange (lowest tier).')
        self.precision_mode: int = self.exchange.precisionMode
        self.precision_mode_price: int = self.exchange.precision_mode_price
        if self.config.get('freqai_backtest_live_models', False):
            from freqtrade.freqai.utils import get_timerange_backtest_live_models
            self.config['timerange'] = get_timerange_backtest_live_models(self.config)
        self.timerange: TimeRange = TimeRange.parse_timerange(None if self.config.get('timerange') is None else str(self.config.get('timerange')))
        self.required_startup: int = max([strat.startup_candle_count for strat in self.strategylist])
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)
        self.config['startup_candle_count'] = self.required_startup
        if self.config.get('freqai', {}).get('enabled', False):
            self.required_startup = self.dataprovider.get_required_startup(self.timeframe)
        self.trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
        self.margin_mode: MarginMode = config.get('margin_mode', MarginMode.ISOLATED)
        self._can_short: bool = self.trading_mode != TradingMode.SPOT
        self._position_stacking: bool = self.config.get('position_stacking', False)
        self.enable_protections: bool = self.config.get('enable_protections', False)
        migrate_data(config, self.exchange)
        self.init_backtest()

    def _validate_pairlists_for_backtesting(self) -> None:
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException('VolumePairList not allowed for backtesting. Please use StaticPairList instead.')
        if len(self.strategylist) > 1 and 'PrecisionFilter' in self.pairlists.name_list:
            raise OperationalException('PrecisionFilter not allowed for backtesting multiple strategies.')

    @staticmethod
    def cleanup() -> None:
        LoggingMixin.show_output = True
        enable_database_use()

    def init_backtest_detail(self) -> None:
        self.timeframe_detail: str = str(self.config.get('timeframe_detail', ''))
        if self.timeframe_detail:
            timeframe_detail_secs: int = timeframe_to_seconds(self.timeframe_detail)
            self.timeframe_detail_td: timedelta = timedelta(seconds=timeframe_detail_secs)
            if self.timeframe_secs <= timeframe_detail_secs:
                raise OperationalException('Detail timeframe must be smaller than strategy timeframe.')
        else:
            self.timeframe_detail_td = timedelta(seconds=0)
        self.detail_data: Dict[str, DataFrame] = {}
        self.futures_data: Dict[str, DataFrame] = {}

    def init_backtest(self) -> None:
        self.prepare_backtest(False)
        self.wallets: Wallets = Wallets(self.config, self.exchange, is_backtest=True)
        self.progress: BTProgress = BTProgress()
        self.abort: bool = False

    def _set_strategy(self, strategy: IStrategy) -> None:
        """
        Load strategy into backtesting
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        strategy.wallets = self.wallets
        self.strategy.order_types['stoploss_on_exchange'] = False
        self._can_short = self.trading_mode != TradingMode.SPOT and strategy.can_short
        self.strategy.ft_bot_start()

    def _load_protections(self, strategy: IStrategy) -> None:
        if self.config.get('enable_protections', False):
            self.protections: ProtectionManager = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        """
        self.progress.init_step(BacktestState.DATALOAD, 1)
        data: Dict[str, DataFrame] = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=self.timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config['dataformat_ohlcv'],
            candle_type=self.config.get('candle_type_def', CandleType.SPOT)
        )
        min_date, max_date = history.get_timerange(data)
        logger.info(f'Loading data from {min_date.strftime(DATETIME_PRINT_FORMAT)} up to {max_date.strftime(DATETIME_PRINT_FORMAT)} ({(max_date - min_date).days} days).')
        self.timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe), self.required_startup, min_date)
        self.progress.set_new_value(1)
        return (data, self.timerange)

    def load_bt_data_detail(self) -> None:
        """
        Loads backtest detail data (smaller timeframe) if necessary.
        """
        if self.timeframe_detail:
            self.detail_data = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.timeframe_detail,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=self.config.get('candle_type_def', CandleType.SPOT)
            )
        else:
            self.detail_data = {}
        if self.trading_mode == TradingMode.FUTURES:
            funding_fee_timeframe = self.exchange.get_option('funding_fee_timeframe')
            self.funding_fee_timeframe_secs = timeframe_to_seconds(funding_fee_timeframe)
            mark_timeframe = self.exchange.get_option('mark_ohlcv_timeframe')
            funding_rates_dict = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=funding_fee_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.FUNDING_RATE)
            mark_rates_dict = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=mark_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.from_string(self.exchange.get_option('mark_ohlcv_price')))
            unavailable_pairs = []
            for pair in self.pairlists.whitelist:
                if pair not in self.exchange._leverage_tiers:
                    unavailable_pairs.append(pair)
                    continue
                self.futures_data[pair] = self.exchange.combine_funding_and_mark(
                    funding_rates=funding_rates_dict[pair],
                    mark_rates=mark_rates_dict[pair],
                    futures_funding_rate=self.config.get('futures_funding_rate', None))
            if unavailable_pairs:
                raise OperationalException(f'Pairs {", ".join(unavailable_pairs)} got no leverage tiers available. It is therefore impossible to backtest with this pair at the moment.')
        else:
            self.futures_data = {}

    def disable_database_use(self) -> None:
        disable_database_use(self.timeframe)

    def prepare_backtest(self, enable_protections: bool) -> None:
        """
        Backtesting setup method - called once for every call to "backtest()".
        """
        self.disable_database_use()
        PairLocks.reset_locks()
        Trade.reset_trades()
        CustomDataWrapper.reset_custom_data()
        self.rejected_trades: int = 0
        self.timedout_entry_orders: int = 0
        self.timedout_exit_orders: int = 0
        self.canceled_trade_entries: int = 0
        self.canceled_entry_orders: int = 0
        self.replaced_entry_orders: int = 0
        self.dataprovider.clear_cache()
        if enable_protections:
            self._load_protections(self.strategy)

    def check_abort(self) -> None:
        """
        Check if abort was requested, raise DependencyException if that's the case
        Only applies to Interactive backtest mode (webserver mode)
        """
        if self.abort:
            self.abort = False
            raise DependencyException('Stop requested')

    def _get_ohlcv_as_lists(self, processed: Dict[str, DataFrame]) -> Dict[str, List[List[Any]]]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """
        data: Dict[str, List[List[Any]]] = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))
        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()
            if not pair_data.empty:
                pair_data.drop(HEADERS[5:] + ['buy', 'sell'], axis=1, errors='ignore')
            df_analyzed = self.strategy.ft_advise_signals(pair_data, {'pair': pair})
            self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed, self.config['candle_type_def'])
            df_analyzed = processed[pair] = pair_data = trim_dataframe(df_analyzed, self.timerange, startup_candles=self.required_startup)
            df_analyzed = df_analyzed.copy()
            for col in HEADERS[5:]:
                tag_col = col in ('enter_tag', 'exit_tag')
                if col in df_analyzed.columns:
                    df_analyzed[col] = df_analyzed.loc[:, col].replace([nan], [0 if not tag_col else None]).shift(1)
                elif not df_analyzed.empty:
                    df_analyzed[col] = 0 if not tag_col else None
            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)
            data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        return data

    def _get_close_rate(self, row: List[Any], trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        if exit_.exit_type in (ExitType.STOP_LOSS, ExitType.TRAILING_STOP_LOSS, ExitType.LIQUIDATION):
            return self._get_close_rate_for_stoploss(row, trade, exit_, trade_dur)
        elif exit_.exit_type == ExitType.ROI:
            return self._get_close_rate_for_roi(row, trade, exit_, trade_dur)
        else:
            return row[OPEN_IDX]

    def _get_close_rate_for_stoploss(self, row: List[Any], trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int) -> float:
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        if exit_.exit_type == ExitType.LIQUIDATION and trade.liquidation_price:
            stoploss_value = trade.liquidation_price
        else