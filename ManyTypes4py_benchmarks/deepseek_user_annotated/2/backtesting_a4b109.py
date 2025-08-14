# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

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
from freqtrade.enums import (
    BacktestState,
    CandleType,
    ExitCheckTuple,
    ExitType,
    MarginMode,
    RunMode,
    TradingMode,
)
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import (
    amount_to_contract_precision,
    price_to_precision,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange import Exchange
from freqtrade.ft_types import BacktestResultType, get_BacktestResultType_default
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.backtest_caching import get_strategy_run_id
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import (
    generate_backtest_stats,
    generate_rejected_signals,
    generate_trade_signal_candles,
    show_backtest_results,
    store_backtest_results,
)
from freqtrade.persistence import (
    CustomDataWrapper,
    LocalTrade,
    Order,
    PairLocks,
    Trade,
    disable_database_use,
    enable_database_use,
)
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import FtPrecise, dt_now
from freqtrade.util.migrations import migrate_data
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)

# Indexes for backtest tuples
DATE_IDX = 0
OPEN_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
CLOSE_IDX = 4
LONG_IDX = 5
ELONG_IDX = 6  # Exit long
SHORT_IDX = 7
ESHORT_IDX = 8  # Exit short
ENTER_TAG_IDX = 9
EXIT_TAG_IDX = 10

# Every change to this headers list must evaluate further usages of the resulting tuple
# and eventually change the constants for indexes at the top
HEADERS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "enter_long",
    "exit_long",
    "enter_short",
    "exit_short",
    "enter_tag",
    "exit_tag",
]


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Config, exchange: Optional[Exchange] = None) -> None:
        LoggingMixin.show_output = False
        self.config = config
        self.results: BacktestResultType = get_BacktestResultType_default()
        self.trade_id_counter: int = 0
        self.order_id_counter: int = 0

        config["dry_run"] = True
        self.run_ids: Dict[str, str] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}
        self.analysis_results: Dict[str, Dict[str, DataFrame]] = {
            "signals": {},
            "rejected": {},
            "exited": {},
        }
        self.rejected_dict: Dict[str, List] = {}

        self._exchange_name = self.config["exchange"]["name"]
        if not exchange:
            exchange = ExchangeResolver.load_exchange(self.config, load_leverage_tiers=True)
        self.exchange = exchange

        self.dataprovider = DataProvider(self.config, self.exchange)

        if self.config.get("strategy_list"):
            if self.config.get("freqai", {}).get("enabled", False):
                logger.warning(
                    "Using --strategy-list with FreqAI REQUIRES all strategies "
                    "to have identical feature_engineering_* functions."
                )
            for strat in list(self.config["strategy_list"]):
                stratconf = deepcopy(self.config)
                stratconf["strategy"] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)

        else:
            # No strategy list specified, only one strategy
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)

        if "timeframe" not in self.config:
            raise OperationalException(
                "Timeframe needs to be set in either "
                "configuration or as cli argument `--timeframe 5m`"
            )
        self.timeframe = str(self.config.get("timeframe"))
        self.timeframe_secs = timeframe_to_seconds(self.timeframe)
        self.timeframe_min = self.timeframe_secs // 60
        self.timeframe_td = timedelta(seconds=self.timeframe_secs)
        self.disable_database_use()
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config, self.dataprovider)
        self._validate_pairlists_for_backtesting()

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get("fee", None) is not None:
            self.fee = config["fee"]
            logger.info(f"Using fee {self.fee:.4%} from config.")
        else:
            fees = [
                self.exchange.get_fee(
                    symbol=self.pairlists.whitelist[0],
                    taker_or_maker=mt,  # type: ignore
                )
                for mt in ("taker", "maker")
            ]
            self.fee = max(fee for fee in fees if fee is not None)
            logger.info(f"Using fee {self.fee:.4%} - worst case fee from exchange (lowest tier).")
        self.precision_mode = self.exchange.precisionMode
        self.precision_mode_price = self.exchange.precision_mode_price

        if self.config.get("freqai_backtest_live_models", False):
            from freqtrade.freqai.utils import get_timerange_backtest_live_models

            self.config["timerange"] = get_timerange_backtest_live_models(self.config)

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get("timerange") is None else str(self.config.get("timerange"))
        )

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)

        # Add maximum startup candle count to configuration for informative pairs support
        self.config["startup_candle_count"] = self.required_startup

        if self.config.get("freqai", {}).get("enabled", False):
            # For FreqAI, increase the required_startup to includes the training data
            # This value should NOT be written to startup_candle_count
            self.required_startup = self.dataprovider.get_required_startup(self.timeframe)

        self.trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = config.get("margin_mode", MarginMode.ISOLATED)
        # strategies which define "can_short=True" will fail to load in Spot mode.
        self._can_short = self.trading_mode != TradingMode.SPOT
        self._position_stacking: bool = self.config.get("position_stacking", False)
        self.enable_protections: bool = self.config.get("enable_protections", False)
        migrate_data(config, self.exchange)

        self.init_backtest()

    def _validate_pairlists_for_backtesting(self) -> None:
        if "VolumePairList" in self.pairlists.name_list:
            raise OperationalException(
                "VolumePairList not allowed for backtesting. Please use StaticPairList instead."
            )

        if len(self.strategylist) > 1 and "PrecisionFilter" in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

    @staticmethod
    def cleanup() -> None:
        LoggingMixin.show_output = True
        enable_database_use()

    def init_backtest_detail(self) -> None:
        # Load detail timeframe if specified
        self.timeframe_detail = str(self.config.get("timeframe_detail", ""))
        if self.timeframe_detail:
            timeframe_detail_secs = timeframe_to_seconds(self.timeframe_detail)
            self.timeframe_detail_td = timedelta(seconds=timeframe_detail_secs)
            if self.timeframe_secs <= timeframe_detail_secs:
                raise OperationalException(
                    "Detail timeframe must be smaller than strategy timeframe."
                )

        else:
            self.timeframe_detail_td = timedelta(seconds=0)
        self.detail_data: Dict[str, DataFrame] = {}
        self.futures_data: Dict[str, DataFrame] = {}

    def init_backtest(self) -> None:
        self.prepare_backtest(False)

        self.wallets = Wallets(self.config, self.exchange, is_backtest=True)

        self.progress = BTProgress()
        self.abort = False

    def _set_strategy(self, strategy: IStrategy) -> None:
        """
        Load strategy into backtesting
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        strategy.wallets = self.wallets
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-exit is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types["stoploss_on_exchange"] = False
        # Update can_short flag
        self._can_short = self.trading_mode != TradingMode.SPOT and strategy.can_short

        self.strategy.ft_bot_start()

    def _load_protections(self, strategy: IStrategy) -> None:
        if self.config.get("enable_protections", False):
            self.protections = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        """
        self.progress.init_step(BacktestState.DATALOAD, 1)

        data = history.load_data(
            datadir=self.config["datadir"],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=self.timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config["dataformat_ohlcv"],
            candle_type=self.config.get("candle_type_def", CandleType.SPOT),
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(
            f"Loading data from {min_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"up to {max_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"({(max_date - min_date).days} days)."
        )

        # Adjust startts forward if not enough data is available
        self.timerange.adjust_start_if_necessary(
            timeframe_to_seconds(self.timeframe), self.required_startup, min_date
        )

        self.progress.set_new_value(1)
        return data, self.timerange

    def load_bt_data_detail(self) -> None:
        """
        Loads backtest detail data (smaller timeframe) if necessary.
        """
        if self.timeframe_detail:
            self.detail_data = history.load_data(
                datadir=self.config["datadir"],
                pairs=self.pairlists.whitelist,
                timeframe=self.timeframe_detail,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config["dataformat_ohlcv"],
                candle_type=self.config.get("candle_type_def", CandleType.SPOT),
            )
        else:
            self.detail_data = {}
        if self.trading_mode == TradingMode.FUTURES:
            funding_fee_timeframe: str = self.exchange.get_option("funding_fee_timeframe")
            self.funding_fee_timeframe_secs: int = timeframe_to_seconds(funding_fee_timeframe)
            mark_timeframe: str = self.exchange.get_option("mark_ohlcv_timeframe")

            # Load additional futures data.
            funding_rates_dict = history.load_data(
                datadir=self.config["datadir"],
                pairs=self.pairlists.whitelist,
                timeframe=funding_fee_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config["dataformat_ohlcv"],
                candle_type=CandleType.FUNDING_RATE,
            )

            # For simplicity, assign to CandleType.Mark (might contain index candles!)
            mark_rates_dict = history.load_data(
                datadir=self.config["datadir"],
                pairs=self.pairlists.whitelist,
                timeframe=mark_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config["dataformat_ohlcv"],
                candle_type=CandleType.from_string(self.exchange.get_option("mark_ohlcv_price")),
            )
            # Combine data to avoid combining the data per trade.
            unavailable_pairs = []
            for pair in self.pairlists.whitelist:
                if pair not in self.exchange._leverage_tiers:
                    unavailable_pairs.append(pair)
                    continue

                self.futures_data[pair] = self.exchange.combine_funding_and_mark(
                    funding_rates=funding_rates_dict[pair],
                    mark_rates=mark_rates_dict[pair],
                    futures_funding_rate=self.config.get("futures_funding_rate", None),
                )

            if unavailable_pairs:
                raise OperationalException(
                    f"Pairs {', '.join(unavailable_pairs)} got no leverage tiers available. "
                    "It is therefore impossible to backtest with this pair at the moment."
                )
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
        self.rejected_trades = 0
        self.timedout_entry_orders = 0
        self.timedout_exit_orders = 0
        self.canceled_trade_entries = 0
        self.canceled_entry_orders = 0
        self.replaced_entry_orders = 0
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
            raise DependencyException("Stop requested")

    def _get_ohlcv_as_lists(self, processed: Dict[str, DataFrame]) -> Dict[str, Tuple]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """

        data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        # Create dict with data
        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()

            if not pair_data.empty:
                # Cleanup from prior runs
                pair_data.drop(HEADERS[5:] + ["buy", "sell"], axis=1, errors="ignore")
            df_analyzed = self.strategy.ft_advise_signals(pair_data, {"pair": pair})
            # Update dataprovider cache
            self.dataprovider._set_cached_df(
                pair, self.timeframe, df_analyzed, self.config["candle_type_def"]
            )

            # Trim startup period from analyzed dataframe
            df_analyzed = processed[pair] = pair_data = trim_dataframe(
                df_analyzed, self.timerange, startup_candles=self.required_startup
            )

            # Create a copy of the dataframe before shifting, that way the entry signal/tag
            # remains on the correct candle for callbacks.
            df_analyzed = df_analyzed.copy()

            # To avoid using data from future, we use entry/exit signals shifted
            # from the previous candle
            for col in