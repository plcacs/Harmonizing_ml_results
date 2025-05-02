from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd

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
        self.analysis_results: Dict[str, Dict[str, pd.DataFrame]] = {
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
        self.detail_data: Dict[str, pd.DataFrame] = {}
        self.futures_data: Dict[str, pd.DataFrame] = {}

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

    def load_bt_data(self) -> Tuple[Dict[str, pd.DataFrame], TimeRange]:
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

    def _get_ohlcv_as_lists(self, processed: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple]]:
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
            for col in HEADERS[5:]:
                tag_col = col in ("enter_tag", "exit_tag")
                if col in df_analyzed.columns:
                    df_analyzed[col] = (
                        df_analyzed.loc[:, col]
                        .replace([np.nan], [0 if not tag_col else None])
                        .shift(1)
                    )
                elif not df_analyzed.empty:
                    df_analyzed[col] = 0 if not tag_col else None

            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        return data

    def _get_close_rate(
        self, row: Tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if exit_.exit_type in (
            ExitType.STOP_LOSS,
            ExitType.TRAILING_STOP_LOSS,
            ExitType.LIQUIDATION,
        ):
            return self._get_close_rate_for_stoploss(row, trade, exit_, trade_dur)
        elif exit_.exit_type == (ExitType.ROI):
            return self._get_close_rate_for_roi(row, trade, exit_, trade_dur)
        else:
            return row[OPEN_IDX]

    def _get_close_rate_for_stoploss(
        self, row: Tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        # our stoploss was already lower than candle high,
        # possibly due to a cancelled trade exit.
        # exit at open price.
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        if exit_.exit_type == ExitType.LIQUIDATION and trade.liquidation_price:
            stoploss_value = trade.liquidation_price
        else:
            stoploss_value = trade.stop_loss

        if is_short:
            if stoploss_value < row[LOW_IDX]:
                return row[OPEN_IDX]
        else:
            if stoploss_value > row[HIGH_IDX]:
                return row[OPEN_IDX]

        # Special case: trailing triggers within same candle as trade opened. Assume most
        # pessimistic price movement, which is moving just enough to arm stoploss and
        # immediately going down to stop price.
        if exit_.exit_type == ExitType.TRAILING_STOP_LOSS and trade_dur == 0:
            if (
                not self.strategy.use_custom_stoploss
                and self.strategy.trailing_stop
                and self.strategy.trailing_only_offset_is_reached
                and self.strategy.trailing_stop_positive_offset is not None
                and self.strategy.trailing_stop_positive
            ):
                # Worst case: price reaches stop_positive_offset and dives down.
                stop_rate = row[OPEN_IDX] * (
                    1
                    + side_1 * abs(self.strategy.trailing_stop_positive_offset)
                    - side_1 * abs(self.strategy.trailing_stop_positive / leverage)
                )
            else:
                # Worst case: price ticks tiny bit above open and dives down.
                stop_rate = row[OPEN_IDX] * (
                    1 - side_1 * abs((trade.stop_loss_pct or 0.0) / leverage)
                )

            # Limit lower-end to candle low to avoid exits below the low.
            # This still remains "worst case" - but "worst realistic case".
            if is_short:
                return min(row[HIGH_IDX], stop_rate)
            else:
                return max(row[LOW_IDX], stop_rate)

        # Set close_rate to stoploss
        return stoploss_value

    def _get_close_rate_for_roi(
        self, row: Tuple, trade: LocalTrade, exit_: ExitCheckTuple, trade_dur: int
    ) -> float:
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
        if roi is not None and roi_entry is not None:
            if roi == -1 and roi_entry % self.timeframe_min == 0:
                # When force_exiting with ROI=-1, the roi time will always be equal to trade_dur.
                # If that entry is a multiple of the timeframe (so on candle open)
                # - we'll use open instead of close
                return row[OPEN_IDX]

            # - (Expected abs profit - open_rate - open_fee) / (fee_close -1)
            roi_rate = trade.open_rate * roi / leverage
            open_fee_rate = side_1 * trade.open_rate * (1 + side_1 * trade.fee_open)
            close_rate = -(roi_rate + open_fee_rate) / ((trade.fee_close or 0.0) - side_1 * 1)
            if is_short:
                is_new_roi = row[OPEN_IDX] < close_rate
            else:
                is_new_roi = row[OPEN_IDX] > close_rate
            if (
                trade_dur > 0
                and trade_dur == roi_entry
                and roi_entry % self.timeframe_min == 0
                and is_new_roi
            ):
                # new ROI entry came into effect.
                # use Open rate if open_rate > calculated exit rate
                return row[OPEN_IDX]

            if trade_dur == 0 and (
                (
                    is_short
                    # Red candle (for longs)
                    and row[OPEN_IDX] < row[CLOSE_IDX]  # Red candle
                    and trade.open_rate > row[OPEN_IDX]  # trade-open above open_rate
                    and close_rate < row[CLOSE_IDX]  # closes below close
                )
                or (
                    not is_short
                    # green candle (for shorts)
                    and row[OPEN_IDX] > row[CLOSE_IDX]  # green candle
                    and trade.open_rate < row[OPEN_IDX]  # trade-open below open_rate
                    and close_rate > row[CLOSE_IDX]  # closes above close
                )
            ):
                # ROI on opening candles with custom pricing can only
                # trigger if the entry was at Open or lower wick.
                # details: https: // github.com/freqtrade/freqtrade/issues/6261
                # If open_rate is < open, only allow exits below the close on red candles.
                raise ValueError("Opening candle ROI on red candles.")

            # Use the maximum between close_rate and low as we
            # cannot exit outside of a candle.
            # Applies when a new ROI setting comes in place and the whole candle is above that.
            return min(max(close_rate, row[LOW_IDX]), row[HIGH_IDX])

        else:
            # This should not be reached...
            return row[OPEN_IDX]

    def _get_adjust_trade_entry_for_candle(
        self, trade: LocalTrade, row: Tuple, current_time: datetime
    ) -> LocalTrade:
        current_rate: float = row[OPEN_IDX]
        current_profit = trade.calc_profit_ratio(current_rate)
        min_stake = self.exchange.get_min_pair_stake_amount(trade.pair, current_rate, -0.1)
        max_stake = self.exchange.get_max_pair_stake_amount(trade.pair, current_rate)
        stake_available = self.wallets.get_available_stake_amount()
        stake_amount, order_tag = self.strategy._adjust_trade_position_internal(
            trade=trade,  # type: ignore[arg-type]
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=min(max_stake, stake_available),
            current_entry_rate=current_rate,
            current_exit_rate=current_rate,
            current_entry_profit=current_profit,
            current_exit_profit=current_profit,
        )

        # Check if we should increase our position
        if stake_amount is not None and stake_amount > 0.0:
            check_adjust_entry = True
            if self.strategy.max_entry_position_adjustment > -1:
                entry_count = trade.nr_of_successful_entries
                check_adjust_entry = entry_count <= self.strategy.max_entry_position_adjustment
            if check_adjust_entry:
                pos_trade = self._enter_trade(
                    trade.pair,
                    row,
                    "short" if trade.is_short else "long",
                    stake_amount,
                    trade,
                    entry_tag1=order_tag,
                )
                if pos_trade is not None:
                    self.wallets.update()
                    return pos_trade

        if stake_amount is not None and stake_amount < 0.0:
            amount = amount_to_contract_precision(
                abs(
                    float(
                        FtPrecise(stake_amount)
                        * FtPrecise(trade.amount)
                        / FtPrecise(trade.stake_amount)
                    )
                ),
                trade.amount_precision,
                self.precision_mode,
                trade.contract_size,
            )
            if amount == 0.0:
                return trade
            remaining = (trade.amount - amount) * current_rate
            if min_stake and remaining != 0 and remaining < min_stake:
                # Remaining stake is too low to be sold.
                return trade
            exit_ = ExitCheckTuple(ExitType.PARTIAL_EXIT, order_tag)
            pos_trade = self._get_exit_for_signal(trade, row, exit_, current_time, amount)
            if pos_trade is not None:
                order = pos_trade.orders[-1]
                # If the order was filled and for the full trade amount, we need to close the trade.
                self._process_exit_order(order, pos_trade, current_time, row, trade.pair)
                return pos_trade

        return trade

    def _get_order_filled(self, rate: float, row: Tuple) -> bool:
        """Rate is within candle, therefore filled"""
        return row[LOW_IDX] <= rate <= row[HIGH_IDX]

    def _call_adjust_stop(self, current_date: datetime, trade: LocalTrade, current_rate: float):
        profit = trade.calc_profit_ratio(current_rate)
        self.strategy.ft_stoploss_adjust(
            current_rate,
            trade,  # type: ignore
            current_date,
            profit,
            0,
            after_fill=True,
        )

    def _try_close_open_order(
        self, order: Optional[Order], trade: LocalTrade, current_date: datetime, row: Tuple
    ) -> bool:
        """
        Check if an order is open and if it should've filled.
        :return:  True if the order filled.
        """
        if order and self._get_order_filled(order.ft_price, row):
            order.close_bt_order(current_date, trade)
            self._run_funding_fees(trade, current_date, force=True)
            strategy_safe_wrapper(self.strategy.order_filled, default_retval=None)(
                pair=trade.pair,
                trade=trade,  # type: ignore[arg-type]
                order=order,
                current_time=current_date,
            )

            if self.margin_mode == MarginMode.CROSS or not (
                order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount
            ):
                # trade is still open or we are in cross margin mode and
                # must update all liquidation prices
                update_liquidation_prices(
                    trade,
                    exchange=self.exchange,
                    wallets=self.wallets,
                    stake_currency=self.config["stake_currency"],
                    dry_run=self.config["dry_run"],
                )
            if not (order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount):
                self._call_adjust_stop(current_date, trade, order.ft_price)
            return True
        return False

    def _process_exit_order(
        self, order: Order, trade: LocalTrade, current_time: datetime, row: Tuple, pair: str
    ):
        """
        Takes an exit order and processes it, potentially closing the trade.
        """
        if self._try_close_open_order(order, trade, current_time, row):
            sub_trade = order.safe_amount_after_fee != trade.amount
            if sub_trade:
                trade.recalc_trade_from_orders()
            else:
                trade.close_date = current_time
                trade.close(order.ft_price, show_msg=False)

                LocalTrade.close_bt_trade(trade)
            self.wallets.update()
            self.run_protections(pair, current_time, trade.trade_direction)

    def _get_exit_for_signal(
        self,
        trade: LocalTrade,
        row: Tuple,
        exit_: ExitCheckTuple,
        current_time: datetime,
        amount: Optional[float] = None,
    ) -> Optional[LocalTrade]:
        if exit_.exit_flag:
            trade.close_date = current_time
            exit_reason = exit_.exit_reason
            amount_ = amount if amount is not None else trade.amount
            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            try:
                close_rate = self._get_close_rate(row, trade, exit_, trade_dur)
            except ValueError:
                return None
            # call the custom exit price,with default value as previous close_rate
            current_profit = trade.calc_profit_ratio(close_rate)
            order_type = self.strategy.order_types["exit"]
            if exit_.exit_type in (
                ExitType.EXIT_SIGNAL,
                ExitType.CUSTOM_EXIT,
                ExitType.PARTIAL_EXIT,
            ):
                # Checks and adds an exit tag, after checking that the length of the
                # row has the length for an exit tag column
                if (
                    len(row) > EXIT_TAG_IDX
                    and row[EXIT_TAG_IDX] is not None
                    and len(row[EXIT_TAG_IDX]) > 0
                    and exit_.exit_type in (ExitType.EXIT_SIGNAL,)
                ):
                    exit_reason = row[EXIT_TAG_IDX]
                # Custom exit pricing only for exit-signals
                if order_type == "limit":
                    rate = strategy_safe_wrapper(
                        self.strategy.custom_exit_price, default_retval=close_rate
                    )(
                        pair=trade.pair,
                        trade=trade,  # type: ignore[arg-type]
                        current_time=current_time,
                        proposed_rate=close_rate,
                        current_profit=current_profit,
                        exit_tag=exit_reason,
                    )
                    if rate is not None and rate != close_rate:
                        close_rate = price_to_precision(
                            rate, trade.price_precision, self.precision_mode_price
                        )
                    # We can't place orders lower than current low.
                    # freqtrade does not support this in live, and the order would fill immediately
                    if trade.is_short:
                        close_rate = min(close_rate, row[HIGH_IDX])
                    else:
                        close_rate = max(close_rate, row[LOW_IDX])
            # Confirm trade exit:
            time_in_force = self.strategy.order_time_in_force["exit"]

            if exit_.exit_type not in (
                ExitType.LIQUIDATION,
                ExitType.PARTIAL_EXIT,
            ) and not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair,
                trade=trade,  # type: ignore[arg-type]
                order_type=order_type,
                amount=amount_,
                rate=close_rate,
                time_in_force=time_in_force,
                sell_reason=exit_reason,  # deprecated
                exit_reason=exit_reason,
                current_time=current_time,
            ):
                return None

            trade.exit_reason = exit_reason

            return self._exit_trade(trade, row, close_rate, amount_, exit_reason)
        return None

    def _exit_trade(
        self,
        trade: LocalTrade,
        sell_row: Tuple,
        close_rate: float,
        amount: float,
        exit_reason: Optional[str],
    ) -> Optional[LocalTrade]:
        self.order_id_counter += 1
        exit_candle_time = sell_row[DATE_IDX].to_pydatetime()
        order_type = self.strategy.order_types["exit"]
        # amount = amount or trade.amount
        amount = amount_to_contract_precision(
            amount or trade.amount, trade.amount_precision, self.precision_mode, trade.contract_size
        )

        if self.handle_similar_order(trade, close_rate, amount, trade.exit_side, exit_candle_time):
            return None

        order = Order(
            id=self.order_id_counter,
            ft_trade_id=trade.id,
            order_date=exit_candle_time,
            order_update_date=exit_candle_time,
            ft_is_open=True,
            ft_pair=trade.pair,
            order_id=str(self.order_id_counter),
            symbol=trade.pair,
            ft_order_side=trade.exit_side,
            side=trade.exit_side,
            order_type=order_type,
            status="open",
            ft_price=close_rate,
            price=close_rate,
            average=