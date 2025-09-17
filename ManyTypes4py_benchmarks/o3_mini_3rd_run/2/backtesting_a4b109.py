#!/usr/bin/env python3
"""
This module contains the backtesting logic
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Generator
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
HEADERS: List[str] = ['date', 'open', 'high', 'low', 'close', 'enter_long', 'exit_long', 'enter_short', 'exit_short', 'enter_tag', 'exit_tag']


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
        self.run_ids: Dict[str, Any] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Any] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {'signals': {}, 'rejected': {}, 'exited': {}}
        self.rejected_dict: Dict[str, List[List[Any]]] = {}
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
        self.precision_mode = self.exchange.precisionMode
        self.precision_mode_price = self.exchange.precision_mode_price
        if self.config.get('freqai_backtest_live_models', False):
            from freqtrade.freqai.utils import get_timerange_backtest_live_models
            self.config['timerange'] = get_timerange_backtest_live_models(self.config)
        self.timerange: TimeRange = TimeRange.parse_timerange(
            None if self.config.get('timerange') is None else str(self.config.get('timerange'))
        )
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
        self.detail_data: Dict[str, Any] = {}
        self.futures_data: Dict[str, Any] = {}

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
        return data, self.timerange

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
            funding_fee_timeframe: str = self.exchange.get_option('funding_fee_timeframe')
            self.funding_fee_timeframe_secs: int = timeframe_to_seconds(funding_fee_timeframe)
            mark_timeframe: str = self.exchange.get_option('mark_ohlcv_timeframe')
            funding_rates_dict: Dict[str, DataFrame] = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=funding_fee_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.FUNDING_RATE
            )
            mark_rates_dict: Dict[str, DataFrame] = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=mark_timeframe,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.from_string(self.exchange.get_option('mark_ohlcv_price'))
            )
            unavailable_pairs: List[str] = []
            for pair in self.pairlists.whitelist:
                if pair not in self.exchange._leverage_tiers:
                    unavailable_pairs.append(pair)
                    continue
                self.futures_data[pair] = self.exchange.combine_funding_and_mark(
                    funding_rates=funding_rates_dict[pair],
                    mark_rates=mark_rates_dict[pair],
                    futures_funding_rate=self.config.get('futures_funding_rate', None)
                )
            if unavailable_pairs:
                raise OperationalException(f"Pairs {', '.join(unavailable_pairs)} got no leverage tiers available. It is therefore impossible to backtest with this pair at the moment.")
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
        """
        data: Dict[str, List[List[Any]]] = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))
        for pair in processed.keys():
            pair_data: DataFrame = processed[pair]
            self.check_abort()
            self.progress.increment()
            if not pair_data.empty:
                pair_data.drop(HEADERS[5:] + ['buy', 'sell'], axis=1, errors='ignore', inplace=True)
            df_analyzed: DataFrame = self.strategy.ft_advise_signals(pair_data, {'pair': pair})
            self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed, self.config['candle_type_def'])
            df_analyzed = processed[pair] = pair_data = trim_dataframe(df_analyzed, self.timerange, startup_candles=self.required_startup)
            df_analyzed = df_analyzed.copy()
            for col in HEADERS[5:]:
                tag_col: bool = (col in ('enter_tag', 'exit_tag'))
                if col in df_analyzed.columns:
                    df_analyzed[col] = df_analyzed.loc[:, col].replace([nan], [0 if not tag_col else None]).shift(1)
                elif not df_analyzed.empty:
                    df_analyzed[col] = 0 if not tag_col else None
            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)
            data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        return data

    def _get_close_rate(self, row: List[Any], trade: Trade, exit_: ExitCheckTuple, trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        if exit_.exit_type in (ExitType.STOP_LOSS, ExitType.TRAILING_STOP_LOSS, ExitType.LIQUIDATION):
            return self._get_close_rate_for_stoploss(row, trade, exit_, trade_dur)
        elif exit_.exit_type == ExitType.ROI:
            return self._get_close_rate_for_roi(row, trade, exit_, trade_dur)
        else:
            return row[OPEN_IDX]

    def _get_close_rate_for_stoploss(self, row: List[Any], trade: Trade, exit_: ExitCheckTuple, trade_dur: int) -> float:
        is_short: bool = trade.is_short or False
        leverage: float = trade.leverage or 1.0
        side_1: int = -1 if is_short else 1
        if exit_.exit_type == ExitType.LIQUIDATION and trade.liquidation_price:
            stoploss_value: float = trade.liquidation_price
        else:
            stoploss_value = trade.stop_loss
        if is_short:
            if stoploss_value < row[LOW_IDX]:
                return row[OPEN_IDX]
        elif stoploss_value > row[HIGH_IDX]:
            return row[OPEN_IDX]
        if exit_.exit_type == ExitType.TRAILING_STOP_LOSS and trade_dur == 0:
            if not self.strategy.use_custom_stoploss and self.strategy.trailing_stop and self.strategy.trailing_only_offset_is_reached and (self.strategy.trailing_stop_positive_offset is not None) and self.strategy.trailing_stop_positive:
                stop_rate: float = row[OPEN_IDX] * (1 + side_1 * abs(self.strategy.trailing_stop_positive_offset) - side_1 * abs(self.strategy.trailing_stop_positive / leverage))
            else:
                stop_rate = row[OPEN_IDX] * (1 - side_1 * abs((trade.stop_loss_pct or 0.0) / leverage))
            if is_short:
                return min(row[HIGH_IDX], stop_rate)
            else:
                return max(row[LOW_IDX], stop_rate)
        return stoploss_value

    def _get_close_rate_for_roi(self, row: List[Any], trade: Trade, exit_: ExitCheckTuple, trade_dur: int) -> float:
        is_short: bool = trade.is_short or False
        leverage: float = trade.leverage or 1.0
        side_1: int = -1 if is_short else 1
        roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
        if roi is not None and roi_entry is not None:
            if roi == -1 and roi_entry % self.timeframe_min == 0:
                return row[OPEN_IDX]
            roi_rate: float = trade.open_rate * roi / leverage
            open_fee_rate: float = side_1 * trade.open_rate * (1 + side_1 * trade.fee_open)
            close_rate: float = -(roi_rate + open_fee_rate) / ((trade.fee_close or 0.0) - side_1 * 1)
            if is_short:
                is_new_roi: bool = row[OPEN_IDX] < close_rate
            else:
                is_new_roi = row[OPEN_IDX] > close_rate
            if trade_dur > 0 and trade_dur == roi_entry and (roi_entry % self.timeframe_min == 0) and is_new_roi:
                return row[OPEN_IDX]
            if trade_dur == 0 and (is_short and row[OPEN_IDX] < row[CLOSE_IDX] and (trade.open_rate > row[OPEN_IDX]) and (close_rate < row[CLOSE_IDX]) or (not is_short and row[OPEN_IDX] > row[CLOSE_IDX] and (trade.open_rate < row[OPEN_IDX]) and (close_rate > row[CLOSE_IDX]))):
                raise ValueError('Opening candle ROI on red candles.')
            return min(max(close_rate, row[LOW_IDX]), row[HIGH_IDX])
        else:
            return row[OPEN_IDX]

    def _get_adjust_trade_entry_for_candle(self, trade: Trade, row: List[Any], current_time: datetime) -> Trade:
        current_rate: float = row[OPEN_IDX]
        current_profit: float = trade.calc_profit_ratio(current_rate)
        min_stake: float = self.exchange.get_min_pair_stake_amount(trade.pair, current_rate, -0.1)
        max_stake: float = self.exchange.get_max_pair_stake_amount(trade.pair, current_rate)
        stake_available: float = self.wallets.get_available_stake_amount()
        stake_amount, order_tag = self.strategy._adjust_trade_position_internal(
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=min(max_stake, stake_available),
            current_entry_rate=current_rate,
            current_exit_rate=current_rate,
            current_entry_profit=current_profit,
            current_exit_profit=current_profit
        )
        if stake_amount is not None and stake_amount > 0.0:
            check_adjust_entry: bool = True
            if self.strategy.max_entry_position_adjustment > -1:
                entry_count: int = trade.nr_of_successful_entries
                check_adjust_entry = entry_count <= self.strategy.max_entry_position_adjustment
            if check_adjust_entry:
                pos_trade: Optional[Trade] = self._enter_trade(trade.pair, row, 'short' if trade.is_short else 'long', stake_amount, trade, entry_tag1=order_tag)
                if pos_trade is not None:
                    self.wallets.update()
                    return pos_trade
        if stake_amount is not None and stake_amount < 0.0:
            amount: float = amount_to_contract_precision(
                abs(float(FtPrecise(stake_amount) * FtPrecise(trade.amount) / FtPrecise(trade.stake_amount))),
                trade.amount_precision,
                self.precision_mode,
                trade.contract_size
            )
            if amount == 0.0:
                return trade
            remaining: float = (trade.amount - amount) * current_rate
            if min_stake and remaining != 0 and (remaining < min_stake):
                return trade
            exit_ = ExitCheckTuple(ExitType.PARTIAL_EXIT, order_tag)
            pos_trade = self._get_exit_for_signal(trade, row, exit_, current_time, amount)
            if pos_trade is not None:
                order = pos_trade.orders[-1]
                self._process_exit_order(order, pos_trade, current_time, row, trade.pair)
                return pos_trade
        return trade

    def _get_order_filled(self, rate: float, row: List[Any]) -> bool:
        """Rate is within candle, therefore filled"""
        return row[LOW_IDX] <= rate <= row[HIGH_IDX]

    def _call_adjust_stop(self, current_date: datetime, trade: Trade, current_rate: float) -> None:
        profit: float = trade.calc_profit_ratio(current_rate)
        self.strategy.ft_stoploss_adjust(current_rate, trade, current_date, profit, 0, after_fill=True)

    def _try_close_open_order(self, order: Optional[Order], trade: Trade, current_date: datetime, row: List[Any]) -> bool:
        """
        Check if an order is open and if it should've filled.
        :return: True if the order filled.
        """
        if order and self._get_order_filled(order.ft_price, row):
            order.close_bt_order(current_date, trade)
            self._run_funding_fees(trade, current_date, force=True)
            strategy_safe_wrapper(self.strategy.order_filled, default_retval=None)(
                pair=trade.pair,
                trade=trade,
                order=order,
                current_time=current_date
            )
            if self.margin_mode == MarginMode.CROSS or not (order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount):
                update_liquidation_prices(trade, exchange=self.exchange, wallets=self.wallets, stake_currency=self.config['stake_currency'], dry_run=self.config['dry_run'])
            if not (order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount):
                self._call_adjust_stop(current_date, trade, order.ft_price)
            return True
        return False

    def _process_exit_order(self, order: Order, trade: Trade, current_time: datetime, row: List[Any], pair: str) -> None:
        """
        Takes an exit order and processes it, potentially closing the trade.
        """
        if self._try_close_open_order(order, trade, current_time, row):
            sub_trade: bool = order.safe_amount_after_fee != trade.amount
            if sub_trade:
                trade.recalc_trade_from_orders()
            else:
                trade.close_date = current_time
                trade.close(order.ft_price, show_msg=False)
                LocalTrade.close_bt_trade(trade)
            self.wallets.update()
            self.run_protections(pair, current_time, trade.trade_direction)

    def _get_exit_for_signal(self, trade: Trade, row: List[Any], exit_: ExitCheckTuple, current_time: datetime, amount: Optional[float] = None) -> Optional[Trade]:
        if exit_.exit_flag:
            trade.close_date = current_time
            exit_reason: str = exit_.exit_reason
            amount_ = amount if amount is not None else trade.amount
            trade_dur: int = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            try:
                close_rate: float = self._get_close_rate(row, trade, exit_, trade_dur)
            except ValueError:
                return None
            current_profit: float = trade.calc_profit_ratio(close_rate)
            order_type: str = self.strategy.order_types['exit']
            if exit_.exit_type in (ExitType.EXIT_SIGNAL, ExitType.CUSTOM_EXIT, ExitType.PARTIAL_EXIT):
                if len(row) > EXIT_TAG_IDX and row[EXIT_TAG_IDX] is not None and (len(row[EXIT_TAG_IDX]) > 0) and (exit_.exit_type in (ExitType.EXIT_SIGNAL,)):
                    exit_reason = row[EXIT_TAG_IDX]
                if order_type == 'limit':
                    rate: Optional[float] = strategy_safe_wrapper(self.strategy.custom_exit_price, default_retval=close_rate)(
                        pair=trade.pair,
                        trade=trade,
                        current_time=current_time,
                        proposed_rate=close_rate,
                        current_profit=current_profit,
                        exit_tag=exit_reason
                    )
                    if rate is not None and rate != close_rate:
                        close_rate = price_to_precision(rate, trade.price_precision, self.precision_mode_price)
                    if trade.is_short:
                        close_rate = min(close_rate, row[HIGH_IDX])
                    else:
                        close_rate = max(close_rate, row[LOW_IDX])
            time_in_force: str = self.strategy.order_time_in_force['exit']
            if exit_.exit_type not in (ExitType.LIQUIDATION, ExitType.PARTIAL_EXIT) and (not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair,
                trade=trade,
                order_type=order_type,
                amount=amount_,
                rate=close_rate,
                time_in_force=time_in_force,
                sell_reason=exit_reason,
                exit_reason=exit_reason,
                current_time=current_time
            )):
                return None
            trade.exit_reason = exit_reason
            return self._exit_trade(trade, row, close_rate, amount_, exit_reason)
        return None

    def _exit_trade(self, trade: Trade, sell_row: List[Any], close_rate: float, amount: float, exit_reason: str) -> Trade:
        self.order_id_counter += 1
        exit_candle_time: datetime = sell_row[DATE_IDX].to_pydatetime()
        order_type: str = self.strategy.order_types['exit']
        amount = amount_to_contract_precision(amount or trade.amount, trade.amount_precision, self.precision_mode, trade.contract_size)
        if self.handle_similar_order(trade, close_rate, amount, trade.exit_side, exit_candle_time):
            return trade
        order: Order = Order(
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
            status='open',
            ft_price=close_rate,
            price=close_rate,
            average=close_rate,
            amount=amount,
            filled=0,
            remaining=amount,
            cost=amount * close_rate * (1 + self.fee),
            ft_order_tag=exit_reason
        )
        order._trade_bt = trade
        trade.orders.append(order)
        return trade

    def _check_trade_exit(self, trade: Trade, row: List[Any], current_time: datetime) -> Optional[Trade]:
        self._run_funding_fees(trade, current_time)
        if self.strategy.position_adjustment_enable:
            trade = self._get_adjust_trade_entry_for_candle(trade, row, current_time)
        if trade.is_open:
            enter: Any = row[SHORT_IDX] if trade.is_short else row[LONG_IDX]
            exit_sig: Any = row[ESHORT_IDX] if trade.is_short else row[ELONG_IDX]
            exits: List[ExitCheckTuple] = self.strategy.should_exit(trade, row[OPEN_IDX], row[DATE_IDX].to_pydatetime(), enter=enter, exit_=exit_sig, low=row[LOW_IDX], high=row[HIGH_IDX])
            for exit_ in exits:
                t: Optional[Trade] = self._get_exit_for_signal(trade, row, exit_, current_time)
                if t:
                    return t
        return None

    def _run_funding_fees(self, trade: Trade, current_time: datetime, force: bool = False) -> None:
        """
        Calculate funding fees if necessary and add them to the trade.
        """
        if self.trading_mode == TradingMode.FUTURES:
            if force or current_time.timestamp() % self.funding_fee_timeframe_secs == 0:
                trade.set_funding_fees(
                    self.exchange.calculate_funding_fees(
                        self.futures_data[trade.pair],
                        amount=trade.amount,
                        is_short=trade.is_short,
                        open_date=trade.date_last_filled_utc,
                        close_date=current_time
                    )
                )

    def get_valid_price_and_stake(self, pair: str, row: List[Any], propose_rate: float, stake_amount: float, direction: str, current_time: datetime, entry_tag: Any, trade: Optional[Trade], order_type: str, price_precision: int) -> Tuple[float, float, float, float]:
        if order_type == 'limit':
            new_rate: Optional[float] = strategy_safe_wrapper(self.strategy.custom_entry_price, default_retval=propose_rate)(
                pair=pair,
                trade=trade,
                current_time=current_time,
                proposed_rate=propose_rate,
                entry_tag=entry_tag,
                side=direction
            )
            if new_rate is not None and new_rate != propose_rate:
                propose_rate = price_to_precision(new_rate, price_precision, self.precision_mode_price)
            if direction == 'short':
                propose_rate = max(propose_rate, row[LOW_IDX])
            else:
                propose_rate = min(propose_rate, row[HIGH_IDX])
        pos_adjust: bool = trade is not None
        leverage: float = trade.leverage if trade else 1.0
        if not pos_adjust:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(pair, self.strategy.max_open_trades, update=False)
            except DependencyException:
                return (0, 0, 0, 0)
            max_leverage: float = self.exchange.get_max_leverage(pair, stake_amount)
            leverage = strategy_safe_wrapper(self.strategy.leverage, default_retval=1.0)(
                pair=pair,
                current_time=current_time,
                current_rate=row[OPEN_IDX],
                proposed_leverage=1.0,
                max_leverage=max_leverage,
                side=direction,
                entry_tag=entry_tag
            ) if self.trading_mode != TradingMode.SPOT else 1.0
            leverage = min(max(leverage, 1.0), max_leverage)
        min_stake_amount: float = self.exchange.get_min_pair_stake_amount(pair, propose_rate, -0.05 if not pos_adjust else 0.0, leverage=leverage) or 0
        max_stake_amount: float = self.exchange.get_max_pair_stake_amount(pair, propose_rate, leverage=leverage)
        stake_available: float = self.wallets.get_available_stake_amount()
        if not pos_adjust:
            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount, default_retval=stake_amount)(
                pair=pair,
                current_time=current_time,
                current_rate=propose_rate,
                proposed_stake=stake_amount,
                min_stake=min_stake_amount,
                max_stake=min(stake_available, max_stake_amount),
                leverage=leverage,
                entry_tag=entry_tag,
                side=direction
            )
        stake_amount_val: float = self.wallets.validate_stake_amount(
            pair=pair,
            stake_amount=stake_amount,
            min_stake_amount=min_stake_amount,
            max_stake_amount=max_stake_amount,
            trade_amount=trade.stake_amount if trade else None
        )
        return propose_rate, stake_amount_val, leverage, min_stake_amount

    def _enter_trade(self, pair: str, row: List[Any], direction: str, stake_amount: Optional[float] = None, trade: Optional[Trade] = None, requested_rate: Optional[float] = None, requested_stake: Optional[float] = None, entry_tag1: Optional[Any] = None) -> Optional[Trade]:
        """
        :param trade: Trade to adjust - initial entry if None
        :param requested_rate: Adjusted entry rate
        :param requested_stake: Stake amount for adjusted orders (`adjust_entry_price`).
        """
        current_time: datetime = row[DATE_IDX].to_pydatetime()
        entry_tag: Any = entry_tag1 or (row[ENTER_TAG_IDX] if len(row) >= ENTER_TAG_IDX + 1 else None)
        order_type: str = self.strategy.order_types['entry']
        pos_adjust: bool = trade is not None and requested_rate is None
        stake_amount_ = stake_amount or (trade.stake_amount if trade else 0.0)
        precision_price: int = self.exchange.get_precision_price(pair)
        propose_rate, stake_amount_new, leverage, min_stake_amount = self.get_valid_price_and_stake(pair, row, row[OPEN_IDX], stake_amount_, direction, current_time, entry_tag, trade, order_type, precision_price)
        propose_rate = requested_rate if requested_rate else propose_rate
        stake_amount_new = requested_stake if requested_stake else stake_amount_new
        if not stake_amount_new:
            return trade
        time_in_force: str = self.strategy.order_time_in_force['entry']
        if stake_amount_new and (not min_stake_amount or stake_amount_new >= min_stake_amount):
            self.order_id_counter += 1
            base_currency: str = self.exchange.get_pair_base_currency(pair)
            amount_p: float = stake_amount_new / propose_rate * leverage
            contract_size: float = self.exchange.get_contract_size(pair)
            precision_amount: int = self.exchange.get_precision_amount(pair)
            amount: float = amount_to_contract_precision(amount_p, precision_amount, self.precision_mode, contract_size)
            if not amount:
                return trade
            stake_amount_new = amount * propose_rate / leverage
            if not pos_adjust:
                if not strategy_safe_wrapper(self.strategy.confirm_trade_entry, default_retval=True)(
                    pair=pair,
                    order_type=order_type,
                    amount=amount,
                    rate=propose_rate,
                    time_in_force=time_in_force,
                    current_time=current_time,
                    entry_tag=entry_tag,
                    side=direction
                ):
                    return trade
            is_short: bool = direction == 'short'
            if trade is None:
                self.trade_id_counter += 1
                trade = LocalTrade(
                    id=self.trade_id_counter,
                    pair=pair,
                    base_currency=base_currency,
                    stake_currency=self.config['stake_currency'],
                    open_rate=propose_rate,
                    open_rate_requested=propose_rate,
                    open_date=current_time,
                    stake_amount=stake_amount_new,
                    amount=0,
                    amount_requested=amount,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                    enter_tag=entry_tag,
                    timeframe=self.timeframe_min,
                    exchange=self._exchange_name,
                    is_short=is_short,
                    trading_mode=self.trading_mode,
                    leverage=leverage,
                    amount_precision=precision_amount,
                    price_precision=precision_price,
                    precision_mode=self.precision_mode,
                    precision_mode_price=self.precision_mode_price,
                    contract_size=contract_size,
                    orders=[]
                )
                LocalTrade.add_bt_trade(trade)
            elif self.handle_similar_order(trade, propose_rate, amount, trade.entry_side, current_time):
                return None
            trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)
            order: Order = Order(
                id=self.order_id_counter,
                ft_trade_id=trade.id,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_id=str(self.order_id_counter),
                symbol=trade.pair,
                ft_order_side=trade.entry_side,
                side=trade.entry_side,
                order_type=order_type,
                status='open',
                order_date=current_time,
                order_filled_date=current_time,
                order_update_date=current_time,
                ft_price=propose_rate,
                price=propose_rate,
                average=propose_rate,
                amount=amount,
                filled=0,
                remaining=amount,
                cost=amount * propose_rate * (1 + self.fee),
                ft_order_tag=entry_tag
            )
            order._trade_bt = trade
            trade.orders.append(order)
            self._try_close_open_order(order, trade, current_time, row)
            trade.recalc_trade_from_orders()
        return trade

    def handle_left_open(self, open_trades: Dict[str, List[Trade]], data: Dict[str, List[List[Any]]]) -> None:
        """
        Handling of left open trades at the end of backtesting
        """
        for pair in open_trades.keys():
            for trade in list(open_trades[pair]):
                if trade.has_open_orders and trade.nr_of_successful_entries == 0 or not trade.has_open_position:
                    LocalTrade.remove_bt_trade(trade)
                    continue
                exit_row: List[Any] = data[pair][-1]
                self._exit_trade(trade, exit_row, exit_row[OPEN_IDX], trade.amount, ExitType.FORCE_EXIT.value)
                trade.exit_reason = ExitType.FORCE_EXIT.value
                self._process_exit_order(trade.orders[-1], trade, exit_row[DATE_IDX].to_pydatetime(), exit_row, pair)

    def trade_slot_available(self, open_trade_count: int) -> bool:
        max_open_trades: int = self.strategy.max_open_trades
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        self.rejected_trades += 1
        return False

    def check_for_trade_entry(self, row: List[Any]) -> Optional[str]:
        enter_long: bool = row[LONG_IDX] == 1
        exit_long: bool = row[ELONG_IDX] == 1
        enter_short: bool = self._can_short and row[SHORT_IDX] == 1
        exit_short: bool = self._can_short and row[ESHORT_IDX] == 1
        if enter_long and (not any([exit_long, enter_short])):
            return 'long'
        if enter_short and (not any([exit_short, enter_long])):
            return 'short'
        return None

    def run_protections(self, pair: str, current_time: datetime, side: Any) -> None:
        if self.enable_protections:
            self.protections.stop_per_pair(pair, current_time, side)
            self.protections.global_stop(current_time, side)

    def manage_open_orders(self, trade: Trade, current_time: datetime, row: List[Any]) -> bool:
        """
        Check if any open order needs to be cancelled or replaced.
        Returns True if the trade should be deleted.
        """
        for order in [o for o in trade.orders if o.ft_is_open]:
            oc: Optional[bool] = self.check_order_cancel(trade, order, current_time)
            if oc:
                return True
            elif oc is None and self.check_order_replace(trade, order, current_time, row):
                self.canceled_trade_entries += 1
                return True
        return False

    def cancel_open_orders(self, trade: Trade, current_time: datetime) -> None:
        """
        Cancel all open orders for the given trade.
        """
        for order in [o for o in trade.orders if o.ft_is_open]:
            if order.side == trade.entry_side:
                self.canceled_entry_orders += 1
            del trade.orders[trade.orders.index(order)]

    def handle_similar_order(self, trade: Trade, price: float, amount: float, side: str, current_time: datetime) -> bool:
        """
        Handle similar order for the given trade.
        """
        if trade.has_open_orders:
            oo: Optional[Order] = trade.select_order(side, True)
            if oo:
                if price == oo.price and side == oo.side and (amount == oo.amount):
                    return True
            self.cancel_open_orders(trade, current_time)
        return False

    def check_order_cancel(self, trade: Trade, order: Order, current_time: datetime) -> Optional[bool]:
        """
        Check if current analyzed order has to be canceled.
        Returns True if the trade should be Deleted (initial order was canceled),
                False if it's Canceled,
                None if the order is still active.
        """
        timedout: bool = self.strategy.ft_check_timed_out(trade, order, current_time)
        if timedout:
            if order.side == trade.entry_side:
                self.timedout_entry_orders += 1
                if trade.nr_of_successful_entries == 0:
                    return True
                else:
                    del trade.orders[trade.orders.index(order)]
                    return False
            if order.side == trade.exit_side:
                self.timedout_exit_orders += 1
                del trade.orders[trade.orders.index(order)]
                return False
        return None

    def check_order_replace(self, trade: Trade, order: Order, current_time: datetime, row: List[Any]) -> bool:
        """
        Check if current analyzed entry order has to be replaced and do so.
        If user requested cancellation and there are no filled orders in the trade will
        instruct caller to delete the trade.
        Returns True if the trade should be deleted.
        """
        if order.side == trade.entry_side and current_time > order.order_date_utc:
            requested_rate: Optional[float] = strategy_safe_wrapper(self.strategy.adjust_entry_price, default_retval=order.ft_price)(
                trade=trade,
                order=order,
                pair=trade.pair,
                current_time=current_time,
                proposed_rate=row[OPEN_IDX],
                current_order_rate=order.ft_price,
                entry_tag=trade.enter_tag,
                side=trade.trade_direction
            )
            if requested_rate == order.ft_price:
                return False
            else:
                del trade.orders[trade.orders.index(order)]
                self.canceled_entry_orders += 1
            if requested_rate:
                self._enter_trade(pair=trade.pair, row=row, trade=trade, requested_rate=requested_rate, requested_stake=order.safe_remaining * order.ft_price / trade.leverage, direction='short' if trade.is_short else 'long')
                if not trade.has_open_orders and trade.nr_of_successful_entries == 0:
                    return True
                self.replaced_entry_orders += 1
            else:
                return trade.nr_of_successful_entries == 0
        return False

    def validate_row(self, data: Dict[str, List[List[Any]]], pair: str, row_index: int, current_time: datetime) -> Optional[List[Any]]:
        try:
            row: List[Any] = data[pair][row_index]
        except IndexError:
            return None
        if row[DATE_IDX] > current_time:
            return None
        return row

    def _collate_rejected(self, pair: str, row: List[Any]) -> None:
        """
        Temporarily store rejected signal information for downstream use in backtesting_analysis
        """
        if self.config.get('export', 'none') == 'signals' and self.dataprovider.runmode == RunMode.BACKTEST:
            if pair not in self.rejected_dict:
                self.rejected_dict[pair] = []
            self.rejected_dict[pair].append([row[DATE_IDX], row[ENTER_TAG_IDX]])

    def backtest_loop(self, row: List[Any], pair: str, current_time: datetime, trade_dir: Optional[str], can_enter: bool) -> Optional[str]:
        """
        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.

        Backtesting processing for one candle/pair.
        """
        exiting_dir: Optional[str] = None
        if not self._position_stacking and len(LocalTrade.bt_trades_open_pp[pair]) > 0:
            exiting_dir = 'short' if LocalTrade.bt_trades_open_pp[pair][0].is_short else 'long'
        for t in list(LocalTrade.bt_trades_open_pp[pair]):
            if self.manage_open_orders(t, current_time, row):
                LocalTrade.remove_bt_trade(t)
                self.wallets.update()
        if can_enter and trade_dir is not None and (self._position_stacking or len(LocalTrade.bt_trades_open_pp[pair]) == 0) and (not PairLocks.is_pair_locked(pair, row[DATE_IDX], trade_dir)):
            if self.trade_slot_available(LocalTrade.bt_open_open_trade_count):
                trade: Optional[Trade] = self._enter_trade(pair, row, trade_dir)
                if trade:
                    self.wallets.update()
            else:
                self._collate_rejected(pair, row)
        for trade in list(LocalTrade.bt_trades_open_pp[pair]):
            order: Optional[Order] = trade.select_order(trade.entry_side, is_open=True)
            if self._try_close_open_order(order, trade, current_time, row):
                self.wallets.update()
            if trade.has_open_position:
                self._check_trade_exit(trade, row, current_time)
            order = trade.select_order(trade.exit_side, is_open=True)
            if order:
                self._process_exit_order(order, trade, current_time, row, pair)
        if exiting_dir and len(LocalTrade.bt_trades_open_pp[pair]) == 0:
            return exiting_dir
        return None

    def get_detail_data(self, pair: str, row: List[Any]) -> Optional[List[List[Any]]]:
        """
        Spread into detail data
        """
        current_detail_time: datetime = row[DATE_IDX].to_pydatetime()
        exit_candle_end: datetime = current_detail_time + self.timeframe_td
        detail_data = self.detail_data[pair]
        detail_data = detail_data.loc[(detail_data['date'] >= current_detail_time) & (detail_data['date'] < exit_candle_end)].copy()
        if len(detail_data) == 0:
            return None
        detail_data.loc[:, 'enter_long'] = row[LONG_IDX]
        detail_data.loc[:, 'exit_long'] = row[ELONG_IDX]
        detail_data.loc[:, 'enter_short'] = row[SHORT_IDX]
        detail_data.loc[:, 'exit_short'] = row[ESHORT_IDX]
        detail_data.loc[:, 'enter_tag'] = row[ENTER_TAG_IDX]
        detail_data.loc[:, 'exit_tag'] = row[EXIT_TAG_IDX]
        return detail_data[HEADERS].values.tolist()

    def _time_generator(self, start_date: datetime, end_date: datetime) -> Generator[datetime, None, None]:
        current_time: datetime = start_date + self.timeframe_td
        while current_time <= end_date:
            yield current_time
            current_time += self.timeframe_td

    def _time_generator_det(self, start_date: datetime, end_date: datetime) -> Generator[Tuple[datetime, bool, bool, int], None, None]:
        """
        Loop for each detail candle.
        Yields only the start date if no detail timeframe is set.
        """
        if not self.timeframe_detail_td:
            yield (start_date, True, False, 0)
            return
        current_time: datetime = start_date
        i: int = 0
        while current_time <= end_date:
            yield (current_time, i == 0, True, i)
            i += 1
            current_time += self.timeframe_detail_td

    def _time_pair_generator_det(self, current_time: datetime, pairs: List[str]) -> Generator[Tuple[datetime, bool, bool, int, str], None, None]:
        for current_time_det, is_first, has_detail, idx in self._time_generator_det(current_time, current_time + self.timeframe_td):
            new_pairlist: List[str] = list(dict.fromkeys([t.pair for t in LocalTrade.bt_trades_open] + pairs))
            for pair in new_pairlist:
                yield (current_time_det, is_first, has_detail, idx, pair)

    def time_pair_generator(self, start_date: datetime, end_date: datetime, pairs: List[str], data: Dict[str, List[List[Any]]]) -> Generator[Tuple[datetime, str, List[Any], bool, Optional[str]], None, None]:
        """
        Backtest time and pair generator
        :returns: generator of (current_time, pair, row, is_last_row, trade_dir)
            where is_last_row is a boolean indicating if this is the data end date.
        """
        current_time: datetime = start_date + self.timeframe_td
        self.progress.init_step(BacktestState.BACKTEST, int((end_date - start_date) / self.timeframe_td))
        indexes: Dict[str, int] = defaultdict(int)
        for current_time in self._time_generator(start_date, end_date):
            self.check_abort()
            strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(current_time=current_time)
            pair_detail_cache: Dict[str, List[List[Any]]] = {}
            pair_tradedir_cache: Dict[str, Optional[str]] = {}
            pairs_with_open_trades: List[str] = [t.pair for t in LocalTrade.bt_trades_open]
            for current_time_det, is_first, has_detail, idx, pair in self._time_pair_generator_det(current_time, pairs):
                trade_dir: Optional[str] = None
                if is_first:
                    row_index: int = indexes[pair]
                    row: Optional[List[Any]] = self.validate_row(data, pair, row_index, current_time)
                    if not row:
                        continue
                    row_index += 1
                    indexes[pair] = row_index
                    is_last_row: bool = current_time == end_date
                    self.dataprovider._set_dataframe_max_index(self.required_startup + row_index)
                    trade_dir = self.check_for_trade_entry(row)
                    pair_tradedir_cache[pair] = trade_dir
                else:
                    detail_data: Optional[List[List[Any]]] = pair_detail_cache.get(pair)
                    if detail_data is None or len(detail_data) <= idx:
                        continue
                    row = detail_data[idx]
                    trade_dir = pair_tradedir_cache.get(pair)
                    if self.strategy.ignore_expired_candle(current_time - self.timeframe_td, current_time_det, self.timeframe_secs, trade_dir is not None):
                        trade_dir = None
                self.dataprovider._set_dataframe_max_date(current_time_det)
                pair_has_open_trades: bool = len(LocalTrade.bt_trades_open_pp[pair]) > 0
                if pair in pairs_with_open_trades and (not pair_has_open_trades):
                    continue
                if pair_has_open_trades and pair not in pairs_with_open_trades:
                    pairs_with_open_trades.append(pair)
                if is_first and (trade_dir is not None or pair_has_open_trades) and has_detail and (pair not in pair_detail_cache) and (pair in self.detail_data) and row:
                    pair_detail: Optional[List[List[Any]]] = self.get_detail_data(pair, row)
                    if pair_detail is not None:
                        pair_detail_cache[pair] = pair_detail
                    row = pair_detail_cache[pair][idx]
                is_last_row = current_time_det == end_date
                yield (current_time_det, pair, row, is_last_row, trade_dir)
            self.progress.increment()

    def backtest(self, processed: Dict[str, DataFrame], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :return: Dict with trades (results of backtesting) and additional information.
        """
        self.prepare_backtest(self.enable_protections)
        self.wallets.update()
        data: Dict[str, List[List[Any]]] = self._get_ohlcv_as_lists(processed)
        for current_time, pair, row, is_last_row, trade_dir in self.time_pair_generator(start_date, end_date, list(data.keys()), data):
            if not self._can_short or trade_dir is None:
                self.backtest_loop(row, pair, current_time, trade_dir, not is_last_row)
            else:
                for _ in (0, 1):
                    a: Optional[str] = self.backtest_loop(row, pair, current_time, trade_dir, not is_last_row)
                    if not a or a == trade_dir:
                        break
        self.handle_left_open(LocalTrade.bt_trades_open_pp, data=data)
        self.wallets.update()
        results: DataFrame = trade_list_to_dataframe(LocalTrade.bt_trades)
        return {
            'results': results,
            'config': self.strategy.config,
            'locks': PairLocks.get_all_locks(),
            'rejected_signals': self.rejected_trades,
            'timedout_entry_orders': self.timedout_entry_orders,
            'timedout_exit_orders': self.timedout_exit_orders,
            'canceled_trade_entries': self.canceled_trade_entries,
            'canceled_entry_orders': self.canceled_entry_orders,
            'replaced_entry_orders': self.replaced_entry_orders,
            'final_balance': self.wallets.get_total(self.strategy.config['stake_currency'])
        }

    def backtest_one_strategy(self, strat: IStrategy, data: Dict[str, DataFrame], timerange: TimeRange) -> Tuple[datetime, datetime]:
        self.progress.init_step(BacktestState.ANALYZE, 0)
        strategy_name: str = strat.get_strategy_name()
        logger.info(f'Running backtesting for Strategy {strategy_name}')
        backtest_start_time: datetime = dt_now()
        self._set_strategy(strat)
        preprocessed: Dict[str, DataFrame] = self.strategy.advise_all_indicators(data)
        preprocessed_tmp: Dict[str, DataFrame] = trim_dataframes(preprocessed, timerange, self.required_startup)
        if not preprocessed_tmp:
            raise OperationalException('No data left after adjusting for startup candles.')
        min_date, max_date = history.get_timerange(preprocessed_tmp)
        logger.info(f'Backtesting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} up to {max_date.strftime(DATETIME_PRINT_FORMAT)} ({(max_date - min_date).days} days).')
        results: Dict[str, Any] = self.backtest(processed=preprocessed, start_date=min_date, end_date=max_date)
        backtest_end_time: datetime = dt_now()
        results.update({
            'run_id': self.run_ids.get(strategy_name, ''),
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp())
        })
        self.all_results[strategy_name] = results
        if self.config.get('export', 'none') == 'signals' and self.dataprovider.runmode == RunMode.BACKTEST:
            signals = generate_trade_signal_candles(preprocessed_tmp, results, 'open_date')
            rejected = generate_rejected_signals(preprocessed_tmp, self.rejected_dict)
            exited = generate_trade_signal_candles(preprocessed_tmp, results, 'close_date')
            self.analysis_results['signals'][strategy_name] = signals
            self.analysis_results['rejected'][strategy_name] = rejected
            self.analysis_results['exited'][strategy_name] = exited
        return min_date, max_date

    def _get_min_cached_backtest_date(self) -> Optional[datetime]:
        min_backtest_date: Optional[datetime] = None
        backtest_cache_age: Any = self.config.get('backtest_cache', constants.BACKTEST_CACHE_DEFAULT)
        if self.timerange.stopts == 0 or self.timerange.stopdt > dt_now():
            logger.warning('Backtest result caching disabled due to use of open-ended timerange.')
        elif backtest_cache_age == 'day':
            min_backtest_date = dt_now() - timedelta(days=1)
        elif backtest_cache_age == 'week':
            min_backtest_date = dt_now() - timedelta(weeks=1)
        elif backtest_cache_age == 'month':
            min_backtest_date = dt_now() - timedelta(weeks=4)
        return min_backtest_date

    def load_prior_backtest(self) -> None:
        self.run_ids = {strategy.get_strategy_name(): get_strategy_run_id(strategy) for strategy in self.strategylist}
        min_backtest_date: Optional[datetime] = self._get_min_cached_backtest_date()
        if min_backtest_date is not None:
            self.results = find_existing_backtest_stats(self.config['user_data_dir'] / 'backtest_results', self.run_ids, min_backtest_date)

    def start(self) -> None:
        """
        Run backtesting end-to-end
        """
        data, timerange = self.load_bt_data()
        self.load_bt_data_detail()
        logger.info('Dataload complete. Calculating indicators')
        self.load_prior_backtest()
        for strat in self.strategylist:
            if self.results and strat.get_strategy_name() in self.results['strategy']:
                logger.info(f'Reusing result of previous backtest for {strat.get_strategy_name()}')
                continue
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)
        if len(self.all_results) > 0:
            results = generate_backtest_stats(data, self.all_results, min_date=min_date, max_date=max_date)
            if self.results:
                self.results['metadata'].update(results['metadata'])
                self.results['strategy'].update(results['strategy'])
                self.results['strategy_comparison'].extend(results['strategy_comparison'])
            else:
                self.results = results
            dt_appendix: str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if self.config.get('export', 'none') in ('trades', 'signals'):
                combined_res = combined_dataframes_with_rel_mean(data, min_date, max_date)
                store_backtest_results(self.config, self.results, dt_appendix, market_change_data=combined_res, analysis_results=self.analysis_results)
        if 'strategy_list' in self.config and len(self.results) > 0:
            self.results['strategy_comparison'] = sorted(self.results['strategy_comparison'], key=lambda c: self.config['strategy_list'].index(c['key']))
            self.results['strategy'] = dict(sorted(self.results['strategy'].items(), key=lambda kv: self.config['strategy_list'].index(kv[0])))
        if len(self.strategylist) > 0:
            show_backtest_results(self.config, self.results)
