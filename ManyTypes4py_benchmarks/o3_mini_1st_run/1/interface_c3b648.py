#!/usr/bin/env python3
"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from math import isinf, isnan
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pandas import DataFrame
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, Config, IntOrInf, ListPairsWithTimeframes
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, ExitCheckTuple, ExitType, MarketDirection, RunMode, SignalDirection, SignalTagType, SignalType, TradingMode
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_next_date, timeframe_to_seconds
from freqtrade.misc import remove_entry_exit_signals
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.strategy.hyper import HyperStrategyMixin
from freqtrade.strategy.informative_decorator import InformativeData, PopulateIndicators, _create_and_merge_informative_pair, _format_pair_name
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import dt_now
from freqtrade.wallets import Wallets
from pandas import Timestamp

logger: logging.Logger = logging.getLogger(__name__)


class IStrategy(ABC, HyperStrategyMixin):
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        timeframe -> str: value of the timeframe to use with the strategy
    """
    INTERFACE_VERSION: int = 3
    minimal_roi: Dict[Any, Any] = {}
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    use_custom_stoploss: bool = False
    can_short: bool = False
    order_types: Dict[str, Any] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60
    }
    order_time_in_force: Dict[str, str] = {'entry': 'GTC', 'exit': 'GTC'}
    process_only_new_candles: bool = True
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1
    ignore_buying_expired_candle_after: int = 0
    disable_dataframe_checks: bool = False
    startup_candle_count: int = 0
    protections: List[Any] = []
    wallets: Optional[Wallets] = None
    __source__: str = ''
    plot_config: Dict[Any, Any] = {}
    market_direction: MarketDirection = MarketDirection.NONE
    _cached_grouped_trades_per_pair: Dict[str, Any] = {}

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self._last_candle_seen_per_pair: Dict[str, Any] = {}
        super().__init__(config)
        self._ft_informative: List[Tuple[InformativeData, Callable]] = []
        for attr_name in dir(self.__class__):
            cls_method: Any = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, '_ft_informative', None)
            if not isinstance(informative_data_list, list):
                continue
            strategy_timeframe_minutes: int = timeframe_to_minutes(self.timeframe)  # type: ignore
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException('Informative timeframe must be equal or higher than strategy timeframe!')
                if not informative_data.candle_type:
                    informative_data.candle_type = config['candle_type_def']
                self._ft_informative.append((informative_data, cls_method))

    def load_freqAI_model(self) -> None:
        if self.config.get('freqai', {}).get('enabled', False):
            from freqtrade.freqai.utils import download_all_data_for_training
            from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
            self.freqai = FreqaiModelResolver.load_freqaimodel(self.config)
            self.freqai_info = self.config['freqai']
            if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
                logger.info('Downloading all training data for all pairs in whitelist and corr_pairlist, this may take a while if the data is not already on disk.')
                download_all_data_for_training(self.dp, self.config)
        else:
            class DummyClass:
                def start(self, *args: Any, **kwargs: Any) -> None:
                    raise OperationalException('freqAI is not enabled. Please enable it in your config to use this strategy.')

                def shutdown(self, *args: Any, **kwargs: Any) -> None:
                    pass
            self.freqai = DummyClass()

    def ft_bot_start(self, **kwargs: Any) -> None:
        """
        Strategy init - runs after dataprovider has been added.
        Must call bot_start()
        """
        self.load_freqAI_model()
        strategy_safe_wrapper(self.bot_start)()
        self.ft_load_hyper_params(self.config.get('runmode') == RunMode.HYPEROPT)

    def ft_bot_cleanup(self) -> None:
        """
        Clean up FreqAI and child threads
        """
        self.freqai.shutdown()

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Populate indicators that will be used in the Buy, Sell, Short, Exit_short strategy
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        DEPRECATED - please migrate to populate_entry_trend
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        return self.populate_buy_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        DEPRECATED - please migrate to populate_exit_trend
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return self.populate_sell_trend(dataframe, metadata)

    def bot_start(self, **kwargs: Any) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def bot_loop_start(self, current_time: datetime, **kwargs: Any) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def check_buy_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any) -> bool:
        """
        DEPRECATED: Please use `check_entry_timeout` instead.
        """
        return False

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any) -> bool:
        """
        Check entry timeout function callback.
        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Additional parameters.
        :return bool: When True is returned, then the entry order is cancelled.
        """
        return self.check_buy_timeout(pair=pair, trade=trade, order=order, current_time=current_time)

    def check_sell_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any) -> bool:
        """
        DEPRECATED: Please use `check_exit_timeout` instead.
        """
        return False

    def check_exit_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any) -> bool:
        """
        Check exit timeout function callback.
        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Additional parameters.
        :return bool: When True is returned, then the exit-order is cancelled.
        """
        return self.check_sell_timeout(pair=pair, trade=trade, order=order, current_time=current_time)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: Optional[str], side: str, **kwargs: Any) -> bool:
        """
        Called right before placing an entry order.
        Returns True to place the order.
        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type.
        :param amount: Amount to trade.
        :param rate: Order rate.
        :param time_in_force: Time in force.
        :param current_time: Current datetime.
        :param entry_tag: Optional entry tag.
        :param side: 'long' or 'short' trade direction.
        :param **kwargs: Additional parameters.
        :return bool: True if the order is confirmed.
        """
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs: Any) -> bool:
        """
        Called right before placing an exit order.
        Returns True to place the order.
        :param pair: Pair for the exiting trade.
        :param trade: Trade object.
        :param order_type: Order type.
        :param amount: Amount in base currency.
        :param rate: Order rate.
        :param time_in_force: Time in force.
        :param exit_reason: Reason for exit.
        :param current_time: Current datetime.
        :param **kwargs: Additional parameters.
        :return bool: True if the order is confirmed.
        """
        return True

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs: Any) -> None:
        """
        Called right after an order fills.
        :param pair: Pair string.
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: Current datetime.
        :param **kwargs: Additional parameters.
        """
        pass

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        """
        Custom stoploss logic, returning the new stoploss value.
        When not implemented, returns the initial stoploss value.
        :param pair: Pair string.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param current_profit: Current profit as a ratio.
        :param after_fill: Whether after fill.
        :param **kwargs: Additional parameters.
        :return float: New stoploss ratio.
        """
        return self.stoploss  # type: ignore

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs: Any) -> float:
        """
        Custom entry price logic.
        :param pair: Pair string.
        :param trade: Trade object or None.
        :param current_time: Current datetime.
        :param proposed_rate: Proposed entry rate.
        :param entry_tag: Optional entry tag.
        :param side: 'long' or 'short'.
        :param **kwargs: Additional parameters.
        :return float: New entry price.
        """
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime, proposed_rate: float, current_profit: float, exit_tag: Optional[str], **kwargs: Any) -> float:
        """
        Custom exit price logic.
        :param pair: Pair string.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param proposed_rate: Proposed exit rate.
        :param current_profit: Current profit ratio.
        :param exit_tag: Exit reason tag.
        :param **kwargs: Additional parameters.
        :return float: New exit price.
        """
        return proposed_rate

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> Union[str, bool, None]:
        """
        DEPRECATED - please use custom_exit instead.
        :param pair: Pair string.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param current_profit: Current profit as a ratio.
        :param **kwargs: Additional parameters.
        :return: Custom exit signal.
        """
        return None

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> Union[str, bool, None]:
        """
        Custom exit signal logic.
        :param pair: Pair string.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param current_profit: Current profit ratio.
        :param **kwargs: Additional parameters.
        :return: Custom exit signal.
        """
        return self.custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, leverage: float, entry_tag: Optional[str], side: str, **kwargs: Any) -> float:
        """
        Customize stake size for each new trade.
        :param pair: Pair string.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param proposed_stake: Proposed stake amount.
        :param min_stake: Minimum stake.
        :param max_stake: Maximum stake.
        :param leverage: Leverage.
        :param entry_tag: Optional entry tag.
        :param side: Trade side.
        :param **kwargs: Additional parameters.
        :return float: Stake size.
        """
        return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: float, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> Optional[Union[float, Tuple[Optional[float], str]]]:
        """
        Custom trade adjustment logic.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param current_profit: Current profit ratio.
        :param min_stake: Minimum stake.
        :param max_stake: Maximum stake.
        :param current_entry_rate: Current entry rate.
        :param current_exit_rate: Current exit rate.
        :param current_entry_profit: Current entry profit ratio.
        :param current_exit_profit: Current exit profit ratio.
        :param **kwargs: Additional parameters.
        :return: Stake adjustment amount or tuple (amount, order_reason) or None.
        """
        return None

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str, current_time: datetime, proposed_rate: float, current_order_rate: float, entry_tag: Optional[str], side: str, **kwargs: Any) -> Optional[float]:
        """
        Entry price re-adjustment logic.
        :param trade: Trade object.
        :param order: Order object.
        :param pair: Pair string.
        :param current_time: Current datetime.
        :param proposed_rate: Proposed new rate.
        :param current_order_rate: Existing order rate.
        :param entry_tag: Optional entry tag.
        :param side: Trade side.
        :param **kwargs: Additional parameters.
        :return: New entry price value if provided.
        """
        return current_order_rate

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs: Any) -> float:
        """
        Customize leverage for each new trade. (For futures)
        :param pair: Pair string.
        :param current_time: Current datetime.
        :param current_rate: Current rate.
        :param proposed_leverage: Proposed leverage.
        :param max_leverage: Maximum allowed leverage.
        :param entry_tag: Optional entry tag.
        :param side: Trade side.
        :param **kwargs: Additional parameters.
        :return float: Leverage amount.
        """
        return 1.0

    def informative_pairs(self) -> List[Tuple[str, str, CandleType]]:
        """
        Define additional, informative pair/interval combinations.
        :return: List of tuples (pair, interval, candle_type)
        """
        return []

    def version(self) -> Optional[str]:
        """
        Returns version of the strategy.
        """
        return None

    def populate_any_indicators(self, pair: str, df: DataFrame, tf: str, informative: Optional[DataFrame] = None, set_generalized_indicators: bool = False) -> DataFrame:
        """
        DEPRECATED - USE FEATURE ENGINEERING FUNCTIONS INSTEAD.
        :param pair: Trading pair.
        :param df: Strategy dataframe.
        :param tf: Timeframe.
        :param informative: Optional informative dataframe.
        :param set_generalized_indicators: Flag.
        :return: New dataframe.
        """
        return df

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        Expand features for FreqAI.
        :param dataframe: Strategy dataframe.
        :param period: Period of indicator.
        :param metadata: Metadata dictionary.
        :param **kwargs: Additional parameters.
        :return: Modified dataframe.
        """
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        Expand basic features for FreqAI.
        :param dataframe: Strategy dataframe.
        :param metadata: Metadata dictionary.
        :param **kwargs: Additional parameters.
        :return: Modified dataframe.
        """
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        Standard feature engineering for FreqAI.
        :param dataframe: Strategy dataframe.
        :param metadata: Metadata dictionary.
        :param **kwargs: Additional parameters.
        :return: Modified dataframe.
        """
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        Set FreqAI targets.
        :param dataframe: Strategy dataframe.
        :param metadata: Metadata dictionary.
        :param **kwargs: Additional parameters.
        :return: Modified dataframe.
        """
        return dataframe

    _ft_stop_uses_after_fill: bool = False

    def _adjust_trade_position_internal(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: float, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> Tuple[Optional[float], str]:
        """
        Internal wrapper around adjust_trade_position.
        :return: Tuple of (stake_amount, order_tag)
        """
        resp: Any = strategy_safe_wrapper(
            self.adjust_trade_position,
            default_retval=(None, ''),
            supress_error=True
        )(
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=max_stake,
            current_entry_rate=current_entry_rate,
            current_exit_rate=current_exit_rate,
            current_entry_profit=current_entry_profit,
            current_exit_profit=current_exit_profit,
            **kwargs
        )
        order_tag: str = ''
        stake_amount: Optional[float] = None
        if isinstance(resp, tuple):
            if len(resp) >= 1:
                stake_amount = resp[0]
            if len(resp) > 1:
                order_tag = resp[1] or ''
        else:
            stake_amount = resp
        return (stake_amount, order_tag)

    def __informative_pairs_freqai(self) -> List[Tuple[str, str, CandleType]]:
        """
        Create informative-pairs needed for FreqAI.
        :return: List of tuples.
        """
        if self.config.get('freqai', {}).get('enabled', False):
            whitelist_pairs: List[str] = self.dp.current_whitelist()  # type: ignore
            candle_type: CandleType = self.config.get('candle_type_def', CandleType.SPOT)
            corr_pairs: List[str] = self.config['freqai']['feature_parameters']['include_corr_pairlist']
            informative_pairs: List[Tuple[str, str, CandleType]] = []
            for tf in self.config['freqai']['feature_parameters']['include_timeframes']:
                for pair in set(whitelist_pairs + corr_pairs):
                    informative_pairs.append((pair, tf, candle_type))
            return informative_pairs
        return []

    def gather_informative_pairs(self) -> List[Tuple[str, str, CandleType]]:
        """
        Gather all informative pairs.
        :return: List of tuples.
        """
        informative_pairs = self.informative_pairs()
        informative_pairs = [
            (
                p[0],
                p[1],
                CandleType.from_string(p[2]) if len(p) > 2 and p[2] != '' else self.config.get('candle_type_def', CandleType.SPOT)
            )
            for p in informative_pairs
        ]
        for inf_data, _ in self._ft_informative:
            candle_type: CandleType = inf_data.candle_type if inf_data.candle_type else self.config.get('candle_type_def', CandleType.SPOT)
            if inf_data.asset:
                if any((s in inf_data.asset for s in ('{BASE}', '{base}'))):
                    for pair in self.dp.current_whitelist():  # type: ignore
                        pair_tf = (_format_pair_name(self.config, inf_data.asset, self.dp.market(pair)), inf_data.timeframe, candle_type)  # type: ignore
                        informative_pairs.append(pair_tf)
                else:
                    pair_tf = (_format_pair_name(self.config, inf_data.asset), inf_data.timeframe, candle_type)
                    informative_pairs.append(pair_tf)
            else:
                for pair in self.dp.current_whitelist():  # type: ignore
                    informative_pairs.append((pair, inf_data.timeframe, candle_type))
        informative_pairs.extend(self.__informative_pairs_freqai())
        return list(set(informative_pairs))

    def get_strategy_name(self) -> str:
        """
        Returns strategy class name.
        """
        return self.__class__.__name__

    def lock_pair(self, pair: str, until: datetime, reason: Optional[str] = None, side: str = '*') -> None:
        """
        Lock a pair.
        :param pair: Pair string.
        :param until: Datetime until the lock expires.
        :param reason: Optional reason.
        :param side: Trade side.
        """
        PairLocks.lock_pair(pair, until, reason, side=side)

    def unlock_pair(self, pair: str) -> None:
        """
        Unlock a pair.
        :param pair: Pair string.
        """
        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:
        """
        Unlock pairs with specified reason.
        :param reason: Reason string.
        """
        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(self, pair: str, *, candle_date: Optional[datetime] = None, side: str = '*') -> bool:
        """
        Check if a pair is locked.
        :param pair: Pair string.
        :param candle_date: Optional candle date.
        :param side: Trade side.
        :return: True if locked.
        """
        if not candle_date:
            return PairLocks.is_pair_locked(pair, side=side)
        else:
            lock_time: datetime = timeframe_to_next_date(self.timeframe, candle_date)  # type: ignore
            return PairLocks.is_pair_locked(pair, lock_time, side=side)

    def analyze_ticker(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Parse given OHLCV data and return populated DataFrame.
        :param dataframe: DataFrame with exchange data.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        logger.debug('TA Analysis Launched')
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        logger.debug('TA Analysis Ended')
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Internal analysis of OHLCV data.
        :param dataframe: DataFrame with exchange data.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        pair: str = str(metadata.get('pair'))
        new_candle: bool = self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']
        if not self.process_only_new_candles or new_candle:
            dataframe = self.analyze_ticker(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
            candle_type: CandleType = self.config.get('candle_type_def', CandleType.SPOT)
            self.dp._set_cached_df(pair, self.timeframe, dataframe, candle_type=candle_type)  # type: ignore
            self.dp._emit_df((pair, self.timeframe, candle_type), dataframe, new_candle)  # type: ignore
        else:
            logger.debug('Skipping TA Analysis for already analyzed candle')
            dataframe = remove_entry_exit_signals(dataframe)
        logger.debug('Loop Analysis Launched')
        return dataframe

    def analyze_pair(self, pair: str) -> None:
        """
        Analyze a single pair.
        :param pair: Pair string.
        """
        dataframe: DataFrame = self.dp.ohlcv(pair, self.timeframe, candle_type=self.config.get('candle_type_def', CandleType.SPOT))  # type: ignore
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning('Empty candle (OHLCV) data for pair %s', pair)
            return
        try:
            df_len, df_close, df_date = self.preserve_df(dataframe)
            dataframe = strategy_safe_wrapper(self._analyze_ticker_internal, message='')(dataframe, {'pair': pair})
            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            logger.warning(f'Unable to analyze candle (OHLCV) data for pair {pair}: {error}')
            return
        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return

    def analyze(self, pairs: List[str]) -> None:
        """
        Analyze all pairs.
        :param pairs: List of pair strings.
        """
        for pair in pairs:
            self.analyze_pair(pair)

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, Any]:
        """
        Preserve dataframe parameters.
        :param dataframe: DataFrame.
        :return: Tuple (length, last close value, last date).
        """
        return (len(dataframe), dataframe['close'].iloc[-1], dataframe['date'].iloc[-1])

    def assert_df(self, dataframe: DataFrame, df_len: int, df_close: float, df_date: Any) -> None:
        """
        Validate dataframe integrity.
        :param dataframe: DataFrame.
        :param df_len: Expected length.
        :param df_close: Expected last close value.
        :param df_date: Expected last date.
        :raises StrategyError: If validation fails.
        """
        message_template: str = 'Dataframe returned from strategy has mismatching {}.'
        message: str = ''
        if dataframe is None:
            message = 'No dataframe returned (return statement missing?).'
        elif df_len != len(dataframe):
            message = message_template.format('length')
        elif df_close != dataframe['close'].iloc[-1]:
            message = message_template.format('last close price')
        elif df_date != dataframe['date'].iloc[-1]:
            message = message_template.format('last date')
        if message:
            if self.disable_dataframe_checks:
                logger.warning(message)
            else:
                raise StrategyError(message)

    def get_latest_candle(self, pair: str, timeframe: str, dataframe: DataFrame) -> Tuple[Optional[DataFrame], Optional[datetime]]:
        """
        Get latest candle.
        :param pair: Pair string.
        :param timeframe: Timeframe.
        :param dataframe: DataFrame.
        :return: Tuple (latest row as DataFrame, latest date) or (None, None)
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning(f'Empty candle (OHLCV) data for pair {pair}')
            return (None, None)
        try:
            latest_date_pd: Timestamp = dataframe['date'].max()  # type: ignore
            latest: DataFrame = dataframe.loc[dataframe['date'] == latest_date_pd].iloc[-1:]
        except Exception as e:
            logger.warning(f'Unable to get latest candle (OHLCV) data for pair {pair} - {e}')
            return (None, None)
        latest_date: datetime = latest_date_pd.to_pydatetime()  # type: ignore
        timeframe_minutes: int = timeframe_to_minutes(timeframe)
        offset: int = self.config.get('exchange', {}).get('outdated_offset', 5)
        if latest_date < dt_now() - timedelta(minutes=timeframe_minutes * 2 + offset):
            logger.warning('Outdated history for pair %s. Last tick is %s minutes old', pair, int((dt_now() - latest_date).total_seconds() // 60))
            return (None, None)
        return (latest, latest_date)

    def get_exit_signal(self, pair: str, timeframe: str, dataframe: DataFrame, is_short: Optional[bool] = None) -> Tuple[bool, bool, Optional[str]]:
        """
        Calculate exit signal.
        :param pair: Pair string.
        :param timeframe: Timeframe.
        :param dataframe: DataFrame.
        :param is_short: Optional flag for short trades.
        :return: Tuple (enter, exit, exit_tag)
        """
        latest, _latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None:
            return (False, False, None)
        if is_short:
            enter: bool = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
            exit_: bool = latest.get(SignalType.EXIT_SHORT.value, 0) == 1
        else:
            enter = latest.get(SignalType.ENTER_LONG.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        exit_tag: Optional[Any] = latest.get(SignalTagType.EXIT_TAG.value, None)
        exit_tag = exit_tag if isinstance(exit_tag, str) and exit_tag != 'nan' else None
        logger.debug(f"exit-trigger: {latest.iloc[0]['date']} (pair={pair}) enter={enter} exit={exit_}")
        return (enter, exit_, exit_tag)

    def get_entry_signal(self, pair: str, timeframe: str, dataframe: DataFrame) -> Tuple[Optional[SignalDirection], Optional[str]]:
        """
        Calculate entry signal.
        :param pair: Pair string.
        :param timeframe: Timeframe.
        :param dataframe: DataFrame.
        :return: Tuple (SignalDirection, entry_tag)
        """
        latest, latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None or latest_date is None:
            return (None, None)
        enter_long: bool = latest.get(SignalType.ENTER_LONG.value, 0) == 1
        exit_long: bool = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        enter_short: bool = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
        exit_short: bool = latest.get(SignalType.EXIT_SHORT.value, 0) == 1
        enter_signal: Optional[SignalDirection] = None
        enter_tag: Optional[Any] = None
        if enter_long and (not any([exit_long, enter_short])):
            enter_signal = SignalDirection.LONG
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        if self.config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT and self.can_short and (enter_short) and (not any([exit_short, enter_long])):
            enter_signal = SignalDirection.SHORT
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        enter_tag = enter_tag if isinstance(enter_tag, str) and enter_tag != 'nan' else None
        timeframe_seconds: int = timeframe_to_seconds(timeframe)
        if self.ignore_expired_candle(latest_date=latest_date, current_time=dt_now(), timeframe_seconds=timeframe_seconds, enter=bool(enter_signal)):
            return (None, enter_tag)
        logger.debug(f"entry trigger: {latest.iloc[0]['date']} (pair={pair}) enter={enter_long} enter_tag_value={enter_tag}")
        return (enter_signal, enter_tag)

    def ignore_expired_candle(self, latest_date: datetime, current_time: datetime, timeframe_seconds: int, enter: bool) -> bool:
        if self.ignore_buying_expired_candle_after and enter:
            time_delta = current_time - (latest_date + timedelta(seconds=timeframe_seconds))
            return time_delta.total_seconds() > self.ignore_buying_expired_candle_after
        else:
            return False

    def should_exit(self, trade: Trade, rate: float, current_time: datetime, *, enter: bool, exit_: bool, low: Optional[float] = None, high: Optional[float] = None, force_stoploss: int = 0) -> List[ExitCheckTuple]:
        """
        Evaluate if exit conditions are met.
        :param trade: Trade object.
        :param rate: Current rate.
        :param current_time: Current datetime.
        :param enter: Whether entry signal is active.
        :param exit_: Whether exit signal is active.
        :param low: Optional low value for backtesting.
        :param high: Optional high value for backtesting.
        :param force_stoploss: External stoploss.
        :return: List of exit reasons.
        """
        exits: List[ExitCheckTuple] = []
        current_rate = rate
        current_profit: float = trade.calc_profit_ratio(current_rate)
        current_profit_best: float = current_profit
        if low is not None or high is not None:
            current_rate_best: float = (low if trade.is_short else high) or rate
            current_profit_best = trade.calc_profit_ratio(current_rate_best)
        trade.adjust_min_max_rates(high or current_rate, low or current_rate)
        stoplossflag: ExitCheckTuple = self.ft_stoploss_reached(current_rate=current_rate, trade=trade, current_time=current_time, current_profit=current_profit, force_stoploss=force_stoploss, low=low, high=high)
        roi_reached: bool = not (enter and self.ignore_roi_if_entry_signal) and self.min_roi_reached(trade=trade, current_profit=current_profit_best, current_time=current_time)
        exit_signal: ExitType = ExitType.NONE
        custom_reason: str = ''
        if self.use_exit_signal:
            if exit_ and (not enter):
                exit_signal = ExitType.EXIT_SIGNAL
            else:
                reason_cust: Union[str, bool] = strategy_safe_wrapper(self.custom_exit, default_retval=False)(pair=trade.pair, trade=trade, current_time=current_time, current_rate=current_rate, current_profit=current_profit)
                if reason_cust:
                    exit_signal = ExitType.CUSTOM_EXIT
                    if isinstance(reason_cust, str):
                        custom_reason = reason_cust
                        if len(reason_cust) > CUSTOM_TAG_MAX_LENGTH:
                            logger.warning(f'Custom exit reason returned from custom_exit is too long and was trimmed to {CUSTOM_TAG_MAX_LENGTH} characters.')
                            custom_reason = reason_cust[:CUSTOM_TAG_MAX_LENGTH]
                    else:
                        custom_reason = ''
            if exit_signal == ExitType.CUSTOM_EXIT or (exit_signal == ExitType.EXIT_SIGNAL and (not self.exit_profit_only or current_profit > self.exit_profit_offset)):
                logger.debug(f'{trade.pair} - Sell signal received. exit_type=ExitType.{exit_signal.name}' + (f', custom_reason={custom_reason}' if custom_reason else ''))
                exits.append(ExitCheckTuple(exit_type=exit_signal, exit_reason=custom_reason))
        if stoplossflag.exit_type in (ExitType.STOP_LOSS, ExitType.LIQUIDATION):
            logger.debug(f'{trade.pair} - Stoploss hit. exit_type={stoplossflag.exit_type}')
            exits.append(stoplossflag)
        if roi_reached:
            logger.debug(f'{trade.pair} - Required profit reached. exit_type=ExitType.ROI')
            exits.append(ExitCheckTuple(exit_type=ExitType.ROI))
        if stoplossflag.exit_type == ExitType.TRAILING_STOP_LOSS:
            logger.debug(f'{trade.pair} - Trailing stoploss hit.')
            exits.append(stoplossflag)
        return exits

    def ft_stoploss_adjust(self, current_rate: float, trade: Trade, current_time: datetime, current_profit: float, force_stoploss: int, low: Optional[float] = None, high: Optional[float] = None, after_fill: bool = False) -> None:
        """
        Adjust stoploss dynamically.
        :param current_rate: Current rate.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_profit: Current profit ratio.
        :param force_stoploss: External stoploss.
        :param low: Optional low value.
        :param high: Optional high value.
        :param after_fill: Flag for after fill.
        """
        if after_fill and (not self._ft_stop_uses_after_fill):
            return
        stop_loss_value: float = force_stoploss if force_stoploss else self.stoploss  # type: ignore
        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)
        dir_correct: bool = trade.stop_loss < (low or current_rate) if not trade.is_short else trade.stop_loss > (high or current_rate)
        bound: Optional[float] = low if trade.is_short else high
        bound_profit: float = current_profit if not bound else trade.calc_profit_ratio(bound)
        if self.use_custom_stoploss and dir_correct:
            stop_loss_value_custom: Optional[float] = strategy_safe_wrapper(self.custom_stoploss, default_retval=None, supress_error=True)(
                pair=trade.pair, trade=trade, current_time=current_time, current_rate=bound or current_rate, current_profit=bound_profit, after_fill=after_fill
            )
            if stop_loss_value_custom and (not (isnan(stop_loss_value_custom) or isinf(stop_loss_value_custom))):
                stop_loss_value = stop_loss_value_custom
                trade.adjust_stop_loss(bound or current_rate, stop_loss_value, allow_refresh=after_fill)
            else:
                logger.debug('CustomStoploss function did not return valid stoploss')
        if self.trailing_stop and dir_correct:
            sl_offset: float = self.trailing_stop_positive_offset
            if not (self.trailing_only_offset_is_reached and bound_profit < sl_offset):
                if self.trailing_stop_positive is not None and bound_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive
                    logger.debug(f'{trade.pair} - Using positive stoploss: {stop_loss_value} offset: {sl_offset:.4g} profit: {bound_profit:.2%}')
                trade.adjust_stop_loss(bound or current_rate, stop_loss_value)

    def ft_stoploss_reached(self, current_rate: float, trade: Trade, current_time: datetime, current_profit: float, force_stoploss: int, low: Optional[float] = None, high: Optional[float] = None) -> ExitCheckTuple:
        """
        Determine if stoploss is reached.
        :param current_rate: Current rate.
        :param trade: Trade object.
        :param current_time: Current datetime.
        :param current_profit: Current profit ratio.
        :param force_stoploss: External stoploss.
        :param low: Optional low value.
        :param high: Optional high value.
        :return: ExitCheckTuple.
        """
        self.ft_stoploss_adjust(current_rate, trade, current_time, current_profit, force_stoploss, low, high)
        sl_higher_long: bool = trade.stop_loss >= (low or current_rate) and (not trade.is_short)
        sl_lower_short: bool = trade.stop_loss <= (high or current_rate) and trade.is_short
        liq_higher_long: bool = trade.liquidation_price and trade.liquidation_price >= (low or current_rate) and (not trade.is_short)
        liq_lower_short: bool = trade.liquidation_price and trade.liquidation_price <= (high or current_rate) and trade.is_short
        if (sl_higher_long or sl_lower_short) and (not self.order_types.get('stoploss_on_exchange') or self.config['dry_run']):
            exit_type: ExitType = ExitType.STOP_LOSS
            if trade.is_stop_loss_trailing:
                exit_type = ExitType.TRAILING_STOP_LOSS
                logger.debug(f'{trade.pair} - HIT STOP: current price at {(high if trade.is_short else low) or current_rate:.6f}, stoploss is {trade.stop_loss:.6f}, initial stoploss was at {trade.initial_stop_loss:.6f}, trade opened at {trade.open_rate:.6f}')
            return ExitCheckTuple(exit_type=exit_type)
        if liq_higher_long or liq_lower_short:
            logger.debug(f'{trade.pair} - Liquidation price hit. exit_type=ExitType.LIQUIDATION')
            return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)
        return ExitCheckTuple(exit_type=ExitType.NONE)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Determine minimal ROI based on trade duration.
        :param trade_dur: Duration in minutes.
        :return: Tuple (roi_key, roi_value)
        """
        roi_list: List[int] = [x for x in self.minimal_roi.keys() if x <= trade_dur]
        if not roi_list:
            return (None, None)
        roi_entry: int = max(roi_list)
        return (roi_entry, self.minimal_roi[roi_entry])

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        Evaluate if ROI is reached.
        :param trade: Trade object.
        :param current_profit: Current profit ratio.
        :param current_time: Current datetime.
        :return: True if ROI reached.
        """
        trade_dur: int = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ft_check_timed_out(self, trade: Trade, order: Order, current_time: datetime) -> bool:
        """
        Check if order has timed out.
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: Current datetime.
        :return: True if timed out.
        """
        side: str = 'entry' if order.ft_order_side == trade.entry_side else 'exit'
        timeout: Optional[int] = self.config.get('unfilledtimeout', {}).get(side)
        if timeout is not None:
            timeout_unit: str = self.config.get('unfilledtimeout', {}).get('unit', 'minutes')
            timeout_kwargs: Dict[str, int] = {timeout_unit: -timeout}
            timeout_threshold: datetime = current_time + timedelta(**timeout_kwargs)
            timedout: bool = order.status == 'open' and order.order_date_utc < timeout_threshold
            if timedout:
                return True
        time_method: Callable = self.check_exit_timeout if order.ft_order_side == trade.exit_side else self.check_entry_timeout
        return strategy_safe_wrapper(time_method, default_retval=False)(pair=trade.pair, trade=trade, order=order, current_time=current_time)

    def advise_all_indicators(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Populate indicators for multiple pairs.
        :param data: Dictionary of pair to DataFrame.
        :return: Dictionary of pair to modified DataFrame.
        """
        return {pair: self.advise_indicators(pair_data.copy(), {'pair': pair}).copy() for pair, pair_data in data.items()}

    def ft_advise_signals(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Call advise_entry and advise_exit.
        :param dataframe: DataFrame.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        return dataframe

    def _if_enabled_populate_trades(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Add trade data to dataframe if enabled.
        :param dataframe: DataFrame.
        :param metadata: Metadata dictionary.
        """
        use_public_trades: bool = self.config.get('exchange', {}).get('use_public_trades', False)
        if use_public_trades:
            trades: Any = self.dp.trades(pair=metadata['pair'], copy=False)  # type: ignore
            pair: str = metadata['pair']
            cached_grouped_trades: Any = self._cached_grouped_trades_per_pair.get(pair)
            dataframe, cached_grouped_trades = populate_dataframe_with_trades(cached_grouped_trades, self.config, dataframe, trades)
            if pair in self._cached_grouped_trades_per_pair:
                del self._cached_grouped_trades_per_pair[pair]
            self._cached_grouped_trades_per_pair[pair] = cached_grouped_trades
            logger.debug('Populated dataframe with trades.')

    def advise_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Populate indicators.
        :param dataframe: DataFrame.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        logger.debug(f"Populating indicators for pair {metadata.get('pair')}.")
        for inf_data, populate_fn in self._ft_informative:
            dataframe = _create_and_merge_informative_pair(self, dataframe, metadata, inf_data, populate_fn)
        self._if_enabled_populate_trades(dataframe, metadata)
        return self.populate_indicators(dataframe, metadata)

    def advise_entry(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Populate entry signals.
        :param dataframe: DataFrame.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        logger.debug(f"Populating enter signals for pair {metadata.get('pair')}.")
        dataframe.loc[:, 'enter_tag'] = ''
        df: DataFrame = self.populate_entry_trend(dataframe, metadata)
        if 'enter_long' not in df.columns:
            df = df.rename(columns={'buy': 'enter_long', 'buy_tag': 'enter_tag'})
        return df

    def advise_exit(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        Populate exit signals.
        :param dataframe: DataFrame.
        :param metadata: Metadata dictionary.
        :return: Modified DataFrame.
        """
        dataframe.loc[:, 'exit_tag'] = ''
        logger.debug(f"Populating exit signals for pair {metadata.get('pair')}.")
        df: DataFrame = self.populate_exit_trend(dataframe, metadata)
        if 'exit_long' not in df.columns:
            df = df.rename(columns={'sell': 'exit_long'})
        return df
