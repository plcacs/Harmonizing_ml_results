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
from freqtrade.constants import (
    CUSTOM_TAG_MAX_LENGTH,
    Config,
    IntOrInf,
    ListPairsWithTimeframes,
)
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import (
    CandleType,
    ExitCheckTuple,
    ExitType,
    MarketDirection,
    RunMode,
    SignalDirection,
    SignalTagType,
    SignalType,
    TradingMode,
)
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.exchange import (
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_seconds,
)
from freqtrade.misc import remove_entry_exit_signals
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.strategy.hyper import HyperStrategyMixin
from freqtrade.strategy.informative_decorator import (
    InformativeData,
    PopulateIndicators,
    _create_and_merge_informative_pair,
    _format_pair_name,
)
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import dt_now
from freqtrade.wallets import Wallets

logger = logging.getLogger(__name__)


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
    minimal_roi: Dict[str, Union[int, float]] = {}
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    use_custom_stoploss: bool = False
    can_short: bool = False
    order_types: Dict[str, Union[str, bool, int]] = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
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
    plot_config: Dict[str, Any] = {}
    market_direction: MarketDirection = MarketDirection.NONE
    _cached_grouped_trades_per_pair: Dict[str, Any] = {}

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        super().__init__(config)
        self._ft_informative: List[Tuple[InformativeData, Callable]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, '_ft_informative', None)
            if not isinstance(informative_data_list, list):
                continue
            strategy_timeframe_minutes = timeframe_to_minutes(self.timeframe)
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException(
                        'Informative timeframe must be equal or higher than strategy timeframe!'
                    )
                if not informative_data.candle_type:
                    informative_data.candle_type = config['candle_type_def']
                self._ft_informative.append((informative_data, cls_method))

    def load_freqAI_model(self) -> None:
        if self.config.get('freqai', {}).get('enabled', False):
            from freqtrade.freqai.utils import download_all_data_for_training
            from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

            self.freqai = FreqaiModelResolver.load_freqaimodel(self.config)
            self.freqai_info: Dict[str, Any] = self.config['freqai']
            if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
                logger.info(
                    'Downloading all training data for all pairs in whitelist and corr_pairlist, this may take a while if the data is not already on disk.'
                )
                download_all_data_for_training(self.dp, self.config)
        else:

            class DummyClass:

                def start(self, *args: Any, **kwargs: Any) -> None:
                    raise OperationalException(
                        'freqAI is not enabled. Please enable it in your config to use this strategy.'
                    )

                def shutdown(self, *args: Any, **kwargs: Any) -> None:
                    pass

            self.freqai: Any = DummyClass()

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
    def populate_indicators(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Populate indicators that will be used in the Buy, Sell, Short, Exit_short strategy
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return dataframe

    def populate_buy_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        DEPRECATED - please migrate to populate_entry_trend
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        return self.populate_buy_trend(dataframe, metadata)

    def populate_sell_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        DEPRECATED - please migrate to populate_exit_trend
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
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

    def bot_loop_start(
        self, current_time: datetime, **kwargs: Any
    ) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def check_buy_timeout(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """
        DEPRECATED: Please use `check_entry_timeout` instead.
        """
        return False

    def check_entry_timeout(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """
        Check entry timeout function callback.
        This method can be used to override the entry-timeout.
        It is called whenever a limit entry order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the entry order is cancelled.
        """
        return self.check_buy_timeout(
            pair=pair, trade=trade, order=order, current_time=current_time
        )

    def check_sell_timeout(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """
        DEPRECATED: Please use `check_exit_timeout` instead.
        """
        return False

    def check_exit_timeout(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """
        Check exit timeout function callback.
        This method can be used to override the exit-timeout.
        It is called whenever a limit exit order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the exit-order is cancelled.
        """
        return self.check_sell_timeout(
            pair=pair, trade=trade, order=order, current_time=current_time
        )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        return True

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        """
        Called right after an order fills.
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value.
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param after_fill: True if the stoploss is called after the order was filled.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """
        return self.stoploss

    def custom_entry_price(
        self,
        pair: str,
        trade: Optional[Trade],
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> Optional[float]:
        """
        Custom entry price logic, returning the new entry price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set entry price

        :param pair: Pair that's currently analyzed
        :param trade: trade object (None for initial entries).
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate that's going to be used when using limit orders
                               or current rate for market orders.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided
        """
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: Optional[str],
        **kwargs: Any,
    ) -> Optional[float]:
        """
        Custom exit price logic, returning the new exit price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set exit price

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate that's going to be used when using limit orders
                               or current rate for market orders.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param exit_tag: Exit reason.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New exit price value if provided
        """
        return proposed_rate

    def custom_sell(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[Union[str, bool]]:
        """
        DEPRECATED - please use custom_exit instead.
        Custom exit signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting exit signal on a candle at specified
        time. This method is not called when exit signal is set.

        This method should be overridden to create exit signals that depend on trade parameters. For
        example you could implement an exit relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom exit reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate that's going to be used when using limit orders
                           or current rate for market orders.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute exit, return a string with custom exit reason or True. Otherwise return
        None or False.
        """
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[Union[str, bool]]:
        """
        Custom exit signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting exit signal on a candle at specified
        time. This method is not called when exit signal is set.

        This method should be overridden to create exit signals that depend on trade parameters. For
        example you could implement an exit relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom exit reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate that's going to be used when using limit orders
                           or current rate for market orders.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute exit, return a string with custom exit reason or True. Otherwise return
        None or False.
        """
        return self.custom_sell(
            pair, trade, current_time, current_rate, current_profit, **kwargs
        )

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        """
        Customize stake size for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate that's going to be used when using limit orders
                           or current rate for market orders.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param leverage: Leverage selected for this trade.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A stake size, which is between min_stake and max_stake.
        """
        return proposed_stake

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> Optional[Union[float, Tuple[Optional[float], str]]]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra entry or exit orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current entry rate (same as current_entry_profit)
        :param current_profit: Current profit (as ratio), calculated based on current_rate
                               (same as current_entry_profit).
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
                       Optionally, return a tuple with a 2nd element with an order reason
        """
        return None

    def adjust_entry_price(
        self,
        trade: Trade,
        order: Order,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> Optional[float]:
        """
        Entry price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-callbacks/

        When not implemented by a strategy, returns current_order_rate as default.
        If current_order_rate is returned then the existing order is maintained.
        If None is returned then order gets canceled but not replaced by a new one.

        :param pair: Pair that's currently analyzed
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate that's going to be used when using limit orders
                               or current rate for market orders.
        :param current_order_rate: Rate of the existing order in place.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided

        """
        return current_order_rate

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate that's going to be used when using limit orders
                           or current rate for market orders.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return float: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0

    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def version(self) -> Optional[str]:
        """
        Returns version of the strategy.
        """
        return None

    def populate_any_indicators(
        self,
        pair: str,
        df: DataFrame,
        tf: str,
        informative: Optional[DataFrame] = None,
        set_generalized_indicators: bool = False,
    ) -> DataFrame:
        """
        DEPRECATED - USE FEATURE ENGINEERING FUNCTIONS INSTEAD
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User can add
        additional features here, but must follow the naming convention.
        This method is *only* used in FreqaiDataKitchen class and therefore
        it is only called if FreqAI is active.
        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        """
        return df

    def feature_engineering_expand_all(
        self,
        dataframe: DataFrame,
        period: int,
        metadata: Dict[str, Any],
        **kwargs: Any,
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        return dataframe

    _ft_stop_uses_after_fill: bool = False

    def _adjust_trade_position_internal(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> Tuple[Optional[float], str]:
        """
        wrapper around adjust_trade_position to handle the return value
        """
        resp: Optional[Union[float, Tuple[Optional[float], Optional[str]]]] = strategy_safe_wrapper(
            self.adjust_trade_position,
            default_retval=(None, ''),
            supress_error=True,
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
            **kwargs,
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
        Create informative-pairs needed for FreqAI
        """
        if self.config.get('freqai', {}).get('enabled', False):
            whitelist_pairs: List[str] = self.dp.current_whitelist()
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
        Internal method which gathers all informative pairs (user or automatically defined).
        """
        informative_pairs: List[Tuple[str, str, CandleType]] = self.informative_pairs()
        informative_pairs = [
            (
                p[0],
                p[1],
                CandleType.from_string(p[2])
                if len(p) > 2 and p[2] != ''
                else self.config.get('candle_type_def', CandleType.SPOT),
            )
            for p in informative_pairs
        ]
        for inf_data, _ in self._ft_informative:
            candle_type: CandleType = (
                inf_data.candle_type
                if inf_data.candle_type
                else self.config.get('candle_type_def', CandleType.SPOT)
            )
            if inf_data.asset:
                if any((s in inf_data.asset for s in ('{BASE}', '{base}'))):
                    for pair in self.dp.current_whitelist():
                        pair_tf = (
                            _format_pair_name(self.config, inf_data.asset, self.dp.market(pair)),
                            inf_data.timeframe,
                            candle_type,
                        )
                        informative_pairs.append(pair_tf)
                else:
                    pair_tf = (
                        _format_pair_name(self.config, inf_data.asset),
                        inf_data.timeframe,
                        candle_type,
                    )
                    informative_pairs.append(pair_tf)
            else:
                for pair in self.dp.current_whitelist():
                    informative_pairs.append((pair, inf_data.timeframe, candle_type))
        informative_pairs.extend(self.__informative_pairs_freqai())
        return list(set(informative_pairs))

    def get_strategy_name(self) -> str:
        """
        Returns strategy class name
        """
        return self.__class__.__name__

    def lock_pair(
        self,
        pair: str,
        until: datetime,
        reason: Optional[str] = None,
        side: str = '*',
    ) -> None:
        """
        Locks pair until a given timestamp happens.
        Locked pairs are not analyzed, and are prevented from opening new trades.
        Locks can only count up (allowing users to lock pairs for a longer period of time).
        To remove a lock from a pair, use `unlock_pair()`
        :param pair: Pair to lock
        :param until: datetime in UTC until the pair should be blocked from opening new trades.
                Needs to be timezone aware `datetime.now(timezone.utc)`
        :param reason: Optional string explaining why the pair was locked.
        :param side: Side to check, can be long, short or '*'
        """
        PairLocks.lock_pair(pair, until, reason, side=side)

    def unlock_pair(self, pair: str) -> None:
        """
        Unlocks a pair previously locked using lock_pair.
        Not used by freqtrade itself, but intended to be used if users lock pairs
        manually from within the strategy, to allow an easy way to unlock pairs.
        :param pair: Unlock pair to allow trading again
        """
        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:
        """
        Unlocks all pairs previously locked using lock_pair with specified reason.
        Not used by freqtrade itself, but intended to be used if users lock pairs
        manually from within the strategy, to allow an easy way to unlock pairs.
        :param reason: Unlock pairs to allow trading again
        """
        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(
        self,
        pair: str,
        *,
        candle_date: Optional[datetime] = None,
        side: str = '*',
    ) -> bool:
        """
        Checks if a pair is currently locked
        The 2nd, optional parameter ensures that locks are applied until the new candle arrives,
        and not stop at 14:00:00 - while the next candle arrives at 14:00:02 leaving a gap
        of 2 seconds for an entry order to happen on an old signal.
        :param pair: "Pair to check"
        :param candle_date: Date of the last candle. Optional, defaults to current date
        :param side: Side to check, can be long, short or '*'
        :returns: locking state of the pair in question.
        """
        if not candle_date:
            return PairLocks.is_pair_locked(pair, side=side)
        else:
            lock_time: datetime = timeframe_to_next_date(
                self.timeframe, candle_date
            )
            return PairLocks.is_pair_locked(pair, lock_time, side=side)

    def analyze_ticker(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and entry order signal to it
        Should only be used in live.
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        logger.debug('TA Analysis Launched')
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        logger.debug('TA Analysis Ended')
        return dataframe

    def _analyze_ticker_internal(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and buy signal to it
        WARNING: Used internally only, may skip analysis if `process_only_new_candles` is set.
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        pair: str = str(metadata.get('pair'))
        new_candle: bool = self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']
        if not self.process_only_new_candles or new_candle:
            dataframe = self.analyze_ticker(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
            candle_type: CandleType = self.config.get('candle_type_def', CandleType.SPOT)
            self.dp._set_cached_df(pair, self.timeframe, dataframe, candle_type=candle_type)
            self.dp._emit_df((pair, self.timeframe, candle_type), dataframe, new_candle)
        else:
            logger.debug('Skipping TA Analysis for already analyzed candle')
            dataframe = remove_entry_exit_signals(dataframe)
        logger.debug('Loop Analysis Launched')
        return dataframe

    def analyze_pair(self, pair: str) -> None:
        """
        Fetch data for this pair from dataprovider and analyze.
        Stores the dataframe into the dataprovider.
        The analyzed dataframe is then accessible via `dp.get_analyzed_dataframe()`.
        :param pair: Pair to analyze.
        """
        dataframe: Optional[DataFrame] = self.dp.ohlcv(
            pair, self.timeframe, candle_type=self.config.get('candle_type_def', CandleType.SPOT)
        )
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning('Empty candle (OHLCV) data for pair %s', pair)
            return
        try:
            df_len: int
            df_close: float
            df_date: datetime
            df_len, df_close, df_date = self.preserve_df(dataframe)
            dataframe = strategy_safe_wrapper(
                self._analyze_ticker_internal, message=''
            )(dataframe, {'pair': pair})
            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            logger.warning(f'Unable to analyze candle (OHLCV) data for pair {pair}: {error}')
            return
        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return

    def analyze(self, pairs: List[str]) -> None:
        """
        Analyze all pairs using analyze_pair().
        :param pairs: List of pairs to analyze
        """
        for pair in pairs:
            self.analyze_pair(pair)

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, datetime]:
        """keep some data for dataframes"""
        return (len(dataframe), dataframe['close'].iloc[-1], dataframe['date'].iloc[-1])

    def assert_df(
        self,
        dataframe: Optional[DataFrame],
        df_len: int,
        df_close: float,
        df_date: datetime,
    ) -> None:
        """
        Ensure dataframe (length, last candle) was not modified, and has all elements we need.
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

    def get_latest_candle(
        self, pair: str, timeframe: str, dataframe: DataFrame
    ) -> Tuple[Optional[Any], Optional[datetime]]:
        """
        Calculates current signal based based on the entry order or exit order
        columns of the dataframe.
        Used by Bot to get the signal to enter, or exit
        :param pair: pair in format ANT/BTC
        :param timeframe: timeframe to use
        :param dataframe: Analyzed dataframe to get signal from.
        :return: (None, None) or (Dataframe, latest_date) - corresponding to the last candle
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning(f'Empty candle (OHLCV) data for pair {pair}')
            return (None, None)
        try:
            latest_date_pd: Any = dataframe['date'].max()
            latest: Any = dataframe.loc[dataframe['date'] == latest_date_pd].iloc[-1]
        except Exception as e:
            logger.warning(f'Unable to get latest candle (OHLCV) data for pair {pair} - {e}')
            return (None, None)
        latest_date: datetime = latest_date_pd.to_pydatetime()
        timeframe_minutes: int = timeframe_to_minutes(timeframe)
        offset: int = self.config.get('exchange', {}).get('outdated_offset', 5)
        if latest_date < dt_now() - timedelta(minutes=timeframe_minutes * 2 + offset):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                int((dt_now() - latest_date).total_seconds() // 60),
            )
            return (None, None)
        return (latest, latest_date)

    def get_exit_signal(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
        is_short: Optional[bool] = None,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Calculates current exit signal based based on the dataframe
        columns of the dataframe.
        Used by Bot to get the signal to exit.
        depending on is_short, looks at "short" or "long" columns.
        :param pair: pair in format ANT/BTC
        :param timeframe: timeframe to use
        :param dataframe: Analyzed dataframe to get signal from.
        :param is_short: Indicating existing trade direction.
        :return: (enter, exit) A bool-tuple with enter / exit values.
        """
        latest: Optional[Any]
        _latest_date: Optional[datetime]
        latest, _latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None:
            return (False, False, None)
        if is_short:
            enter: bool = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
            exit_: bool = latest.get(SignalType.EXIT_SHORT.value, 0) == 1
        else:
            enter = latest.get(SignalType.ENTER_LONG.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        exit_tag: Optional[str] = latest.get(SignalTagType.EXIT_TAG.value, None)
        exit_tag = exit_tag if isinstance(exit_tag, str) and exit_tag != 'nan' else None
        logger.debug(
            f"exit-trigger: {latest['date']} (pair={pair}) enter={enter} exit={exit_}"
        )
        return (enter, exit_, exit_tag)

    def get_entry_signal(
        self, pair: str, timeframe: str, dataframe: DataFrame
    ) -> Tuple[Optional[SignalDirection], Optional[str]]:
        """
        Calculates current entry signal based based on the dataframe signals
        columns of the dataframe.
        Used by Bot to get the signal to enter trades.
        :param pair: pair in format ANT/BTC
        :param timeframe: timeframe to use
        :param dataframe: Analyzed dataframe to get signal from.
        :return: (SignalDirection, entry_tag)
        """
        latest: Optional[Any]
        latest_date: Optional[datetime]
        latest, latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None or latest_date is None:
            return (None, None)
        enter_long: bool = latest.get(SignalType.ENTER_LONG.value, 0) == 1
        exit_long: bool = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        enter_short: bool = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
        exit_short: bool = latest.get(SignalType.EXIT_SHORT.value, 0) == 1
        enter_signal: Optional[SignalDirection] = None
        enter_tag: Optional[str] = None
        if enter_long and not any([exit_long, enter_short]):
            enter_signal = SignalDirection.LONG
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        if (
            self.config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT
            and self.can_short
            and enter_short
            and not any([exit_short, enter_long])
        ):
            enter_signal = SignalDirection.SHORT
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        enter_tag = enter_tag if isinstance(enter_tag, str) and enter_tag != 'nan' else None
        timeframe_seconds: int = timeframe_to_seconds(timeframe)
        if self.ignore_expired_candle(
            latest_date=latest_date,
            current_time=dt_now(),
            timeframe_seconds=timeframe_seconds,
            enter=bool(enter_signal),
        ):
            return (None, enter_tag)
        logger.debug(
            f"entry trigger: {latest['date']} (pair={pair}) enter={enter_long} enter_tag_value={enter_tag}"
        )
        return (enter_signal, enter_tag)

    def ignore_expired_candle(
        self,
        latest_date: datetime,
        current_time: datetime,
        timeframe_seconds: int,
        enter: bool,
    ) -> bool:
        if self.ignore_buying_expired_candle_after and enter:
            time_delta: timedelta = current_time - (
                latest_date + timedelta(seconds=timeframe_seconds)
            )
            return time_delta.total_seconds() > self.ignore_buying_expired_candle_after
        else:
            return False

    def should_exit(
        self,
        trade: Trade,
        rate: float,
        current_time: datetime,
        *,
        enter: bool,
        exit_: bool,
        low: Optional[float] = None,
        high: Optional[float] = None,
        force_stoploss: float = 0.0,
    ) -> List[ExitCheckTuple]:
        """
        This function evaluates if one of the conditions required to trigger an exit order
        has been reached, which can either be a stop-loss, ROI or exit-signal.
        :param low: Only used during backtesting to simulate (long)stoploss/(short)ROI
        :param high: Only used during backtesting, to simulate (short)stoploss/(long)ROI
        :param force_stoploss: Externally provided stoploss
        :return: List of exit reasons - or empty list.
        """
        exits: List[ExitCheckTuple] = []
        current_rate_best: float = rate
        current_profit_best: float = trade.calc_profit_ratio(rate)
        if low is not None or high is not None:
            current_rate_best = (low if trade.is_short else high) or rate
            current_profit_best = trade.calc_profit_ratio(current_rate_best)
        trade.adjust_min_max_rates(high or rate, low or rate)
        stoplossflag: ExitCheckTuple = self.ft_stoploss_reached(
            current_rate=rate,
            trade=trade,
            current_time=current_time,
            current_profit=current_profit,
            force_stoploss=force_stoploss,
            low=low,
            high=high,
        )
        roi_reached: bool = not (
            enter and self.ignore_roi_if_entry_signal
        ) and self.min_roi_reached(
            trade=trade, current_profit=current_profit_best, current_time=current_time
        )
        exit_signal: ExitType = ExitType.NONE
        custom_reason: str = ''
        if self.use_exit_signal:
            if exit_ and not enter:
                exit_signal = ExitType.EXIT_SIGNAL
            else:
                reason_cust: Union[str, bool] = strategy_safe_wrapper(
                    self.custom_exit,
                    default_retval=False,
                )(
                    pair=trade.pair,
                    trade=trade,
                    current_time=current_time,
                    current_rate=rate,
                    current_profit=current_profit,
                )
                if reason_cust:
                    exit_signal = ExitType.CUSTOM_EXIT
                    if isinstance(reason_cust, str):
                        if len(reason_cust) > CUSTOM_TAG_MAX_LENGTH:
                            logger.warning(
                                f'Custom exit reason returned from custom_exit is too long and was trimmedto {CUSTOM_TAG_MAX_LENGTH} characters.'
                            )
                            custom_reason = reason_cust[:CUSTOM_TAG_MAX_LENGTH]
                        else:
                            custom_reason = reason_cust
                    else:
                        custom_reason = ''
            if (
                exit_signal == ExitType.CUSTOM_EXIT
                or (
                    exit_signal == ExitType.EXIT_SIGNAL
                    and (not self.exit_profit_only or current_profit > self.exit_profit_offset)
                )
            ):
                logger.debug(
                    f"{trade.pair} - Sell signal received. exit_type=ExitType.{exit_signal.name}"
                    + (f', custom_reason={custom_reason}' if custom_reason else '')
                )
                exits.append(
                    ExitCheckTuple(exit_type=exit_signal, exit_reason=custom_reason)
                )
        if stoplossflag.exit_type in (ExitType.STOP_LOSS, ExitType.LIQUIDATION):
            logger.debug(
                f"{trade.pair} - Stoploss hit. exit_type={stoplossflag.exit_type}"
            )
            exits.append(stoplossflag)
        if roi_reached:
            logger.debug(
                f"{trade.pair} - Required profit reached. exit_type=ExitType.ROI"
            )
            exits.append(ExitCheckTuple(exit_type=ExitType.ROI))
        if stoplossflag.exit_type == ExitType.TRAILING_STOP_LOSS:
            logger.debug(f"{trade.pair} - Trailing stoploss hit.")
            exits.append(stoplossflag)
        return exits

    def ft_stoploss_adjust(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float,
        low: Optional[float],
        high: Optional[float],
    ) -> None:
        """
        Adjust stop-loss dynamically if configured to do so.
        :param current_profit: current profit as ratio
        :param low: Low value of this candle, only set in backtesting
        :param high: High value of this candle, only set in backtesting
        """
        if force_stoploss:
            stop_loss_value = force_stoploss
        else:
            stop_loss_value = self.stoploss

        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        if not trade.is_short:
            dir_correct: bool = trade.stop_loss < (low or current_rate)
        else:
            dir_correct = trade.stop_loss > (high or current_rate)

        bound: Optional[float] = low if not trade.is_short else high
        bound_profit: float = current_profit if not bound else trade.calc_profit_ratio(bound)

        if self.use_custom_stoploss and dir_correct:
            stop_loss_value_custom: Optional[float] = strategy_safe_wrapper(
                self.custom_stoploss,
                default_retval=None,
                supress_error=True,
            )(
                pair=trade.pair,
                trade=trade,
                current_time=current_time,
                current_rate=bound or current_rate,
                current_profit=bound_profit,
                after_fill=False,  # assuming after_fill is handled elsewhere
            )
            if (
                stop_loss_value_custom is not None
                and not (isnan(stop_loss_value_custom) or isinf(stop_loss_value_custom))
            ):
                stop_loss_value = stop_loss_value_custom
                trade.adjust_stop_loss(
                    bound or current_rate, stop_loss_value, allow_refresh=False
                )
            else:
                logger.debug('CustomStoploss function did not return valid stoploss')

        if self.trailing_stop and dir_correct:
            sl_offset: float = self.trailing_stop_positive_offset
            if not (self.trailing_only_offset_is_reached and bound_profit < sl_offset):
                if self.trailing_stop_positive is not None and bound_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive
                    logger.debug(
                        f"{trade.pair} - Using positive stoploss: {stop_loss_value} "
                        f"offset: {sl_offset:.4g} profit: {bound_profit:.2%}"
                    )
                trade.adjust_stop_loss(bound or current_rate, stop_loss_value)

    def ft_stoploss_reached(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float,
        low: Optional[float],
        high: Optional[float],
    ) -> ExitCheckTuple:
        """
        Based on current profit of the trade and configured (trailing) stoploss,
        decides to exit or not
        :param current_profit: current profit as ratio
        :param low: Low value of this candle, only set in backtesting
        :param high: High value of this candle, only set in backtesting
        """
        self.ft_stoploss_adjust(
            current_rate, trade, current_time, current_profit, force_stoploss, low, high
        )
        if not trade.is_short:
            sl_higher_long: bool = trade.stop_loss >= (low or current_rate)
            liq_higher_long: bool = (
                trade.liquidation_price is not None
                and trade.liquidation_price >= (low or current_rate)
            )
        else:
            sl_lower_short: bool = trade.stop_loss <= (high or current_rate)
            liq_lower_short: bool = (
                trade.liquidation_price is not None
                and trade.liquidation_price <= (high or current_rate)
            )

        if not trade.is_short:
            sl_condition: bool = trade.stop_loss >= (low or current_rate)
            liq_condition: bool = (
                trade.liquidation_price is not None
                and trade.liquidation_price >= (low or current_rate)
            )
            sl_higher_long = sl_condition
            liq_higher_long = liq_condition
        else:
            sl_condition: bool = trade.stop_loss <= (high or current_rate)
            liq_condition: bool = (
                trade.liquidation_price is not None
                and trade.liquidation_price <= (high or current_rate)
            )
            sl_lower_short = sl_condition
            liq_lower_short = liq_condition

        if (
            (not trade.is_short and sl_higher_long)
            or (trade.is_short and sl_lower_short)
        ) and (
            not self.order_types.get('stoploss_on_exchange', False)
            or self.config['dry_run']
        ):
            exit_type: ExitType = ExitType.STOP_LOSS
            if trade.is_stop_loss_trailing:
                exit_type = ExitType.TRAILING_STOP_LOSS
                logger.debug(
                    f"{trade.pair} - HIT STOP: current price at {(high if trade.is_short else low) or current_rate:.6f}, "
                    f"stoploss is {trade.stop_loss:.6f}, initial stoploss was at {trade.initial_stop_loss:.6f}, "
                    f"trade opened at {trade.open_rate:.6f}"
                )
            return ExitCheckTuple(exit_type=exit_type)
        if (not trade.is_short and liq_higher_long) or (
            trade.is_short and liq_lower_short
        ):
            logger.debug(f"{trade.pair} - Liquidation price hit. exit_type=ExitType.LIQUIDATION")
            return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)
        return ExitCheckTuple(exit_type=ExitType.NONE)

    def min_roi_reached_entry(
        self, trade_dur: int
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Based on trade duration defines the ROI entry that may have been reached.
        :param trade_dur: trade duration in minutes
        :return: minimal ROI entry value or None if none proper ROI entry was found.
        """
        roi_list: List[int] = [x for x in self.minimal_roi.keys() if x <= trade_dur]
        if not roi_list:
            return (None, None)
        roi_entry: int = max(roi_list)
        return (roi_entry, self.minimal_roi[roi_entry])

    def min_roi_reached(
        self, trade: Trade, current_profit: float, current_time: datetime
    ) -> bool:
        """
        Based on trade duration, current profit of the trade and ROI configuration,
        decides whether bot should exit.
        :param current_profit: current profit as ratio
        :return: True if bot should exit at current rate
        """
        trade_dur: int = int(
            (current_time.timestamp() - trade.open_date_utc.timestamp()) // 60
        )
        _, roi: Tuple[Optional[int], Optional[float]] = self.min_roi_reached_entry(
            trade_dur
        )
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ft_check_timed_out(
        self, trade: Trade, order: Order, current_time: datetime
    ) -> bool:
        """
        FT Internal method.
        Check if timeout is active, and if the order is still open and timed out
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
        time_method: Callable[..., bool] = (
            self.check_exit_timeout
            if order.ft_order_side == trade.exit_side
            else self.check_entry_timeout
        )
        return strategy_safe_wrapper(time_method, default_retval=False)(
            pair=trade.pair, trade=trade, order=order, current_time=current_time
        )

    def advise_all_indicators(
        self, data: Dict[str, DataFrame]
    ) -> Dict[str, DataFrame]:
        """
        Populates indicators for given candle (OHLCV) data (for multiple pairs)
        Does not run advise_entry or advise_exit!
        Used by optimize operations only, not during dry / live runs.
        Using .copy() to get a fresh copy of the dataframe for every strategy run.
        Also copy on output to avoid PerformanceWarnings pandas 1.3.0 started to show.
        Has positive effects on memory usage for whatever reason - also when
        using only one strategy.
        """
        return {
            pair: self.advise_indicators(pair_data.copy(), {'pair': pair}).copy()
            for pair, pair_data in data.items()
        }

    def ft_advise_signals(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Call advise_entry and advise_exit and return the resulting dataframe.
        :param dataframe: Dataframe containing data from exchange, as well as pre-calculated
                          indicators
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added

        """
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        return dataframe

    def _if_enabled_populate_trades(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> None:
        use_public_trades: bool = self.config.get('exchange', {}).get(
            'use_public_trades', False
        )
        if use_public_trades:
            trades: Any = self.dp.trades(pair=metadata['pair'], copy=False)
            pair: str = metadata['pair']
            cached_grouped_trades: Any = self._cached_grouped_trades_per_pair.get(pair)
            dataframe, cached_grouped_trades = populate_dataframe_with_trades(
                cached_grouped_trades, self.config, dataframe, trades
            )
            if pair in self._cached_grouped_trades_per_pair:
                del self._cached_grouped_trades_per_pair[pair]
            self._cached_grouped_trades_per_pair[pair] = cached_grouped_trades
            logger.debug('Populated dataframe with trades.')

    def advise_indicators(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Populate indicators that will be used in the Buy, Sell, short, exit_short strategy
        This method should not be overridden.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        logger.debug(f"Populating indicators for pair {metadata.get('pair')}.")
        for inf_data, populate_fn in self._ft_informative:
            dataframe = _create_and_merge_informative_pair(
                self, dataframe, metadata, inf_data, populate_fn
            )
        self._if_enabled_populate_trades(dataframe, metadata)
        return self.populate_indicators(dataframe, metadata)

    def advise_entry(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Based on TA indicators, populates the entry order signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param metadata: Additional information dictionary, with details like the
            currently traded pair
        :return: DataFrame with buy column
        """
        logger.debug(f"Populating enter signals for pair {metadata.get('pair')}.")
        dataframe.loc[:, 'enter_tag'] = ''
        df: DataFrame = self.populate_entry_trend(dataframe, metadata)
        if 'enter_long' not in df.columns:
            df = df.rename(
                {'buy': 'enter_long', 'buy_tag': 'enter_tag'}, axis='columns'
            )
        return df

    def advise_exit(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Based on TA indicators, populates the exit order signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param metadata: Additional information dictionary, with details like the
            currently traded pair
        :return: DataFrame with exit column
        """
        dataframe.loc[:, 'exit_tag'] = ''
        logger.debug(f"Populating exit signals for pair {metadata.get('pair')}.")
        df: DataFrame = self.populate_exit_trend(dataframe, metadata)
        if 'exit_long' not in df.columns:
            df = df.rename({'sell': 'exit_long'}, axis='columns')
        return df
