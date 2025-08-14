"""
IStrategy interface
This module defines the interface to apply for strategies
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from math import isinf, isnan
from typing import Any, Dict, List, Optional, Tuple, Union

from pandas import DataFrame

from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, Config, IntOrInf, ListPairsWithTimeframes
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
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_next_date, timeframe_to_seconds
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

    # Strategy interface version
    # Default to version 2
    # Version 1 is the initial interface without metadata dict - deprecated and no longer supported.
    # Version 2 populate_* include metadata dict
    # Version 3 - First version with short and leverage support
    INTERFACE_VERSION: int = 3

    _ft_params_from_file: Dict[str, Any]
    # associated minimal roi
    minimal_roi: Dict[int, float] = {}

    # associated stoploss
    stoploss: float

    # max open trades for the strategy
    max_open_trades: IntOrInf

    # trailing stoploss
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    use_custom_stoploss: bool = False

    # Can this strategy go short?
    can_short: bool = False

    # associated timeframe
    timeframe: str

    # Optional order types
    order_types: Dict[str, str] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    }

    # Optional time in force
    order_time_in_force: Dict[str, str] = {
        "entry": "GTC",
        "exit": "GTC",
    }

    # run "populate_indicators" only for new candle
    process_only_new_candles: bool = True

    use_exit_signal: bool
    exit_profit_only: bool
    exit_profit_offset: float
    ignore_roi_if_entry_signal: bool

    # Position adjustment is disabled by default
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1

    # Number of seconds after which the candle will no longer result in a buy on expired candles
    ignore_buying_expired_candle_after: int = 0

    # Disable checking the dataframe (converts the error into a warning message)
    disable_dataframe_checks: bool = False

    # Count of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0

    # Protections
    protections: List[Dict[str, Any]] = []

    # Class level variables (intentional) containing
    # the dataprovider (dp) (access to other candles, historic data, ...)
    # and wallets - access to the current balance.
    dp: DataProvider
    wallets: Optional[Wallets] = None
    # Filled from configuration
    stake_currency: str
    # container variable for strategy source code
    __source__: str = ""

    # Definition of plot_config. See plotting documentation for more details.
    plot_config: Dict[str, Any] = {}

    # A self set parameter that represents the market direction. filled from configuration
    market_direction: MarketDirection = MarketDirection.NONE

    # Global cache dictionary
    _cached_grouped_trades_per_pair: Dict[str, DataFrame] = {}

    def __init__(self, config: Config) -> None:
        self.config = config
        # Dict to determine if analysis is necessary
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        super().__init__(config)

        # Gather informative pairs from @informative-decorated methods.
        self._ft_informative: List[Tuple[InformativeData, PopulateIndicators]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, "_ft_informative", None)
            if not isinstance(informative_data_list, list):
                # Type check is required because mocker would return a mock object that evaluates to
                # True, confusing this code.
                continue
            strategy_timeframe_minutes = timeframe_to_minutes(self.timeframe)
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException(
                        "Informative timeframe must be equal or higher than strategy timeframe!"
                    )
                if not informative_data.candle_type:
                    informative_data.candle_type = config["candle_type_def"]
                self._ft_informative.append((informative_data, cls_method))

    def load_freqAI_model(self) -> None:
        if self.config.get("freqai", {}).get("enabled", False):
            # Import here to avoid importing this if freqAI is disabled
            from freqtrade.freqai.utils import download_all_data_for_training
            from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

            self.freqai = FreqaiModelResolver.load_freqaimodel(self.config)
            self.freqai_info = self.config["freqai"]

            # download the desired data in dry/live
            if self.config.get("runmode") in (RunMode.DRY_RUN, RunMode.LIVE):
                logger.info(
                    "Downloading all training data for all pairs in whitelist and "
                    "corr_pairlist, this may take a while if the data is not "
                    "already on disk."
                )
                download_all_data_for_training(self.dp, self.config)
        else:
            # Gracious failures if freqAI is disabled but "start" is called.
            class DummyClass:
                def start(self, *args, **kwargs):
                    raise OperationalException(
                        "freqAI is not enabled. "
                        "Please enable it in your config to use this strategy."
                    )

                def shutdown(self, *args, **kwargs):
                    pass

            self.freqai = DummyClass()  # type: ignore

    def ft_bot_start(self, **kwargs) -> None:
        """
        Strategy init - runs after dataprovider has been added.
        Must call bot_start()
        """
        self.load_freqAI_model()

        strategy_safe_wrapper(self.bot_start)()

        self.ft_load_hyper_params(self.config.get("runmode") == RunMode.HYPEROPT)

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
        :param dataframe: DataFrame
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
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return self.populate_sell_trend(dataframe, metadata)

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def check_buy_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        DEPRECATED: Please use `check_entry_timeout` instead.
        """
        return False

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
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
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        DEPRECATED: Please use `check_exit_timeout` instead.
        """
        return False

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
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
        **kwargs,
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
        **kwargs,
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
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
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
        **kwargs,
    ) -> Optional[float]:
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
        :param current_profit: Current profit (as ratio), calculated based on current_rate