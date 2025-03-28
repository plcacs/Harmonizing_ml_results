from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from math import isinf, isnan
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
    INTERFACE_VERSION: int = 3

    _ft_params_from_file: Dict
    minimal_roi: Dict = {}
    stoploss: float
    max_open_trades: IntOrInf
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False
    use_custom_stoploss: bool = False
    can_short: bool = False
    timeframe: str
    order_types: Dict = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    }
    order_time_in_force: Dict = {
        "entry": "GTC",
        "exit": "GTC",
    }
    process_only_new_candles: bool = True
    use_exit_signal: bool
    exit_profit_only: bool
    exit_profit_offset: float
    ignore_roi_if_entry_signal: bool
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1
    ignore_buying_expired_candle_after: int = 0
    disable_dataframe_checks: bool = False
    startup_candle_count: int = 0
    protections: List = []
    dp: DataProvider
    wallets: Optional[Wallets] = None
    stake_currency: str
    __source__: str = ""
    plot_config: Dict = {}
    market_direction: MarketDirection = MarketDirection.NONE
    _cached_grouped_trades_per_pair: Dict[str, DataFrame] = {}

    def __init__(self, config: Config) -> None:
        self.config = config
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        super().__init__(config)
        self._ft_informative: List[Tuple[InformativeData, PopulateIndicators]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, "_ft_informative", None)
            if not isinstance(informative_data_list, list):
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
            from freqtrade.freqai.utils import download_all_data_for_training
            from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

            self.freqai = FreqaiModelResolver.load_freqaimodel(self.config)
            self.freqai_info = self.config["freqai"]

            if self.config.get("runmode") in (RunMode.DRY_RUN, RunMode.LIVE):
                logger.info(
                    "Downloading all training data for all pairs in whitelist and "
                    "corr_pairlist, this may take a while if the data is not "
                    "already on disk."
                )
                download_all_data_for_training(self.dp, self.config)
        else:
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
        self.load_freqAI_model()
        strategy_safe_wrapper(self.bot_start))()
        self.ft_load_hyper_params(self.config.get("runmode") == RunMode.HYPEROPT)

    def ft_bot_cleanup(self) -> None:
        self.freqai.shutdown()

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        return self.populate_buy_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        return self.populate_sell_trend(dataframe, metadata)

    def bot_start(self, **kwargs) -> None:
        pass

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        pass

    def check_buy_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return False

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return self.check_buy_timeout(
            pair=pair, trade=trade, order=order, current_time=current_time
        )

    def check_sell_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return False

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
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
        return True

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
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
        return self.stoploss

    def custom_entry_price(
        self,
        pair: str,
        trade: Optional[Trade],
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: Optional[str],
        **kwargs,
    ) -> float:
        return proposed_rate

    def custom_sell(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Union[str, bool, None]:
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Union[str, bool, None]:
        return self.custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return proposed_stake

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Union[float, None, Tuple[Optional[float], Optional[str]]]:
        return None

    def adjust_entry_price(
        self,
        trade: Trade,
        order: Optional[Order],
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
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
        **kwargs,
    ) -> float:
        return 1.0

    def informative_pairs(self) -> ListPairsWithTimeframes:
        return []

    def version(self) -> Optional[str]:
        return None

    def populate_any_indicators(
        self,
        pair: str,
        df: DataFrame,
        tf: str,
        informative: Optional[DataFrame] = None,
        set_generalized_indicators: bool = False,
    ) -> DataFrame:
        return df

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        return dataframe

    _ft_stop_uses_after_fill = False

    def _adjust_trade_position_internal(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Tuple[Optional[float], str]:
        resp = strategy_safe_wrapper(
            self.adjust_trade_position, default_retval=(None, ""), supress_error=True
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
        order_tag = ""
        if isinstance(resp, tuple):
            if len(resp) >= 1:
                stake_amount = resp[0]
            if len(resp) > 1:
                order_tag = resp[1] or ""
        else:
            stake_amount = resp
        return stake_amount, order_tag

    def __informative_pairs_freqai(self) -> ListPairsWithTimeframes:
        if self.config.get("freqai", {}).get("enabled", False):
            whitelist_pairs = self.dp.current_whitelist()
            candle_type = self.config.get("candle_type_def", CandleType.SPOT)
            corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
            informative_pairs = []
            for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
                for pair in set(whitelist_pairs + corr_pairs):
                    informative_pairs.append((pair, tf, candle_type))
            return informative_pairs

        return []

    def gather_informative_pairs(self) -> ListPairsWithTimeframes:
        informative_pairs = self.informative_pairs()
        informative_pairs = [
            (
                p[0],
                p[1],
                (
                    CandleType.from_string(p[2])
                    if len(p) > 2 and p[2] != ""
                    else self.config.get("candle_type_def", CandleType.SPOT)
                ),
            )
            for p in informative_pairs
        ]
        for inf_data, _ in self._ft_informative:
            candle_type = (
                inf_data.candle_type
                if inf_data.candle_type
                else self.config.get("candle_type_def", CandleType.SPOT)
            )
            if inf_data.asset:
                if any(s in inf_data.asset for s in ("{BASE}", "{base}")):
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
        return self.__class__.__name__

    def lock_pair(
        self, pair: str, until: datetime, reason: Optional[str] = None, side: str = "*"
    ) -> None:
        PairLocks.lock_pair(pair, until, reason, side=side)

    def unlock_pair(self, pair: str) -> None:
        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:
        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(
        self, pair: str, *, candle_date: Optional[datetime] = None, side: str = "*"
    ) -> bool:
        if not candle_date:
            return PairLocks.is_pair_locked(pair, side=side)
        else:
            lock_time = timeframe_to_next_date(self.timeframe, candle_date)
            return PairLocks.is_pair_locked(pair, lock_time, side=side)

    def analyze_ticker(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        logger.debug("TA Analysis Launched")
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        logger.debug("TA Analysis Ended")
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        pair = str(metadata.get("pair"))

        new_candle = self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]["date"]
        if not self.process_only_new_candles or new_candle:
            dataframe = self.analyze_ticker(dataframe, metadata)

            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]["date"]

            candle_type = self.config.get("candle_type_def", CandleType.SPOT)
            self.dp._set_cached_df(pair, self.timeframe, dataframe, candle_type=candle_type)
            self.dp._emit_df((pair, self.timeframe, candle_type), dataframe, new_candle)

        else:
            logger.debug("Skipping TA Analysis for already analyzed candle")
            dataframe = remove_entry_exit_signals(dataframe)

        logger.debug("Loop Analysis Launched")

        return dataframe

    def analyze_pair(self, pair: str) -> None:
        dataframe = self.dp.ohlcv(
            pair, self.timeframe, candle_type=self.config.get("candle_type_def", CandleType.SPOT)
        )
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning("Empty candle (OHLCV) data for pair %s", pair)
            return

        try:
            df_len, df_close, df_date = self.preserve_df(dataframe)

            dataframe = strategy_safe_wrapper(self._analyze_ticker_internal, message="")(
                dataframe, {"pair": pair}
            )

            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            logger.warning(f"Unable to analyze candle (OHLCV) data for pair {pair}: {error}")
            return

        if dataframe.empty:
            logger.warning("Empty dataframe for pair %s", pair)
            return

    def analyze(self, pairs: List[str]) -> None:
        for pair in pairs:
            self.analyze_pair(pair)

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, datetime]:
        return len(dataframe), dataframe["close"].iloc[-1], dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame, df_len: int, df_close: float, df_date: datetime):
        message_template = "Dataframe returned from strategy has mismatching {}."
        message = ""
        if dataframe is None:
            message = "No dataframe returned (return statement missing?)."
        elif df_len != len(dataframe):
            message = message_template.format("length")
        elif df_close != dataframe["close"].iloc[-1]:
            message = message_template.format("last close price")
        elif df_date != dataframe["date"].iloc[-1]:
            message = message_template.format("last date")
        if message:
            if self.disable_dataframe_checks:
                logger.warning(message)
            else:
                raise StrategyError(message)

    def get_latest_candle(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
    ) -> Tuple[Optional[DataFrame], Optional[datetime]]:
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning(f"Empty candle (OHLCV) data for pair {pair}")
            return None, None

        try:
            latest_date_pd = dataframe["date"].max()
            latest = dataframe.loc[dataframe["date"] == latest_date_pd].iloc[-1]
        except Exception as e:
            logger.warning(f"Unable to get latest candle (OHLCV) data for pair {pair} - {e}")
            return None, None
        latest_date: datetime = latest_date_pd.to_pydatetime()

        timeframe_minutes = timeframe_to_minutes(timeframe)
        offset = self.config.get("exchange", {}).get("outdated_offset", 5)
        if latest_date < (dt_now() - timedelta(minutes=timeframe_minutes * 2 + offset)):
            logger.warning(
                "Outdated history for pair %s. Last tick is %s minutes old",
                pair,
                int((dt_now() - latest_date).total_seconds() // 60),
            )
            return None, None
        return latest, latest_date

    def get_exit_signal(
        self, pair: str, timeframe: str, dataframe: DataFrame, is_short: Optional[bool] = None
    ) -> Tuple[bool, bool, Optional[str]]:
        latest, _latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None:
            return False, False, None

        if is_short:
            enter = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_SHORT.value, 0) == 1

        else:
            enter = latest.get(SignalType.ENTER_LONG.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        exit_tag = latest.get(SignalTagType.EXIT_TAG.value, None)
        exit_tag = exit_tag if isinstance(exit_tag, str) and exit_tag != "nan" else None

        logger.debug(f"exit-trigger: {latest['date']} (pair={pair}) enter={enter} exit={exit_}")

        return enter, exit_, exit_tag

    def get_entry_signal(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
    ) -> Tuple[Optional[SignalDirection], Optional[str]]:
        latest, latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None or latest_date is None:
            return None, None

        enter_long = latest.get(SignalType.ENTER_LONG.value, 0) == 1
        exit_long = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        enter_short = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
        exit_short = latest.get(SignalType.EXIT_SHORT.value, 0) == 1

        enter_signal: Optional[SignalDirection] = None
        enter_tag: Optional[str] = None
        if enter_long == 1 and not any([exit_long, enter_short]):
            enter_signal = SignalDirection.LONG
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        if (
            self.config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT
            and self.can_short
            and enter_short == 1
            and not any([exit_short, enter_long])
        ):
            enter_signal = SignalDirection.SHORT
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)

        enter_tag = enter_tag if isinstance(enter_tag, str) and enter_tag != "nan" else None

        timeframe_seconds = timeframe_to_seconds(timeframe)

        if self.ignore_expired_candle(
            latest_date=latest_date,
            current_time=dt_now(),
            timeframe_seconds=timeframe_seconds,
            enter=bool(enter_signal),
        ):
            return None, enter_tag

        logger.debug(
            f"entry trigger: {latest['date']} (pair={pair}) "
            f"enter={enter_long} enter_tag_value={enter_tag}"
        )
        return enter_signal, enter_tag

    def ignore_expired_candle(
        self, latest_date: datetime, current_time: datetime, timeframe_seconds: int, enter: bool
    ) -> bool:
        if self.ignore_buying_expired_candle_after and enter:
            time_delta = current_time - (latest_date + timedelta(seconds=timeframe_seconds))
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
        force_stoploss: float = 0,
    ) -> List[ExitCheckTuple]:
        exits: List[ExitCheckTuple] = []
        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)
        current_profit_best = current_profit
        if low is not None or high is not None:
            current_rate_best = (low if trade.is_short else high) or rate
            current_profit_best = trade.calc_profit_ratio(current_rate_best)

        trade.adjust_min_max_rates(high or current_rate, low or current_rate)

        stoplossflag = self.ft_stoploss_reached(
            current_rate=current_rate,
            trade=trade,
            current_time=current_time,
            current_profit=current_profit,
            force_stoploss=force_stoploss,
            low=low,
            high=high,
        )

        roi_reached = not (enter and self.ignore_roi_if_entry_signal) and self.min_roi_reached(
            trade=trade, current_profit=current_profit_best, current_time=current_time
        )

        exit_signal = ExitType.NONE
        custom_reason = ""

        if self.use_exit_signal:
            if exit_ and not enter:
                exit_signal = ExitType.EXIT_SIGNAL
            else:
                reason_cust = strategy_safe_wrapper(self.custom_exit, default_retval=False)(
                    pair=trade.pair,
                    trade=trade,
                    current_time=current_time,
                    current_rate=current_rate,
                    current_profit=current_profit,
                )
                if reason_cust:
                    exit_signal = ExitType.CUSTOM_EXIT
                    if isinstance(reason_cust, str):
                        custom_reason = reason_cust
                        if len(reason_cust) > CUSTOM_TAG_MAX_LENGTH:
                            logger.warning(
                                f"Custom exit reason returned from "
                                f"custom_exit is too long and was trimmed"
                                f"to {CUSTOM_TAG_MAX_LENGTH} characters."
                            )
                            custom_reason = reason_cust[:CUSTOM_TAG_MAX_LENGTH]
                    else:
                        custom_reason = ""
            if exit_signal == ExitType.CUSTOM_EXIT or (
                exit_signal == ExitType.EXIT_SIGNAL
                and (not self.exit_profit_only or current_profit > self.exit_profit_offset)
            ):
                logger.debug(
                    f"{trade.pair} - Sell signal received. "
                    f"exit_type=ExitType.{exit_signal.name}"
                    + (f", custom_reason={custom_reason}" if custom_reason else "")
                )
                exits.append(ExitCheckTuple(exit_type=exit_signal, exit_reason=custom_reason))

        if stoplossflag.exit_type in (ExitType.STOP_LOSS, ExitType.LIQUIDATION):
            logger.debug(f"{trade.pair} - Stoploss hit. exit_type={stoplossflag.exit_type}")
            exits.append(stoplossflag)

        if roi_reached:
            logger.debug(f"{trade.pair} - Required profit reached. exit_type=ExitType.ROI")
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
        low: Optional[float] = None,
        high: Optional[float] = None,
        after_fill: bool = False,
    ) -> None:
        if after_fill and not self._ft_stop_uses_after_fill:
            return

        stop_loss_value = force_stoploss if force_stoploss else self.stoploss

        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        dir_correct = (
            trade.stop_loss < (low or current_rate)
            if not trade.is_short
            else trade.stop_loss > (high or current_rate)
        )

        bound = low if trade.is_short else high
        bound_profit = current_profit if not bound else trade.calc_profit_ratio(bound)
        if self.use_custom_stoploss and dir_correct:
            stop_loss_value_custom = strategy_safe_wrapper(
                self.custom_stoploss, default_retval=None, supress_error=True
            )(
                pair=trade.pair,
                trade=trade,
                current_time=current_time,
                current_rate=(bound or current_rate),
                current_profit=bound_profit,
                after_fill=after_fill,
            )
            if stop_loss_value_custom and not (
                isnan(stop_loss_value_custom) or isinf(stop_loss_value_custom)
            ):
                stop_loss_value = stop_loss_value_custom
                trade.adjust_stop_loss(
                    bound or current_rate, stop_loss_value, allow_refresh=after_fill
                )
            else:
                logger.debug("CustomStoploss function did not return valid stoploss")

        if self.trailing_stop and dir_correct:
            sl_offset = self.trailing_stop_positive_offset

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
        low: Optional[float] = None,
        high: Optional[float] = None,
    ) -> ExitCheckTuple:
        self.ft_stoploss_adjust(
            current_rate, trade, current_time, current_profit, force_stoploss, low, high
        )

        sl_higher_long = trade.stop_loss >= (low or current_rate) and not trade.is_short
        sl_lower_short = trade.stop_loss <= (high or current_rate) and trade.is_short
        liq_higher_long = (
            trade.liquidation_price
            and trade.liquidation_price >= (low or current_rate)
            and not trade.is_short
        )
        liq_lower_short = (
            trade.liquidation_price
            and trade.liquidation_price <= (high or current_rate)
            and trade.is_short
        )

        if (sl_higher_long or sl_lower_short) and (
            not self.order_types.get("stoploss_on_exchange") or self.config["dry_run"]
        ):
            exit_type = ExitType.STOP_LOSS

            if trade.is_stop_loss_trailing:
                exit_type = ExitType.TRAILING_STOP_LOSS
                logger.debug(
                    f"{trade.pair} - HIT STOP: current price at "
                    f"{((high if trade.is_short else low) or current_rate):.6f}, "
                    f"stoploss is {trade.stop_loss:.6f}, "
                    f"initial stoploss was at {trade.initial_stop_loss:.6f}, "
                    f"trade opened at {trade.open_rate:.6f}"
                )

            return ExitCheckTuple(exit_type=exit_type)

        if liq_higher_long or liq_lower_short:
            logger.debug(f"{trade.pair} - Liquidation price hit. exit_type=ExitType.LIQUIDATION")
            return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)

        return ExitCheckTuple(exit_type=ExitType.NONE)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        roi_list = [x for x in self.minimal_roi.keys() if x <= trade_dur]
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        return roi_entry, self.minimal_roi[roi_entry]

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ft_check_timed_out(self, trade: Trade, order: Order, current_time: datetime) -> bool:
        side = "entry" if order.ft_order_side == trade.entry_side else "exit"

        timeout = self.config.get("unfilledtimeout", {}).get(side)
        if timeout is not None:
            timeout_unit = self.config.get("unfilledtimeout", {}).get("unit", "minutes")
            timeout_kwargs = {timeout_unit: -timeout}
            timeout_threshold = current_time + timedelta(**timeout_kwargs)
            timedout = order.status == "open" and order.order_date_utc < timeout_threshold
            if timedout:
                return True
        time_method = (
            self.check_exit_timeout
            if order.ft_order_side == trade.exit_side
            else self.check_entry_timeout
        )

        return strategy_safe_wrapper(time_method, default_retval=False)(
            pair=trade.pair, trade=trade, order=order, current_time=current_time
        )

    def advise_all_indicators(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return {
            pair: self.advise_indicators(pair_data.copy(), {"pair": pair}).copy()
            for pair, pair_data in data.items()
        }

    def ft_advise_signals(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.adv