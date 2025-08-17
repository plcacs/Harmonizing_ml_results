from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, time, timedelta, timezone
from math import isclose
from threading import Lock
from time import sleep
from copy import deepcopy
import logging
import traceback

from schedule import Scheduler

from freqtrade import constants
from freqtrade.configuration import validate_config_consistency
from freqtrade.constants import BuySell, Config, EntryExecuteMode, ExchangeConfig, LongShort
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.edge import Edge
from freqtrade.enums import (
    ExitCheckTuple,
    ExitType,
    MarginMode,
    RPCMessageType,
    SignalDirection,
    State,
    TradingMode,
)
from freqtrade.exceptions import (
    DependencyException,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    PricingError,
)
from freqtrade.exchange import (
    ROUND_DOWN,
    ROUND_UP,
    remove_exchange_credentials,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange_types import CcxtOrder
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.misc import safe_value_fallback, safe_value_fallback2
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import Order, PairLocks, Trade, init_db
from freqtrade.persistence.key_value_store import set_startup_time
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.rpc import RPCManager
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from freqtrade.rpc.rpc_types import (
    ProfitLossStr,
    RPCCancelMsg,
    RPCEntryMsg,
    RPCExitCancelMsg,
    RPCExitMsg,
    RPCProtectionMsg,
)
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import FtPrecise, MeasureTime, dt_from_ts
from freqtrade.util.migrations.binance_mig import migrate_binance_futures_names
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


class FreqtradeBot(LoggingMixin):
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Config) -> None:
        """
        Init all variables and objects the bot needs to work
        :param config: configuration dict, you can use Configuration.get_config()
        to get the config dict.
        """
        self.active_pair_whitelist: List[str] = []

        # Init bot state
        self.state = State.STOPPED

        # Init objects
        self.config = config
        exchange_config: ExchangeConfig = deepcopy(config["exchange"])
        # Remove credentials from original exchange config to avoid accidental credential exposure
        remove_exchange_credentials(config["exchange"], True)

        self.strategy: IStrategy = StrategyResolver.load_strategy(self.config)

        # Check config consistency here since strategies can set certain options
        validate_config_consistency(config)

        self.exchange = ExchangeResolver.load_exchange(
            self.config, exchange_config=exchange_config, load_leverage_tiers=True
        )

        init_db(self.config["db_url"])

        self.wallets = Wallets(self.config, self.exchange)

        PairLocks.timeframe = self.config["timeframe"]

        self.trading_mode: TradingMode = self.config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = self.config.get("margin_mode", MarginMode.NONE)
        self.last_process: Optional[datetime] = None

        # RPC runs in separate threads, can start handling external commands just after
        # initialization, even before Freqtradebot has a chance to start its throttling,
        # so anything in the Freqtradebot instance should be ready (initialized), including
        # the initial state of the bot.
        # Keep this at the end of this initialization method.
        self.rpc: RPCManager = RPCManager(self)

        self.dataprovider = DataProvider(self.config, self.exchange, rpc=self.rpc)
        self.pairlists = PairListManager(self.exchange, self.config, self.dataprovider)

        self.dataprovider.add_pairlisthandler(self.pairlists)

        # Attach Dataprovider to strategy instance
        self.strategy.dp = self.dataprovider
        # Attach Wallets to strategy instance
        self.strategy.wallets = self.wallets

        # Initializing Edge only if enabled
        self.edge = (
            Edge(self.config, self.exchange, self.strategy)
            if self.config.get("edge", {}).get("enabled", False)
            else None
        )

        # Init ExternalMessageConsumer if enabled
        self.emc = (
            ExternalMessageConsumer(self.config, self.dataprovider)
            if self.config.get("external_message_consumer", {}).get("enabled", False)
            else None
        )

        self.active_pair_whitelist = self._refresh_active_whitelist()

        # Set initial bot state from config
        initial_state = self.config.get("initial_state")
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED

        # Protect exit-logic from forcesell and vice versa
        self._exit_lock = Lock()
        timeframe_secs = timeframe_to_seconds(self.strategy.timeframe)
        LoggingMixin.__init__(self, logger, timeframe_secs)

        self._schedule = Scheduler()

        if self.trading_mode == TradingMode.FUTURES:

            def update():
                self.update_funding_fees()
                self.update_all_liquidation_prices()
                self.wallets.update()

            # This would be more efficient if scheduled in utc time, and performed at each
            # funding interval, specified by funding_fee_times on the exchange classes
            # However, this reduces the precision - and might therefore lead to problems.
            for time_slot in range(0, 24):
                for minutes in [1, 31]:
                    t = str(time(time_slot, minutes, 2))
                    self._schedule.every().day.at(t).do(update)

        self._schedule.every().day.at("00:02").do(self.exchange.ws_connection_reset)

        self.strategy.ft_bot_start()
        # Initialize protections AFTER bot start - otherwise parameters are not loaded.
        self.protections = ProtectionManager(self.config, self.strategy.protections)

        def log_took_too_long(duration: float, time_limit: float):
            logger.warning(
                f"Strategy analysis took {duration:.2f}s, more than 25% of the timeframe "
                f"({time_limit:.2f}s). This can lead to delayed orders and missed signals."
                "Consider either reducing the amount of work your strategy performs "
                "or reduce the amount of pairs in the Pairlist."
            )

        self._measure_execution = MeasureTime(log_took_too_long, timeframe_secs * 0.25)

    def notify_status(self, msg: str, msg_type: RPCMessageType = RPCMessageType.STATUS) -> None:
        """
        Public method for users of this class (worker, etc.) to send notifications
        via RPC about changes in the bot status.
        """
        self.rpc.send_msg({"type": msg_type, "status": msg})

    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info("Cleaning up modules ...")
        try:
            # Wrap db activities in shutdown to avoid problems if database is gone,
            # and raises further exceptions.
            if self.config["cancel_open_orders_on_exit"]:
                self.cancel_all_open_orders()

            self.check_for_open_trades()
        except Exception as e:
            logger.warning(f"Exception during cleanup: {e.__class__.__name__} {e}")

        finally:
            self.strategy.ft_bot_cleanup()

        self.rpc.cleanup()
        if self.emc:
            self.emc.shutdown()
        self.exchange.close()
        try:
            Trade.commit()
        except Exception:
            # Exceptions here will be happening if the db disappeared.
            # At which point we can no longer commit anyway.
            logger.exception("Error during cleanup")

    def startup(self) -> None:
        """
        Called on startup and after reloading the bot - triggers notifications and
        performs startup tasks
        """
        migrate_binance_futures_names(self.config)
        set_startup_time()

        self.rpc.startup_messages(self.config, self.pairlists, self.protections)
        # Update older trades with precision and precision mode
        self.startup_backpopulate_precision()
        if not self.edge:
            # Adjust stoploss if it was changed
            Trade.stoploss_reinitialization(self.strategy.stoploss)

        # Only update open orders on startup
        # This will update the database after the initial migration
        self.startup_update_open_orders()
        self.update_all_liquidation_prices()
        self.update_funding_fees()

    def process(self) -> None:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :return: True if one or more trades has been created or closed, False otherwise
        """

        # Check whether markets have to be reloaded and reload them when it's needed
        self.exchange.reload_markets()

        self.update_trades_without_assigned_fees()

        # Query trades from persistence layer
        trades: List[Trade] = Trade.get_open_trades()

        self.active_pair_whitelist = self._refresh_active_whitelist(trades)

        # Refreshing candles
        self.dataprovider.refresh(
            self.pairlists.create_pair_list(self.active_pair_whitelist),
            self.strategy.gather_informative_pairs(),
        )

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(
            current_time=datetime.now(timezone.utc)
        )

        with self._measure_execution:
            self.strategy.analyze(self.active_pair_whitelist)

        with self._exit_lock:
            # Check for exchange cancellations, timeouts and user requested replace
            self.manage_open_orders()

        # Protect from collisions with force_exit.
        # Without this, freqtrade may try to recreate stoploss_on_exchange orders
        # while exiting is in process, since telegram messages arrive in an different thread.
        with self._exit_lock:
            trades = Trade.get_open_trades()
            # First process current opened trades (positions)
            self.exit_positions(trades)

        # Check if we need to adjust our current positions before attempting to enter new trades.
        if self.strategy.position_adjustment_enable:
            with self._exit_lock:
                self.process_open_trade_positions()

        # Then looking for entry opportunities
        if self.get_free_open_trades():
            self.enter_positions()
        self._schedule.run_pending()
        Trade.commit()
        self.rpc.process_msg_queue(self.dataprovider._msg_queue)
        self.last_process = datetime.now(timezone.utc)

    def process_stopped(self) -> None:
        """
        Close all orders that were left open
        """
        if self.config["cancel_open_orders_on_exit"]:
            self.cancel_all_open_orders()

    def check_for_open_trades(self) -> None:
        """
        Notify the user when the bot is stopped (not reloaded)
        and there are still open trades active.
        """
        open_trades = Trade.get_open_trades()

        if len(open_trades) != 0 and self.state != State.RELOAD_CONFIG:
            msg = {
                "type": RPCMessageType.WARNING,
                "status": f"{len(open_trades)} open trades active.\n\n"
                f"Handle these trades manually on {self.exchange.name}, "
                f"or '/start' the bot again and use '/stopentry' "
                f"to handle open trades gracefully. \n"
                f"{'Note: Trades are simulated (dry run).' if self.config['dry_run'] else ''}",
            }
            self.rpc.send_msg(msg)

    def _refresh_active_whitelist(self, trades: Optional[List[Trade]] = None) -> List[str]:
        """
        Refresh active whitelist from pairlist or edge and extend it with
        pairs that have open trades.
        """
        # Refresh whitelist
        _prev_whitelist = self.pairlists.whitelist
        self.pairlists.refresh_pairlist()
        _whitelist = self.pairlists.whitelist

        # Calculating Edge positioning
        if self.edge:
            self.edge.calculate(_whitelist)
            _whitelist = self.edge.adjust(_whitelist)

        if trades:
            # Extend active-pair whitelist with pairs of open trades
            # It ensures that candle (OHLCV) data are downloaded for open trades as well
            _whitelist.extend([trade.pair for trade in trades if trade.pair not in _whitelist])

        # Called last to include the included pairs
        if _prev_whitelist != _whitelist:
            self.rpc.send_msg({"type": RPCMessageType.WHITELIST, "data": _whitelist})

        return _whitelist

    def get_free_open_trades(self) -> int:
        """
        Return the number of free open trades slots or 0 if
        max number of open trades reached
        """
        open_trades = Trade.get_open_trade_count()
        return max(0, self.config["max_open_trades"] - open_trades)

    def update_all_liquidation_prices(self) -> None:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.CROSS:
            # Update liquidation prices for all trades in cross margin mode
            update_liquidation_prices(
                exchange=self.exchange,
                wallets=self.wallets,
                stake_currency=self.config["stake_currency"],
                dry_run=self.config["dry_run"],
            )

    def update_funding_fees(self) -> None:
        if self.trading_mode == TradingMode.FUTURES:
            trades: List[Trade] = Trade.get_open_trades()
            for trade in trades:
                trade.set_funding_fees(
                    self.exchange.get_funding_fees(
                        pair=trade.pair,
                        amount=trade.amount,
                        is_short=trade.is_short,
                        open_date=trade.date_last_filled_utc,
                    )
                )

    def startup_backpopulate_precision(self) -> None:
        trades = Trade.get_trades([Trade.contract_size.is_(None)])
        for trade in trades:
            if trade.exchange != self.exchange.id:
                continue
            trade.precision_mode = self.exchange.precisionMode
            trade.precision_mode_price = self.exchange.precision_mode_price
            trade.amount_precision = self.exchange.get_precision_amount(trade.pair)
            trade.price_precision = self.exchange.get_precision_price(trade.pair)
            trade.contract_size = self.exchange.get_contract_size(trade.pair)
        Trade.commit()

    def startup_update_open_orders(self) -> None:
        """
        Updates open orders based on order list kept in the database.
        Mainly updates the state of orders - but may also close trades
        """
        if self.config["dry_run"] or self.config["exchange"].get("skip_open_order_update", False):
            # Updating open orders in dry-run does not make sense and will fail.
            return

        orders = Order.get_open_orders()
        logger.info(f"Updating {len(orders)} open orders.")
        for order in orders:
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(
                    order.order_id, order.ft_pair, order.ft_order_side == "stoploss"
                )
                if not order.trade:
                    # This should not happen, but it does if trades were deleted manually.
                    # This can only incur on sqlite, which doesn't enforce foreign constraints.
                    logger.warning(
                        f"Order {order.order_id} has no trade attached. "
                        "This may suggest a database corruption. "
                        f"The expected trade ID is {order.ft_trade_id}. Ignoring this order."
                    )
                    continue
                self.update_trade_state(
                    order.trade,
                    order.order_id,
                    fo,
                    stoploss_order=(order.ft_order_side == "stoploss"),
                )

            except InvalidOrderException as e:
                logger.warning(f"Error updating Order {order.order_id} due to {e}.")
                if order.order_date_utc - timedelta(days=5) < datetime.now(timezone.utc):
                    logger.warning(
                        "Order is older than 5 days. Assuming order was fully cancelled."
                    )
                    fo = order.to_ccxt_object()
                    fo["status"] = "canceled"
                    self.handle_cancel_order(
                        fo, order, order.trade, constants.CANCEL_REASON["TIMEOUT"]
                    )

            except ExchangeError as e:
                logger.warning(f"Error updating Order {order.order_id} due to {e}")

    def update_trades_without_assigned_fees(self) -> None:
        """
        Update closed trades without close fees assigned.
        Only acts when Orders are in the database, otherwise the last order-id is unknown.
        """
        if self.config["dry_run"]:
            # Updating open orders in dry-run does not make sense and will fail.
            return

        trades: List[Trade] = Trade.get_closed_trades_without_assigned_fees()
        for trade in trades:
            if not trade.is_open and not trade.fee_updated(trade.exit_side):
                # Get sell fee
                order = trade.select_order(trade.exit_side, False, only_filled=True)
                if not order:
                    order = trade.select_order("stoploss", False)
                if order:
