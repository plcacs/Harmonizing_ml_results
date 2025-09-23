"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""
import logging
import traceback
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from math import isclose
from threading import Lock
from time import sleep
from typing import Any, Optional, List, Dict, Tuple, Union
from schedule import Scheduler
from freqtrade import constants
from freqtrade.configuration import validate_config_consistency
from freqtrade.constants import BuySell, Config, EntryExecuteMode, ExchangeConfig, LongShort
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.edge import Edge
from freqtrade.enums import ExitCheckTuple, ExitType, MarginMode, RPCMessageType, SignalDirection, State, TradingMode
from freqtrade.exceptions import DependencyException, ExchangeError, InsufficientFundsError, InvalidOrderException, PricingError
from freqtrade.exchange import ROUND_DOWN, ROUND_UP, remove_exchange_credentials, timeframe_to_minutes, timeframe_to_next_date, timeframe_to_seconds
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
from freqtrade.rpc.rpc_types import ProfitLossStr, RPCCancelMsg, RPCEntryMsg, RPCExitCancelMsg, RPCExitMsg, RPCProtectionMsg
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
        self.state: State = State.STOPPED
        self.config: Config = config
        exchange_config: ExchangeConfig = deepcopy(config['exchange'])
        remove_exchange_credentials(config['exchange'], True)
        self.strategy: IStrategy = StrategyResolver.load_strategy(self.config)
        validate_config_consistency(config)
        self.exchange = ExchangeResolver.load_exchange(self.config, exchange_config=exchange_config, load_leverage_tiers=True)
        init_db(self.config['db_url'])
        self.wallets: Wallets = Wallets(self.config, self.exchange)
        PairLocks.timeframe = self.config['timeframe']
        self.trading_mode: TradingMode = self.config.get('trading_mode', TradingMode.SPOT)
        self.margin_mode: MarginMode = self.config.get('margin_mode', MarginMode.NONE)
        self.last_process: Optional[datetime] = None
        self.rpc: RPCManager = RPCManager(self)
        self.dataprovider: DataProvider = DataProvider(self.config, self.exchange, rpc=self.rpc)
        self.pairlists: PairListManager = PairListManager(self.exchange, self.config, self.dataprovider)
        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.strategy.dp = self.dataprovider
        self.strategy.wallets = self.wallets
        self.edge: Optional[Edge] = Edge(self.config, self.exchange, self.strategy) if self.config.get('edge', {}).get('enabled', False) else None
        self.emc: Optional[ExternalMessageConsumer] = ExternalMessageConsumer(self.config, self.dataprovider) if self.config.get('external_message_consumer', {}).get('enabled', False) else None
        self.active_pair_whitelist = self._refresh_active_whitelist()
        initial_state = self.config.get('initial_state')
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED
        self._exit_lock: Lock = Lock()
        timeframe_secs: int = timeframe_to_seconds(self.strategy.timeframe)
        LoggingMixin.__init__(self, logger, timeframe_secs)
        self._schedule: Scheduler = Scheduler()
        if self.trading_mode == TradingMode.FUTURES:

            def update() -> None:
                self.update_funding_fees()
                self.update_all_liquidation_prices()
                self.wallets.update()
            for time_slot in range(0, 24):
                for minutes in [1, 31]:
                    t = str(time(time_slot, minutes, 2))
                    self._schedule.every().day.at(t).do(update)
        self._schedule.every().day.at('00:02').do(self.exchange.ws_connection_reset)
        self.strategy.ft_bot_start()
        self.protections: ProtectionManager = ProtectionManager(self.config, self.strategy.protections)

        def log_took_too_long(duration: float, time_limit: float) -> None:
            logger.warning(f'Strategy analysis took {duration:.2f}s, more than 25% of the timeframe ({time_limit:.2f}s). This can lead to delayed orders and missed signals.Consider either reducing the amount of work your strategy performs or reduce the amount of pairs in the Pairlist.')
        self._measure_execution: MeasureTime = MeasureTime(log_took_too_long, timeframe_secs * 0.25)

    def notify_status(self, msg: str, msg_type: RPCMessageType = RPCMessageType.STATUS) -> None:
        """
        Public method for users of this class (worker, etc.) to send notifications
        via RPC about changes in the bot status.
        """
        self.rpc.send_msg({'type': msg_type, 'status': msg})

    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')
        try:
            if self.config['cancel_open_orders_on_exit']:
                self.cancel_all_open_orders()
            self.check_for_open_trades()
        except Exception as e:
            logger.warning(f'Exception during cleanup: {e.__class__.__name__} {e}')
        finally:
            self.strategy.ft_bot_cleanup()
        self.rpc.cleanup()
        if self.emc:
            self.emc.shutdown()
        self.exchange.close()
        try:
            Trade.commit()
        except Exception:
            logger.exception('Error during cleanup')

    def startup(self) -> None:
        """
        Called on startup and after reloading the bot - triggers notifications and
        performs startup tasks
        """
        migrate_binance_futures_names(self.config)
        set_startup_time()
        self.rpc.startup_messages(self.config, self.pairlists, self.protections)
        self.startup_backpopulate_precision()
        if not self.edge:
            Trade.stoploss_reinitialization(self.strategy.stoploss)
        self.startup_update_open_orders()
        self.update_all_liquidation_prices()
        self.update_funding_fees()

    def process(self) -> bool:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :return: True if one or more trades has been created or closed, False otherwise
        """
        self.exchange.reload_markets()
        self.update_trades_without_assigned_fees()
        trades: List[Trade] = Trade.get_open_trades()
        self.active_pair_whitelist = self._refresh_active_whitelist(trades)
        self.dataprovider.refresh(self.pairlists.create_pair_list(self.active_pair_whitelist), self.strategy.gather_informative_pairs())
        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(current_time=datetime.now(timezone.utc))
        with self._measure_execution:
            self.strategy.analyze(self.active_pair_whitelist)
        with self._exit_lock:
            self.manage_open_orders()
        with self._exit_lock:
            trades = Trade.get_open_trades()
            self.exit_positions(trades)
        if self.strategy.position_adjustment_enable:
            with self._exit_lock:
                self.process_open_trade_positions()
        if self.get_free_open_trades():
            self.enter_positions()
        self._schedule.run_pending()
        Trade.commit()
        self.rpc.process_msg_queue(self.dataprovider._msg_queue)
        self.last_process = datetime.now(timezone.utc)
        return True

    def process_stopped(self) -> None:
        """
        Close all orders that were left open
        """
        if self.config['cancel_open_orders_on_exit']:
            self.cancel_all_open_orders()

    def check_for_open_trades(self) -> None:
        """
        Notify the user when the bot is stopped (not reloaded)
        and there are still open trades active.
        """
        open_trades = Trade.get_open_trades()
        if len(open_trades) != 0 and self.state != State.RELOAD_CONFIG:
            msg = {'type': RPCMessageType.WARNING, 'status': f"{len(open_trades)} open trades active.\n\nHandle these trades manually on {self.exchange.name}, or '/start' the bot again and use '/stopentry' to handle open trades gracefully. \n{('Note: Trades are simulated (dry run).' if self.config['dry_run'] else '')}"}
            self.rpc.send_msg(msg)

    def _refresh_active_whitelist(self, trades: Optional[List[Trade]] = None) -> List[str]:
        """
        Refresh active whitelist from pairlist or edge and extend it with
        pairs that have open trades.
        """
        _prev_whitelist = self.pairlists.whitelist
        self.pairlists.refresh_pairlist()
        _whitelist = self.pairlists.whitelist
        if self.edge:
            self.edge.calculate(_whitelist)
            _whitelist = self.edge.adjust(_whitelist)
        if trades:
            _whitelist.extend([trade.pair for trade in trades if trade.pair not in _whitelist])
        if _prev_whitelist != _whitelist:
            self.rpc.send_msg({'type': RPCMessageType.WHITELIST, 'data': _whitelist})
        return _whitelist

    def get_free_open_trades(self) -> int:
        """
        Return the number of free open trades slots or 0 if
        max number of open trades reached
        """
        open_trades = Trade.get_open_trade_count()
        return max(0, self.config['max_open_trades'] - open_trades)

    def update_all_liquidation_prices(self) -> None:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.CROSS:
            update_liquidation_prices(exchange=self.exchange, wallets=self.wallets, stake_currency=self.config['stake_currency'], dry_run=self.config['dry_run'])

    def update_funding_fees(self) -> None:
        if self.trading_mode == TradingMode.FUTURES:
            trades: List[Trade] = Trade.get_open_trades()
            for trade in trades:
                trade.set_funding_fees(self.exchange.get_funding_fees(pair=trade.pair, amount=trade.amount, is_short=trade.is_short, open_date=trade.date_last_filled_utc))

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
        if self.config['dry_run'] or self.config['exchange'].get('skip_open_order_update', False):
            return
        orders = Order.get_open_orders()
        logger.info(f'Updating {len(orders)} open orders.')
        for order in orders:
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair, order.ft_order_side == 'stoploss')
                if not order.trade:
                    logger.warning(f'Order {order.order_id} has no trade attached. This may suggest a database corruption. The expected trade ID is {order.ft_trade_id}. Ignoring this order.')
                    continue
                self.update_trade_state(order.trade, order.order_id, fo, stoploss_order=order.ft_order_side == 'stoploss')
            except InvalidOrderException as e:
                logger.warning(f'Error updating Order {order.order_id} due to {e}.')
                if order.order_date_utc - timedelta(days=5) < datetime.now(timezone.utc):
                    logger.warning('Order is older than 5 days. Assuming order was fully cancelled.')
                    fo = order.to_ccxt_object()
                    fo['status'] = 'canceled'
                    self.handle_cancel_order(fo, order, order.trade, constants.CANCEL_REASON['TIMEOUT'])
            except ExchangeError as e:
                logger.warning(f'Error updating Order {order.order_id} due to {e}')

    def update_trades_without_assigned_fees(self) -> None:
        """
        Update closed trades without close fees assigned.
        Only acts when Orders are in the database, otherwise the last order-id is unknown.
        """
        if self.config['dry_run']:
            return
        trades: List[Trade] = Trade.get_closed_trades_without_assigned_fees()
        for trade in trades:
            if not trade.is_open and (not trade.fee_updated(trade.exit_side)):
                order = trade.select_order(trade.exit_side, False, only_filled=True)
                if not order:
                    order = trade.select_order('stoploss', False)
                if order:
                    logger.info(f'Updating {trade.exit_side}-fee on trade {trade}for order {order.order_id}.')
                    self.update_trade_state(trade, order.order_id, stoploss_order=order.ft_order_side == 'stoploss', send_msg=False)
        trades = Trade.get_open_trades_without_assigned_fees()
        for trade in trades:
            with self._exit_lock:
                if trade.is_open and (not trade.fee_updated(trade.entry_side)):
                    order = trade.select_order(trade.entry_side, False, only_filled=True)
                    open_order = trade.select_order(trade.entry_side, True)
                    if order and open_order is None:
                        logger.info(f'Updating {trade.entry_side}-fee on trade {trade}for order {order.order_id}.')
                        self.update_trade_state(trade, order.order_id, send_msg=False)

    def handle_insufficient_funds(self, trade: Trade) -> None:
        """
        Try refinding a lost trade.
        Only used when InsufficientFunds appears on exit orders (stoploss or long sell/short buy).
        Tries to walk the stored orders and updates the trade state if necessary.
        """
        logger.info(f'Trying to refind lost order for {trade}')
        for order in trade.orders:
            logger.info(f'Trying to refind {order}')
            fo = None
            if not order.ft_is_open:
                logger.debug(f'Order {order} is no longer open.')
                continue
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair, order.ft_order_side == 'stoploss')
                if fo:
                    logger.info(f'Found {order} for trade {trade}.')
                    self.update_trade_state(trade, order.order_id, fo, stoploss_order=order.ft_order_side == 'stoploss')
            except ExchangeError:
                logger.warning(f'Error updating {order.order_id}.')

    def handle_onexchange_order(self, trade: Trade) -> bool:
        """
        Try refinding a order that is not in the database.
        Only used balance disappeared, which would make exiting impossible.
        :return: True if the trade was deleted, False otherwise
        """
        try:
            orders = self.exchange.fetch_orders(trade.pair, trade.open_date_utc - timedelta(seconds=10))
            prev_exit_reason = trade.exit_reason
            prev_trade_state = trade.is_open
            prev_trade_amount = trade.amount
            for order in orders:
                trade_order = [o for o in trade.orders if o.order_id == order['id']]
                if trade_order:
                    order_obj = trade_order[0]
                else:
                    logger.info(f"Found previously unknown order {order['id']} for {trade.pair}.")
                    order_obj = Order.parse_from_ccxt_object(order, trade.pair, order['side'])
                    order_obj.order_filled_date = dt_from_ts(safe_value_fallback(order, 'lastTradeTimestamp', 'timestamp'))
                    trade.orders.append