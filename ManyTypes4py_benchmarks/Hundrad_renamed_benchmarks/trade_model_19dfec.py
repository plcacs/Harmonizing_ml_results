"""
This module contains the class to persist trades into SQLite
"""
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose
from typing import Any, ClassVar, Optional, cast
from sqlalchemy import Enum, Float, ForeignKey, Integer, ScalarResult, Select, String, UniqueConstraint, case, desc, func, select
from sqlalchemy.orm import Mapped, lazyload, mapped_column, relationship, validates
from typing_extensions import Self
from freqtrade.constants import CANCELED_EXCHANGE_STATES, CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT, MATH_CLOSE_PREC, NON_OPEN_EXCHANGE_STATES, BuySell, LongShort
from freqtrade.enums import ExitType, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import ROUND_DOWN, ROUND_UP, amount_to_contract_precision, price_to_precision
from freqtrade.exchange.exchange_types import CcxtOrder
from freqtrade.leverage import interest
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.persistence.custom_data import CustomDataWrapper, _CustomData
from freqtrade.util import FtPrecise, dt_from_ts, dt_now, dt_ts, dt_ts_none
logger = logging.getLogger(__name__)


@dataclass
class ProfitStruct:
    pass


class Order(ModelBase):
    """
    Order database model
    Keeps a record of all orders placed on the exchange

    One to many relationship with Trades:
      - One trade can have many orders
      - One Order can only be associated with one Trade

    Mirrors CCXT Order structure
    """
    __tablename__ = 'orders'
    __allow_unmapped__ = True
    __table_args__ = UniqueConstraint('ft_pair', 'order_id', name=
        '_order_pair_order_id'),
    id = mapped_column(Integer, primary_key=True)
    ft_trade_id = mapped_column(Integer, ForeignKey('trades.id'), index=True)
    _trade_live = relationship('Trade', back_populates='orders', lazy=
        'immediate')
    _trade_bt = None
    ft_order_side = mapped_column(String(25), nullable=False)
    ft_pair = mapped_column(String(25), nullable=False)
    ft_is_open = mapped_column(nullable=False, default=True, index=True)
    ft_amount = mapped_column(Float(), nullable=False)
    ft_price = mapped_column(Float(), nullable=False)
    ft_cancel_reason = mapped_column(String(CUSTOM_TAG_MAX_LENGTH),
        nullable=True)
    order_id = mapped_column(String(255), nullable=False, index=True)
    status = mapped_column(String(255), nullable=True)
    symbol = mapped_column(String(25), nullable=True)
    order_type = mapped_column(String(50), nullable=True)
    side = mapped_column(String(25), nullable=True)
    price = mapped_column(Float(), nullable=True)
    average = mapped_column(Float(), nullable=True)
    amount = mapped_column(Float(), nullable=True)
    filled = mapped_column(Float(), nullable=True)
    remaining = mapped_column(Float(), nullable=True)
    cost = mapped_column(Float(), nullable=True)
    stop_price = mapped_column(Float(), nullable=True)
    order_date = mapped_column(nullable=True, default=dt_now)
    order_filled_date = mapped_column(nullable=True)
    order_update_date = mapped_column(nullable=True)
    funding_fee = mapped_column(Float(), nullable=True)
    ft_fee_base = mapped_column(Float(), nullable=True)
    ft_order_tag = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)

    @property
    def func_ph1g1z9k(self):
        """Order-date with UTC timezoneinfo"""
        return self.order_date.replace(tzinfo=timezone.utc)

    @property
    def func_py63qbhv(self):
        """last order-date with UTC timezoneinfo"""
        return self.order_filled_date.replace(tzinfo=timezone.utc
            ) if self.order_filled_date else None

    @property
    def func_saejuyae(self):
        return self.amount or self.ft_amount

    @property
    def func_83r41mn9(self):
        """Price at which the order was placed"""
        return self.price or self.stop_price or self.ft_price

    @property
    def func_5kxwhoiq(self):
        return self.average or self.price or self.stop_price or self.ft_price

    @property
    def func_vcjvak6y(self):
        return self.filled if self.filled is not None else 0.0

    @property
    def func_qlrq1myw(self):
        return self.cost or 0.0

    @property
    def func_5opci4hn(self):
        return (self.remaining if self.remaining is not None else self.
            safe_amount - (self.filled or 0.0))

    @property
    def func_3ujdo76q(self):
        return self.ft_fee_base or 0.0

    @property
    def func_ipy8aimf(self):
        return self.safe_filled - self.safe_fee_base

    @property
    def func_0yut655f(self):
        return self._trade_bt or self._trade_live

    @property
    def func_czh0rg28(self):
        """Amount in stake currency used for this order"""
        return float(FtPrecise(self.safe_amount) * FtPrecise(self.
            safe_price) / FtPrecise(self.trade.leverage))

    @property
    def func_ogu3o02s(self):
        """Filled Amount in stake currency used for this order"""
        return float(FtPrecise(self.safe_filled) * FtPrecise(self.
            safe_price) / FtPrecise(self.trade.leverage))

    def __repr__(self):
        return (
            f'Order(id={self.id}, trade={self.ft_trade_id}, order_id={self.order_id}, side={self.side}, filled={self.safe_filled}, price={self.safe_price}, amount={self.amount}, status={self.status}, date={self.order_date_utc:{DATETIME_PRINT_FORMAT}})'
            )

    def func_b2bjn7vt(self, order):
        """
        Update Order from ccxt response
        Only updates if fields are available from ccxt -
        """
        if self.order_id != str(order['id']):
            raise DependencyException("Order-id's don't match")
        self.status = safe_value_fallback(order, 'status', default_value=
            self.status)
        self.symbol = safe_value_fallback(order, 'symbol', default_value=
            self.symbol)
        self.order_type = safe_value_fallback(order, 'type', default_value=
            self.order_type)
        self.side = safe_value_fallback(order, 'side', default_value=self.side)
        self.price = safe_value_fallback(order, 'price', default_value=self
            .price)
        self.amount = safe_value_fallback(order, 'amount', default_value=
            self.amount)
        self.filled = safe_value_fallback(order, 'filled', default_value=
            self.filled)
        self.average = safe_value_fallback(order, 'average', default_value=
            self.average)
        self.remaining = safe_value_fallback(order, 'remaining',
            default_value=self.remaining)
        self.cost = safe_value_fallback(order, 'cost', default_value=self.cost)
        self.stop_price = safe_value_fallback(order, 'stopPrice',
            default_value=self.stop_price)
        order_date = safe_value_fallback(order, 'timestamp')
        if order_date:
            self.order_date = dt_from_ts(order_date)
        elif not self.order_date:
            self.order_date = dt_now()
        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get('filled', 0.0) or 0.0
                ) > 0 and not self.order_filled_date:
                self.order_filled_date = dt_from_ts(safe_value_fallback(
                    order, 'lastTradeTimestamp', default_value=dt_ts()))
        self.order_update_date = datetime.now(timezone.utc)

    def func_0lkg933u(self, stopPriceName='stopPrice'):
        order = {'id': self.order_id, 'symbol': self.ft_pair, 'price': self
            .price, 'average': self.average, 'amount': self.amount, 'cost':
            self.cost, 'type': self.order_type, 'side': self.ft_order_side,
            'filled': self.filled, 'remaining': self.remaining, 'datetime':
            self.order_date_utc.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'timestamp': int(self.order_date_utc.timestamp() * 1000),
            'status': self.status, 'fee': None, 'info': {}}
        if self.ft_order_side == 'stoploss':
            order.update({stopPriceName: self.stop_price, 'ft_order_type':
                'stoploss'})
        return order

    def func_fu7eccei(self, entry_side, minified=False):
        """
        :param minified: If True, only return a subset of the data is returned.
                         Only used for backtesting.
        """
        resp = {'amount': self.safe_amount, 'safe_price': self.safe_price,
            'ft_order_side': self.ft_order_side, 'order_filled_timestamp':
            dt_ts_none(self.order_filled_utc), 'ft_is_entry': self.
            ft_order_side == entry_side, 'ft_order_tag': self.ft_order_tag,
            'cost': self.cost if self.cost else 0}
        if not minified:
            resp.update({'pair': self.ft_pair, 'order_id': self.order_id,
                'status': self.status, 'average': round(self.average, 8) if
                self.average else 0, 'filled': self.filled, 'is_open': self
                .ft_is_open, 'order_date': self.order_date.strftime(
                DATETIME_PRINT_FORMAT) if self.order_date else None,
                'order_timestamp': int(self.order_date.replace(tzinfo=
                timezone.utc).timestamp() * 1000) if self.order_date else
                None, 'order_filled_date': self.order_filled_date.strftime(
                DATETIME_PRINT_FORMAT) if self.order_filled_date else None,
                'order_type': self.order_type, 'price': self.price,
                'remaining': self.remaining, 'ft_fee_base': self.
                ft_fee_base, 'funding_fee': self.funding_fee})
        return resp

    def func_cmcrlm6f(self, close_date, trade):
        self.order_filled_date = close_date
        self.filled = self.amount
        self.remaining = 0
        self.status = 'closed'
        self.ft_is_open = False
        self.funding_fee = trade.funding_fee_running
        trade.funding_fee_running = 0.0
        if self.ft_order_side == trade.entry_side and self.price:
            trade.open_rate = self.price
            func_0yut655f.recalc_trade_from_orders()
            if trade.nr_of_successful_entries == 1:
                trade.initial_stop_loss_pct = None
                trade.is_stop_loss_trailing = False
            func_0yut655f.adjust_stop_loss(trade.open_rate, trade.stop_loss_pct
                )

    @staticmethod
    def func_met6mbvm(orders, order):
        """
        Get all non-closed orders - useful when trying to batch-update orders
        """
        if not isinstance(order, dict):
            logger.warning(f'{order} is not a valid response object.')
            return
        filtered_orders = [o for o in orders if o.order_id == order.get('id')]
        if filtered_orders:
            oobj = filtered_orders[0]
            oobj.update_from_ccxt_object(order)
            Trade.commit()
        else:
            logger.warning(f'Did not find order for {order}.')

    @classmethod
    def func_irsibpf1(cls, order, pair, side, amount=None, price=None):
        """
        Parse an order from a ccxt object and return a new order Object.
        Optional support for overriding amount and price is only used for test simplification.
        """
        o = cls(order_id=str(order['id']), ft_order_side=side, ft_pair=pair,
            ft_amount=amount or order.get('amount', None) or 0.0, ft_price=
            price or order.get('price', None))
        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def func_3v4jn7m4():
        """
        Retrieve open orders from the database
        :return: List of open orders
        """
        return Order.session.scalars(select(Order).filter(Order.ft_is_open.
            is_(True))).all()

    @staticmethod
    def func_55d96i6y(order_id):
        """
        Retrieve order based on order_id
        :return: Order or None
        """
        return Order.session.scalars(select(Order).filter(Order.order_id ==
            order_id)).first()


class LocalTrade:
    """
    Trade database model.
    Used in backtesting - must be aligned to Trade model!
    """
    use_db = False
    bt_trades = []
    bt_trades_open = []
    bt_trades_open_pp = defaultdict(list)
    bt_open_open_trade_count = 0
    bt_total_profit = 0
    realized_profit = 0
    id = 0
    orders = []
    exchange = ''
    pair = ''
    base_currency = ''
    stake_currency = ''
    is_open = True
    fee_open = 0.0
    fee_open_cost = None
    fee_open_currency = ''
    fee_close = 0.0
    fee_close_cost = None
    fee_close_currency = ''
    open_rate = 0.0
    open_rate_requested = None
    open_trade_value = 0.0
    close_rate = None
    close_rate_requested = None
    close_profit = None
    close_profit_abs = None
    stake_amount = 0.0
    max_stake_amount = 0.0
    amount = 0.0
    amount_requested = None
    close_date = None
    stop_loss = 0.0
    stop_loss_pct = 0.0
    initial_stop_loss = 0.0
    initial_stop_loss_pct = None
    is_stop_loss_trailing = False
    max_rate = None
    min_rate = None
    exit_reason = ''
    exit_order_status = ''
    strategy = ''
    enter_tag = None
    timeframe = None
    trading_mode = TradingMode.SPOT
    amount_precision = None
    price_precision = None
    precision_mode = None
    precision_mode_price = None
    contract_size = None
    liquidation_price = None
    is_short = False
    leverage = 1.0
    interest_rate = 0.0
    funding_fees = None
    funding_fee_running = None

    @property
    def func_a2r58kgv(self):
        if self.liquidation_price:
            if self.is_short:
                return min(self.stop_loss, self.liquidation_price)
            else:
                return max(self.stop_loss, self.liquidation_price)
        return self.stop_loss

    @property
    def func_dknukcxb(self):
        """
        Compatibility between buy_tag (old) and enter_tag (new)
        Consider buy_tag deprecated
        """
        return self.enter_tag

    @property
    def func_p75ma5bj(self):
        """Returns true if this is a non-leverage, non-short trade"""
        return (self.leverage == 1.0 or self.leverage is None
            ) and not self.is_short

    @property
    def func_9fjk7adr(self):
        """
        The amount of currency borrowed from the exchange for leverage trades
        If a long trade, the amount is in base currency
        If a short trade, the amount is in the other currency being traded
        """
        if self.has_no_leverage:
            return 0.0
        elif not self.is_short:
            return self.amount * self.open_rate * ((self.leverage - 1) /
                self.leverage)
        else:
            return self.amount

    @property
    def func_8k0ytshm(self):
        """Date of the last filled order"""
        orders = self.select_filled_orders()
        if orders:
            return max(o.order_filled_utc for o in orders if o.order_filled_utc
                )
        return None

    @property
    def func_cyn69dxj(self):
        """Date of the last filled order - or open_date if no orders are filled"""
        dt_last_filled = self._date_last_filled_utc
        if not dt_last_filled:
            return self.open_date_utc
        return max([self.open_date_utc, dt_last_filled])

    @property
    def func_ay251u2o(self):
        """Date of the first filled order"""
        orders = self.select_filled_orders(self.entry_side)
        if orders and len((filled_date := [o.order_filled_utc for o in
            orders if o.order_filled_utc])):
            return min(filled_date)
        return None

    @property
    def func_q0vpwwm1(self):
        return self.open_date.replace(tzinfo=timezone.utc)

    @property
    def func_vmhlj2k0(self):
        if self.has_open_sl_orders:
            return max(o.order_date_utc for o in self.open_sl_orders)
        return None

    @property
    def func_83wrfbyo(self):
        return self.close_date.replace(tzinfo=timezone.utc
            ) if self.close_date else None

    @property
    def func_rf4a0qnh(self):
        if self.is_short:
            return 'sell'
        else:
            return 'buy'

    @property
    def func_w17l4m4g(self):
        if self.is_short:
            return 'buy'
        else:
            return 'sell'

    @property
    def func_gpudcu7f(self):
        if self.is_short:
            return 'short'
        else:
            return 'long'

    @property
    def func_44r19kb8(self):
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.base_currency or self.pair.split('/')[0]
        except IndexError:
            return ''

    @property
    def func_7et8wdf7(self):
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.stake_currency or self.pair.split('/')[1].split(':')[0]
        except IndexError:
            return ''

    @property
    def func_03ooz2nt(self):
        """
        All open orders for this trade excluding stoploss orders
        """
        return [o for o in self.orders if o.ft_is_open and o.ft_order_side !=
            'stoploss']

    @property
    def func_wukljspp(self):
        """
        True if there are open orders for this trade excluding stoploss orders
        """
        open_orders_wo_sl = [o for o in self.orders if o.ft_order_side not in
            ['stoploss'] and o.ft_is_open]
        return len(open_orders_wo_sl) > 0

    @property
    def func_1yjspd0b(self):
        """
        True if there is an open position for this trade
        """
        return self.amount > 0

    @property
    def func_qj0cf0hc(self):
        """
        All open stoploss orders for this trade
        """
        return [o for o in self.orders if o.ft_order_side in ['stoploss'] and
            o.ft_is_open]

    @property
    def func_5j7qg4ue(self):
        """
        True if there are open stoploss orders for this trade
        """
        open_sl_orders = [o for o in self.orders if o.ft_order_side in [
            'stoploss'] and o.ft_is_open]
        return len(open_sl_orders) > 0

    @property
    def func_xokhq7lp(self):
        """
        All stoploss orders for this trade
        """
        return [o for o in self.orders if o.ft_order_side in ['stoploss']]

    @property
    def func_1uo9xp8s(self):
        open_orders_ids_wo_sl = [oo.order_id for oo in self.open_orders if 
            oo.ft_order_side not in ['stoploss']]
        return open_orders_ids_wo_sl

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.recalc_open_trade_value()
        self.orders = []
        if (self.trading_mode == TradingMode.MARGIN and self.interest_rate is
            None):
            raise OperationalException(
                f'{self.trading_mode} trading requires param interest_rate on trades'
                )

    def __repr__(self):
        open_since = self.open_date_utc.strftime(DATETIME_PRINT_FORMAT
            ) if self.is_open else 'closed'
        return (
            f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, is_short={self.is_short or False}, leverage={self.leverage or 1.0}, open_rate={self.open_rate:.8f}, open_since={open_since})'
            )

    def func_fu7eccei(self, minified=False):
        """
        :param minified: If True, only return a subset of the data is returned.
                         Only used for backtesting.
        :return: Dictionary with trade data
        """
        filled_or_open_orders = self.select_filled_or_open_orders()
        orders_json = [order.to_json(self.entry_side, minified) for order in
            filled_or_open_orders]
        return {'trade_id': self.id, 'pair': self.pair, 'base_currency':
            self.safe_base_currency, 'quote_currency': self.
            safe_quote_currency, 'is_open': self.is_open, 'exchange': self.
            exchange, 'amount': round(self.amount, 8), 'amount_requested': 
            round(self.amount_requested, 8) if self.amount_requested else
            None, 'stake_amount': round(self.stake_amount, 8),
            'max_stake_amount': round(self.max_stake_amount, 8) if self.
            max_stake_amount else None, 'strategy': self.strategy,
            'enter_tag': self.enter_tag, 'timeframe': self.timeframe,
            'fee_open': self.fee_open, 'fee_open_cost': self.fee_open_cost,
            'fee_open_currency': self.fee_open_currency, 'fee_close': self.
            fee_close, 'fee_close_cost': self.fee_close_cost,
            'fee_close_currency': self.fee_close_currency, 'open_date':
            self.open_date.strftime(DATETIME_PRINT_FORMAT),
            'open_timestamp': dt_ts_none(self.open_date_utc),
            'open_fill_date': self.date_entry_fill_utc.strftime(
            DATETIME_PRINT_FORMAT) if self.date_entry_fill_utc else None,
            'open_fill_timestamp': dt_ts_none(self.date_entry_fill_utc),
            'open_rate': self.open_rate, 'open_rate_requested': self.
            open_rate_requested, 'open_trade_value': round(self.
            open_trade_value, 8), 'close_date': self.close_date.strftime(
            DATETIME_PRINT_FORMAT) if self.close_date else None,
            'close_timestamp': dt_ts_none(self.close_date_utc),
            'realized_profit': self.realized_profit or 0.0,
            'realized_profit_ratio': self.close_profit or None,
            'close_rate': self.close_rate, 'close_rate_requested': self.
            close_rate_requested, 'close_profit': self.close_profit,
            'close_profit_pct': round(self.close_profit * 100, 2) if self.
            close_profit else None, 'close_profit_abs': self.
            close_profit_abs, 'trade_duration_s': int((self.close_date_utc -
            self.open_date_utc).total_seconds()) if self.close_date else
            None, 'trade_duration': int((self.close_date_utc - self.
            open_date_utc).total_seconds() // 60) if self.close_date else
            None, 'profit_ratio': self.close_profit, 'profit_pct': round(
            self.close_profit * 100, 2) if self.close_profit else None,
            'profit_abs': self.close_profit_abs, 'exit_reason': self.
            exit_reason, 'exit_order_status': self.exit_order_status,
            'stop_loss_abs': self.stop_loss, 'stop_loss_ratio': self.
            stop_loss_pct if self.stop_loss_pct else None, 'stop_loss_pct':
            self.stop_loss_pct * 100 if self.stop_loss_pct else None,
            'stoploss_last_update': self.stoploss_last_update_utc.strftime(
            DATETIME_PRINT_FORMAT) if self.stoploss_last_update_utc else
            None, 'stoploss_last_update_timestamp': dt_ts_none(self.
            stoploss_last_update_utc), 'initial_stop_loss_abs': self.
            initial_stop_loss, 'initial_stop_loss_ratio': self.
            initial_stop_loss_pct if self.initial_stop_loss_pct else None,
            'initial_stop_loss_pct': self.initial_stop_loss_pct * 100 if
            self.initial_stop_loss_pct else None, 'min_rate': self.min_rate,
            'max_rate': self.max_rate, 'leverage': self.leverage,
            'interest_rate': self.interest_rate, 'liquidation_price': self.
            liquidation_price, 'is_short': self.is_short, 'trading_mode':
            self.trading_mode, 'funding_fees': self.funding_fees,
            'amount_precision': self.amount_precision, 'price_precision':
            self.price_precision, 'precision_mode': self.precision_mode,
            'precision_mode_price': self.precision_mode_price,
            'contract_size': self.contract_size, 'has_open_orders': self.
            has_open_orders, 'orders': orders_json}

    @staticmethod
    def func_07aelsay():
        """
        Resets all trades. Only active for backtesting mode.
        """
        LocalTrade.bt_trades = []
        LocalTrade.bt_trades_open = []
        LocalTrade.bt_trades_open_pp = defaultdict(list)
        LocalTrade.bt_open_open_trade_count = 0
        LocalTrade.bt_total_profit = 0

    def func_hmiobm6m(self, current_price, current_price_low):
        """
        Adjust the max_rate and min_rate.
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price_low, self.min_rate or self.open_rate)

    def func_mh1lzhdr(self, liquidation_price):
        """
        Method you should use to set self.liquidation price.
        Assures stop_loss is not passed the liquidation price
        """
        if liquidation_price is None:
            return
        self.liquidation_price = price_to_precision(liquidation_price, self
            .price_precision, self.precision_mode_price)

    def func_vybtqbsn(self, funding_fee):
        """
        Assign funding fees to Trade.
        """
        if funding_fee is None:
            return
        self.funding_fee_running = funding_fee
        prior_funding_fees = sum([o.funding_fee for o in self.orders if o.
            funding_fee])
        self.funding_fees = prior_funding_fees + funding_fee

    def __set_stop_loss(self, stop_loss, percent):
        """
        Method used internally to set self.stop_loss.
        """
        if not self.stop_loss:
            self.initial_stop_loss = stop_loss
        self.stop_loss = stop_loss
        self.stop_loss_pct = -1 * abs(percent)

    def func_ke2ws2dd(self, current_price, stoploss, initial=False,
        allow_refresh=False):
        """
        This adjusts the stop loss to it's most recently observed setting
        :param current_price: Current rate the asset is traded
        :param stoploss: Stoploss as factor (sample -0.05 -> -5% below current price).
        :param initial: Called to initiate stop_loss.
            Skips everything if self.stop_loss is already set.
        :param refresh: Called to refresh stop_loss, allows adjustment in both directions
        """
        if stoploss is None or initial and not (self.stop_loss is None or 
            self.stop_loss == 0):
            return
        leverage = self.leverage or 1.0
        if self.is_short:
            new_loss = float(current_price * (1 + abs(stoploss / leverage)))
        else:
            new_loss = float(current_price * (1 - abs(stoploss / leverage)))
        stop_loss_norm = price_to_precision(new_loss, self.price_precision,
            self.precision_mode_price, rounding_mode=ROUND_DOWN if self.
            is_short else ROUND_UP)
        if self.initial_stop_loss_pct is None:
            self.__set_stop_loss(stop_loss_norm, stoploss)
            self.initial_stop_loss = price_to_precision(stop_loss_norm,
                self.price_precision, self.precision_mode_price,
                rounding_mode=ROUND_DOWN if self.is_short else ROUND_UP)
            self.initial_stop_loss_pct = -1 * abs(stoploss)
        else:
            higher_stop = stop_loss_norm > self.stop_loss
            lower_stop = stop_loss_norm < self.stop_loss
            if (allow_refresh or higher_stop and not self.is_short or 
                lower_stop and self.is_short):
                logger.debug(f'{self.pair} - Adjusting stoploss...')
                if not allow_refresh:
                    self.is_stop_loss_trailing = True
                self.__set_stop_loss(stop_loss_norm, stoploss)
            else:
                logger.debug(f'{self.pair} - Keeping current stoploss...')
        logger.debug(
            f'{self.pair} - Stoploss adjusted. current_price={current_price:.8f}, open_rate={self.open_rate:.8f}, max_rate={self.max_rate or self.open_rate:.8f}, initial_stop_loss={self.initial_stop_loss:.8f}, stop_loss={self.stop_loss:.8f}. Trailing stoploss saved us: {float(self.stop_loss) - float(self.initial_stop_loss or 0.0):.8f}.'
            )

    def func_75ulpu7a(self, order, recalculating=False):
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.fetch_order()
        :return: None
        """
        if order.status == 'open' or order.safe_price is None:
            return
        logger.info(f'Updating trade (id={self.id}) ...')
        if order.ft_order_side != 'stoploss':
            order.funding_fee = self.funding_fee_running
            self.funding_fee_running = 0.0
        order_type = order.order_type.upper() if order.order_type else None
        if order.ft_order_side == self.entry_side:
            self.open_rate = order.safe_price
            self.amount = order.safe_amount_after_fee
            if self.is_open:
                payment = 'SELL' if self.is_short else 'BUY'
                logger.info(
                    f'{order_type}_{payment} has been fulfilled for {self}.')
            self.recalc_trade_from_orders()
        elif order.ft_order_side == self.exit_side:
            if self.is_open:
                payment = 'BUY' if self.is_short else 'SELL'
                logger.info(
                    f'{order_type}_{payment} has been fulfilled for {self}.')
        elif order.ft_order_side == 'stoploss' and order.status not in ('open',
            ):
            self.close_rate_requested = self.stop_loss
            self.exit_reason = ExitType.STOPLOSS_ON_EXCHANGE.value
            if self.is_open and order.safe_filled > 0:
                logger.info(f'{order_type} is hit for {self}.')
        else:
            raise ValueError(f'Unknown order type: {order.order_type}')
        if order.ft_order_side != self.entry_side:
            amount_tr = amount_to_contract_precision(self.amount, self.
                amount_precision, self.precision_mode, self.contract_size)
            if isclose(order.safe_amount_after_fee, amount_tr, abs_tol=
                MATH_CLOSE_PREC
                ) or not recalculating and order.safe_amount_after_fee > amount_tr:
                self.close(order.safe_price)
            else:
                self.recalc_trade_from_orders()
        Trade.commit()

    def func_xeizj1l2(self, rate, *, show_msg=True):
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = rate
        self.close_date = (self.close_date or self._date_last_filled_utc or
            dt_now())
        self.is_open = False
        self.exit_order_status = 'closed'
        self.recalc_trade_from_orders(is_closing=True)
        if show_msg:
            logger.info(
                f'Marking {self} as closed as the trade is fulfilled and found no open orders for it.'
                )

    def func_lwswy6bn(self, fee_cost, fee_currency, fee_rate, side):
        """
        Update Fee parameters. Only acts once per side
        """
        if self.entry_side == side and self.fee_open_currency is None:
            self.fee_open_cost = fee_cost
            self.fee_open_currency = fee_currency
            if fee_rate is not None:
                self.fee_open = fee_rate
                self.fee_close = fee_rate
        elif self.exit_side == side and self.fee_close_currency is None:
            self.fee_close_cost = fee_cost
            self.fee_close_currency = fee_currency
            if fee_rate is not None:
                self.fee_close = fee_rate

    def func_v9huziqi(self, side):
        """
        Verify if this side (buy / sell) has already been updated
        """
        if self.entry_side == side:
            return self.fee_open_currency is not None
        elif self.exit_side == side:
            return self.fee_close_currency is not None
        else:
            return False

    def func_c8a0ueb6(self, order):
        Order.update_orders(self.orders, order)

    @property
    def func_gcezj2kr(self):
        """
        Get amount of failed exiting orders
        assumes full exits.
        """
        return len([o for o in self.orders if o.ft_order_side == self.
            entry_side and o.status in CANCELED_EXCHANGE_STATES and o.
            filled == 0])

    @property
    def func_eor3u930(self):
        """
        Get amount of failed exiting orders
        assumes full exits.
        """
        return len([o for o in self.orders if o.ft_order_side == self.
            exit_side and o.status in CANCELED_EXCHANGE_STATES])

    def func_t7tvehi1(self):
        """
        Get amount of failed exiting orders
        assumes full exits.
        """
        return self.canceled_exit_order_count

    def func_ng8j2sv8(self, amount, open_rate):
        """
        Calculate the open_rate including open_fee.
        :return: Price in of the open trade incl. Fees
        """
        open_trade = FtPrecise(amount) * FtPrecise(open_rate)
        fees = open_trade * FtPrecise(self.fee_open)
        if self.is_short:
            return float(open_trade - fees)
        else:
            return float(open_trade + fees)

    def func_hq4qr05i(self):
        """
        Recalculate open_trade_value.
        Must be called whenever open_rate, fee_open is changed.
        """
        self.open_trade_value = self._calc_open_trade_value(self.amount,
            self.open_rate)

    def func_aq1wr05g(self):
        """
        Calculate interest for this trade. Only applicable for Margin trading.
        """
        zero = FtPrecise(0.0)
        if self.trading_mode != TradingMode.MARGIN or self.has_no_leverage:
            return zero
        open_date = self.open_date.replace(tzinfo=None)
        now = (self.close_date or datetime.now(timezone.utc)).replace(tzinfo
            =None)
        sec_per_hour = FtPrecise(3600)
        total_seconds = FtPrecise((now - open_date).total_seconds())
        hours = total_seconds / sec_per_hour or zero
        rate = FtPrecise(self.interest_rate)
        borrowed = FtPrecise(self.borrowed)
        return interest(exchange_name=self.exchange, borrowed=borrowed,
            rate=rate, hours=hours)

    def func_un7qhpis(self, amount, rate, fee):
        close_trade = amount * FtPrecise(rate)
        fees = close_trade * FtPrecise(fee or 0.0)
        if self.is_short:
            return close_trade + fees
        else:
            return close_trade - fees

    def func_cvlnlifd(self, rate, amount=None):
        """
        Calculate the Trade's close value including fees
        :param rate: rate to compare with.
        :return: value in stake currency of the open trade
        """
        if rate is None and not self.close_rate:
            return 0.0
        amount1 = FtPrecise(amount or self.amount)
        trading_mode = self.trading_mode or TradingMode.SPOT
        if trading_mode == TradingMode.SPOT:
            return float(self._calc_base_close(amount1, rate, self.fee_close))
        elif trading_mode == TradingMode.MARGIN:
            total_interest = self.calculate_interest()
            if self.is_short:
                amount1 = amount1 + total_interest
                return float(self._calc_base_close(amount1, rate, self.
                    fee_close))
            else:
                return float(self._calc_base_close(amount1, rate, self.
                    fee_close) - total_interest)
        elif trading_mode == TradingMode.FUTURES:
            funding_fees = self.funding_fees or 0.0
            if self.is_short:
                return float(self._calc_base_close(amount1, rate, self.
                    fee_close)) - funding_fees
            else:
                return float(self._calc_base_close(amount1, rate, self.
                    fee_close)) + funding_fees
        else:
            raise OperationalException(
                f'{self.trading_mode} trading is not yet available using freqtrade'
                )

    def func_u6e2opfv(self, rate, amount=None, open_rate=None):
        """
        Calculate the absolute profit in stake currency between Close and Open trade
        Deprecated - only available for backwards compatibility
        :param rate: close rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: profit in stake currency as float
        """
        prof = self.calculate_profit(rate, amount, open_rate)
        return prof.profit_abs

    def func_qa6tb8jm(self, rate, amount=None, open_rate=None):
        """
        Calculate profit metrics (absolute, ratio, total, total ratio).
        All calculations include fees.
        :param rate: close rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: Profit structure, containing absolute and relative profits.
        """
        close_trade_value = self.calc_close_trade_value(rate, amount)
        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)
        if self.is_short:
            profit_abs = open_trade_value - close_trade_value
        else:
            profit_abs = close_trade_value - open_trade_value
        try:
            if self.is_short:
                profit_ratio = (1 - close_trade_value / open_trade_value
                    ) * self.leverage
            else:
                profit_ratio = (close_trade_value / open_trade_value - 1
                    ) * self.leverage
            profit_ratio = float(f'{profit_ratio:.8f}')
        except ZeroDivisionError:
            profit_ratio = 0.0
        total_profit_abs = profit_abs + self.realized_profit
        total_profit_ratio = (total_profit_abs / self.max_stake_amount *
            self.leverage if self.max_stake_amount else 0.0)
        total_profit_ratio = float(f'{total_profit_ratio:.8f}')
        profit_abs = float(f'{profit_abs:.8f}')
        return ProfitStruct(profit_abs=profit_abs, profit_ratio=
            profit_ratio, total_profit=profit_abs + self.realized_profit,
            total_profit_ratio=total_profit_ratio)

    def func_rdye57oe(self, rate, amount=None, open_rate=None):
        """
        Calculates the profit as ratio (including fee).
        :param rate: rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: profit ratio as float
        """
        close_trade_value = self.calc_close_trade_value(rate, amount)
        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)
        if open_trade_value == 0.0:
            return 0.0
        elif self.is_short:
            profit_ratio = (1 - close_trade_value / open_trade_value
                ) * self.leverage
        else:
            profit_ratio = (close_trade_value / open_trade_value - 1
                ) * self.leverage
        return float(f'{profit_ratio:.8f}')

    def func_o8t4luci(self, *, is_closing=False):
        ZERO = FtPrecise(0.0)
        current_amount = FtPrecise(0.0)
        current_stake = FtPrecise(0.0)
        max_stake_amount = FtPrecise(0.0)
        total_stake = 0.0
        avg_price = FtPrecise(0.0)
        close_profit = 0.0
        close_profit_abs = 0.0
        self.funding_fees = 0.0
        funding_fees = 0.0
        ordercount = len(self.orders) - 1
        for i, o in enumerate(self.orders):
            if o.ft_is_open or not o.filled:
                continue
            funding_fees += o.funding_fee or 0.0
            tmp_amount = FtPrecise(o.safe_amount_after_fee)
            tmp_price = FtPrecise(o.safe_price)
            is_exit = o.ft_order_side != self.entry_side
            side = FtPrecise(-1 if is_exit else 1)
            if tmp_amount > ZERO and tmp_price is not None:
                current_amount += tmp_amount * side
                price = avg_price if is_exit else tmp_price
                current_stake += price * tmp_amount * side
                if current_amount > ZERO and not is_exit:
                    avg_price = current_stake / current_amount
            if is_exit:
                if i == ordercount and is_closing:
                    self.funding_fees = funding_fees
                exit_rate = o.safe_price
                exit_amount = o.safe_amount_after_fee
                prof = self.calculate_profit(exit_rate, exit_amount, float(
                    avg_price))
                close_profit_abs += prof.profit_abs
                if total_stake > 0:
                    close_profit = (close_profit_abs / total_stake * self.
                        leverage)
            else:
                total_stake += self._calc_open_trade_value(tmp_amount, price)
                max_stake_amount += tmp_amount * price
        self.funding_fees = funding_fees
        self.max_stake_amount = float(max_stake_amount)
        if close_profit:
            self.close_profit = close_profit
            self.realized_profit = close_profit_abs
            self.close_profit_abs = prof.profit_abs
        current_amount_tr = amount_to_contract_precision(float(
            current_amount), self.amount_precision, self.precision_mode,
            self.contract_size)
        if current_amount_tr > 0.0:
            self.open_rate = price_to_precision(float(current_stake /
                current_amount), self.price_precision, self.
                precision_mode_price)
            self.amount = current_amount_tr
            self.stake_amount = float(current_stake) / (self.leverage or 1.0)
            self.fee_open_cost = self.fee_open * float(self.max_stake_amount)
            self.recalc_open_trade_value()
            if self.stop_loss_pct is not None and self.open_rate is not None:
                self.adjust_stop_loss(self.open_rate, self.stop_loss_pct)
        elif is_closing and total_stake > 0:
            self.close_profit = close_profit_abs / total_stake * self.leverage
            self.close_profit_abs = close_profit_abs

    def func_02hzx7wa(self, order_id):
        """
        Finds order object by Order id.
        :param order_id: Exchange order id
        """
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def func_lte5guot(self, order_side=None, is_open=None, only_filled=False):
        """
        Finds latest order for this orderside and status
        :param order_side: ft_order_side of the order (either 'buy', 'sell' or 'stoploss')
        :param is_open: Only search for open orders?
        :param only_filled: Only search for Filled orders (only valid with is_open=False).
        :return: latest Order object if it exists, else None
        """
        orders = self.orders
        if order_side:
            orders = [o for o in orders if o.ft_order_side == order_side]
        if is_open is not None:
            orders = [o for o in orders if o.ft_is_open == is_open]
        if is_open is False and only_filled:
            orders = [o for o in orders if o.filled and o.status in
                NON_OPEN_EXCHANGE_STATES]
        if len(orders) > 0:
            return orders[-1]
        else:
            return None

    def func_75u89juw(self, order_side=None):
        """
        Finds filled orders for this order side.
        Will not return open orders which already partially filled.
        :param order_side: Side of the order (either 'buy', 'sell', or None)
        :return: array of Order objects
        """
        return [o for o in self.orders if (o.ft_order_side == order_side or
            order_side is None) and o.ft_is_open is False and o.filled and 
            o.status in NON_OPEN_EXCHANGE_STATES]

    def func_t0jssapy(self):
        """
        Finds filled or open orders
        :param order_side: Side of the order (either 'buy', 'sell', or None)
        :return: array of Order objects
        """
        return [o for o in self.orders if o.ft_is_open is False and (o.
            filled or 0) > 0 and o.status in NON_OPEN_EXCHANGE_STATES or o.
            ft_is_open is True and o.status is not None]

    def func_g9w7etxo(self, key, value):
        """
        Set custom data for this trade
        :param key: key of the custom data
        :param value: value of the custom data (must be JSON serializable)
        """
        CustomDataWrapper.set_custom_data(trade_id=self.id, key=key, value=
            value)

    def func_f8alj9hl(self, key, default=None):
        """
        Get custom data for this trade
        :param key: key of the custom data
        """
        data = CustomDataWrapper.get_custom_data(trade_id=self.id, key=key)
        if data:
            return data[0].value
        return default

    def func_kcby3wfq(self, key):
        """
        Get custom data for this trade
        :param key: key of the custom data
        """
        data = CustomDataWrapper.get_custom_data(trade_id=self.id, key=key)
        if data:
            return data[0]
        return None

    def func_y58b04be(self):
        """
        Get all custom data for this trade
        """
        return CustomDataWrapper.get_custom_data(trade_id=self.id)

    @property
    def func_gl7exjuw(self):
        """
        Helper function to count the number of entry orders that have been filled.
        :return: int count of entry orders that have been filled for this trade.
        """
        return len(self.select_filled_orders(self.entry_side))

    @property
    def func_1e73uzxg(self):
        """
        Helper function to count the number of exit orders that have been filled.
        :return: int count of exit orders that have been filled for this trade.
        """
        return len(self.select_filled_orders(self.exit_side))

    @property
    def func_7gtblj0e(self):
        """
        Helper function to count the number of buy orders that have been filled.
        WARNING: Please use nr_of_successful_entries for short support.
        :return: int count of buy orders that have been filled for this trade.
        """
        return len(self.select_filled_orders('buy'))

    @property
    def func_cfj92scj(self):
        """
        Helper function to count the number of sell orders that have been filled.
        WARNING: Please use nr_of_successful_exits for short support.
        :return: int count of sell orders that have been filled for this trade.
        """
        return len(self.select_filled_orders('sell'))

    @property
    def func_i95z4838(self):
        """DEPRECATED! Please use exit_reason instead."""
        return self.exit_reason

    @property
    def func_aw6gyw3v(self):
        return self.close_rate or self.close_rate_requested or 0.0

    @staticmethod
    def func_t8pk2h2i(*, pair=None, is_open=None, open_date=None,
        close_date=None):
        """
        Helper function to query Trades.
        Returns a List of trades, filtered on the parameters given.
        In live mode, converts the filter to a database query and returns all rows
        In Backtest mode, uses filters on Trade.bt_trades to get the result.

        :param pair: Filter by pair
        :param is_open: Filter by open/closed status
        :param open_date: Filter by open_date (filters via trade.open_date > input)
        :param close_date: Filter by close_date (filters via trade.close_date > input)
                           Will implicitly only return closed trades.
        :return: unsorted List[Trade]
        """
        if is_open is not None:
            if is_open:
                sel_trades = LocalTrade.bt_trades_open
            else:
                sel_trades = LocalTrade.bt_trades
        else:
            sel_trades = list(LocalTrade.bt_trades + LocalTrade.bt_trades_open)
        if pair:
            sel_trades = [trade for trade in sel_trades if trade.pair == pair]
        if open_date:
            sel_trades = [trade for trade in sel_trades if trade.open_date >
                open_date]
        if close_date:
            sel_trades = [trade for trade in sel_trades if trade.close_date and
                trade.close_date > close_date]
        return sel_trades

    @staticmethod
    def func_nhq82lbi(trade):
        LocalTrade.bt_trades_open.remove(trade)
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1
        LocalTrade.bt_trades.append(trade)
        LocalTrade.bt_total_profit += trade.close_profit_abs

    @staticmethod
    def func_c1qqm0f4(trade):
        if trade.is_open:
            LocalTrade.bt_trades_open.append(trade)
            LocalTrade.bt_trades_open_pp[trade.pair].append(trade)
            LocalTrade.bt_open_open_trade_count += 1
        else:
            LocalTrade.bt_trades.append(trade)

    @staticmethod
    def func_4z73hznw(trade):
        LocalTrade.bt_trades_open.remove(trade)
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1

    @staticmethod
    def func_joafq9zo():
        """
        Retrieve open trades
        """
        return Trade.get_trades_proxy(is_open=True)

    @staticmethod
    def func_tk8fdilm():
        """
        get open trade count
        """
        if Trade.use_db:
            return Trade.session.execute(select(func.count(Trade.id)).
                filter(Trade.is_open.is_(True))).scalar_one()
        else:
            return LocalTrade.bt_open_open_trade_count

    @staticmethod
    def func_0nw07ax1(desired_stoploss):
        """
        Adjust initial Stoploss to desired stoploss for all open trades.
        """
        for trade in Trade.get_open_trades():
            logger.info(f'Found open trade: {trade}')
            if (not trade.is_stop_loss_trailing and trade.
                initial_stop_loss_pct != desired_stoploss):
                logger.info(f'Stoploss for {trade} needs adjustment...')
                trade.stop_loss = 0.0
                trade.initial_stop_loss_pct = None
                func_0yut655f.adjust_stop_loss(trade.open_rate,
                    desired_stoploss)
                logger.info(f'New stoploss: {trade.stop_loss}.')

    @classmethod
    def func_lbtascky(cls, json_str):
        """
        Create a Trade instance from a json string.

        Used for debugging purposes - please keep.
        :param json_str: json string to parse
        :return: Trade instance
        """
        import rapidjson
        data = rapidjson.loads(json_str)
        trade = cls(__FROM_JSON=True, id=data['trade_id'], pair=data['pair'
            ], base_currency=data['base_currency'], stake_currency=data[
            'quote_currency'], is_open=data['is_open'], exchange=data[
            'exchange'], amount=data['amount'], amount_requested=data[
            'amount_requested'], stake_amount=data['stake_amount'],
            strategy=data['strategy'], enter_tag=data['enter_tag'],
            timeframe=data['timeframe'], fee_open=data['fee_open'],
            fee_open_cost=data['fee_open_cost'], fee_open_currency=data[
            'fee_open_currency'], fee_close=data['fee_close'],
            fee_close_cost=data['fee_close_cost'], fee_close_currency=data[
            'fee_close_currency'], open_date=datetime.fromtimestamp(data[
            'open_timestamp'] // 1000, tz=timezone.utc), open_rate=data[
            'open_rate'], open_rate_requested=data['open_rate_requested'],
            open_trade_value=data['open_trade_value'], close_date=datetime.
            fromtimestamp(data['close_timestamp'] // 1000, tz=timezone.utc) if
            data['close_timestamp'] else None, realized_profit=data[
            'realized_profit'], close_rate=data['close_rate'],
            close_rate_requested=data['close_rate_requested'], close_profit
            =data['close_profit'], close_profit_abs=data['close_profit_abs'
            ], exit_reason=data['exit_reason'], exit_order_status=data[
            'exit_order_status'], stop_loss=data['stop_loss_abs'],
            stop_loss_pct=data['stop_loss_ratio'], initial_stop_loss=data[
            'initial_stop_loss_abs'], initial_stop_loss_pct=data[
            'initial_stop_loss_ratio'], min_rate=data['min_rate'], max_rate
            =data['max_rate'], leverage=data['leverage'], interest_rate=
            data['interest_rate'], liquidation_price=data[
            'liquidation_price'], is_short=data['is_short'], trading_mode=
            data['trading_mode'], funding_fees=data['funding_fees'],
            amount_precision=data.get('amount_precision', None),
            price_precision=data.get('price_precision', None),
            precision_mode=data.get('precision_mode', None),
            precision_mode_price=data.get('precision_mode_price', data.get(
            'precision_mode', None)), contract_size=data.get(
            'contract_size', None))
        for order in data['orders']:
            order_obj = Order(amount=order['amount'], ft_amount=order[
                'amount'], ft_order_side=order['ft_order_side'], ft_pair=
                order['pair'], ft_is_open=order['is_open'], order_id=order[
                'order_id'], status=order['status'], average=order[
                'average'], cost=order['cost'], filled=order['filled'],
                order_date=datetime.strptime(order['order_date'],
                DATETIME_PRINT_FORMAT), order_filled_date=datetime.
                fromtimestamp(order['order_filled_timestamp'] // 1000, tz=
                timezone.utc) if order['order_filled_timestamp'] else None,
                order_type=order['order_type'], price=order['price'],
                ft_price=order['price'], remaining=order['remaining'],
                funding_fee=order.get('funding_fee', None), ft_order_tag=
                order.get('ft_order_tag', None))
            trade.orders.append(order_obj)
        return trade


class Trade(ModelBase, LocalTrade):
    """
    Trade database model.
    Also handles updating and querying trades

    Note: Fields must be aligned with LocalTrade class
    """
    __tablename__ = 'trades'
    use_db = True
    id = mapped_column(Integer, primary_key=True)
    orders = relationship('Order', order_by='Order.id', cascade=
        'all, delete-orphan', lazy='selectin', innerjoin=True)
    custom_data = relationship('_CustomData', cascade='all, delete-orphan',
        lazy='raise')
    exchange = mapped_column(String(25), nullable=False)
    pair = mapped_column(String(25), nullable=False, index=True)
    base_currency = mapped_column(String(25), nullable=True)
    stake_currency = mapped_column(String(25), nullable=True)
    is_open = mapped_column(nullable=False, default=True, index=True)
    fee_open = mapped_column(Float(), nullable=False, default=0.0)
    fee_open_cost = mapped_column(Float(), nullable=True)
    fee_open_currency = mapped_column(String(25), nullable=True)
    fee_close = mapped_column(Float(), nullable=False, default=0.0)
    fee_close_cost = mapped_column(Float(), nullable=True)
    fee_close_currency = mapped_column(String(25), nullable=True)
    open_rate = mapped_column(Float())
    open_rate_requested = mapped_column(Float(), nullable=True)
    open_trade_value = mapped_column(Float(), nullable=True)
    close_rate = mapped_column(Float())
    close_rate_requested = mapped_column(Float())
    realized_profit = mapped_column(Float(), default=0.0, nullable=True)
    close_profit = mapped_column(Float())
    close_profit_abs = mapped_column(Float())
    stake_amount = mapped_column(Float(), nullable=False)
    max_stake_amount = mapped_column(Float())
    amount = mapped_column(Float())
    amount_requested = mapped_column(Float())
    open_date = mapped_column(nullable=False, default=datetime.now)
    close_date = mapped_column()
    stop_loss = mapped_column(Float(), nullable=True, default=0.0)
    stop_loss_pct = mapped_column(Float(), nullable=True)
    initial_stop_loss = mapped_column(Float(), nullable=True, default=0.0)
    initial_stop_loss_pct = mapped_column(Float(), nullable=True)
    is_stop_loss_trailing = mapped_column(nullable=False, default=False)
    max_rate = mapped_column(Float(), nullable=True, default=0.0)
    min_rate = mapped_column(Float(), nullable=True)
    exit_reason = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)
    exit_order_status = mapped_column(String(100), nullable=True)
    strategy = mapped_column(String(100), nullable=True)
    enter_tag = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)
    timeframe = mapped_column(Integer, nullable=True)
    trading_mode = mapped_column(Enum(TradingMode), nullable=True)
    amount_precision = mapped_column(Float(), nullable=True)
    price_precision = mapped_column(Float(), nullable=True)
    precision_mode = mapped_column(Integer, nullable=True)
    precision_mode_price = mapped_column(Integer, nullable=True)
    contract_size = mapped_column(Float(), nullable=True)
    leverage = mapped_column(Float(), nullable=True, default=1.0)
    is_short = mapped_column(nullable=False, default=False)
    liquidation_price = mapped_column(Float(), nullable=True)
    interest_rate = mapped_column(Float(), nullable=False, default=0.0)
    funding_fees = mapped_column(Float(), nullable=True, default=None)
    funding_fee_running = mapped_column(Float(), nullable=True, default=None)

    def __init__(self, **kwargs):
        from_json = kwargs.pop('__FROM_JSON', None)
        super().__init__(**kwargs)
        if not from_json:
            self.realized_profit = 0
            self.recalc_open_trade_value()

    @validates('enter_tag', 'exit_reason')
    def func_orxvtgaa(self, key, value):
        max_len = getattr(self.__class__, key).prop.columns[0].type.length
        if value and len(value) > max_len:
            return value[:max_len]
        return value

    def func_vhao404r(self):
        for order in self.orders:
            Order.session.delete(order)
        CustomDataWrapper.delete_custom_data(trade_id=self.id)
        Trade.session.delete(self)
        Trade.commit()

    @staticmethod
    def func_bgqgml4g():
        Trade.session.commit()

    @staticmethod
    def func_zzv2lrq7():
        Trade.session.rollback()

    @staticmethod
    def func_t8pk2h2i(*, pair=None, is_open=None, open_date=None,
        close_date=None):
        """
        Helper function to query Trades.j
        Returns a List of trades, filtered on the parameters given.
        In live mode, converts the filter to a database query and returns all rows
        In Backtest mode, uses filters on Trade.bt_trades to get the result.

        :return: unsorted List[Trade]
        """
        if Trade.use_db:
            trade_filter = []
            if pair:
                trade_filter.append(Trade.pair == pair)
            if open_date:
                trade_filter.append(Trade.open_date > open_date)
            if close_date:
                trade_filter.append(Trade.close_date > close_date)
            if is_open is not None:
                trade_filter.append(Trade.is_open.is_(is_open))
            return cast(list[LocalTrade], Trade.get_trades(trade_filter).all())
        else:
            return LocalTrade.get_trades_proxy(pair=pair, is_open=is_open,
                open_date=open_date, close_date=close_date)

    @staticmethod
    def func_mlmlb1xw(trade_filter=None, include_orders=True):
        """
        Helper function to query Trades using filters.
        NOTE: Not supported in Backtesting.
        :param trade_filter: Optional filter to apply to trades
                             Can be either a Filter object, or a List of filters
                             e.g. `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             e.g. `(trade_filter=Trade.id == trade_id)`
        :return: unsorted query object
        """
        if not Trade.use_db:
            raise NotImplementedError(
                '`Trade.get_trades()` not supported in backtesting mode.')
        if trade_filter is not None:
            if not isinstance(trade_filter, list):
                trade_filter = [trade_filter]
            this_query = select(Trade).filter(*trade_filter)
        else:
            this_query = select(Trade)
        if not include_orders:
            this_query = this_query.options(lazyload(Trade.orders))
        return this_query

    @staticmethod
    def func_wsd0fbl2(trade_filter=None, include_orders=True):
        """
        Helper function to query Trades using filters.
        NOTE: Not supported in Backtesting.
        :param trade_filter: Optional filter to apply to trades
                             Can be either a Filter object, or a List of filters
                             e.g. `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             e.g. `(trade_filter=Trade.id == trade_id)`
        :return: unsorted query object
        """
        query = Trade.get_trades_query(trade_filter, include_orders)
        return Trade.session.scalars(query)

    @staticmethod
    def func_ju6nb0mh():
        """
        Returns all open trades which don't have open fees set correctly
        NOTE: Not supported in Backtesting.
        """
        return Trade.get_trades([Trade.fee_open_currency.is_(None), Trade.
            orders.any(), Trade.is_open.is_(True)]).all()

    @staticmethod
    def func_e6rhyb1f():
        """
        Returns all closed trades which don't have fees set correctly
        NOTE: Not supported in Backtesting.
        """
        return Trade.get_trades([Trade.fee_close_currency.is_(None), Trade.
            orders.any(), Trade.is_open.is_(False)]).all()

    @staticmethod
    def func_pvzv51ad():
        """
        Retrieves total realized profit
        """
        if Trade.use_db:
            total_profit = Trade.session.execute(select(func.sum(Trade.
                close_profit_abs)).filter(Trade.is_open.is_(False))
                ).scalar_one()
        else:
            total_profit = sum(t.close_profit_abs for t in LocalTrade.
                get_trades_proxy(is_open=False))
        return total_profit or 0

    @staticmethod
    def func_sag5w5dy():
        """
        Calculates total invested amount in open trades
        in stake currency
        """
        if Trade.use_db:
            total_open_stake_amount = Trade.session.scalar(select(func.sum(
                Trade.stake_amount)).filter(Trade.is_open.is_(True)))
        else:
            total_open_stake_amount = sum(t.stake_amount for t in
                LocalTrade.get_trades_proxy(is_open=True))
        return total_open_stake_amount or 0

    @staticmethod
    def func_9lkb6mrr(columns, filters, fallback=''):
        """
        Retrieve a generic select object to calculate performance grouped on `columns`.
        Returns the following columns:
        - columns
        - profit_ratio
        - profit_sum_abs
        - count
        NOTE: Not supported in Backtesting.
        """
        columns_coal = [func.coalesce(c, fallback).label(c.name) for c in
            columns]
        pair_costs = select(*columns_coal, func.sum(func.coalesce(Order.
            filled, Order.amount) * func.coalesce(Order.average, Order.
            price, Order.ft_price) / func.coalesce(Trade.leverage, 1)).
            label('cost_per_pair')).join(Order, Trade.id == Order.ft_trade_id
            ).filter(*filters, Order.ft_order_side == case((Trade.is_short.
            is_(True), 'sell'), else_='buy'), Order.filled > 0).group_by(*
            columns).cte('pair_costs')
        trades_grouped = select(*columns_coal, func.sum(Trade.
            close_profit_abs).label('profit_sum_abs'), func.count(*
            columns_coal).label('count')).filter(*filters).group_by(*
            columns_coal).cte('trades_grouped')
        q = select(*[trades_grouped.c[x.name] for x in columns], (
            trades_grouped.c.profit_sum_abs / pair_costs.c.cost_per_pair).
            label('profit_ratio'), trades_grouped.c.profit_sum_abs,
            trades_grouped.c.count).join(pair_costs, *[(trades_grouped.c[x.
            name] == pair_costs.c[x.name]) for x in columns]).order_by(desc
            ('profit_sum_abs'))
        return q

    @staticmethod
    def func_87o96o40(start_date=None):
        """
        Returns List of dicts containing all Trades, including profit and trade count
        NOTE: Not supported in Backtesting.
        """
        filters = [Trade.is_open.is_(False)]
        if start_date:
            filters.append(Trade.close_date >= start_date)
        pair_rates_query = Trade._generic_performance_query([Trade.pair],
            filters)
        pair_rates = Trade.session.execute(pair_rates_query).all()
        return [{'pair': pair, 'profit_ratio': profit, 'profit': round(
            profit * 100, 2), 'profit_pct': round(profit * 100, 2),
            'profit_abs': profit_abs, 'count': count} for pair, profit,
            profit_abs, count in pair_rates]

    @staticmethod
    def func_xt3el3j6(pair):
        """
        Returns List of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """
        filters = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)
        pair_rates_query = Trade._generic_performance_query([Trade.
            enter_tag], filters, 'Other')
        enter_tag_perf = Trade.session.execute(pair_rates_query).all()
        return [{'enter_tag': enter_tag if enter_tag is not None else
            'Other', 'profit_ratio': profit, 'profit_pct': round(profit * 
            100, 2), 'profit_abs': profit_abs, 'count': count} for 
            enter_tag, profit, profit_abs, count in enter_tag_perf]

    @staticmethod
    def func_ag4a4zze(pair):
        """
        Returns List of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """
        filters = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)
        pair_rates_query = Trade._generic_performance_query([Trade.
            exit_reason], filters, 'Other')
        sell_tag_perf = Trade.session.execute(pair_rates_query).all()
        return [{'exit_reason': exit_reason if exit_reason is not None else
            'Other', 'profit_ratio': profit, 'profit_pct': round(profit * 
            100, 2), 'profit_abs': profit_abs, 'count': count} for 
            exit_reason, profit, profit_abs, count in sell_tag_perf]

    @staticmethod
    def func_ri3oh988(pair):
        """
        Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """
        filters = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)
        mix_tag_perf = Trade.session.execute(select(Trade.id, Trade.
            enter_tag, Trade.exit_reason, func.sum(Trade.close_profit).
            label('profit_sum'), func.sum(Trade.close_profit_abs).label(
            'profit_sum_abs'), func.count(Trade.pair).label('count')).
            filter(*filters).group_by(Trade.id).order_by(desc(
            'profit_sum_abs'))).all()
        resp = []
        for _, enter_tag, exit_reason, profit, profit_abs, count in mix_tag_perf:
            enter_tag = enter_tag if enter_tag is not None else 'Other'
            exit_reason = exit_reason if exit_reason is not None else 'Other'
            if exit_reason is not None and enter_tag is not None:
                mix_tag = enter_tag + ' ' + exit_reason
                i = 0
                if not any(item['mix_tag'] == mix_tag for item in resp):
                    resp.append({'mix_tag': mix_tag, 'profit_ratio': profit,
                        'profit_pct': round(profit * 100, 2), 'profit_abs':
                        profit_abs, 'count': count})
                else:
                    while i < len(resp):
                        if resp[i]['mix_tag'] == mix_tag:
                            resp[i] = {'mix_tag': mix_tag, 'profit_ratio': 
                                profit + resp[i]['profit_ratio'],
                                'profit_pct': round(profit + resp[i][
                                'profit_ratio'] * 100, 2), 'profit_abs': 
                                profit_abs + resp[i]['profit_abs'], 'count':
                                1 + resp[i]['count']}
                        i += 1
        return resp

    @staticmethod
    def func_sdchmvsu(start_date=None):
        """
        Get best pair with closed trade.
        NOTE: Not supported in Backtesting.
        :returns: Tuple containing (pair, profit_sum)
        """
        filters = [Trade.is_open.is_(False)]
        if start_date:
            filters.append(Trade.close_date >= start_date)
        pair_rates_query = Trade._generic_performance_query([Trade.pair],
            filters)
        best_pair = Trade.session.execute(pair_rates_query).first()
        return best_pair

    @staticmethod
    def func_1pk1n1zd(start_date=None):
        """
        Get Trade volume based on Orders
        NOTE: Not supported in Backtesting.
        :returns: Tuple containing (pair, profit_sum)
        """
        filters = [Order.status == 'closed']
        if start_date:
            filters.append(Order.order_filled_date >= start_date)
        trading_volume = Trade.session.execute(select(func.sum(Order.cost).
            label('volume')).filter(*filters)).scalar_one()
        return trading_volume or 0.0
