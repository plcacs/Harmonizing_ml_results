"""
This module contains the class to persist trades into SQLite
"""
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose
from typing import Any, ClassVar, Optional, cast, List, Dict, Tuple, Union
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
    profit_abs: float
    profit_ratio: float
    total_profit: float
    total_profit_ratio: float

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
    __table_args__ = (UniqueConstraint('ft_pair', 'order_id', name='_order_pair_order_id'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ft_trade_id: Mapped[int] = mapped_column(Integer, ForeignKey('trades.id'), index=True)
    _trade_live: Mapped['Trade'] = relationship('Trade', back_populates='orders', lazy='immediate')
    _trade_bt: Optional[Any] = None
    ft_order_side: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_pair: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_is_open: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
    ft_amount: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_price: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_cancel_reason: Mapped[Optional[str]] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)
    order_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    order_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    side: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    average: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    filled: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    remaining: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    cost: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    order_date: Mapped[Optional[datetime]] = mapped_column(nullable=True, default=dt_now)
    order_filled_date: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    order_update_date: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    funding_fee: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    ft_fee_base: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    ft_order_tag: Mapped[Optional[str]] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)

    @property
    def order_date_utc(self) -> datetime:
        """Order-date with UTC timezoneinfo"""
        return self.order_date.replace(tzinfo=timezone.utc)

    @property
    def order_filled_utc(self) -> Optional[datetime]:
        """last order-date with UTC timezoneinfo"""
        return self.order_filled_date.replace(tzinfo=timezone.utc) if self.order_filled_date else None

    @property
    def safe_amount(self) -> float:
        return self.amount or self.ft_amount

    @property
    def safe_placement_price(self) -> float:
        """Price at which the order was placed"""
        return self.price or self.stop_price or self.ft_price

    @property
    def safe_price(self) -> float:
        return self.average or self.price or self.stop_price or self.ft_price

    @property
    def safe_filled(self) -> float:
        return self.filled if self.filled is not None else 0.0

    @property
    def safe_cost(self) -> float:
        return self.cost or 0.0

    @property
    def safe_remaining(self) -> float:
        return self.remaining if self.remaining is not None else self.safe_amount - (self.filled or 0.0)

    @property
    def safe_fee_base(self) -> float:
        return self.ft_fee_base or 0.0

    @property
    def safe_amount_after_fee(self) -> float:
        return self.safe_filled - self.safe_fee_base

    @property
    def trade(self) -> 'Trade':
        return self._trade_bt or self._trade_live

    @property
    def stake_amount(self) -> float:
        """Amount in stake currency used for this order"""
        return float(FtPrecise(self.safe_amount) * FtPrecise(self.safe_price) / FtPrecise(self.trade.leverage))

    @property
    def stake_amount_filled(self) -> float:
        """Filled Amount in stake currency used for this order"""
        return float(FtPrecise(self.safe_filled) * FtPrecise(self.safe_price) / FtPrecise(self.trade.leverage))

    def __repr__(self) -> str:
        return f'Order(id={self.id}, trade={self.ft_trade_id}, order_id={self.order_id}, side={self.side}, filled={self.safe_filled}, price={self.safe_price}, amount={self.amount}, status={self.status}, date={self.order_date_utc:{DATETIME_PRINT_FORMAT}})'

    def update_from_ccxt_object(self, order: Dict[str, Any]) -> None:
        """
        Update Order from ccxt response
        Only updates if fields are available from ccxt -
        """
        if self.order_id != str(order['id']):
            raise DependencyException("Order-id's don't match")
        self.status = safe_value_fallback(order, 'status', default_value=self.status)
        self.symbol = safe_value_fallback(order, 'symbol', default_value=self.symbol)
        self.order_type = safe_value_fallback(order, 'type', default_value=self.order_type)
        self.side = safe_value_fallback(order, 'side', default_value=self.side)
        self.price = safe_value_fallback(order, 'price', default_value=self.price)
        self.amount = safe_value_fallback(order, 'amount', default_value=self.amount)
        self.filled = safe_value_fallback(order, 'filled', default_value=self.filled)
        self.average = safe_value_fallback(order, 'average', default_value=self.average)
        self.remaining = safe_value_fallback(order, 'remaining', default_value=self.remaining)
        self.cost = safe_value_fallback(order, 'cost', default_value=self.cost)
        self.stop_price = safe_value_fallback(order, 'stopPrice', default_value=self.stop_price)
        order_date = safe_value_fallback(order, 'timestamp')
        if order_date:
            self.order_date = dt_from_ts(order_date)
        elif not self.order_date:
            self.order_date = dt_now()
        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get('filled', 0.0) or 0.0) > 0 and (not self.order_filled_date):
                self.order_filled_date = dt_from_ts(safe_value_fallback(order, 'lastTradeTimestamp', default_value=dt_ts()))
        self.order_update_date = datetime.now(timezone.utc)

    def to_ccxt_object(self, stopPriceName: str = 'stopPrice') -> Dict[str, Any]:
        order = {'id': self.order_id, 'symbol': self.ft_pair, 'price': self.price, 'average': self.average, 'amount': self.amount, 'cost': self.cost, 'type': self.order_type, 'side': self.ft_order_side, 'filled': self.filled, 'remaining': self.remaining, 'datetime': self.order_date_utc.strftime('%Y-%m-%dT%H:%M:%S.%f'), 'timestamp': int(self.order_date_utc.timestamp() * 1000), 'status': self.status, 'fee': None, 'info': {}}
        if self.ft_order_side == 'stoploss':
            order.update({stopPriceName: self.stop_price, 'ft_order_type': 'stoploss'})
        return order

    def to_json(self, entry_side: str, minified: bool = False) -> Dict[str, Any]:
        """
        :param minified: If True, only return a subset of the data is returned.
                         Only used for backtesting.
        """
        resp = {'amount': self.safe_amount, 'safe_price': self.safe_price, 'ft_order_side': self.ft_order_side, 'order_filled_timestamp': dt_ts_none(self.order_filled_utc), 'ft_is_entry': self.ft_order_side == entry_side, 'ft_order_tag': self.ft_order_tag, 'cost': self.cost if self.cost else 0}
        if not minified:
            resp.update({'pair': self.ft_pair, 'order_id': self.order_id, 'status': self.status, 'average': round(self.average, 8) if self.average else 0, 'filled': self.filled, 'is_open': self.ft_is_open, 'order_date': self.order_date.strftime(DATETIME_PRINT_FORMAT) if self.order_date else None, 'order_timestamp': int(self.order_date.replace(tzinfo=timezone.utc).timestamp() * 1000) if self.order_date else None, 'order_filled_date': self.order_filled_date.strftime(DATETIME_PRINT_FORMAT) if self.order_filled_date else None, 'order_type': self.order_type, 'price': self.price, 'remaining': self.remaining, 'ft_fee_base': self.ft_fee_base, 'funding_fee': self.funding_fee})
        return resp

    def close_bt_order(self, close_date: datetime, trade: 'Trade') -> None:
        self.order_filled_date = close_date
        self.filled = self.amount
        self.remaining = 0
        self.status = 'closed'
        self.ft_is_open = False
        self.funding_fee = trade.funding_fee_running
        trade.funding_fee_running = 0.0
        if self.ft_order_side == trade.entry_side and self.price:
            trade.open_rate = self.price
            trade.recalc_trade_from_orders()
            if trade.nr_of_successful_entries == 1:
                trade.initial_stop_loss_pct = None
                trade.is_stop_loss_trailing = False
            trade.adjust_stop_loss(trade.open_rate, trade.stop_loss_pct)

    @staticmethod
    def update_orders(orders: List['Order'], order: Dict[str, Any]) -> None:
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
    def parse_from_ccxt_object(cls, order: Dict[str, Any], pair: str, side: str, amount: Optional[float] = None, price: Optional[float] = None) -> 'Order':
        """
        Parse an order from a ccxt object and return a new order Object.
        Optional support for overriding amount and price is only used for test simplification.
        """
        o = cls(order_id=str(order['id']), ft_order_side=side, ft_pair=pair, ft_amount=amount or order.get('amount', None) or 0.0, ft_price=price or order.get('price', None))
        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def get_open_orders() -> List['Order']:
        """
        Retrieve open orders from the database
        :return: List of open orders
        """
        return Order.session.scalars(select(Order).filter(Order.ft_is_open.is_(True))).all()

    @staticmethod
    def order_by_id(order_id: str) -> Optional['Order']:
        """
        Retrieve order based on order_id
        :return: Order or None
        """
        return Order.session.scalars(select(Order).filter(Order.order_id == order_id)).first()

class LocalTrade:
    """
    Trade database model.
    Used in backtesting - must be aligned to Trade model!
    """
    use_db: bool = False
    bt_trades: List['LocalTrade'] = []
    bt_trades_open: List['LocalTrade'] = []
    bt_trades_open_pp: Dict[str, List['LocalTrade']] = defaultdict(list)
    bt_open_open_trade_count: int = 0
    bt_total_profit: float = 0
    realized_profit: float = 0
    id: int = 0
    orders: List[Order] = []
    exchange: str = ''
    pair: str = ''
    base_currency: str = ''
    stake_currency: str = ''
    is_open: bool = True
    fee_open: float = 0.0
    fee_open_cost: Optional[float] = None
    fee_open_currency: str = ''
    fee_close: float = 0.0
    fee_close_cost: Optional[float] = None
    fee_close_currency: str = ''
    open_rate: float = 0.0
    open_rate_requested: Optional[float] = None
    open_trade_value: float = 0.0
    close_rate: Optional[float] = None
    close_rate_requested: Optional[float] = None
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    stake_amount: float = 0.0
    max_stake_amount: float = 0.0
    amount: float = 0.0
    amount_requested: Optional[float] = None
    close_date: Optional[datetime] = None
    stop_loss: float = 0.0
    stop_loss_pct: float = 0.0
    initial_stop_loss: float = 0.0
    initial_stop_loss_pct: Optional[float] = None
    is_stop_loss_trailing: bool = False
    max_rate: Optional[float] = None
    min_rate: Optional[float] = None
    exit_reason: str = ''
    exit_order_status: str = ''
    strategy: str = ''
    enter_tag: Optional[str] = None
    timeframe: Optional[int] = None
    trading_mode: TradingMode = TradingMode.SPOT
    amount_precision: Optional[float] = None
    price_precision: Optional[float] = None
    precision_mode: Optional[int] = None
    precision_mode_price: Optional[int] = None
    contract_size: Optional[float] = None
    liquidation_price: Optional[float] = None
    is_short: bool = False
    leverage: float = 1.0
    interest_rate: float = 0.0
    funding_fees: Optional[float] = None
    funding_fee_running: Optional[float] = None

    @property
    def stoploss_or_liquidation(self) -> float:
        if self.liquidation_price:
            if self.is_short:
                return min(self.stop_loss, self.liquidation_price)
            else:
                return max(self.stop_loss, self.liquidation_price)
        return self.stop_loss

    @property
    def buy_tag(self) -> Optional[str]:
        """
        Compatibility between buy_tag (old) and enter_tag (new)
        Consider buy_tag deprecated
        """
        return self.enter_tag

    @property
    def has_no_leverage(self) -> bool:
        """Returns true if this is a non-leverage, non-short trade"""
        return (self.leverage == 1.0 or self.leverage is None) and (not self.is_short)

    @property
    def borrowed(self) -> float:
        """
        The amount of currency borrowed from the