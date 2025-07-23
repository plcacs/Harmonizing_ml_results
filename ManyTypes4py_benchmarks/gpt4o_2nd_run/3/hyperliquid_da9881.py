"""Hyperliquid exchange subclass"""
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Union
from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util.datetime_helpers import dt_from_ts

logger = logging.getLogger(__name__)

class Hyperliquid(Exchange):
    """Hyperliquid exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """
    _ft_has: Dict[str, Union[bool, List[int], Dict[str, bool]]] = {
        'ohlcv_has_history': False,
        'l2_limit_range': [20],
        'trades_has_history': False,
        'tickers_have_bid_ask': False,
        'stoploss_on_exchange': False,
        'exchange_has_overrides': {'fetchTrades': False},
        'marketOrderRequiresPrice': True
    }
    _ft_has_futures: Dict[str, Union[bool, Dict[str, str], str, int]] = {
        'stoploss_on_exchange': True,
        'stoploss_order_types': {'limit': 'limit'},
        'stop_price_prop': 'stopPrice',
        'funding_fee_timeframe': '1h',
        'funding_fee_candle_limit': 500
    }
    _supported_trading_mode_margin_pairs: List[tuple] = [
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    @property
    def _ccxt_config(self) -> Dict[str, Dict[str, str]]:
        config: Dict[str, Dict[str, str]] = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({'options': {'defaultType': 'spot'}})
        config.update(super()._ccxt_config)
        return config

    def get_max_leverage(self, pair: str, stake_amount: float) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]['limits']['leverage']['max']
        else:
            return 1.0

    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != TradingMode.SPOT:
            leverage = int(leverage)
            self.set_margin_mode(pair, self.margin_mode, params={'leverage': leverage})

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Dict]) -> float:
        isolated_margin: float = wallet_balance
        position_size: float = amount
        price: float = open_rate
        position_value: float = price * position_size
        max_leverage: float = self.markets[pair]['limits']['leverage']['max']
        maintenance_margin_required: float = position_value / max_leverage / 2
        margin_available: float = isolated_margin - maintenance_margin_required
        maintenance_leverage: float = max_leverage * 2
        ll: float = 1 / maintenance_leverage
        side: int = -1 if is_short else 1
        liq_price: float = price - side * margin_available / position_size / (1 - ll * side)
        if self.trading_mode == TradingMode.FUTURES:
            return liq_price
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f'Could not update funding fees for {pair}.')
        return 0.0

    def _adjust_hyperliquid_order(self, order: Dict) -> Dict:
        if order['average'] is None and order['status'] in ('canceled', 'closed') and (order['filled'] > 0):
            trades = self.get_trades_for_order(order['id'], order['symbol'], since=dt_from_ts(order['timestamp']))
            if trades:
                total_amount: float = sum((t['amount'] for t in trades))
                order['average'] = sum((t['price'] * t['amount'] for t in trades)) / total_amount if total_amount else None
        return order

    def fetch_order(self, order_id: str, pair: str, params: Optional[Dict] = None) -> Dict:
        order: Dict = super().fetch_order(order_id, pair, params)
        order = self._adjust_hyperliquid_order(order)
        self._log_exchange_response('fetch_order2', order)
        return order

    def fetch_orders(self, pair: str, since: Optional[datetime], params: Optional[Dict] = None) -> List[Dict]:
        orders: List[Dict] = super().fetch_orders(pair, since, params)
        for idx, order in enumerate(deepcopy(orders)):
            order2: Dict = self._adjust_hyperliquid_order(order)
            orders[idx] = order2
        self._log_exchange_response('fetch_orders2', orders)
        return orders
