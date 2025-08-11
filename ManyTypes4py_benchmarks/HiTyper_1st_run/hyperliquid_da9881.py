"""Hyperliquid exchange subclass"""
import logging
from copy import deepcopy
from datetime import datetime
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
    _ft_has = {'ohlcv_has_history': False, 'l2_limit_range': [20], 'trades_has_history': False, 'tickers_have_bid_ask': False, 'stoploss_on_exchange': False, 'exchange_has_overrides': {'fetchTrades': False}, 'marketOrderRequiresPrice': True}
    _ft_has_futures = {'stoploss_on_exchange': True, 'stoploss_order_types': {'limit': 'limit'}, 'stop_price_prop': 'stopPrice', 'funding_fee_timeframe': '1h', 'funding_fee_candle_limit': 500}
    _supported_trading_mode_margin_pairs = [(TradingMode.FUTURES, MarginMode.ISOLATED)]

    @property
    def _ccxt_config(self) -> dict[typing.Text, dict[typing.Text, typing.Text]]:
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({'options': {'defaultType': 'spot'}})
        config.update(super()._ccxt_config)
        return config

    def get_max_leverage(self, pair: Union[str, int], stake_amount: Union[int, list[int], bytes]) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]['limits']['leverage']['max']
        else:
            return 1.0

    def _lev_prep(self, pair: Union[str, list[str]], leverage: Any, side: Union[bool, list[str]], accept_fail: bool=False) -> None:
        if self.trading_mode != TradingMode.SPOT:
            leverage = int(leverage)
            self.set_margin_mode(pair, self.margin_mode, params={'leverage': leverage})

    def dry_run_liquidation_price(self, pair: int, open_rate: Union[int, float], is_short: Union[list[str], int, str], amount: Union[int, float], stake_amount: Union[str, float, int], leverage: Union[str, float, int], wallet_balance: int, open_trades: Union[str, float, int]) -> float:
        """
        Optimized
        Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations
        Below can be done in fewer lines of code, but like this it matches the documentation.

        Tested with 196 unique ccxt fetch_positions() position outputs
        - Only first output per position where pnl=0.0
        - Compare against returned liquidation price
        Positions: 197 Average deviation: 0.00028980% Max deviation: 0.01309453%
        Positions info:
        {'leverage': {1.0: 23, 2.0: 155, 3.0: 8, 4.0: 7, 5.0: 4},
        'side': {'long': 133, 'short': 64},
        'symbol': {'BTC/USDC:USDC': 81,
                   'DOGE/USDC:USDC': 20,
                   'ETH/USDC:USDC': 53,
                   'SOL/USDC:USDC': 43}}
        """
        isolated_margin = wallet_balance
        position_size = amount
        price = open_rate
        position_value = price * position_size
        max_leverage = self.markets[pair]['limits']['leverage']['max']
        maintenance_margin_required = position_value / max_leverage / 2
        margin_available = isolated_margin - maintenance_margin_required
        maintenance_leverage = max_leverage * 2
        ll = 1 / maintenance_leverage
        side = -1 if is_short else 1
        liq_price = price - side * margin_available / position_size / (1 - ll * side)
        if self.trading_mode == TradingMode.FUTURES:
            return liq_price
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_funding_fees(self, pair: Union[str, int], amount: Union[str, None, float], is_short: Union[str, None, float], open_date: Union[str, None, float]) -> Union[bool, float]:
        """
        Fetch funding fees, either from the exchange (live) or calculates them
        based on funding rate/mark price history
        :param pair: The quote/base pair of the trade
        :param is_short: trade direction
        :param amount: Trade amount
        :param open_date: Open date of the trade
        :return: funding fee since open_date
        :raises: ExchangeError if something goes wrong.
        """
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f'Could not update funding fees for {pair}.')
        return 0.0

    def _adjust_hyperliquid_order(self, order: Any):
        """
        Adjusts order response for Hyperliquid
        :param order: Order response from Hyperliquid
        :return: Adjusted order response
        """
        if order['average'] is None and order['status'] in ('canceled', 'closed') and (order['filled'] > 0):
            trades = self.get_trades_for_order(order['id'], order['symbol'], since=dt_from_ts(order['timestamp']))
            if trades:
                total_amount = sum((t['amount'] for t in trades))
                order['average'] = sum((t['price'] * t['amount'] for t in trades)) / total_amount if total_amount else None
        return order

    def fetch_order(self, order_id: Union[dict[str, typing.Any], dict, dict[str, str]], pair: Union[dict[str, typing.Any], dict, dict[str, str]], params: Union[None, dict[str, typing.Any], dict, dict[str, str]]=None) -> Union[tuple[str], int]:
        order = super().fetch_order(order_id, pair, params)
        order = self._adjust_hyperliquid_order(order)
        self._log_exchange_response('fetch_order2', order)
        return order

    def fetch_orders(self, pair: Union[dict[str, typing.Any], dict, PRecord], since: Union[dict[str, typing.Any], dict, PRecord], params: Union[None, dict[str, typing.Any], dict, PRecord]=None) -> Union[list[str], list]:
        orders = super().fetch_orders(pair, since, params)
        for idx, order in enumerate(deepcopy(orders)):
            order2 = self._adjust_hyperliquid_order(order)
            orders[idx] = order2
        self._log_exchange_response('fetch_orders2', orders)
        return orders