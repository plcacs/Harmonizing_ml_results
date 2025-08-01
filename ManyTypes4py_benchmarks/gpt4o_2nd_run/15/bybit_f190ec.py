"""Bybit exchange subclass"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import ccxt
from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util.datetime_helpers import dt_now, dt_ts

logger = logging.getLogger(__name__)

class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """
    unified_account: bool = False
    _ft_has: Dict[str, Any] = {
        'ohlcv_has_history': True,
        'order_time_in_force': ['GTC', 'FOK', 'IOC', 'PO'],
        'ws_enabled': True,
        'trades_has_history': False,
        'exchange_has_overrides': {'fetchOrder': False}
    }
    _ft_has_futures: Dict[str, Any] = {
        'ohlcv_has_history': True,
        'mark_ohlcv_timeframe': '4h',
        'funding_fee_timeframe': '8h',
        'funding_fee_candle_limit': 200,
        'stoploss_on_exchange': True,
        'stoploss_order_types': {'limit': 'limit', 'market': 'market'},
        'stop_price_prop': 'stopPrice',
        'stop_price_type_field': 'triggerBy',
        'stop_price_type_value_mapping': {
            PriceType.LAST: 'LastPrice',
            PriceType.MARK: 'MarkPrice',
            PriceType.INDEX: 'IndexPrice'
        },
        'exchange_has_overrides': {'fetchOrder': True}
    }
    _supported_trading_mode_margin_pairs: List[tuple] = [(TradingMode.FUTURES, MarginMode.ISOLATED)]

    @property
    def _ccxt_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({'options': {'defaultType': 'spot'}})
        config.update(super()._ccxt_config)
        return config

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        main: bool = super().market_is_future(market)
        return main and market['settle'] == 'USDT'

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if not self._config['dry_run']:
                if self.trading_mode == TradingMode.FUTURES:
                    position_mode = self._api.set_position_mode(False)
                    self._log_exchange_response('set_position_mode', position_mode)
                is_unified = self._api.is_unified_enabled()
                if is_unified and len(is_unified) > 1 and is_unified[1]:
                    self.unified_account = True
                    logger.info('Bybit: Unified account. Assuming dedicated subaccount for this bot.')
                else:
                    self.unified_account = False
                    logger.info('Bybit: Standard account.')
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != TradingMode.SPOT:
            params: Dict[str, Any] = {'leverage': leverage}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)
            self._set_leverage(leverage, pair, accept_fail=True)

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict[str, Any]:
        params: Dict[str, Any] = super()._get_params(side=side, ordertype=ordertype, leverage=leverage, reduceOnly=reduceOnly, time_in_force=time_in_force)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['position_idx'] = 0
        return params

    def _order_needs_price(self, side: str, ordertype: str) -> bool:
        return ordertype != 'market' or (side == 'buy' and (not self.unified_account) and (self.trading_mode == TradingMode.SPOT)) or self._ft_has.get('marketOrderRequiresPrice', False)

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Dict[str, Any]]) -> float:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!
        PERPETUAL:
         bybit:
          https://www.bybithelp.com/HelpCenterKnowledge/bybitHC_Article?language=en_US&id=000001067

        Long:
        Liquidation Price = (
            Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate)
            - Extra Margin Added/ Contract)
        Short:
        Liquidation Price = (
            Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate)
            + Extra Margin Added/ Contract)

        Implementation Note: Extra margin is currently not used.

        :param pair: Pair to calculate liquidation price for
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param leverage: Leverage used for this position.
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param margin_mode: Either ISOLATED or CROSS
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        :param open_trades: List of other open trades in the same wallet
        """
        market: Dict[str, Any] = self.markets[pair]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market['inverse']:
                raise OperationalException('Freqtrade does not yet support inverse contracts')
            initial_margin_rate: float = 1 / leverage
            if is_short:
                return open_rate * (1 + initial_margin_rate - mm_ratio)
            else:
                return open_rate * (1 - initial_margin_rate + mm_ratio)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
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

    def fetch_orders(self, pair: str, since: datetime, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch all orders for a pair "since"
        :param pair: Pair for the query
        :param since: Starting time for the query
        """
        orders: List[Dict[str, Any]] = []
        while since < dt_now():
            until: datetime = since + timedelta(days=7, minutes=-1)
            orders += super().fetch_orders(pair, since, params={'until': dt_ts(until)})
            since = until
        return orders

    def fetch_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.exchange_has('fetchOrder'):
            params = {'acknowledged': True}
        order: Dict[str, Any] = super().fetch_order(order_id, pair, params)
        if not order:
            order = self.fetch_order_emulated(order_id, pair, {})
        if order.get('status') == 'canceled' and order.get('filled') == 0.0 and (order.get('remaining') == 0.0):
            order['remaining'] = None
        return order

    @retrier
    def get_leverage_tiers(self) -> List[Dict[str, Any]]:
        """
        Cache leverage tiers for 1 day, since they are not expected to change often, and
        bybit requires pagination to fetch all tiers.
        """
        tiers_cached: Optional[List[Dict[str, Any]]] = self.load_cached_leverage_tiers(self._config['stake_currency'], timedelta(days=1))
        if tiers_cached:
            return tiers_cached
        tiers: List[Dict[str, Any]] = super().get_leverage_tiers()
        self.cache_leverage_tiers(tiers, self._config['stake_currency'])
        return tiers
