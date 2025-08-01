import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional
import ccxt
from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, RetryableOrderError, TemporaryError
from freqtrade.exchange import Exchange, date_minus_candles
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.misc import safe_value_fallback2
from freqtrade.util import dt_now, dt_ts

logger = logging.getLogger(__name__)


class Okx(Exchange):
    """Okx exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """
    _ft_has: Dict[str, Any] = {
        'ohlcv_candle_limit': 100,
        'mark_ohlcv_timeframe': '4h',
        'funding_fee_timeframe': '8h',
        'stoploss_order_types': {'limit': 'limit'},
        'stoploss_on_exchange': True,
        'trades_has_history': False,
        'ws_enabled': True,
    }
    _ft_has_futures: Dict[str, Any] = {
        'tickers_have_quoteVolume': False,
        'stop_price_type_field': 'slTriggerPxType',
        'stop_price_type_value_mapping': {PriceType.LAST: 'last', PriceType.MARK: 'index', PriceType.INDEX: 'mark'},
        'ws_enabled': True,
    }
    _supported_trading_mode_margin_pairs = [(TradingMode.FUTURES, MarginMode.ISOLATED)]
    net_only: bool = True
    _ccxt_params: Dict[str, Any] = {'options': {'brokerId': 'ffb5405ad327SUDE'}}

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        """
        Exchange ohlcv candle limit
        OKX has the following behaviour:
        * 300 candles for up-to-date data
        * 100 candles for historic data
        * 100 candles for additional candles (not futures or spot).
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        if candle_type in (CandleType.FUTURES, CandleType.SPOT) and (not since_ms or since_ms > date_minus_candles(timeframe, 300).timestamp() * 1000):
            return 300
        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and (not self._config['dry_run']):
                accounts: Any = self._api.fetch_accounts()
                self._log_exchange_response('fetch_accounts', accounts)
                if len(accounts) > 0:
                    self.net_only = accounts[0].get('info', {}).get('posMode') == 'net_mode'
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_posSide(self, side: str, reduceOnly: bool) -> str:
        if self.net_only:
            return 'net'
        if not reduceOnly:
            return 'long' if side == 'buy' else 'short'
        else:
            return 'long' if side == 'sell' else 'short'

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> Dict[str, Any]:
        params: Dict[str, Any] = super()._get_params(side=side, ordertype=ordertype, leverage=leverage, reduceOnly=reduceOnly, time_in_force=time_in_force)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['tdMode'] = self.margin_mode.value
            params['posSide'] = self._get_posSide(side, reduceOnly)
        return params

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: str) -> bool:
        try:
            res_lev: Any = self._api.fetch_leverage(symbol=pair, params={'mgnMode': self.margin_mode.value, 'posSide': self._get_posSide(side, False)})
            self._log_exchange_response('get_leverage', res_lev)
            already_set: bool = all((float(x['lever']) == leverage for x in res_lev['data']))
            return already_set
        except ccxt.BaseError:
            return False

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != TradingMode.SPOT and self.margin_mode is not None:
            try:
                res: Any = self._api.set_leverage(leverage=leverage, symbol=pair, params={'mgnMode': self.margin_mode.value, 'posSide': self._get_posSide(side, False)})
                self._log_exchange_response('set_leverage', res)
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                already_set: bool = self.__fetch_leverage_already_set(pair, leverage, side)
                if not already_set:
                    raise TemporaryError(f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        if self.trading_mode == TradingMode.SPOT:
            return float('inf')
        if pair not in self._leverage_tiers:
            return float('inf')
        pair_tiers: List[Dict[str, Any]] = self._leverage_tiers[pair]
        return pair_tiers[-1]['maxNotional'] / leverage

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> Dict[str, Any]:
        params: Dict[str, Any] = super()._get_stop_params(side, ordertype, stop_price)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['tdMode'] = self.margin_mode.value
            params['posSide'] = self._get_posSide(side, True)
        return params

    def _convert_stop_order(self, pair: str, order_id: str, order: Dict[str, Any]) -> Dict[str, Any]:
        if order.get('status', 'open') == 'closed' and (real_order_id := order.get('info', {}).get('ordId')) is not None:
            order_reg: Dict[str, Any] = self.fetch_order(real_order_id, pair)
            self._log_exchange_response('fetch_stoploss_order1', order_reg)
            order_reg['id_stop'] = order_reg['id']
            order_reg['id'] = order_id
            order_reg['type'] = 'stoploss'
            order_reg['status_stop'] = 'triggered'
            return order_reg
        order = self._order_contracts_to_amount(order)
        order['type'] = 'stoploss'
        return order

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._config['dry_run']:
            return self.fetch_dry_run_order(order_id)
        try:
            params1: Dict[str, Any] = {'stop': True}
            order_reg: Dict[str, Any] = self._api.fetch_order(order_id, pair, params=params1)
            self._log_exchange_response('fetch_stoploss_order', order_reg)
            return self._convert_stop_order(pair, order_id, order_reg)
        except (ccxt.OrderNotFound, ccxt.InvalidOrder):
            pass
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e
        return self._fetch_stop_order_fallback(order_id, pair)

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> Dict[str, Any]:
        params2: Dict[str, Any] = {'stop': True, 'ordType': 'conditional'}
        for method in (self._api.fetch_open_orders, self._api.fetch_closed_orders, self._api.fetch_canceled_orders):
            try:
                orders: List[Dict[str, Any]] = method(pair, params=params2)
                orders_f: List[Dict[str, Any]] = [order for order in orders if order['id'] == order_id]
                if orders_f:
                    order: Dict[str, Any] = orders_f[0]
                    return self._convert_stop_order(pair, order_id, order)
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                raise TemporaryError(f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        raise RetryableOrderError(f'StoplossOrder not found (pair: {pair} id: {order_id}).')

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        if order.get('type', '') == 'stop':
            return safe_value_fallback2(order, order, 'id_stop', 'id')
        return order['id']

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params1: Dict[str, Any] = {'stop': True}
        return self.cancel_order(order_id=order_id, pair=pair, params=params1)

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        orders = self._api.fetch_closed_orders(pair, since=since_ms)
        if since_ms < dt_ts(dt_now() - timedelta(days=6, hours=23)):
            params: Dict[str, Any] = {'method': 'privateGetTradeOrdersHistoryArchive'}
            orders_hist: List[Dict[str, Any]] = self._api.fetch_closed_orders(pair, since=since_ms, params=params)
            orders.extend(orders_hist)
        orders_open: List[Dict[str, Any]] = self._api.fetch_open_orders(pair, since=since_ms)
        orders.extend(orders_open)
        return orders