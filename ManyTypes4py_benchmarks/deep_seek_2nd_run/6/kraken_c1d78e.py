"""Kraken exchange subclass"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import ccxt
from pandas import DataFrame
from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtBalances, FtHas, Tickers
logger = logging.getLogger(__name__)

class Kraken(Exchange):
    _params: Dict[str, str] = {'trading_agreement': 'agree'}
    _ft_has: FtHas = {
        'stoploss_on_exchange': True,
        'stop_price_param': 'stopLossPrice',
        'stop_price_prop': 'stopLossPrice',
        'stoploss_order_types': {'limit': 'limit', 'market': 'market'},
        'order_time_in_force': ['GTC', 'IOC', 'PO'],
        'ohlcv_has_history': False,
        'trades_pagination': 'id',
        'trades_pagination_arg': 'since',
        'trades_pagination_overlap': False,
        'trades_has_history': True,
        'mark_ohlcv_timeframe': '4h'
    }
    _supported_trading_mode_margin_pairs: List = []

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is darkpool pair.
        """
        parent_check = super().market_is_tradable(market)
        return parent_check and market.get('darkpool', False) is False

    def get_tickers(self, symbols: Optional[List[str]] = None, *, cached: bool = False, market_type: Optional[str] = None) -> Tickers:
        symbols = list(self.get_markets(quote_currencies=[self._config['stake_currency']]))
        return super().get_tickers(symbols=symbols, cached=cached, market_type=market_type)

    def consolidate_balances(self, balances: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Consolidate balances for the same currency.
        Kraken returns ".F" balances if rewards is enabled.
        """
        consolidated: Dict[str, Dict[str, float]] = {}
        for currency, balance in balances.items():
            base_currency = currency[:-2] if currency.endswith('.F') else currency
            base_currency = self._api.commonCurrencies.get(base_currency, base_currency)
            if base_currency in consolidated:
                consolidated[base_currency]['free'] += balance['free']
                consolidated[base_currency]['used'] += balance['used']
                consolidated[base_currency]['total'] += balance['total']
            else:
                consolidated[base_currency] = balance
        return consolidated

    @retrier
    def get_balances(self) -> CcxtBalances:
        if self._config['dry_run']:
            return {}
        try:
            balances = self._api.fetch_balance()
            balances.pop('info', None)
            balances.pop('free', None)
            balances.pop('total', None)
            balances.pop('used', None)
            self._log_exchange_response('fetch_balances', balances)
            balances = self.consolidate_balances(balances)
            orders = self._api.fetch_open_orders()
            order_list: List[Tuple[str, float]] = [
                (x['symbol'].split('/')[0 if x['side'] == 'sell' else 1],
                x['remaining'] if x['side'] == 'sell' else x['remaining'] * x['price'])
                for x in orders
                if x['remaining'] is not None and (x['side'] == 'sell' or x['price'] is not None)
            ]
            for bal in balances:
                if not isinstance(balances[bal], dict):
                    continue
                balances[bal]['used'] = sum((order[1] for order in order_list if order[0] == bal))
                balances[bal]['free'] = balances[bal]['total'] - balances[bal]['used']
            self._log_exchange_response('fetch_balances2', balances)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _set_leverage(self, leverage: float, pair: Optional[str] = None, accept_fail: bool = False) -> None:
        """
        Kraken set's the leverage as an option in the order object, so we need to
        add it to params
        """
        return

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = 'GTC'
    ) -> Dict[str, Any]:
        params = super()._get_params(side=side, ordertype=ordertype, leverage=leverage, reduceOnly=reduceOnly, time_in_force=time_in_force)
        if leverage > 1.0:
            params['leverage'] = round(leverage)
        if time_in_force == 'PO':
            params.pop('timeInForce', None)
            params['postOnly'] = True
        return params

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime,
        time_in_ratio: Optional[float] = None
    ) -> float:
        """
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        """
        if not time_in_ratio:
            raise OperationalException(f'time_in_ratio is required for {self.name}._get_funding_fee')
        fees = 0.0
        if not df.empty:
            df = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df['open_fund'] * df['open_mark'] * amount * time_in_ratio)
        return fees if is_short else -fees

    def _get_trade_pagination_next_value(self, trades: List[Dict[str, Any]]) -> Optional[Union[str, int]]:
        """
        Extract pagination id for the next "from_id" value
        Applies only to fetch_trade_history by id.
        """
        if len(trades) > 0:
            if isinstance(trades[-1].get('info'), list) and len(trades[-1].get('info', [])) > 7:
                return trades[-1].get('info', [])[-1]
            return trades[-1].get('timestamp')
        return None

    def _valid_trade_pagination_id(self, pair: str, from_id: str) -> bool:
        """
        Verify trade-pagination id is valid.
        Workaround for odd Kraken issue where ID is sometimes wrong.
        """
        if len(from_id) >= 19:
            return True
        logger.debug(f'{pair} - trade-pagination id is not valid. Fallback to timestamp.')
        return False
