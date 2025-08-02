"""
A Rest Client for Freqtrade bot

Should not import anything from freqtrade,
so it can be used as a standalone script, and can be installed independently.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, urlparse, urlunparse
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError

logger = logging.getLogger('ft_rest_client')
ParamsT = Dict[str, Any] | None
PostDataT = Union[Dict[str, Any], List[Dict[str, Any]], None]

class FtRestClient:

    def __init__(self, serverurl: str, username: Optional[str] = None, password: Optional[str] = None, *, pool_connections: int = 10, pool_maxsize: int = 10, timeout: int = 10) -> None:
        self._serverurl: str = serverurl
        self._session: requests.Session = requests.Session()
        self._timeout: int = timeout
        adapter: HTTPAdapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount('http://', adapter)
        if username and password:
            self._session.auth = (username, password)

    def _call(self, method: str, apipath: str, params: ParamsT = None, data: PostDataT = None, files: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f'invalid method <{method}>')
        basepath: str = f'{self._serverurl}/api/v1/{apipath}'
        hd: Dict[str, str] = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        query = urlencode(params) if params else ''
        url: str = urlunparse((schema, netloc, path, par, query, fragment))
        try:
            resp: requests.Response = self._session.request(method, url, headers=hd, timeout=self._timeout, data=json.dumps(data))
            return resp.json()
        except RequestConnectionError:
            logger.warning(f'Connection error - could not connect to {netloc}.')
            return None

    def _get(self, apipath: str, params: ParamsT = None) -> Optional[Dict[str, Any]]:
        return self._call('GET', apipath, params=params)

    def _delete(self, apipath: str, params: ParamsT = None) -> Optional[Dict[str, Any]]:
        return self._call('DELETE', apipath, params=params)

    def _post(self, apipath: str, params: ParamsT = None, data: PostDataT = None) -> Optional[Dict[str, Any]]:
        return self._call('POST', apipath, params=params, data=data)

    def start(self) -> Optional[Dict[str, Any]]:
        """Start the bot if it's in the stopped state.

        :return: json object
        """
        return self._post('start')

    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop the bot. Use `start` to restart.

        :return: json object
        """
        return self._post('stop')

    def stopbuy(self) -> Optional[Dict[str, Any]]:
        """Stop buying (but handle sells gracefully). Use `reload_config` to reset.

        :return: json object
        """
        return self._post('stopbuy')

    def reload_config(self) -> Optional[Dict[str, Any]]:
        """Reload configuration.

        :return: json object
        """
        return self._post('reload_config')

    def balance(self) -> Optional[Dict[str, Any]]:
        """Get the account balance.

        :return: json object
        """
        return self._get('balance')

    def count(self) -> Optional[Dict[str, Any]]:
        """Return the amount of open trades.

        :return: json object
        """
        return self._get('count')

    def entries(self, pair: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns List of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('entries', params={'pair': pair} if pair else None)

    def exits(self, pair: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns List of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('exits', params={'pair': pair} if pair else None)

    def mix_tags(self, pair: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('mix_tags', params={'pair': pair} if pair else None)

    def locks(self) -> Optional[Dict[str, Any]]:
        """Return current locks

        :return: json object
        """
        return self._get('locks')

    def delete_lock(self, lock_id: int) -> Optional[Dict[str, Any]]:
        """Delete (disable) lock from the database.

        :param lock_id: ID for the lock to delete
        :return: json object
        """
        return self._delete(f'locks/{lock_id}')

    def lock_add(self, pair: str, until: str, side: str = '*', reason: str = '') -> Optional[Dict[str, Any]]:
        """Lock pair

        :param pair: Pair to lock
        :param until: Lock until this date (format "2024-03-30 16:00:00Z")
        :param side: Side to lock (long, short, *)
        :param reason: Reason for the lock
        :return: json object
        """
        data: List[Dict[str, str]] = [{'pair': pair, 'until': until, 'side': side, 'reason': reason}]
        return self._post('locks', data=data)

    def daily(self, days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the profits for each day, and amount of trades.

        :return: json object
        """
        return self._get('daily', params={'timescale': days} if days else None)

    def weekly(self, weeks: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the profits for each week, and amount of trades.

        :return: json object
        """
        return self._get('weekly', params={'timescale': weeks} if weeks else None)

    def monthly(self, months: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the profits for each month, and amount of trades.

        :return: json object
        """
        return self._get('monthly', params={'timescale': months} if months else None)

    def edge(self) -> Optional[Dict[str, Any]]:
        """Return information about edge.

        :return: json object
        """
        return self._get('edge')

    def profit(self) -> Optional[Dict[str, Any]]:
        """Return the profit summary.

        :return: json object
        """
        return self._get('profit')

    def stats(self) -> Optional[Dict[str, Any]]:
        """Return the stats report (durations, sell-reasons).

        :return: json object
        """
        return self._get('stats')

    def performance(self) -> Optional[Dict[str, Any]]:
        """Return the performance of the different coins.

        :return: json object
        """
        return self._get('performance')

    def status(self) -> Optional[Dict[str, Any]]:
        """Get the status of open trades.

        :return: json object
        """
        return self._get('status')

    def version(self) -> Optional[Dict[str, Any]]:
        """Return the version of the bot.

        :return: json object containing the version
        """
        return self._get('version')

    def show_config(self) -> Optional[Dict[str, Any]]:
        """Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version
        """
        return self._get('show_config')

    def ping(self) -> Dict[str, str]:
        """simple ping"""
        configstatus: Optional[Dict[str, Any]] = self.show_config()
        if not configstatus:
            return {'status': 'not_running'}
        elif configstatus['state'] == 'running':
            return {'status': 'pong'}
        else:
            return {'status': 'not_running'}

    def logs(self, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.
        :return: json object
        """
        return self._get('logs', params={'limit': limit} if limit else {})

    def trades(self, limit: Optional[int] = None, offset: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.
        :return: json object
        """
        params: Dict[str, int] = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get('trades', params)

    def trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Return specific trade

        :param trade_id: Specify which trade to get.
        :return: json object
        """
        return self._get(f'trade/{trade_id}')

    def delete_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Delete trade from the database.
        Tries to close open orders. Requires manual handling of this asset on the exchange.

        :param trade_id: Deletes the trade with this ID from the database.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}')

    def cancel_open_order(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Cancel open order for trade.

        :param trade_id: Cancels open orders for this trade.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}/open-order')

    def whitelist(self) -> Optional[Dict[str, Any]]:
        """Show the current whitelist.

        :return: json object
        """
        return self._get('whitelist')

    def blacklist(self, *args: str) -> Optional[Dict[str, Any]]:
        """Show the current blacklist.

        :param add: List of coins to add (example: "BNB/BTC")
        :return: json object
        """
        if not args:
            return self._get('blacklist')
        else:
            return self._post('blacklist', data={'blacklist': args})

    def forcebuy(self, pair: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Buy an asset.

        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :return: json object of the trade
        """
        data: Dict[str, Any] = {'pair': pair, 'price': price}
        return self._post('forcebuy', data=data)

    def forceenter(self, pair: str, side: str, price: Optional[float] = None, *, order_type: Optional[str] = None, stake_amount: Optional[float] = None, leverage: Optional[float] = None, enter_tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Force entering a trade

        :param pair: Pair to buy (ETH/BTC)
        :param side: 'long' or 'short'
        :param price: Optional - price to buy
        :param order_type: Optional keyword argument - 'limit' or 'market'
        :param stake_amount: Optional keyword argument - stake amount (as float)
        :param leverage: Optional keyword argument - leverage (as float)
        :param enter_tag: Optional keyword argument - entry tag (as string, default: 'force_enter')
        :return: json object of the trade
        """
        data: Dict[str, Any] = {'pair': pair, 'side': side}
        if price:
            data['price'] = price
        if order_type:
            data['ordertype'] = order_type
        if stake_amount:
            data['stakeamount'] = stake_amount
        if leverage:
            data['leverage'] = leverage
        if enter_tag:
            data['entry_tag'] = enter_tag
        return self._post('forceenter', data=data)

    def forceexit(self, tradeid: int, ordertype: Optional[str] = None, amount: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Force-exit a trade.

        :param tradeid: Id of the trade (can be received via status command)
        :param ordertype: Order type to use (must be market or limit)
        :param amount: Amount to sell. Full sell if not given
        :return: json object
        """
        return self._post('forceexit', data={'tradeid': tradeid, 'ordertype': ordertype, 'amount': amount})

    def strategies(self) -> Optional[Dict[str, Any]]:
        """Lists available strategies

        :return: json object
        """
        return self._get('strategies')

    def strategy(self, strategy: str) -> Optional[Dict[str, Any]]:
        """Get strategy details

        :param strategy: Strategy class name
        :return: json object
        """
        return self._get(f'strategy/{strategy}')

    def pairlists_available(self) -> Optional[Dict[str, Any]]:
        """Lists available pairlist providers

        :return: json object
        """
        return self._get('pairlists/available')

    def plot_config(self) -> Optional[Dict[str, Any]]:
        """Return plot configuration if the strategy defines one.

        :return: json object
        """
        return self._get('plot_config')

    def available_pairs(self, timeframe: Optional[str] = None, stake_currency: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return available pair (backtest data) based on timeframe / stake_currency selection

        :param timeframe: Only pairs with this timeframe available.
        :param stake_currency: Only pairs that include this timeframe
        :return: json object
        """
        return self._get('available_pairs', params={'stake_currency': stake_currency if timeframe else '', 'timeframe': timeframe if timeframe else ''})

    def pair_candles(self, pair: str, timeframe: str, limit: Optional[int] = None, columns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Return live dataframe for <pair><timeframe>.

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param limit: Limit result to the last n candles.
        :param columns: List of dataframe columns to return. Empty list will return OHLCV.
        :return: json object
        """
        params: Dict[str, Any] = {'pair': pair, 'timeframe': timeframe}
        if limit:
            params['limit'] = limit
        if columns is not None:
            params['columns'] = columns
            return self._post('pair_candles', data=params)
        return self._get('pair_candles', params=params)

    def pair_history(self, pair: str, timeframe: str, strategy: str, timerange: Optional[str] = None, freqaimodel: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return historic, analyzed dataframe

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param strategy: Strategy to analyze and get values for
        :param freqaimodel: FreqAI model to use for analysis
        :param timerange: Timerange to get data for (same format than --timerange endpoints)
        :return: json object
        """
        return self._get('pair_history', params={'pair': pair, 'timeframe': timeframe, 'strategy': strategy, 'freqaimodel': freqaimodel, 'timerange': timerange if timerange else ''})

    def sysinfo(self) -> Optional[Dict[str, Any]]:
        """Provides system information (CPU, RAM usage)

        :return: json object
        """
        return self._get('sysinfo')

    def health(self) -> Optional[Dict[str, Any]]:
        """Provides a quick health check of the running bot.

        :return: json object
        """
        return self._get('health')
