"""
A Rest Client for Freqtrade bot

Should not import anything from freqtrade,
so it can be used as a standalone script, and can be installed independently.
"""
import json
import logging
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError
logger = logging.getLogger('ft_rest_client')
ParamsT = dict[str, Any] | None
PostDataT = dict[str, Any] | list[dict[str, Any]] | None

class FtRestClient:

    def __init__(self, serverurl: Union[bool, str, None], username: Union[None, str, int]=None, password: Union[None, str, int]=None, *, pool_connections: int=10, pool_maxsize: int=10, timeout: int=10) -> None:
        self._serverurl = serverurl
        self._session = requests.Session()
        self._timeout = timeout
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount('http://', adapter)
        if username and password:
            self._session.auth = (username, password)

    def _call(self, method: Union[str, dict, None], apipath: Union[str, dict[str, typing.Any]], params: Union[None, str, dict[str, str], dict]=None, data: Union[None, str, dict]=None, files: Union[None, str, typing.Iterable[list]]=None) -> Union[dict, requests.Response, typing.Mapping]:
        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f'invalid method <{method}>')
        basepath = f'{self._serverurl}/api/v1/{apipath}'
        hd = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        query = urlencode(params) if params else ''
        url = urlunparse((schema, netloc, path, par, query, fragment))
        try:
            resp = self._session.request(method, url, headers=hd, timeout=self._timeout, data=json.dumps(data))
            return resp.json()
        except RequestConnectionError:
            logger.warning(f'Connection error - could not connect to {netloc}.')

    def _get(self, apipath: Union[dict, list, dict[int, list[str]]], params: Union[None, dict, list, dict[int, list[str]]]=None) -> Union[str, dict[str, dict[str, str]], typing.Callable[KT,VT, bool]]:
        return self._call('GET', apipath, params=params)

    def _delete(self, apipath: Union[dict[str, typing.Any], dict, typing.Mapping], params: Union[None, dict[str, typing.Any], dict, typing.Mapping]=None) -> Union[bool, dict[str, typing.Any], str]:
        return self._call('DELETE', apipath, params=params)

    def _post(self, apipath: Union[dict, dict[str, typing.Any]], params: Union[None, dict, dict[str, typing.Any]]=None, data: Union[None, dict, dict[str, typing.Any]]=None) -> Union[str, dict, typing.Callable]:
        return self._call('POST', apipath, params=params, data=data)

    def start(self) -> Union[str, int, list[dict[str, str]]]:
        """Start the bot if it's in the stopped state.

        :return: json object
        """
        return self._post('start')

    def stop(self) -> Union[str, int]:
        """Stop the bot. Use `start` to restart.

        :return: json object
        """
        return self._post('stop')

    def stopbuy(self) -> str:
        """Stop buying (but handle sells gracefully). Use `reload_config` to reset.

        :return: json object
        """
        return self._post('stopbuy')

    def reload_config(self) -> Union[dict, str]:
        """Reload configuration.

        :return: json object
        """
        return self._post('reload_config')

    def balance(self) -> Union[int, dict, str, None]:
        """Get the account balance.

        :return: json object
        """
        return self._get('balance')

    def count(self) -> Union[int, str, bytes]:
        """Return the amount of open trades.

        :return: json object
        """
        return self._get('count')

    def entries(self, pair: Union[None, dict, list[str], dict[str, str]]=None) -> Union[dict, dict[str, typing.Any], int]:
        """Returns List of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('entries', params={'pair': pair} if pair else None)

    def exits(self, pair: Union[None, list[str], list[dict], str]=None) -> typing.IO:
        """Returns List of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('exits', params={'pair': pair} if pair else None)

    def mix_tags(self, pair: Union[None, list[str], tuple[list[str]], str]=None) -> Union[bool, dict[str, typing.Any]]:
        """Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('mix_tags', params={'pair': pair} if pair else None)

    def locks(self) -> Union[bool, str]:
        """Return current locks

        :return: json object
        """
        return self._get('locks')

    def delete_lock(self, lock_id: str) -> Union[bool, str]:
        """Delete (disable) lock from the database.

        :param lock_id: ID for the lock to delete
        :return: json object
        """
        return self._delete(f'locks/{lock_id}')

    def lock_add(self, pair: Union[str, int, float], until: Union[str, int, float], side: typing.Text='*', reason: typing.Text='') -> Union[str, bytes, dict[str, str]]:
        """Lock pair

        :param pair: Pair to lock
        :param until: Lock until this date (format "2024-03-30 16:00:00Z")
        :param side: Side to lock (long, short, *)
        :param reason: Reason for the lock
        :return: json object
        """
        data = [{'pair': pair, 'until': until, 'side': side, 'reason': reason}]
        return self._post('locks', data=data)

    def daily(self, days: Union[None, int]=None) -> Union[int, float]:
        """Return the profits for each day, and amount of trades.

        :return: json object
        """
        return self._get('daily', params={'timescale': days} if days else None)

    def weekly(self, weeks: Union[None, int, str]=None) -> Union[str, list[str], float, None]:
        """Return the profits for each week, and amount of trades.

        :return: json object
        """
        return self._get('weekly', params={'timescale': weeks} if weeks else None)

    def monthly(self, months: Union[None, int]=None) -> Union[str, dict[int, dict[int, typing.Any]], dict]:
        """Return the profits for each month, and amount of trades.

        :return: json object
        """
        return self._get('monthly', params={'timescale': months} if months else None)

    def edge(self) -> Union[list, bool]:
        """Return information about edge.

        :return: json object
        """
        return self._get('edge')

    def profit(self) -> Union[str, float, None]:
        """Return the profit summary.

        :return: json object
        """
        return self._get('profit')

    def stats(self) -> Union[str, bool, list[str]]:
        """Return the stats report (durations, sell-reasons).

        :return: json object
        """
        return self._get('stats')

    def performance(self) -> Union[float, str]:
        """Return the performance of the different coins.

        :return: json object
        """
        return self._get('performance')

    def status(self) -> Union[int, str, dict, None]:
        """Get the status of open trades.

        :return: json object
        """
        return self._get('status')

    def version(self) -> Union[str, None]:
        """Return the version of the bot.

        :return: json object containing the version
        """
        return self._get('version')

    def show_config(self) -> Union[str, None, bool]:
        """Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version
        """
        return self._get('show_config')

    def ping(self) -> dict[typing.Text, typing.Text]:
        """simple ping"""
        configstatus = self.show_config()
        if not configstatus:
            return {'status': 'not_running'}
        elif configstatus['state'] == 'running':
            return {'status': 'pong'}
        else:
            return {'status': 'not_running'}

    def logs(self, limit: Union[None, int]=None) -> int:
        """Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.
        :return: json object
        """
        return self._get('logs', params={'limit': limit} if limit else {})

    def trades(self, limit: Union[None, int]=None, offset: Union[None, int]=None) -> Union[str, dict[str, typing.Any], dict[tuple[typing.Union[typing.Any,str]], int]]:
        """Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.
        :return: json object
        """
        params = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get('trades', params)

    def trade(self, trade_id: str) -> Union[str, tuple[str]]:
        """Return specific trade

        :param trade_id: Specify which trade to get.
        :return: json object
        """
        return self._get(f'trade/{trade_id}')

    def delete_trade(self, trade_id: str) -> Union[str, tuple[str], None]:
        """Delete trade from the database.
        Tries to close open orders. Requires manual handling of this asset on the exchange.

        :param trade_id: Deletes the trade with this ID from the database.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}')

    def cancel_open_order(self, trade_id: str) -> Union[dict, str, list]:
        """Cancel open order for trade.

        :param trade_id: Cancels open orders for this trade.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}/open-order')

    def whitelist(self) -> Union[str, typing.Callable[dict, None], None]:
        """Show the current whitelist.

        :return: json object
        """
        return self._get('whitelist')

    def blacklist(self, *args) -> Union[str, set, list[str]]:
        """Show the current blacklist.

        :param add: List of coins to add (example: "BNB/BTC")
        :return: json object
        """
        if not args:
            return self._get('blacklist')
        else:
            return self._post('blacklist', data={'blacklist': args})

    def forcebuy(self, pair: Union[float, list[int], int, None], price: Union[None, float, list[int], int]=None) -> Union[str, typing.BinaryIO, logging.LogRecord]:
        """Buy an asset.

        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :return: json object of the trade
        """
        data = {'pair': pair, 'price': price}
        return self._post('forcebuy', data=data)

    def forceenter(self, pair: Union[str, int, float], side: Union[str, int, float], price: Union[None, int, float, str]=None, *, order_type: Union[None, str]=None, stake_amount: Union[None, int, list["SubRate"]]=None, leverage: Union[None, str, list[str], int]=None, enter_tag: Union[None, int, str]=None) -> Union[str, dict, list[str]]:
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
        data = {'pair': pair, 'side': side}
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

    def forceexit(self, tradeid: Union[float, int, str], ordertype: Union[None, float, int, str]=None, amount: Union[None, float, int, str]=None) -> Union[str, bytes]:
        """Force-exit a trade.

        :param tradeid: Id of the trade (can be received via status command)
        :param ordertype: Order type to use (must be market or limit)
        :param amount: Amount to sell. Full sell if not given
        :return: json object
        """
        return self._post('forceexit', data={'tradeid': tradeid, 'ordertype': ordertype, 'amount': amount})

    def strategies(self) -> Union[typing.Sequence[typing.Any], dict[str, typing.Any], dict]:
        """Lists available strategies

        :return: json object
        """
        return self._get('strategies')

    def strategy(self, strategy: Union[typing.Callable, T]) -> Union[str, typing.Callable]:
        """Get strategy details

        :param strategy: Strategy class name
        :return: json object
        """
        return self._get(f'strategy/{strategy}')

    def pairlists_available(self) -> Union[bool, str, typing.Callable[typing.Any, None], None]:
        """Lists available pairlist providers

        :return: json object
        """
        return self._get('pairlists/available')

    def plot_config(self):
        """Return plot configuration if the strategy defines one.

        :return: json object
        """
        return self._get('plot_config')

    def available_pairs(self, timeframe: Union[None, str, list]=None, stake_currency: Union[None, str, list]=None) -> Union[tuple[typing.Union[bool,requests.Response]], str, dict[str, typing.Any]]:
        """Return available pair (backtest data) based on timeframe / stake_currency selection

        :param timeframe: Only pairs with this timeframe available.
        :param stake_currency: Only pairs that include this timeframe
        :return: json object
        """
        return self._get('available_pairs', params={'stake_currency': stake_currency if timeframe else '', 'timeframe': timeframe if timeframe else ''})

    def pair_candles(self, pair: Union[str, list[typing.Union[str,typing.Any]], float], timeframe: Union[str, list[typing.Union[str,typing.Any]], float], limit: Union[None, str, int, typing.MutableSet]=None, columns: Union[None, str]=None) -> Union[str, dict[str, typing.Any], None]:
        """Return live dataframe for <pair><timeframe>.

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param limit: Limit result to the last n candles.
        :param columns: List of dataframe columns to return. Empty list will return OHLCV.
        :return: json object
        """
        params = {'pair': pair, 'timeframe': timeframe}
        if limit:
            params['limit'] = limit
        if columns is not None:
            params['columns'] = columns
            return self._post('pair_candles', data=params)
        return self._get('pair_candles', params=params)

    def pair_history(self, pair: Union[str, dict[str, typing.Any]], timeframe: Union[str, dict[str, typing.Any]], strategy: Union[str, dict[str, typing.Any]], timerange: Union[None, str, dict[str, typing.Any]]=None, freqaimodel: Union[None, str, dict[str, typing.Any]]=None) -> Union[dict, str, dict[int, bytes]]:
        """Return historic, analyzed dataframe

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param strategy: Strategy to analyze and get values for
        :param freqaimodel: FreqAI model to use for analysis
        :param timerange: Timerange to get data for (same format than --timerange endpoints)
        :return: json object
        """
        return self._get('pair_history', params={'pair': pair, 'timeframe': timeframe, 'strategy': strategy, 'freqaimodel': freqaimodel, 'timerange': timerange if timerange else ''})

    def sysinfo(self) -> Union[str, bool, typing.Callable, None]:
        """Provides system information (CPU, RAM usage)

        :return: json object
        """
        return self._get('sysinfo')

    def health(self) -> Union[str, bool]:
        """Provides a quick health check of the running bot.

        :return: json object
        """
        return self._get('health')