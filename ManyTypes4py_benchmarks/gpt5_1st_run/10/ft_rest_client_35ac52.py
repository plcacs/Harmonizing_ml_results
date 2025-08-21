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

logger: logging.Logger = logging.getLogger('ft_rest_client')

ParamsT = dict[str, Any] | None
PostDataT = dict[str, Any] | list[dict[str, Any]] | None
JSONType = dict[str, Any] | list[Any] | None


class FtRestClient:
    _serverurl: str
    _session: requests.Session
    _timeout: float

    def __init__(
        self,
        serverurl: str,
        username: str | None = None,
        password: str | None = None,
        *,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        timeout: float = 10,
    ) -> None:
        self._serverurl = serverurl
        self._session = requests.Session()
        self._timeout = timeout
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount('http://', adapter)
        if username and password:
            self._session.auth = (username, password)

    def _call(
        self,
        method: str,
        apipath: str,
        params: ParamsT = None,
        data: PostDataT = None,
        files: dict[str, Any] | None = None,
    ) -> JSONType:
        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f'invalid method <{method}>')
        basepath = f'{self._serverurl}/api/v1/{apipath}'
        hd = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        query = urlencode(params) if params else ''
        url = urlunparse((schema, netloc, path, par, query, fragment))
        try:
            resp = self._session.request(
                method,
                url,
                headers=hd,
                timeout=self._timeout,
                data=json.dumps(data),
                files=files,
            )
            return resp.json()
        except RequestConnectionError:
            logger.warning(f'Connection error - could not connect to {netloc}.')
            return None

    def _get(self, apipath: str, params: ParamsT = None) -> JSONType:
        return self._call('GET', apipath, params=params)

    def _delete(self, apipath: str, params: ParamsT = None) -> JSONType:
        return self._call('DELETE', apipath, params=params)

    def _post(self, apipath: str, params: ParamsT = None, data: PostDataT = None) -> JSONType:
        return self._call('POST', apipath, params=params, data=data)

    def start(self) -> JSONType:
        """Start the bot if it's in the stopped state.

        :return: json object
        """
        return self._post('start')

    def stop(self) -> JSONType:
        """Stop the bot. Use `start` to restart.

        :return: json object
        """
        return self._post('stop')

    def stopbuy(self) -> JSONType:
        """Stop buying (but handle sells gracefully). Use `reload_config` to reset.

        :return: json object
        """
        return self._post('stopbuy')

    def reload_config(self) -> JSONType:
        """Reload configuration.

        :return: json object
        """
        return self._post('reload_config')

    def balance(self) -> JSONType:
        """Get the account balance.

        :return: json object
        """
        return self._get('balance')

    def count(self) -> JSONType:
        """Return the amount of open trades.

        :return: json object
        """
        return self._get('count')

    def entries(self, pair: str | None = None) -> JSONType:
        """Returns List of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('entries', params={'pair': pair} if pair else None)

    def exits(self, pair: str | None = None) -> JSONType:
        """Returns List of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('exits', params={'pair': pair} if pair else None)

    def mix_tags(self, pair: str | None = None) -> JSONType:
        """Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific pair provided

        :return: json object
        """
        return self._get('mix_tags', params={'pair': pair} if pair else None)

    def locks(self) -> JSONType:
        """Return current locks

        :return: json object
        """
        return self._get('locks')

    def delete_lock(self, lock_id: int) -> JSONType:
        """Delete (disable) lock from the database.

        :param lock_id: ID for the lock to delete
        :return: json object
        """
        return self._delete(f'locks/{lock_id}')

    def lock_add(self, pair: str, until: str, side: str = '*', reason: str = '') -> JSONType:
        """Lock pair

        :param pair: Pair to lock
        :param until: Lock until this date (format "2024-03-30 16:00:00Z")
        :param side: Side to lock (long, short, *)
        :param reason: Reason for the lock
        :return: json object
        """
        data = [{'pair': pair, 'until': until, 'side': side, 'reason': reason}]
        return self._post('locks', data=data)

    def daily(self, days: int | None = None) -> JSONType:
        """Return the profits for each day, and amount of trades.

        :return: json object
        """
        return self._get('daily', params={'timescale': days} if days else None)

    def weekly(self, weeks: int | None = None) -> JSONType:
        """Return the profits for each week, and amount of trades.

        :return: json object
        """
        return self._get('weekly', params={'timescale': weeks} if weeks else None)

    def monthly(self, months: int | None = None) -> JSONType:
        """Return the profits for each month, and amount of trades.

        :return: json object
        """
        return self._get('monthly', params={'timescale': months} if months else None)

    def edge(self) -> JSONType:
        """Return information about edge.

        :return: json object
        """
        return self._get('edge')

    def profit(self) -> JSONType:
        """Return the profit summary.

        :return: json object
        """
        return self._get('profit')

    def stats(self) -> JSONType:
        """Return the stats report (durations, sell-reasons).

        :return: json object
        """
        return self._get('stats')

    def performance(self) -> JSONType:
        """Return the performance of the different coins.

        :return: json object
        """
        return self._get('performance')

    def status(self) -> JSONType:
        """Get the status of open trades.

        :return: json object
        """
        return self._get('status')

    def version(self) -> JSONType:
        """Return the version of the bot.

        :return: json object containing the version
        """
        return self._get('version')

    def show_config(self) -> JSONType:
        """Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version
        """
        return self._get('show_config')

    def ping(self) -> dict[str, str]:
        """simple ping"""
        configstatus = self.show_config()
        if not configstatus:
            return {'status': 'not_running'}
        elif isinstance(configstatus, dict) and configstatus.get('state') == 'running':
            return {'status': 'pong'}
        else:
            return {'status': 'not_running'}

    def logs(self, limit: int | None = None) -> JSONType:
        """Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.
        :return: json object
        """
        return self._get('logs', params={'limit': limit} if limit else {})

    def trades(self, limit: int | None = None, offset: int | None = None) -> JSONType:
        """Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.
        :return: json object
        """
        params: dict[str, Any] = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get('trades', params)

    def trade(self, trade_id: int) -> JSONType:
        """Return specific trade

        :param trade_id: Specify which trade to get.
        :return: json object
        """
        return self._get(f'trade/{trade_id}')

    def delete_trade(self, trade_id: int) -> JSONType:
        """Delete trade from the database.
        Tries to close open orders. Requires manual handling of this asset on the exchange.

        :param trade_id: Deletes the trade with this ID from the database.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}')

    def cancel_open_order(self, trade_id: int) -> JSONType:
        """Cancel open order for trade.

        :param trade_id: Cancels open orders for this trade.
        :return: json object
        """
        return self._delete(f'trades/{trade_id}/open-order')

    def whitelist(self) -> JSONType:
        """Show the current whitelist.

        :return: json object
        """
        return self._get('whitelist')

    def blacklist(self, *args: str) -> JSONType:
        """Show the current blacklist.

        :param add: List of coins to add (example: "BNB/BTC")
        :return: json object
        """
        if not args:
            return self._get('blacklist')
        else:
            return self._post('blacklist', data={'blacklist': args})

    def forcebuy(self, pair: str, price: float | None = None) -> JSONType:
        """Buy an asset.

        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :return: json object of the trade
        """
        data: dict[str, Any] = {'pair': pair, 'price': price}
        return self._post('forcebuy', data=data)

    def forceenter(
        self,
        pair: str,
        side: str,
        price: float | None = None,
        *,
        order_type: str | None = None,
        stake_amount: float | None = None,
        leverage: float | None = None,
        enter_tag: str | None = None,
    ) -> JSONType:
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
        data: dict[str, Any] = {'pair': pair, 'side': side}
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

    def forceexit(self, tradeid: int, ordertype: str | None = None, amount: float | None = None) -> JSONType:
        """Force-exit a trade.

        :param tradeid: Id of the trade (can be received via status command)
        :param ordertype: Order type to use (must be market or limit)
        :param amount: Amount to sell. Full sell if not given
        :return: json object
        """
        return self._post('forceexit', data={'tradeid': tradeid, 'ordertype': ordertype, 'amount': amount})

    def strategies(self) -> JSONType:
        """Lists available strategies

        :return: json object
        """
        return self._get('strategies')

    def strategy(self, strategy: str) -> JSONType:
        """Get strategy details

        :param strategy: Strategy class name
        :return: json object
        """
        return self._get(f'strategy/{strategy}')

    def pairlists_available(self) -> JSONType:
        """Lists available pairlist providers

        :return: json object
        """
        return self._get('pairlists/available')

    def plot_config(self) -> JSONType:
        """Return plot configuration if the strategy defines one.

        :return: json object
        """
        return self._get('plot_config')

    def available_pairs(self, timeframe: str | None = None, stake_currency: str | None = None) -> JSONType:
        """Return available pair (backtest data) based on timeframe / stake_currency selection

        :param timeframe: Only pairs with this timeframe available.
        :param stake_currency: Only pairs that include this timeframe
        :return: json object
        """
        return self._get(
            'available_pairs',
            params={
                'stake_currency': stake_currency if timeframe else '',
                'timeframe': timeframe if timeframe else '',
            },
        )

    def pair_candles(
        self,
        pair: str,
        timeframe: str,
        limit: int | None = None,
        columns: list[str] | None = None,
    ) -> JSONType:
        """Return live dataframe for <pair><timeframe>.

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param limit: Limit result to the last n candles.
        :param columns: List of dataframe columns to return. Empty list will return OHLCV.
        :return: json object
        """
        params: dict[str, Any] = {'pair': pair, 'timeframe': timeframe}
        if limit:
            params['limit'] = limit
        if columns is not None:
            params['columns'] = columns
            return self._post('pair_candles', data=params)
        return self._get('pair_candles', params=params)

    def pair_history(
        self,
        pair: str,
        timeframe: str,
        strategy: str,
        timerange: str | None = None,
        freqaimodel: str | None = None,
    ) -> JSONType:
        """Return historic, analyzed dataframe

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param strategy: Strategy to analyze and get values for
        :param freqaimodel: FreqAI model to use for analysis
        :param timerange: Timerange to get data for (same format than --timerange endpoints)
        :return: json object
        """
        return self._get(
            'pair_history',
            params={
                'pair': pair,
                'timeframe': timeframe,
                'strategy': strategy,
                'freqaimodel': freqaimodel,
                'timerange': timerange if timerange else '',
            },
        )

    def sysinfo(self) -> JSONType:
        """Provides system information (CPU, RAM usage)

        :return: json object
        """
        return self._get('sysinfo')

    def health(self) -> JSONType:
        """Provides a quick health check of the running bot.

        :return: json object
        """
        return self._get('health')