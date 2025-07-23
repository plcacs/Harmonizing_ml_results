import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, urlparse, urlunparse
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError

logger = logging.getLogger('ft_rest_client')
ParamsT = Optional[Dict[str, Any]]
PostDataT = Union[Dict[str, Any], List[Dict[str, Any]], None]

class FtRestClient:

    def __init__(self, serverurl: str, username: Optional[str] = None, password: Optional[str] = None, *,
                 pool_connections: int = 10, pool_maxsize: int = 10, timeout: int = 10) -> None:
        self._serverurl = serverurl
        self._session = requests.Session()
        self._timeout = timeout
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount('http://', adapter)
        if username and password:
            self._session.auth = (username, password)

    def _call(self, method: str, apipath: str, params: ParamsT = None, data: PostDataT = None, files: Any = None) -> Any:
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

    def _get(self, apipath: str, params: ParamsT = None) -> Any:
        return self._call('GET', apipath, params=params)

    def _delete(self, apipath: str, params: ParamsT = None) -> Any:
        return self._call('DELETE', apipath, params=params)

    def _post(self, apipath: str, params: ParamsT = None, data: PostDataT = None) -> Any:
        return self._call('POST', apipath, params=params, data=data)

    def start(self) -> Any:
        return self._post('start')

    def stop(self) -> Any:
        return self._post('stop')

    def stopbuy(self) -> Any:
        return self._post('stopbuy')

    def reload_config(self) -> Any:
        return self._post('reload_config')

    def balance(self) -> Any:
        return self._get('balance')

    def count(self) -> Any:
        return self._get('count')

    def entries(self, pair: Optional[str] = None) -> Any:
        return self._get('entries', params={'pair': pair} if pair else None)

    def exits(self, pair: Optional[str] = None) -> Any:
        return self._get('exits', params={'pair': pair} if pair else None)

    def mix_tags(self, pair: Optional[str] = None) -> Any:
        return self._get('mix_tags', params={'pair': pair} if pair else None)

    def locks(self) -> Any:
        return self._get('locks')

    def delete_lock(self, lock_id: str) -> Any:
        return self._delete(f'locks/{lock_id}')

    def lock_add(self, pair: str, until: str, side: str = '*', reason: str = '') -> Any:
        data = [{'pair': pair, 'until': until, 'side': side, 'reason': reason}]
        return self._post('locks', data=data)

    def daily(self, days: Optional[int] = None) -> Any:
        return self._get('daily', params={'timescale': days} if days else None)

    def weekly(self, weeks: Optional[int] = None) -> Any:
        return self._get('weekly', params={'timescale': weeks} if weeks else None)

    def monthly(self, months: Optional[int] = None) -> Any:
        return self._get('monthly', params={'timescale': months} if months else None)

    def edge(self) -> Any:
        return self._get('edge')

    def profit(self) -> Any:
        return self._get('profit')

    def stats(self) -> Any:
        return self._get('stats')

    def performance(self) -> Any:
        return self._get('performance')

    def status(self) -> Any:
        return self._get('status')

    def version(self) -> Any:
        return self._get('version')

    def show_config(self) -> Any:
        return self._get('show_config')

    def ping(self) -> Dict[str, str]:
        configstatus = self.show_config()
        if not configstatus:
            return {'status': 'not_running'}
        elif configstatus['state'] == 'running':
            return {'status': 'pong'}
        else:
            return {'status': 'not_running'}

    def logs(self, limit: Optional[int] = None) -> Any:
        return self._get('logs', params={'limit': limit} if limit else {})

    def trades(self, limit: Optional[int] = None, offset: Optional[int] = None) -> Any:
        params = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get('trades', params)

    def trade(self, trade_id: str) -> Any:
        return self._get(f'trade/{trade_id}')

    def delete_trade(self, trade_id: str) -> Any:
        return self._delete(f'trades/{trade_id}')

    def cancel_open_order(self, trade_id: str) -> Any:
        return self._delete(f'trades/{trade_id}/open-order')

    def whitelist(self) -> Any:
        return self._get('whitelist')

    def blacklist(self, *args: str) -> Any:
        if not args:
            return self._get('blacklist')
        else:
            return self._post('blacklist', data={'blacklist': args})

    def forcebuy(self, pair: str, price: Optional[float] = None) -> Any:
        data = {'pair': pair, 'price': price}
        return self._post('forcebuy', data=data)

    def forceenter(self, pair: str, side: str, price: Optional[float] = None, *,
                   order_type: Optional[str] = None, stake_amount: Optional[float] = None,
                   leverage: Optional[float] = None, enter_tag: Optional[str] = None) -> Any:
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

    def forceexit(self, tradeid: str, ordertype: Optional[str] = None, amount: Optional[float] = None) -> Any:
        return self._post('forceexit', data={'tradeid': tradeid, 'ordertype': ordertype, 'amount': amount})

    def strategies(self) -> Any:
        return self._get('strategies')

    def strategy(self, strategy: str) -> Any:
        return self._get(f'strategy/{strategy}')

    def pairlists_available(self) -> Any:
        return self._get('pairlists/available')

    def plot_config(self) -> Any:
        return self._get('plot_config')

    def available_pairs(self, timeframe: Optional[str] = None, stake_currency: Optional[str] = None) -> Any:
        return self._get('available_pairs', params={'stake_currency': stake_currency if timeframe else '', 'timeframe': timeframe if timeframe else ''})

    def pair_candles(self, pair: str, timeframe: str, limit: Optional[int] = None, columns: Optional[List[str]] = None) -> Any:
        params = {'pair': pair, 'timeframe': timeframe}
        if limit:
            params['limit'] = limit
        if columns is not None:
            params['columns'] = columns
            return self._post('pair_candles', data=params)
        return self._get('pair_candles', params=params)

    def pair_history(self, pair: str, timeframe: str, strategy: str, timerange: Optional[str] = None, freqaimodel: Optional[str] = None) -> Any:
        return self._get('pair_history', params={'pair': pair, 'timeframe': timeframe, 'strategy': strategy, 'freqaimodel': freqaimodel, 'timerange': timerange if timerange else ''})

    def sysinfo(self) -> Any:
        return self._get('sysinfo')

    def health(self) -> Any:
        return self._get('health')
