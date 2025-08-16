import json
import logging
from typing import Any, Union, List, Dict
from urllib.parse import urlencode, urlparse, urlunparse
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError

logger: logging.Logger = logging.getLogger('ft_rest_client')
ParamsT: Union[Dict[str, Any], None]
PostDataT: Union[Dict[str, Any], List[Dict[str, Any]], None]

class FtRestClient:
    def __init__(self, serverurl: str, username: str = None, password: str = None, *, pool_connections: int = 10, pool_maxsize: int = 10, timeout: int = 10) -> None:
    def _call(self, method: str, apipath: str, params: ParamsT = None, data: PostDataT = None, files: Any = None) -> Any:
    def _get(self, apipath: str, params: ParamsT = None) -> Any:
    def _delete(self, apipath: str, params: ParamsT = None) -> Any:
    def _post(self, apipath: str, params: ParamsT = None, data: PostDataT = None) -> Any:
    def start(self) -> Any:
    def stop(self) -> Any:
    def stopbuy(self) -> Any:
    def reload_config(self) -> Any:
    def balance(self) -> Any:
    def count(self) -> Any:
    def entries(self, pair: str = None) -> Any:
    def exits(self, pair: str = None) -> Any:
    def mix_tags(self, pair: str = None) -> Any:
    def locks(self) -> Any:
    def delete_lock(self, lock_id: str) -> Any:
    def lock_add(self, pair: str, until: str, side: str = '*', reason: str = '') -> Any:
    def daily(self, days: int = None) -> Any:
    def weekly(self, weeks: int = None) -> Any:
    def monthly(self, months: int = None) -> Any:
    def edge(self) -> Any:
    def profit(self) -> Any:
    def stats(self) -> Any:
    def performance(self) -> Any:
    def status(self) -> Any:
    def version(self) -> Any:
    def show_config(self) -> Any:
    def ping(self) -> Any:
    def logs(self, limit: int = None) -> Any:
    def trades(self, limit: int = None, offset: int = None) -> Any:
    def trade(self, trade_id: str) -> Any:
    def delete_trade(self, trade_id: str) -> Any:
    def cancel_open_order(self, trade_id: str) -> Any:
    def whitelist(self) -> Any:
    def blacklist(self, *args: str) -> Any:
    def forcebuy(self, pair: str, price: float = None) -> Any:
    def forceenter(self, pair: str, side: str, price: float = None, order_type: str = None, stake_amount: float = None, leverage: float = None, enter_tag: str = None) -> Any:
    def forceexit(self, tradeid: str, ordertype: str = None, amount: float = None) -> Any:
    def strategies(self) -> Any:
    def strategy(self, strategy: str) -> Any:
    def pairlists_available(self) -> Any:
    def plot_config(self) -> Any:
    def available_pairs(self, timeframe: str = None, stake_currency: str = None) -> Any:
    def pair_candles(self, pair: str, timeframe: str, limit: int = None, columns: List[str] = None) -> Any:
    def pair_history(self, pair: str, timeframe: str, strategy: str, timerange: str = None, freqaimodel: str = None) -> Any:
    def sysinfo(self) -> Any:
    def health(self) -> Any:
