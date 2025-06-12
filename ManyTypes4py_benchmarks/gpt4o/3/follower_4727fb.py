import abc
import datetime
import os
import pickle
import queue
import re
import threading
import time
from typing import List, Dict, Any, Union
import requests
from easytrader import exceptions
from easytrader.log import logger

class BaseFollower(metaclass=abc.ABCMeta):
    """
    slippage: 滑点，取值范围为 [0, 1]
    """
    LOGIN_PAGE: str = ''
    LOGIN_API: str = ''
    TRANSACTION_API: str = ''
    CMD_CACHE_FILE: str = 'cmd_cache.pk'
    WEB_REFERER: str = ''
    WEB_ORIGIN: str = ''

    def __init__(self) -> None:
        self.trade_queue: queue.Queue = queue.Queue()
        self.expired_cmds: set = set()
        self.s: requests.Session = requests.Session()
        self.s.verify = False
        self.slippage: float = 0.0

    def login(self, user: str = None, password: str = None, **kwargs: Any) -> None:
        headers = self._generate_headers()
        self.s.headers.update(headers)
        self.s.get(self.LOGIN_PAGE)
        params = self.create_login_params(user, password, **kwargs)
        rep = self.s.post(self.LOGIN_API, data=params)
        self.check_login_success(rep)
        logger.info('登录成功')

    def _generate_headers(self) -> Dict[str, str]:
        headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.8',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.100 Safari/537.36',
            'Referer': self.WEB_REFERER,
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': self.WEB_ORIGIN,
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
        }
        return headers

    def check_login_success(self, rep: requests.Response) -> None:
        pass

    def create_login_params(self, user: str, password: str, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def follow(self, users: List[Any], strategies: List[str], track_interval: int = 1, trade_cmd_expire_seconds: int = 120, cmd_cache: bool = True, slippage: float = 0.0, **kwargs: Any) -> None:
        self.slippage = slippage

    def _calculate_price_by_slippage(self, action: str, price: float) -> float:
        if action == 'buy':
            return price * (1 + self.slippage)
        if action == 'sell':
            return price * (1 - self.slippage)
        return price

    def load_expired_cmd_cache(self) -> None:
        if os.path.exists(self.CMD_CACHE_FILE):
            with open(self.CMD_CACHE_FILE, 'rb') as f:
                self.expired_cmds = pickle.load(f)

    def start_trader_thread(self, users: List[Any], trade_cmd_expire_seconds: int, entrust_prop: str = 'limit', send_interval: int = 0) -> None:
        trader = threading.Thread(target=self.trade_worker, args=[users], kwargs={'expire_seconds': trade_cmd_expire_seconds, 'entrust_prop': entrust_prop, 'send_interval': send_interval})
        trader.setDaemon(True)
        trader.start()

    @staticmethod
    def warp_list(value: Union[Any, List[Any]]) -> List[Any]:
        if not isinstance(value, list):
            value = [value]
        return value

    @staticmethod
    def extract_strategy_id(strategy_url: str) -> str:
        pass

    def extract_strategy_name(self, strategy_url: str) -> str:
        pass

    def track_strategy_worker(self, strategy: str, name: str, interval: int = 10, **kwargs: Any) -> None:
        while True:
            try:
                transactions = self.query_strategy_transaction(strategy, **kwargs)
            except Exception as e:
                logger.exception('无法获取策略 %s 调仓信息, 错误: %s, 跳过此次调仓查询', name, e)
                time.sleep(3)
                continue
            for transaction in transactions:
                trade_cmd = {
                    'strategy': strategy,
                    'strategy_name': name,
                    'action': transaction['action'],
                    'stock_code': transaction['stock_code'],
                    'amount': transaction['amount'],
                    'price': transaction['price'],
                    'datetime': transaction['datetime']
                }
                if self.is_cmd_expired(trade_cmd):
                    continue
                logger.info('策略 [%s] 发送指令到交易队列, 股票: %s 动作: %s 数量: %s 价格: %s 信号产生时间: %s', name, trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], trade_cmd['price'], trade_cmd['datetime'])
                self.trade_queue.put(trade_cmd)
                self.add_cmd_to_expired_cmds(trade_cmd)
            try:
                for _ in range(interval):
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info('程序退出')
                break

    @staticmethod
    def generate_expired_cmd_key(cmd: Dict[str, Any]) -> str:
        return '{}_{}_{}_{}_{}_{}'.format(cmd['strategy_name'], cmd['stock_code'], cmd['action'], cmd['amount'], cmd['price'], cmd['datetime'])

    def is_cmd_expired(self, cmd: Dict[str, Any]) -> bool:
        key = self.generate_expired_cmd_key(cmd)
        return key in self.expired_cmds

    def add_cmd_to_expired_cmds(self, cmd: Dict[str, Any]) -> None:
        key = self.generate_expired_cmd_key(cmd)
        self.expired_cmds.add(key)
        with open(self.CMD_CACHE_FILE, 'wb') as f:
            pickle.dump(self.expired_cmds, f)

    @staticmethod
    def _is_number(s: Any) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _execute_trade_cmd(self, trade_cmd: Dict[str, Any], users: List[Any], expire_seconds: int, entrust_prop: str, send_interval: int) -> None:
        for user in users:
            now = datetime.datetime.now()
            expire = (now - trade_cmd['datetime']).total_seconds()
            if expire > expire_seconds:
                logger.warning('策略 [%s] 指令(股票: %s 动作: %s 数量: %s 价格: %s)超时，指令产生时间: %s 当前时间: %s, 超过设置的最大过期时间 %s 秒, 被丢弃', trade_cmd['strategy_name'], trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], trade_cmd['price'], trade_cmd['datetime'], now, expire_seconds)
                break
            price = trade_cmd['price']
            if not self._is_number(price) or price <= 0:
                logger.warning('策略 [%s] 指令(股票: %s 动作: %s 数量: %s 价格: %s)超时，指令产生时间: %s 当前时间: %s, 价格无效 , 被丢弃', trade_cmd['strategy_name'], trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], trade_cmd['price'], trade_cmd['datetime'], now)
                break
            if trade_cmd['amount'] <= 0:
                logger.warning('策略 [%s] 指令(股票: %s 动作: %s 数量: %s 价格: %s)超时，指令产生时间: %s 当前时间: %s, 买入股数无效 , 被丢弃', trade_cmd['strategy_name'], trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], trade_cmd['price'], trade_cmd['datetime'], now)
                break
            actual_price = self._calculate_price_by_slippage(trade_cmd['action'], trade_cmd['price'])
            args = {'security': trade_cmd['stock_code'], 'price': actual_price, 'amount': trade_cmd['amount'], 'entrust_prop': entrust_prop}
            try:
                response = getattr(user, trade_cmd['action'])(**args)
            except exceptions.TradeError as e:
                trader_name = type(user).__name__
                err_msg = '{}: {}'.format(type(e).__name__, e.args)
                logger.error('%s 执行 策略 [%s] 指令(股票: %s 动作: %s 数量: %s 价格(考虑滑点): %s 指令产生时间: %s) 失败, 错误信息: %s', trader_name, trade_cmd['strategy_name'], trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], actual_price, trade_cmd['datetime'], err_msg)
            else:
                logger.info('策略 [%s] 指令(股票: %s 动作: %s 数量: %s 价格(考虑滑点): %s 指令产生时间: %s) 执行成功, 返回: %s', trade_cmd['strategy_name'], trade_cmd['stock_code'], trade_cmd['action'], trade_cmd['amount'], actual_price, trade_cmd['datetime'], response)

    def trade_worker(self, users: List[Any], expire_seconds: int = 120, entrust_prop: str = 'limit', send_interval: int = 0) -> None:
        while True:
            trade_cmd = self.trade_queue.get()
            self._execute_trade_cmd(trade_cmd, users, expire_seconds, entrust_prop, send_interval)
            time.sleep(send_interval)

    def query_strategy_transaction(self, strategy: str, **kwargs: Any) -> List[Dict[str, Any]]:
        params = self.create_query_transaction_params(strategy)
        rep = self.s.get(self.TRANSACTION_API, params=params)
        history = rep.json()
        transactions = self.extract_transactions(history)
        self.project_transactions(transactions, **kwargs)
        return self.order_transactions_sell_first(transactions)

    def extract_transactions(self, history: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def create_query_transaction_params(self, strategy: str) -> Dict[str, Any]:
        return {}

    @staticmethod
    def re_find(pattern: str, string: str, dtype: type = str) -> Any:
        return dtype(re.search(pattern, string).group())

    @staticmethod
    def re_search(pattern: str, string: str, dtype: type = str) -> Any:
        return dtype(re.search(pattern, string).group(1))

    def project_transactions(self, transactions: List[Dict[str, Any]], **kwargs: Any) -> None:
        pass

    def order_transactions_sell_first(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sell_first_transactions: List[Dict[str, Any]] = []
        for transaction in transactions:
            if transaction['action'] == 'sell':
                sell_first_transactions.insert(0, transaction)
            else:
                sell_first_transactions.append(transaction)
        return sell_first_transactions
