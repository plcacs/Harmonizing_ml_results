from datetime import datetime
from threading import Thread
from typing import Any, Dict, List, Optional, Union
from easytrader import exceptions
from easytrader.follower import BaseFollower
from easytrader.log import logger

class JoinQuantFollower(BaseFollower):
    LOGIN_PAGE: str = 'https://www.joinquant.com'
    LOGIN_API: str = 'https://www.joinquant.com/user/login/doLogin?ajax=1'
    TRANSACTION_API: str = 'https://www.joinquant.com/algorithm/live/transactionDetail'
    WEB_REFERER: str = 'https://www.joinquant.com/user/login/index'
    WEB_ORIGIN: str = 'https://www.joinquant.com'

    def create_login_params(self, user: str, password: str, **kwargs: Any) -> Dict[str, Union[str, int]]:
        params: Dict[str, Union[str, int]] = {'CyLoginForm[username]': user, 'CyLoginForm[pwd]': password, 'ajax': 1}
        return params

    def check_login_success(self, rep: Any) -> None:
        set_cookie: str = rep.headers['set-cookie']
        if len(set_cookie) < 50:
            raise exceptions.NotLoginError('登录失败，请检查用户名和密码')
        self.s.headers.update({'cookie': set_cookie})

    def follow(
        self,
        users: Union[Any, List[Any]],
        strategies: Union[str, List[str]],
        track_interval: int = 1,
        trade_cmd_expire_seconds: int = 120,
        cmd_cache: bool = True,
        entrust_prop: str = 'limit',
        send_interval: int = 0
    ) -> None:
        users_list: List[Any] = self.warp_list(users)
        strategies_list: List[str] = self.warp_list(strategies)
        if cmd_cache:
            self.load_expired_cmd_cache()
        self.start_trader_thread(users_list, trade_cmd_expire_seconds, entrust_prop, send_interval)
        workers: List[Thread] = []
        for strategy_url in strategies_list:
            try:
                strategy_id: str = self.extract_strategy_id(strategy_url)
                strategy_name: str = self.extract_strategy_name(strategy_url)
            except Exception:
                logger.error('抽取交易id和策略名失败, 无效的模拟交易url: %s', strategy_url)
                raise
            strategy_worker: Thread = Thread(
                target=self.track_strategy_worker,
                args=[strategy_id, strategy_name],
                kwargs={'interval': track_interval}
            )
            strategy_worker.start()
            workers.append(strategy_worker)
            logger.info('开始跟踪策略: %s', strategy_name)
        for worker in workers:
            worker.join()

    def extract_strategy_id(self, strategy_url: str) -> str:
        rep: Any = self.s.get(strategy_url)
        return self.re_search('name="backtest\\[backtestId\\]"\\s+?value="(.*?)">', rep.content.decode('utf8'))

    def extract_strategy_name(self, strategy_url: str) -> str:
        rep: Any = self.s.get(strategy_url)
        return self.re_search('class="backtest_name".+?>(.*?)</span>', rep.content.decode('utf8'))

    def create_query_transaction_params(self, strategy: str) -> Dict[str, Union[str, int]]:
        today_str: str = datetime.today().strftime('%Y-%m-%d')
        params: Dict[str, Union[str, int]] = {'backtestId': strategy, 'date': today_str, 'ajax': 1}
        return params

    def extract_transactions(self, history: Dict[str, Any]) -> List[Dict[str, Any]]:
        transactions: List[Dict[str, Any]] = history['data']['transaction']
        return transactions

    @staticmethod
    def stock_shuffle_to_prefix(stock: str) -> str:
        assert len(stock) == 11, 'stock {} must like 123456.XSHG or 123456.XSHE'.format(stock)
        code: str = stock[:6]
        if stock.find('XSHG') != -1:
            return 'sh' + code
        if stock.find('XSHE') != -1:
            return 'sz' + code
        raise TypeError('not valid stock code: {}'.format(stock))

    def project_transactions(self, transactions: List[Dict[str, Any]], **kwargs: Any) -> None:
        for transaction in transactions:
            transaction['amount'] = self.re_find('\\d+', transaction['amount'], dtype=int)
            time_str: str = '{} {}'.format(transaction['date'], transaction['time'])
            transaction['datetime'] = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            stock: str = self.re_find('\\d{6}\\.\\w{4}', transaction['stock'])
            transaction['stock_code'] = self.stock_shuffle_to_prefix(stock)
            transaction['action'] = 'buy' if transaction['transaction'] == '买' else 'sell'
