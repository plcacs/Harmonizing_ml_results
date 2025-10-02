from datetime import datetime
from threading import Thread
from typing import Any, Dict, List, Union
from easytrader import exceptions
from easytrader.follower import BaseFollower
from easytrader.log import logger


class JoinQuantFollower(BaseFollower):
    LOGIN_PAGE: str = 'https://www.joinquant.com'
    LOGIN_API: str = 'https://www.joinquant.com/user/login/doLogin?ajax=1'
    TRANSACTION_API: str = 'https://www.joinquant.com/algorithm/live/transactionDetail'
    WEB_REFERER: str = 'https://www.joinquant.com/user/login/index'
    WEB_ORIGIN: str = 'https://www.joinquant.com'

    def create_login_params(self, user: str, password: str, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {'CyLoginForm[username]': user, 'CyLoginForm[pwd]': password, 'ajax': 1}
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
        """跟踪joinquant对应的模拟交易，支持多用户多策略

        :param users: 支持easytrader的用户对象，支持使用 [] 指定多个用户
        :param strategies: joinquant 的模拟交易地址，支持使用 [] 指定多个模拟交易,
            地址类似 https://www.joinquant.com/algorithm/live/index?backtestId=xxx
        :param track_interval: 轮训模拟交易时间，单位为秒
        :param trade_cmd_expire_seconds: 交易指令过期时间, 单位为秒
        :param cmd_cache: 是否读取存储历史执行过的指令，防止重启时重复执行已经交易过的指令
        :param entrust_prop: 委托方式, 'limit' 为限价，'market' 为市价, 仅在银河实现
        :param send_interval: 交易发送间隔， 默认为0s。调大可防止卖出买入时卖出单没有及时成交导致的买入金额不足
        """
        users = self.warp_list(users)
        strategies = self.warp_list(strategies)
        if cmd_cache:
            self.load_expired_cmd_cache()
        self.start_trader_thread(users, trade_cmd_expire_seconds, entrust_prop, send_interval)
        workers: List[Thread] = []
        for strategy_url in strategies:
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

    def create_query_transaction_params(self, strategy: str) -> Dict[str, Any]:
        today_str: str = datetime.today().strftime('%Y-%m-%d')
        params: Dict[str, Any] = {'backtestId': strategy, 'date': today_str, 'ajax': 1}
        return params

    def extract_transactions(self, history: Dict[str, Any]) -> List[Any]:
        transactions: List[Any] = history['data']['transaction']
        return transactions

    @staticmethod
    def stock_shuffle_to_prefix(stock: str) -> str:
        assert len(stock) == 11, 'stock {} must like 123456.XSHG or 123456.XSHE'.format(stock)
        code: str = stock[:6]
        if stock.find('XSHG') != -1:
            return 'sh' + code
        if stock.find('XSHE') != -1:
            return 'sz' + code
        raise TypeError('not valid stock code: {}'.format(code))

    def project_transactions(self, transactions: List[Dict[str, Any]], **kwargs: Any) -> None:
        for transaction in transactions:
            transaction['amount'] = self.re_find('\\d+', transaction['amount'], dtype=int)
            time_str: str = '{} {}'.format(transaction['date'], transaction['time'])
            transaction['datetime'] = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            stock: str = self.re_find('\\d{6}\\.\\w{4}', transaction['stock'])
            transaction['stock_code'] = self.stock_shuffle_to_prefix(stock)
            transaction['action'] = 'buy' if transaction['transaction'] == '买' else 'sell'