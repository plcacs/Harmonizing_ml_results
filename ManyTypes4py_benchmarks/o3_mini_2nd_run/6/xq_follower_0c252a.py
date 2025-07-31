from __future__ import division, print_function, unicode_literals
import json
import re
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Dict, List, Optional, Union

from easytrader.follower import BaseFollower
from easytrader.log import logger
from easytrader.utils.misc import parse_cookies_str


class XueQiuFollower(BaseFollower):
    LOGIN_PAGE: str = 'https://www.xueqiu.com'
    LOGIN_API: str = 'https://xueqiu.com/snowman/login'
    TRANSACTION_API: str = 'https://xueqiu.com/cubes/rebalancing/history.json'
    PORTFOLIO_URL: str = 'https://xueqiu.com/p/'
    WEB_REFERER: str = 'https://www.xueqiu.com'

    def __init__(self) -> None:
        super().__init__()
        self._adjust_sell: Optional[bool] = None
        self._users: Optional[List[Any]] = None

    def login(self, user: Optional[Any] = None, password: Optional[str] = None, **kwargs: Any) -> None:
        """
        雪球登陆， 需要设置 cookies
        :param cookies: 雪球登陆需要设置 cookies， 具体见
            https://smalltool.github.io/2016/08/02/cookie/
        :return: None
        """
        cookies = kwargs.get('cookies')
        if cookies is None:
            raise TypeError('雪球登陆需要设置 cookies， 具体见https://smalltool.github.io/2016/08/02/cookie/')
        headers: Dict[str, str] = self._generate_headers()  # type: ignore
        self.s.headers.update(headers)
        self.s.get(self.LOGIN_PAGE)
        cookie_dict: Dict[str, str] = parse_cookies_str(cookies)
        self.s.cookies.update(cookie_dict)
        logger.info('登录成功')

    def follow(self,
               users: Union[Any, List[Any]],
               strategies: Union[str, List[str]],
               total_assets: Union[Number, List[Number]] = 10000,
               initial_assets: Optional[Union[Number, List[Number]]] = None,
               adjust_sell: bool = False,
               track_interval: int = 10,
               trade_cmd_expire_seconds: int = 120,
               cmd_cache: bool = True,
               slippage: float = 0.0) -> None:
        """跟踪 joinquant 对应的模拟交易，支持多用户多策略
        :param users: 支持 easytrader 的用户对象，支持使用 [] 指定多个用户
        :param strategies: 雪球组合名, 类似 ZH123450
        :param total_assets: 雪球组合对应的总资产， 格式 [组合1对应资金, 组合2对应资金]
        :param adjust_sell: 是否根据用户的实际持仓数调整卖出股票数量
        :param initial_assets: 雪球组合对应的初始资产, 格式 [ 组合1对应资金, 组合2对应资金 ]
        :param track_interval: 轮训模拟交易时间，单位为秒
        :param trade_cmd_expire_seconds: 交易指令过期时间, 单位为秒
        :param cmd_cache: 是否读取存储历史执行过的指令，防止重启时重复执行已经交易过的指令
        :param slippage: 滑点，0.0 表示无滑点, 0.05 表示滑点为 5%
        """
        super().follow(users=users, strategies=strategies, track_interval=track_interval,
                       trade_cmd_expire_seconds=trade_cmd_expire_seconds, cmd_cache=cmd_cache, slippage=slippage)
        self._adjust_sell = adjust_sell
        self._users = self.warp_list(users)  # type: ignore
        strategies_list: List[str] = self.warp_list(strategies)  # type: ignore
        total_assets_list: List[Number] = self.warp_list(total_assets)  # type: ignore
        initial_assets_list: List[Optional[Number]] = self.warp_list(initial_assets)  # type: ignore
        if cmd_cache:
            self.load_expired_cmd_cache()  # type: ignore
        self.start_trader_thread(self._users, trade_cmd_expire_seconds)  # type: ignore
        for strategy_url, strategy_total_assets, strategy_initial_assets in zip(
                strategies_list, total_assets_list, initial_assets_list):
            assets: Number = self.calculate_assets(strategy_url, strategy_total_assets, strategy_initial_assets)
            try:
                strategy_id: str = self.extract_strategy_id(strategy_url)
                strategy_name: str = self.extract_strategy_name(strategy_url)
            except Exception:
                logger.error('抽取交易id和策略名失败, 无效模拟交易url: %s', strategy_url)
                raise
            strategy_worker: Thread = Thread(
                target=self.track_strategy_worker,  # type: ignore
                args=[strategy_id, strategy_name],
                kwargs={'interval': track_interval, 'assets': assets}
            )
            strategy_worker.start()
            logger.info('开始跟踪策略: %s', strategy_name)

    def calculate_assets(self,
                         strategy_url: str,
                         total_assets: Optional[Number] = None,
                         initial_assets: Optional[Number] = None) -> Number:
        if total_assets is None and initial_assets is not None:
            net_value: Number = self._get_portfolio_net_value(strategy_url)
            total_assets = initial_assets * net_value
        if not isinstance(total_assets, Number):
            raise TypeError('input assets type must be number(int, float)')
        if total_assets < 1000.0:
            raise ValueError('雪球总资产不能小于1000元，当前预设值 {}'.format(total_assets))
        return total_assets

    @staticmethod
    def extract_strategy_id(strategy_url: str) -> str:
        return strategy_url

    def extract_strategy_name(self, strategy_url: str) -> str:
        base_url: str = 'https://xueqiu.com/cubes/nav_daily/all.json?cube_symbol={}'
        url: str = base_url.format(strategy_url)
        rep = self.s.get(url)
        info_index: int = 0
        return rep.json()[info_index]['name']

    def extract_transactions(self, history: Dict[str, Any]) -> List[Dict[str, Any]]:
        if history.get('count', 0) <= 0:
            return []
        rebalancing_index: int = 0
        raw_transactions: List[Dict[str, Any]] = history['list'][rebalancing_index]['rebalancing_histories']
        transactions: List[Dict[str, Any]] = []
        for transaction in raw_transactions:
            if transaction['price'] is None:
                logger.info('该笔交易无法获取价格，疑似未成交，跳过。交易详情: %s', transaction)
                continue
            transactions.append(transaction)
        return transactions

    def create_query_transaction_params(self, strategy: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {'cube_symbol': strategy, 'page': 1, 'count': 1}
        return params

    def none_to_zero(self, data: Optional[Number]) -> Number:
        if data is None:
            return 0
        return data

    def project_transactions(self, transactions: List[Dict[str, Any]], assets: Number) -> None:
        for transaction in transactions:
            weight_diff: Number = self.none_to_zero(transaction.get('weight')) - self.none_to_zero(transaction.get('prev_weight'))
            initial_amount: float = abs(weight_diff) / 100 * assets / transaction['price']
            transaction['datetime'] = datetime.fromtimestamp(transaction['created_at'] // 1000)
            transaction['stock_code'] = transaction['stock_symbol'].lower()
            transaction['action'] = 'buy' if weight_diff > 0 else 'sell'
            transaction['amount'] = int(round(initial_amount, -2))
            if transaction['action'] == 'sell' and self._adjust_sell:
                transaction['amount'] = self._adjust_sell_amount(transaction['stock_code'], transaction['amount'])

    def _adjust_sell_amount(self, stock_code: str, amount: int) -> int:
        """
        根据实际持仓值计算雪球卖出股数
        :param stock_code: 证券代码
        :param amount: 卖出股份数
        :return: 考虑实际持仓之后的卖出股份数
        """
        stock_code = stock_code[-6:]
        user = self._users[0]  # type: ignore
        position = user.position  # type: ignore
        try:
            stock = next((s for s in position if s['证券代码'] == stock_code))
        except StopIteration:
            logger.info('根据持仓调整 %s 卖出额，发现未持有股票 %s, 不做任何调整', stock_code, stock_code)
            return amount
        available_amount: int = stock['可用余额']
        if available_amount >= amount:
            return amount
        adjust_amount: int = available_amount // 100 * 100
        logger.info('股票 %s 实际可用余额 %s, 指令卖出股数为 %s, 调整为 %s', stock_code, available_amount, amount, adjust_amount)
        return adjust_amount

    def _get_portfolio_info(self, portfolio_code: str) -> Dict[str, Any]:
        """
        获取组合信息
        """
        url: str = self.PORTFOLIO_URL + portfolio_code
        portfolio_page = self.s.get(url)
        match_info = re.search('(?<=SNB.cubeInfo = ).*(?=;\\n)', portfolio_page.text)
        if match_info is None:
            raise Exception('cant get portfolio info, portfolio url : {}'.format(url))
        try:
            portfolio_info: Dict[str, Any] = json.loads(match_info.group())
        except Exception as e:
            raise Exception('get portfolio info error: {}'.format(e))
        return portfolio_info

    def _get_portfolio_net_value(self, portfolio_code: str) -> Number:
        """
        获取组合信息
        """
        portfolio_info: Dict[str, Any] = self._get_portfolio_info(portfolio_code)
        return portfolio_info['net_value']