import json
import numbers
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import requests
from easytrader import exceptions, webtrader
from easytrader.log import logger
from easytrader.utils.misc import parse_cookies_str

class XueQiuTrader(webtrader.WebTrader):
    config_path: str = os.path.dirname(__file__) + '/config/xq.json'
    _HEADERS: Dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36',
        'Host': 'xueqiu.com',
        'Pragma': 'no-cache',
        'Connection': 'keep-alive',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Cache-Control': 'no-cache',
        'Referer': 'https://xueqiu.com/P/ZH004612',
        'X-Requested-With': 'XMLHttpRequest'
    }

    def __init__(self, **kwargs: Any) -> None:
        super(XueQiuTrader, self).__init__()
        self.multiple: float = kwargs['initial_assets'] if 'initial_assets' in kwargs else 1000000
        if not isinstance(self.multiple, numbers.Number):
            raise TypeError('initial assets must be number(int, float)')
        if self.multiple < 1000.0:
            raise ValueError(f'雪球初始资产不能小于1000元，当前预设值 {self.multiple}')
        self.s: requests.Session = requests.Session()
        self.s.verify = False
        self.s.headers.update(self._HEADERS)
        self.account_config: Optional[Dict[str, Any]] = None

    def autologin(self, **kwargs: Any) -> None:
        """
        使用 cookies 之后不需要自动登陆
        :return:
        """
        if self.account_config and 'cookies' in self.account_config:
            self._set_cookies(self.account_config['cookies'])

    def _set_cookies(self, cookies: str) -> None:
        """设置雪球 cookies，代码来自于
        https://github.com/shidenggui/easytrader/issues/269
        :param cookies: 雪球 cookies
        :type cookies: str
        """
        cookie_dict: Dict[str, str] = parse_cookies_str(cookies)
        self.s.cookies.update(cookie_dict)

    def _prepare_account(self, user: str = '', password: str = '', **kwargs: Any) -> None:
        """
        转换参数到登录所需的字典格式
        :param cookies: 雪球登陆需要设置 cookies， 具体见
            https://smalltool.github.io/2016/08/02/cookie/
        :param portfolio_code: 组合代码
        :param portfolio_market: 交易市场， 可选['cn', 'us', 'hk'] 默认 'cn'
        :return:
        """
        if 'portfolio_code' not in kwargs:
            raise TypeError('雪球登录需要设置 portfolio_code(组合代码) 参数')
        if 'portfolio_market' not in kwargs:
            kwargs['portfolio_market'] = 'cn'
        if 'cookies' not in kwargs:
            raise TypeError('雪球登陆需要设置 cookies， 具体见https://smalltool.github.io/2016/08/02/cookie/')
        self.account_config = {
            'cookies': kwargs['cookies'],
            'portfolio_code': kwargs['portfolio_code'],
            'portfolio_market': kwargs['portfolio_market']
        }

    def _virtual_to_balance(self, virtual: float) -> float:
        """
        虚拟净值转化为资金
        :param virtual: 雪球组合净值
        :return: 换算的资金
        """
        return virtual * self.multiple

    def _get_html(self, url: str) -> str:
        return self.s.get(url).text

    def _search_stock_info(self, code: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        通过雪球的接口获取股票详细信息
        :param code: 股票代码 000001
        :return: 查询到的股票 {...}
        """
        data: Dict[str, Any] = {
            'code': str(code),
            'size': '300',
            'key': '47bce5c74f',
            'market': self.account_config['portfolio_market']
        }
        r: requests.Response = self.s.get(self.config['search_stock_url'], params=data)
        stocks: List[Dict[str, Any]] = json.loads(r.text).get('stocks', [])
        return stocks[0] if stocks else None

    def _get_portfolio_info(self, portfolio_code: str) -> Dict[str, Any]:
        """
        获取组合信息
        :return: 字典
        """
        url: str = self.config['portfolio_url'] + portfolio_code
        html: str = self._get_html(url)
        match_info: Optional[re.Match] = re.search(r'(?<=SNB.cubeInfo = ).*(?=;\n)', html)
        if match_info is None:
            raise Exception(f'cant get portfolio info, portfolio html : {html}')
        try:
            portfolio_info: Dict[str, Any] = json.loads(match_info.group())
        except Exception as e:
            raise Exception(f'get portfolio info error: {e}')
        return portfolio_info

    def get_balance(self) -> List[Dict[str, Union[float, str]]]:
        """
        获取账户资金状况
        :return:
        """
        portfolio_code: str = self.account_config.get('portfolio_code', 'ch')
        portfolio_info: Dict[str, Any] = self._get_portfolio_info(portfolio_code)
        asset_balance: float = self._virtual_to_balance(float(portfolio_info['net_value']))
        position: Dict[str, Any] = portfolio_info['view_rebalancing']
        cash: float = asset_balance * float(position['cash']) / 100
        market: float = asset_balance - cash
        return [{
            'asset_balance': asset_balance,
            'current_balance': cash,
            'enable_balance': cash,
            'market_value': market,
            'money_type': '人民币',
            'pre_interest': 0.25
        }]

    def _get_position(self) -> List[Dict[str, Any]]:
        """
        获取雪球持仓
        :return:
        """
        portfolio_code: str = self.account_config['portfolio_code']
        portfolio_info: Dict[str, Any] = self._get_portfolio_info(portfolio_code)
        position: Dict[str, Any] = portfolio_info['view_rebalancing']
        stocks: List[Dict[str, Any]] = position['holdings']
        return stocks

    @staticmethod
    def _time_strftime(time_stamp: int) -> str:
        try:
            local_time = time.localtime(time_stamp / 1000)
            return time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        except Exception:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def get_position(self) -> List[Dict[str, Any]]:
        """
        获取持仓
        :return:
        """
        xq_positions: List[Dict[str, Any]] = self._get_position()
        balance: Dict[str, Any] = self.get_balance()[0]
        position_list: List[Dict[str, Any]] = []
        for pos in xq_positions:
            volume: float = pos['weight'] * balance['asset_balance'] / 100
            position_list.append({
                'cost_price': volume / 100,
                'current_amount': 100,
                'enable_amount': 100,
                'income_balance': 0,
                'keep_cost_price': volume / 100,
                'last_price': volume / 100,
                'market_value': volume,
                'position_str': 'random',
                'stock_code': pos['stock_symbol'],
                'stock_name': pos['stock_name']
            })
        return position_list

    def _get_xq_history(self) -> List[Dict[str, Any]]:
        """
        获取雪球调仓历史
        :return:
        """
        data: Dict[str, Any] = {
            'cube_symbol': str(self.account_config['portfolio_code']),
            'count': 20,
            'page': 1
        }
        resp: requests.Response = self.s.get(self.config['history_url'], params=data)
        res: Dict[str, Any] = json.loads(resp.text)
        return res.get('list', [])

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._get_xq_history()

    def get_entrust(self) -> List[Dict[str, Any]]:
        """
        获取委托单(目前返回20次调仓的结果)
        操作数量都按1手模拟换算的
        :return:
        """
        xq_entrust_list: List[Dict[str, Any]] = self._get_xq_history()
        entrust_list: List[Dict[str, Any]] = []
        replace_none = lambda s: s or 0  # type: ignore
        for xq_entrusts in xq_entrust_list:
            status: str = xq_entrusts['status']
            if status == 'pending':
                status = '已报'
            elif status in ['canceled', 'failed']:
                status = '废单'
            else:
                status = '已成'
            for entrust in xq_entrusts['rebalancing_histories']:
                price: float = entrust['price']
                entrust_list.append({
                    'entrust_no': entrust['id'],
                    'entrust_bs': '买入' if entrust['target_weight'] > replace_none(entrust['prev_weight']) else '卖出',
                    'report_time': self._time_strftime(entrust['updated_at']),
                    'entrust_status': status,
                    'stock_code': entrust['stock_symbol'],
                    'stock_name': entrust['stock_name'],
                    'business_amount': 100,
                    'business_price': price,
                    'entrust_amount': 100,
                    'entrust_price': price
                })
        return entrust_list

    def cancel_entrust(self, entrust_no: Union[str, int]) -> bool:
        """
        对未成交的调仓进行伪撤单
        :param entrust_no:
        :return:
        """
        xq_entrust_list: List[Dict[str, Any]] = self._get_xq_history()
        is_have: bool = False
        for xq_entrusts in xq_entrust_list:
            status: str = xq_entrusts['status']
            for entrust in xq_entrusts['rebalancing_histories']:
                if entrust['id'] == entrust_no and status == 'pending':
                    is_have = True
                    buy_or_sell: str = 'buy' if entrust['target_weight'] < entrust['weight'] else 'sell'
                    if entrust['target_weight'] == 0 and entrust['weight'] == 0:
                        raise exceptions.TradeError('移除的股票操作无法撤销,建议重新买入')
                    balance: Dict[str, Any] = self.get_balance()[0]
                    volume: float = abs(entrust['target_weight'] - entrust['weight']) * balance['asset_balance'] / 100
                    r: Optional[List[Dict[str, Any]]] = self._trade(
                        security=entrust['stock_symbol'],
                        volume=volume,
                        entrust_bs=buy_or_sell
                    )
                    if r and 'error_info' in r[0]:
                        raise exceptions.TradeError(f"撤销失败!{r[0].get('error_info')}")
        if not is_have:
            raise exceptions.TradeError('撤销对象已失效')
        return True

    def adjust_weight(self, stock_code: str, weight: float) -> Optional[List[Dict[str, Any]]]:
        """
        雪球组合调仓, weight 为调整后的仓位比例
        :param stock_code: str 股票代码
        :param weight: float 调整之后的持仓百分比， 0 - 100 之间的浮点数
        """
        stock: Optional[Dict[str, Any]] = self._search_stock_info(stock_code)
        if stock is None:
            raise exceptions.TradeError('没有查询要操作的股票信息')
        if stock['flag'] != 1:
            raise exceptions.TradeError('未上市、停牌、涨跌停、退市的股票无法操作。')
        weight = round(weight, 2)
        position_list: List[Dict[str, Any]] = self._get_position()
        for position in position_list:
            if position['stock_id'] == stock['stock_id']:
                position['proactive'] = True
                position['weight'] = weight
        if weight != 0 and stock['stock_id'] not in [k['stock_id'] for k in position_list]:
            position_list.append({
                'code': stock['code'],
                'name': stock['name'],
                'enName': stock['enName'],
                'hasexist': stock['hasexist'],
                'flag': stock['flag'],
                'type': stock['type'],
                'current': stock['current'],
                'chg': stock['chg'],
                'percent': str(stock['percent']),
                'stock_id': stock['stock_id'],
                'ind_id': stock['ind_id'],
                'ind_name': stock['ind_name'],
                'ind_color': stock['ind_color'],
                'textname': stock['name'],
                'segment_name': stock['ind_name'],
                'weight': weight,
                'url': '/S/' + stock['code'],
                'proactive': True,
                'price': str(stock['current'])
            })
        remain_weight: float = 100 - sum(i.get('weight', 0) for i in position_list)
        cash: float = round(remain_weight, 2)
        logger.info('调仓比例:%f, 剩余持仓 :%f', weight, remain_weight)
        data: Dict[str, Any] = {
            'cash': cash,
            'holdings': str(json.dumps(position_list)),
            'cube_symbol': str(self.account_config['portfolio_code']),
            'segment': 'true',
            'comment': ''
        }
        try:
            resp: requests.Response = self.s.post(self.config['rebalance_url'], data=data)
        except Exception as e:
            logger.warning('调仓失败: %s ', e)
            return None
        logger.info('调仓 %s: 持仓比例%d', stock['name'], weight)
        resp_json: Dict[str, Any] = json.loads(resp.text)
        if 'error_description' in resp_json and resp.status_code != 200:
            logger.error('调仓错误: %s', resp_json['error_description'])
            return [{
                'error_no': resp_json['error_code'],
                'error_info': resp_json['error_description']
            }]
        logger.info('调仓成功 %s: 持仓比例%d', stock['name'], weight)
        return None

    def _trade(
        self,
        security: str,
        price: float = 0,
        amount: int = 0,
        volume: float = 0,
        entrust_bs: str = 'buy'
    ) -> Optional[List[Dict[str, Any]]]:
        """
        调仓
        :param security:
        :param price:
        :param amount:
        :param volume:
        :param entrust_bs:
        :return:
        """
        stock: Optional[Dict[str, Any]] = self._search_stock_info(security)
        balance: Dict[str, Any] = self.get_balance()[0]
        if stock is None:
            raise exceptions.TradeError('没有查询要操作的股票信息')
        if not volume:
            volume = float(price) * amount
        if balance['current_balance'] < volume and entrust_bs == 'buy':
            raise exceptions.TradeError('没有足够的现金进行操作')
        if stock['flag'] != 1:
            raise exceptions.TradeError('未上市、停牌、涨跌停、退市的股票无法操作。')
        if volume == 0:
            raise exceptions.TradeError('操作金额不能为零')
        weight: float = round((volume / balance['asset_balance']) * 100, 2)
        position_list: List[Dict[str, Any]] = self._get_position()
        is_have: bool = False
        for position in position_list:
            if position['stock_id'] == stock['stock_id']:
                is_have = True
                position['proactive'] = True
                old_weight: float = position['weight']
                if entrust_bs == 'buy':
                    position['weight'] = round(weight + old_weight, 2)
                elif weight > old_weight:
                    raise exceptions.TradeError('操作数量大于实际可卖出数量')
                else:
                    position['weight'] = round(old_weight - weight, 2)
        if not is_have:
            if entrust_bs == 'buy':
                position_list.append({
                    'code': stock['code'],
                    'name': stock['name'],
                    'enName': stock['enName'],
                    'hasexist': stock['hasexist'],
                    'flag': stock['flag'],
                    'type': stock['type'],
                    'current': stock['current'],
                    'chg': stock['chg'],
                    'percent': str(stock['percent']),
                    'stock_id': stock['stock_id'],
                    'ind_id': stock['ind_id'],
                    'ind_name': stock['ind_name'],
                    'ind_color': stock['ind_color'],
                    'textname': stock['name'],
                    'segment_name': stock['ind_name'],
                    'weight': round(weight, 2),
                    'url': '/S/' + stock['code'],
                    'proactive': True,
                    'price': str(stock['current'])
                })
            else:
                raise exceptions.TradeError('没有持有要卖出的股票')
        if entrust_bs == 'buy':
            cash: float = round((balance['current_balance'] - volume) / balance['asset_balance'] * 100, 2)
        else:
            cash = round((balance['current_balance'] + volume) / balance['asset_balance'] * 100, 2)
        logger.info('weight:%f, cash:%f', weight, cash)
        data: Dict[str, Any] = {
            'cash': cash,
            'holdings': str(json.dumps(position_list)),
            'cube_symbol': str(self.account_config['portfolio_code']),
            'segment': 1,
            'comment': ''
        }
        try:
            resp: requests.Response = self.s.post(self.config['rebalance_url'], data=data)
        except Exception as e:
            logger.warning('调仓失败: %s ', e)
            return None
        logger.info('调仓 %s%s: %d', entrust_bs, stock['name'], resp.status_code)
        resp_json: Dict[str, Any] = json.loads(resp.text)
        if 'error_description' in resp_json and resp.status_code != 200:
            logger.error('调仓错误: %s', resp_json['error_description'])
            return [{
                'error_no': resp_json['error_code'],
                'error_info': resp_json['error_description']
            }]
        return [{
            'entrust_no': resp_json['id'],
            'init_date': self._time_strftime(resp_json['created_at']),
            'batch_no': '委托批号',
            'report_no': '申报号',
            'seat_no': '席位编号',
            'entrust_time': self._time_strftime(resp_json['updated_at']),
            'entrust_price': price,
            'entrust_amount': amount,
            'stock_code': security,
            'entrust_bs': '买入',
            'entrust_type': '雪球虚拟委托',
            'entrust_status': '-'
        }]

    def buy(
        self,
        security: str,
        price: float = 0,
        amount: int = 0,
        volume: float = 0,
        entrust_prop: float = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """买入股票
        :param security: 股票代码
        :param price: 买入价格
        :param amount: 买入股数
        :param volume: 买入总金额 由 volume / price 取整， 若指定 price 则此参数无效
        :param entrust_prop:
        """
        return self._trade(security, price, amount, volume, 'buy')

    def sell(
        self,
        security: str,
        price: float = 0,
        amount: int = 0,
        volume: float = 0,
        entrust_prop: float = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """卖出股票
        :param security: 股票代码
        :param price: 卖出价格
        :param amount: 卖出股数
        :param volume: 卖出总金额 由 volume / price 取整， 若指定 price 则此参数无效
        :param entrust_prop:
        """
        return self._trade(security, price, amount, volume, 'sell')
