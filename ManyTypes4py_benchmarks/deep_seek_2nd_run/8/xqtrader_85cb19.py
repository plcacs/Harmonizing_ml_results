import json
import numbers
import os
import re
import time
from typing import Any, Dict, List, Optional, Union, Callable
import requests
from easytrader import exceptions, webtrader
from easytrader.log import logger
from easytrader.utils.misc import parse_cookies_str

class XueQiuTrader(webtrader.WebTrader):
    config_path: str = os.path.dirname(__file__) + '/config/xq.json'
    _HEADERS: Dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36',
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
        self.multiple: Union[int, float] = kwargs['initial_assets'] if 'initial_assets' in kwargs else 1000000
        if not isinstance(self.multiple, numbers.Number):
            raise TypeError('initial assets must be number(int, float)')
        if self.multiple < 1000.0:
            raise ValueError('雪球初始资产不能小于1000元，当前预设值 {}'.format(self.multiple))
        self.s: requests.Session = requests.Session()
        self.s.verify = False
        self.s.headers.update(self._HEADERS)
        self.account_config: Optional[Dict[str, Any]] = None

    def autologin(self, **kwargs: Any) -> None:
        self._set_cookies(self.account_config['cookies'])

    def _set_cookies(self, cookies: str) -> None:
        cookie_dict = parse_cookies_str(cookies)
        self.s.cookies.update(cookie_dict)

    def _prepare_account(self, user: str = '', password: str = '', **kwargs: Any) -> None:
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
        return virtual * self.multiple

    def _get_html(self, url: str) -> str:
        return self.s.get(url).text

    def _search_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        data = {
            'code': str(code),
            'size': '300',
            'key': '47bce5c74f',
            'market': self.account_config['portfolio_market']
        }
        r = self.s.get(self.config['search_stock_url'], params=data)
        stocks = json.loads(r.text)
        stocks = stocks['stocks']
        stock = None
        if len(stocks) > 0:
            stock = stocks[0]
        return stock

    def _get_portfolio_info(self, portfolio_code: str) -> Dict[str, Any]:
        url = self.config['portfolio_url'] + portfolio_code
        html = self._get_html(url)
        match_info = re.search('(?<=SNB.cubeInfo = ).*(?=;\\n)', html)
        if match_info is None:
            raise Exception('cant get portfolio info, portfolio html : {}'.format(html))
        try:
            portfolio_info = json.loads(match_info.group())
        except Exception as e:
            raise Exception('get portfolio info error: {}'.format(e))
        return portfolio_info

    def get_balance(self) -> List[Dict[str, Any]]:
        portfolio_code = self.account_config.get('portfolio_code', 'ch')
        portfolio_info = self._get_portfolio_info(portfolio_code)
        asset_balance = self._virtual_to_balance(float(portfolio_info['net_value']))
        position = portfolio_info['view_rebalancing']
        cash = asset_balance * float(position['cash']) / 100
        market = asset_balance - cash
        return [{
            'asset_balance': asset_balance,
            'current_balance': cash,
            'enable_balance': cash,
            'market_value': market,
            'money_type': u'人民币',
            'pre_interest': 0.25
        }]

    def _get_position(self) -> List[Dict[str, Any]]:
        portfolio_code = self.account_config['portfolio_code']
        portfolio_info = self._get_portfolio_info(portfolio_code)
        position = portfolio_info['view_rebalancing']
        stocks = position['holdings']
        return stocks

    @staticmethod
    def _time_strftime(time_stamp: int) -> str:
        try:
            local_time = time.localtime(time_stamp / 1000)
            return time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        except Exception:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def get_position(self) -> List[Dict[str, Any]]:
        xq_positions = self._get_position()
        balance = self.get_balance()[0]
        position_list = []
        for pos in xq_positions:
            volume = pos['weight'] * balance['asset_balance'] / 100
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
        data = {
            'cube_symbol': str(self.account_config['portfolio_code']),
            'count': 20,
            'page': 1
        }
        resp = self.s.get(self.config['history_url'], params=data)
        res = json.loads(resp.text)
        return res['list']

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._get_xq_history()

    def get_entrust(self) -> List[Dict[str, Any]]:
        xq_entrust_list = self._get_xq_history()
        entrust_list = []
        replace_none: Callable[[Any], Any] = lambda s: s or 0
        for xq_entrusts in xq_entrust_list:
            status = xq_entrusts['status']
            if status == 'pending':
                status = '已报'
            elif status in ['canceled', 'failed']:
                status = '废单'
            else:
                status = '已成'
            for entrust in xq_entrusts['rebalancing_histories']:
                price = entrust['price']
                entrust_list.append({
                    'entrust_no': entrust['id'],
                    'entrust_bs': u'买入' if entrust['target_weight'] > replace_none(entrust['prev_weight']) else u'卖出',
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

    def cancel_entrust(self, entrust_no: int) -> bool:
        xq_entrust_list = self._get_xq_history()
        is_have = False
        for xq_entrusts in xq_entrust_list:
            status = xq_entrusts['status']
            for entrust in xq_entrusts['rebalancing_histories']:
                if entrust['id'] == entrust_no and status == 'pending':
                    is_have = True
                    buy_or_sell = 'buy' if entrust['target_weight'] < entrust['weight'] else 'sell'
                    if entrust['target_weight'] == 0 and entrust['weight'] == 0:
                        raise exceptions.TradeError(u'移除的股票操作无法撤销,建议重新买入')
                    balance = self.get_balance()[0]
                    volume = abs(entrust['target_weight'] - entrust['weight']) * balance['asset_balance'] / 100
                    r = self._trade(security=entrust['stock_symbol'], volume=volume, entrust_bs=buy_or_sell)
                    if len(r) > 0 and 'error_info' in r[0]:
                        raise exceptions.TradeError(u'撤销失败!%s' % ('error_info' in r[0]))
        if not is_have:
            raise exceptions.TradeError(u'撤销对象已失效')
        return True

    def adjust_weight(self, stock_code: str, weight: float) -> Optional[List[Dict[str, Any]]]:
        stock = self._search_stock_info(stock_code)
        if stock is None:
            raise exceptions.TradeError(u'没有查询要操作的股票信息')
        if stock['flag'] != 1:
            raise exceptions.TradeError(u'未上市、停牌、涨跌停、退市的股票无法操作。')
        weight = round(weight, 2)
        position_list = self._get_position()
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
        remain_weight = 100 - sum((i.get('weight') for i in position_list))
        cash = round(remain_weight, 2)
        logger.info('调仓比例:%f, 剩余持仓 :%f', weight, remain_weight)
        data = {
            'cash': cash,
            'holdings': str(json.dumps(position_list)),
            'cube_symbol': str(self.account_config['portfolio_code']),
            'segment': 'true',
            'comment': ''
        }
        try:
            resp = self.s.post(self.config['rebalance_url'], data=data)
        except Exception as e:
            logger.warning('调仓失败: %s ', e)
            return None
        logger.info('调仓 %s: 持仓比例%d', stock['name'], weight)
        resp_json = json.loads(resp.text)
        if 'error_description' in resp_json and resp.status_code != 200:
            logger.error('调仓错误: %s', resp_json['error_description'])
            return [{'error_no': resp_json['error_code'], 'error_info': resp_json['error_description']}]
        logger.info('调仓成功 %s: 持仓比例%d', stock['name'], weight)
        return None

    def _trade(
        self,
        security: str,
        price: Union[int, float] = 0,
        amount: Union[int, float] = 0,
        volume: Union[int, float] = 0,
        entrust_bs: str = 'buy'
    ) -> Optional[List[Dict[str, Any]]]:
        stock = self._search_stock_info(security)
        balance = self.get_balance()[0]
        if stock is None:
            raise exceptions.TradeError(u'没有查询要操作的股票信息')
        if not volume:
            volume = int(float(price) * amount)
        if balance['current_balance'] < volume and entrust_bs == 'buy':
            raise exceptions.TradeError(u'没有足够的现金进行操作')
        if stock['flag'] != 1:
            raise exceptions.TradeError(u'未上市、停牌、涨跌停、退市的股票无法操作。')
        if volume == 0:
            raise exceptions.TradeError(u'操作金额不能为零')
        weight = volume / balance['asset_balance'] * 100
        weight = round(weight, 2)
        position_list = self._get_position()
        is_have = False
        for position in position_list:
            if position['stock_id'] == stock['stock_id']:
                is_have = True
                position['proactive'] = True
                old_weight = position['weight']
                if entrust_bs == 'buy':
                    position['weight'] = weight + old_weight
                elif weight > old_weight:
                    raise exceptions.TradeError(u'操作数量大于实际可卖出数量')
                else:
                    position['weight'] = old_weight - weight
                position['weight'] = round(position['weight'], 2)
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
                raise exceptions.TradeError(u'没有持有要卖出的股票')
        if entrust_bs == 'buy':
            cash = (balance['current_balance'] - volume) / balance['asset_balance'] * 100
        else:
            cash = (balance['current_balance'] + volume) / balance['asset_balance'] * 100
        cash = round(cash, 2)
        logger.info('weight:%f, cash:%f', weight, cash)
        data = {
            'cash': cash,
            'holdings': str(json.dumps(position_list)),
            'cube_symbol': str(self.account_config['portfolio_code']),
            'segment': 1,
            'comment': ''
        }
        try:
            resp = self.s.post(self.config['rebalance_url'], data=data)
        except Exception as e:
            logger.warning('调仓失败: %s ', e)
            return None
        else:
            logger.info('调仓 %s%s: %d', entrust_bs, stock['name'], resp.status_code)
            resp_json = json.loads(resp.text)
            if 'error_description' in resp_json and resp.status_code != 200:
                logger.error('调仓错误: %s', resp_json['error_description'])
                return [{'error_no': resp_json['error_code'], 'error_info': resp_json['error_description']}]
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
        price: Union[int, float] = 0,
        amount: Union[int, float] = 0,
        volume: Union[int, float] = 0,
        entrust_prop: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        return self._trade(security, price, amount, volume, 'buy')

    def sell(
        self,
        security: str,
        price: Union[int, float] = 0,
        amount: Union[int, float] = 0,
        volume: Union[int, float] = 0,
        entrust_prop: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        return self._trade(security, price, amount, volume, 'sell')
