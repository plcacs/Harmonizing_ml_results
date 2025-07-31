#!/usr/bin/env python3
"""
获取集思路的数据
"""
import json
import time
from typing import Any, Dict, List, Optional, Union
import requests

class Jsl:
    """
    抓取集思路的分级A数据
    """
    __funda_url: str = 'http://www.jisilu.cn/data/sfnew/funda_list/?___t={ctime:d}'
    __fundb_url: str = 'http://www.jisilu.cn/data/sfnew/fundb_list/?___t={ctime:d}'
    __fundm_url: str = 'https://www.jisilu.cn/data/sfnew/fundm_list/?___t={ctime:d}'
    __fundarb_url: str = 'http://www.jisilu.cn/data/sfnew/arbitrage_vip_list/?___t={ctime:d}'
    __jsl_login_url: str = 'https://www.jisilu.cn/account/ajax/login_process/'
    __etf_index_url: str = 'https://www.jisilu.cn/data/etf/etf_list/?___jsl=LST___t={ctime:d}&rp=25&page=1'
    __etf_gold_url: str = 'https://www.jisilu.cn/jisiludata/etf.php?qtype=pmetf&___t={ctime:d}'
    __etf_money_url: str = 'https://www.jisilu.cn/data/money_fund/list/?___t={ctime:d}'
    __qdii_url: str = 'https://www.jisilu.cn/data/qdii/qdii_list/?___t={ctime:d}'
    __cb_url: str = 'https://www.jisilu.cn/data/cbnew/cb_list/?___t={ctime:d}'

    def __init__(self) -> None:
        self.__funda: Optional[Dict[Any, Any]] = None
        self.__fundm: Optional[Dict[Any, Any]] = None
        self.__fundb: Optional[Dict[Any, Any]] = None
        self.__fundarb: Optional[Dict[Any, Any]] = None
        self.__etfindex: Optional[Dict[Any, Any]] = None
        self.__qdii: Optional[Dict[Any, Any]] = None
        self.__cb: Optional[Dict[Any, Any]] = None
        self._cookie: Optional[str] = None

    def set_cookie(self, cookie: str) -> None:
        self._cookie = cookie

    def _get_headers(self) -> Dict[str, str]:
        default: Dict[str, str] = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'}
        if self._cookie:
            default = {**default, 'Cookie': self._cookie}
        return default

    @staticmethod
    def formatfundajson(fundajson: Dict[str, Any]) -> Dict[Any, Any]:
        """格式化集思录返回的json数据,以字典形式保存"""
        result: Dict[Any, Any] = {}
        for row in fundajson['rows']:
            funda_id = row['id']
            cell = row['cell']
            result[funda_id] = cell
        return result

    @staticmethod
    def formatfundbjson(fundbjson: Dict[str, Any]) -> Dict[Any, Any]:
        """格式化集思录返回的json数据,以字典形式保存"""
        result: Dict[Any, Any] = {}
        for row in fundbjson['rows']:
            cell = row['cell']
            fundb_id = cell['fundb_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatetfindexjson(fundbjson: Dict[str, Any]) -> Dict[Any, Any]:
        """格式化集思录返回 指数ETF 的json数据,以字典形式保存"""
        result: Dict[Any, Any] = {}
        for row in fundbjson['rows']:
            cell = row['cell']
            fundb_id = cell['fund_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatjisilujson(data: Dict[str, Any]) -> Dict[Any, Any]:
        result: Dict[Any, Any] = {}
        for row in data['rows']:
            cell = row['cell']
            id_ = row['id']
            result[id_] = cell
        return result

    @staticmethod
    def percentage2float(per: str) -> float:
        """
        将字符串的百分数转化为浮点数
        :param per:
        :return:
        """
        return float(per.strip('%')) / 100.0

    def funda(self, fields: Optional[List[str]] = None, min_volume: float = 0, min_discount: float = 0, ignore_nodown: bool = False, forever: bool = False) -> Dict[Any, Any]:
        """以字典形式返回分级A数据
        :param fields:利率范围，形如['+3.0%', '6.0%']
        :param min_volume:最小交易量，单位万元
        :param min_discount:最小折价率, 单位%
        :param ignore_nodown:是否忽略无下折品种,默认 False
        :param forever: 是否选择永续品种,默认 False
        """
        if fields is None:
            fields = []
        url: str = self.__funda_url.format(ctime=int(time.time()))
        rep = requests.get(url)
        fundajson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatfundajson(fundajson)
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['funda_volume']) > min_volume}
        if len(fields):
            data = {k: data[k] for k in data if data[k]['coupon_descr_s'] in ''.join(fields)}
        if ignore_nodown:
            data = {k: data[k] for k in data if data[k]['fund_descr'].find('无下折') == -1}
        if forever:
            data = {k: data[k] for k in data if data[k]['funda_left_year'].find('永续') != -1}
        if min_discount:
            data = {k: data[k] for k in data if float(data[k]['funda_discount_rt'][:-1]) > min_discount}
        self.__funda = data
        return self.__funda

    def fundm(self) -> Dict[Any, Any]:
        """以字典形式返回分级母基数据
        """
        url: str = self.__fundm_url.format(ctime=int(time.time()))
        rep = requests.get(url)
        fundmjson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatfundajson(fundmjson)
        self.__fundm = data
        return self.__fundm

    def fundb(self, fields: Optional[List[str]] = None, min_volume: float = 0, min_discount: float = 0, forever: bool = False) -> Dict[Any, Any]:
        """以字典形式返回分级B数据
        :param fields:利率范围，形如['+3.0%', '6.0%']
        :param min_volume:最小交易量，单位万元
        :param min_discount:最小折价率, 单位%
        :param forever: 是否选择永续品种,默认 False
        """
        if fields is None:
            fields = []
        url: str = self.__fundb_url.format(ctime=int(time.time()))
        rep = requests.get(url)
        fundbjson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatfundbjson(fundbjson)
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['fundb_volume']) > min_volume}
        if len(fields):
            data = {k: data[k] for k in data if data[k]['coupon_descr_s'] in ''.join(fields)}
        if forever:
            data = {k: data[k] for k in data if data[k]['fundb_left_year'].find('永续') != -1}
        if min_discount:
            data = {k: data[k] for k in data if float(data[k]['fundb_discount_rt'][:-1]) > min_discount}
        self.__fundb = data
        return self.__fundb

    def fundarb(self, jsl_username: str, jsl_password: str, avolume: Union[int, float] = 100, bvolume: Union[int, float] = 100, ptype: str = 'price') -> Union[Dict[Any, Any], Any]:
        """以字典形式返回分级A数据
        :param jsl_username: 集思录用户名
        :param jsl_password: 集思路登录密码
        :param avolume: A成交额，单位百万
        :param bvolume: B成交额，单位百万
        :param ptype: 溢价计算方式，price=现价，buy=买一，sell=卖一
        """
        session: requests.Session = requests.session()
        session.headers.update(self._get_headers())
        logindata: Dict[str, Any] = dict(return_url='http://www.jisilu.cn/', user_name=jsl_username, password=jsl_password, net_auto_login='1', _post_type='ajax')
        rep = session.post(self.__jsl_login_url, data=logindata)
        if rep.json()['err'] is not None:
            return rep.json()
        fundarb_url: str = self.__fundarb_url.format(ctime=int(time.time()))
        pdata: Dict[str, Any] = dict(avolume=avolume, bvolume=bvolume, ptype=ptype, is_search='1', market=['sh', 'sz'], rp='50')
        rep = session.post(fundarb_url, data=pdata)
        fundajson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatfundajson(fundajson)
        self.__fundarb = data
        return self.__fundarb

    def etfindex(self, index_id: str = '', min_volume: float = 0, max_discount: Optional[Union[str, float]] = None, min_discount: Optional[Union[str, float]] = None) -> Dict[Any, Any]:
        """
        以字典形式返回 指数ETF 数据
        :param index_id: 获取指定的指数
        :param min_volume: 最小成交量
        :param min_discount:最低溢价率, 适用于溢价套利, 格式 "-1.2%", "-1.2", -0.012 三种均可
        :param max_discount: 最高溢价率, 适用于折价套利, 格式 "-1.2%", "-1.2", -0.012 三种均可
        :return: {"fund_id":{}}
        """
        url: str = self.__etf_index_url.format(ctime=int(time.time()))
        etf_json: Dict[str, Any] = requests.get(url).json()
        data: Dict[Any, Any] = self.formatetfindexjson(etf_json)
        if index_id:
            data = {fund_id: cell for fund_id, cell in data.items() if cell['index_id'] == index_id}
        if min_volume:
            data = {fund_id: cell for fund_id, cell in data.items() if float(cell['volume']) >= min_volume}
        if min_discount is not None:
            if isinstance(min_discount, str):
                if min_discount.endswith('%'):
                    min_discount = self.percentage2float(min_discount)
                else:
                    min_discount = float(min_discount) / 100.0
            data = {fund_id: cell for fund_id, cell in data.items() if self.percentage2float(cell['discount_rt']) >= min_discount}
        if max_discount is not None:
            if isinstance(max_discount, str):
                if max_discount.endswith('%'):
                    max_discount = self.percentage2float(max_discount)
                else:
                    max_discount = float(max_discount) / 100.0
            data = {fund_id: cell for fund_id, cell in data.items() if self.percentage2float(cell['discount_rt']) <= max_discount}
        self.__etfindex = data
        return self.__etfindex

    def qdii(self, min_volume: float = 0) -> Dict[Any, Any]:
        """以字典形式返回QDII数据
        :param min_volume:最小交易量，单位万元
        """
        url: str = self.__qdii_url.format(ctime=int(time.time()))
        rep = requests.get(url)
        fundjson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatjisilujson(fundjson)
        data = {x: y for x, y in data.items() if y['notes'] != '估值有问题'}
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['volume']) > min_volume}
        self.__qdii = data
        return self.__qdii

    def cb(self, min_volume: float = 0, cookie: Optional[str] = None) -> Dict[Any, Any]:
        """以字典形式返回可转债数据
        :param min_volume:最小交易量，单位万元
        :param cookie: 登陆凭证，可以从浏览器获取的对应 Cookie
        """
        url: str = self.__cb_url.format(ctime=int(time.time()))
        session: requests.Session = requests.Session()
        rep = session.get(url, headers=self._get_headers())
        fundjson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[Any, Any] = self.formatjisilujson(fundjson)
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['volume']) > min_volume}
        self.__cb = data
        return self.__cb

if __name__ == '__main__':
    Jsl().etfindex(index_id='000016', min_volume=0, max_discount='-0.4', min_discount='-1.3%')