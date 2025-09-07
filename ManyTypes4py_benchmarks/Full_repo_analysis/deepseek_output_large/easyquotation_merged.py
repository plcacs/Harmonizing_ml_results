from typing import Any, Dict, List, Optional, Union, Tuple, Pattern, Match, Iterable, Callable, Type, overload
import abc
import json
import multiprocessing.pool
import os
import re
import time
import warnings
import requests
from datetime import datetime

STOCK_CODE_PATH: str = os.path.join(os.path.dirname(__file__), 'stock_codes.conf')

def update_stock_codes() -> List[str]:
    """获取所有股票 ID 到 all_stock_code 目录下"""
    response: requests.Response = requests.get('http://www.shdjt.com/js/lib/astock.js')
    stock_codes: List[str] = re.findall('~([a-z0-9]*)`', response.text)
    with open(STOCK_CODE_PATH, 'w') as f:
        f.write(json.dumps(dict(stock=stock_codes)))
    return stock_codes

def get_stock_codes(realtime: bool = False) -> List[str]:
    """获取所有股票 ID 到 all_stock_code 目录下"""
    if realtime:
        return update_stock_codes()
    with open(STOCK_CODE_PATH) as f:
        return json.load(f)['stock']

def get_stock_type(stock_code: str) -> str:
    """判断股票ID对应的证券市场
    匹配规则
    ['50', '51', '60', '90', '110'] 为 sh
    ['00', '13', '18', '15', '16', '18', '20', '30', '39', '115'] 为 sz
    ['5', '6', '9'] 开头的为 sh， 其余为 sz
    :param stock_code:股票ID, 若以 'sz', 'sh' 开头直接返回对应类型，否则使用内置规则判断
    :return 'sh' or 'sz'"""
    assert type(stock_code) is str, 'stock code need str type'
    sh_head: Tuple[str, ...] = ('50', '51', '60', '90', '110', '113', '118', '132', '204', '5', '6', '9', '7')
    if stock_code.startswith(('sh', 'sz', 'zz')):
        return stock_code[:2]
    else:
        return 'sh' if stock_code.startswith(sh_head) else 'sz'

class BaseQuotation(metaclass=abc.ABCMeta):
    """行情获取基类"""
    max_num: int = 800

    @property
    @abc.abstractmethod
    def stock_api(self) -> str:
        """
        行情 api 地址
        """
        pass

    def __init__(self) -> None:
        self._session: requests.Session = requests.session()
        stock_codes: List[str] = self.load_stock_codes()
        self.stock_list: List[str] = self.gen_stock_list(stock_codes)

    def gen_stock_list(self, stock_codes: List[str]) -> List[str]:
        stock_with_exchange_list: List[str] = self._gen_stock_prefix(stock_codes)
        if self.max_num > len(stock_with_exchange_list):
            request_list: str = ','.join(stock_with_exchange_list)
            return [request_list]
        stock_list: List[str] = []
        for i in range(0, len(stock_codes), self.max_num):
            request_list = ','.join(stock_with_exchange_list[i:i + self.max_num])
            stock_list.append(request_list)
        return stock_list

    def _gen_stock_prefix(self, stock_codes: List[str]) -> List[str]:
        return [get_stock_type(code) + code[-6:] for code in stock_codes]

    @staticmethod
    def load_stock_codes() -> List[str]:
        with open(STOCK_CODE_PATH) as f:
            return json.load(f)['stock']

    @property
    def all(self) -> Dict[str, Any]:
        warnings.warn('use market_snapshot instead', DeprecationWarning)
        return self.get_stock_data(self.stock_list)

    @property
    def all_market(self) -> Dict[str, Any]:
        """return quotation with stock_code prefix key"""
        return self.get_stock_data(self.stock_list, prefix=True)

    def stocks(self, stock_codes: Union[str, List[str]], prefix: bool = False) -> Dict[str, Any]:
        return self.real(stock_codes, prefix)

    def real(self, stock_codes: Union[str, List[str]], prefix: bool = False) -> Dict[str, Any]:
        """return specific stocks real quotation
        :param stock_codes: stock code or list of stock code,
                when prefix is True, stock code must start with sh/sz
        :param prefix: if prefix i True, stock_codes must contain sh/sz market
            flag. If prefix is False, index quotation can't return
        :return quotation dict, key is stock_code, value is real quotation.
            If prefix with True, key start with sh/sz market flag

        """
        if not isinstance(stock_codes, list):
            stock_codes_list: List[str] = [stock_codes]
        else:
            stock_codes_list = stock_codes
        stock_list: List[str] = self.gen_stock_list(stock_codes_list)
        return self.get_stock_data(stock_list, prefix=prefix)

    def market_snapshot(self, prefix: bool = False) -> Dict[str, Any]:
        """return all market quotation snapshot
        :param prefix: if prefix is True, return quotation dict's  stock_code
             key start with sh/sz market flag
        """
        return self.get_stock_data(self.stock_list, prefix=prefix)

    def get_stocks_by_range(self, params: str) -> str:
        headers: Dict[str, str] = self._get_headers()
        r: requests.Response = self._session.get(self.stock_api + params, headers=headers)
        return r.text

    def _get_headers(self) -> Dict[str, str]:
        return {'Accept-Encoding': 'gzip, deflate, sdch', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.100 Safari/537.36'}

    def get_stock_data(self, stock_list: List[str], **kwargs: Any) -> Dict[str, Any]:
        """获取并格式化股票信息"""
        res: List[str] = self._fetch_stock_data(stock_list)
        return self.format_response_data(res, **kwargs)

    def _fetch_stock_data(self, stock_list: List[str]) -> List[str]:
        """获取股票信息"""
        pool: multiprocessing.pool.ThreadPool = multiprocessing.pool.ThreadPool(len(stock_list))
        try:
            res: List[str] = pool.map(self.get_stocks_by_range, stock_list)
        finally:
            pool.close()
        return [d for d in res if d is not None]

    def format_response_data(self, rep_data: List[str], **kwargs: Any) -> Dict[str, Any]:
        pass

class Boc:
    """中行美元最新汇率"""
    url: str = 'http://www.boc.cn/sourcedb/whpj/'

    def get_exchange_rate(self, currency: str = 'usa') -> Dict[str, str]:
        rep: requests.Response = requests.get(self.url)
        data: List[str] = re.findall('<td>(.*?)</td>', rep.text)
        if currency == 'usa':
            return {'sell': data[-13], 'buy': data[-15]}
        return {}

class DayKline(BaseQuotation):
    """腾讯免费行情获取"""
    max_num: int = 1

    @property
    def stock_api(self) -> str:
        return 'http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?_var=kline_dayqfq&param='

    def _gen_stock_prefix(self, stock_codes: List[str], day: int = 1500) -> List[str]:
        return ['hk{},day,,,{},qfq'.format(code, day) for code in stock_codes]

    def format_response_data(self, rep_data: List[str], **kwargs: Any) -> Dict[str, Any]:
        stock_dict: Dict[str, Any] = {}
        for raw_quotation in rep_data:
            raw_stocks_detail_match: Optional[Match[str]] = re.search('=(.*)', raw_quotation)
            if raw_stocks_detail_match is None:
                continue
            raw_stocks_detail: str = raw_stocks_detail_match.group(1)
            stock_details: Dict[str, Any] = json.loads(raw_stocks_detail)
            for (stock, value) in stock_details['data'].items():
                stock_code: str = stock[2:]
                if 'qfqday' in value:
                    stock_detail: Optional[List[List[str]]] = value['qfqday']
                else:
                    stock_detail = value.get('day')
                if stock_detail is None:
                    print('stock code data not find %s' % stock_code)
                    continue
                stock_dict[stock_code] = stock_detail
                break
        return stock_dict

class HKQuote(BaseQuotation):
    """腾讯免费行情获取"""

    @property
    def stock_api(self) -> str:
        return 'http://sqt.gtimg.cn/utf8/q='

    def _gen_stock_prefix(self, stock_codes: List[str]) -> List[str]:
        return ['r_hk{}'.format(code) for code in stock_codes]

    def format_response_data(self, rep_data: List[str], **kwargs: Any) -> Dict[str, Any]:
        stocks_detail: str = ''.join(rep_data)
        stock_dict: Dict[str, Any] = {}
        for raw_quotation in re.findall('v_r_hk\\d+=".*?"', stocks_detail):
            quotation_match: Optional[Match[str]] = re.search('"(.*?)"', raw_quotation)
            if quotation_match is None:
                continue
            quotation: List[str] = quotation_match.group(1).split('~')
            stock_dict[quotation[2]] = dict(lotSize=float(quotation[0]), name=quotation[1], price=float(quotation[3]), lastPrice=float(quotation[4]), openPrice=float(quotation[5]), amount=float(quotation[6]), time=quotation[30], dtd=float(quotation[32]), high=float(quotation[33]), low=float(quotation[34]))
        return stock_dict

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
        self.__funda: Optional[Dict[str, Any]] = None
        self.__fundm: Optional[Dict[str, Any]] = None
        self.__fundb: Optional[Dict[str, Any]] = None
        self.__fundarb: Optional[Dict[str, Any]] = None
        self.__etfindex: Optional[Dict[str, Any]] = None
        self.__qdii: Optional[Dict[str, Any]] = None
        self.__cb: Optional[Dict[str, Any]] = None
        self._cookie: Optional[str] = None

    def set_cookie(self, cookie: str) -> None:
        self._cookie = cookie

    def _get_headers(self) -> Dict[str, str]:
        default: Dict[str, str] = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'}
        if self._cookie:
            default = {**default, 'Cookie': self._cookie}
        return default

    @staticmethod
    def formatfundajson(fundajson: Dict[str, Any]) -> Dict[str, Any]:
        """格式化集思录返回的json数据,以字典形式保存"""
        result: Dict[str, Any] = {}
        for row in fundajson['rows']:
            funda_id: str = row['id']
            cell: Dict[str, Any] = row['cell']
            result[funda_id] = cell
        return result

    @staticmethod
    def formatfundbjson(fundbjson: Dict[str, Any]) -> Dict[str, Any]:
        """格式化集思录返回的json数据,以字典形式保存"""
        result: Dict[str, Any] = {}
        for row in fundbjson['rows']:
            cell: Dict[str, Any] = row['cell']
            fundb_id: str = cell['fundb_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatetfindexjson(fundbjson: Dict[str, Any]) -> Dict[str, Any]:
        """格式化集思录返回 指数ETF 的json数据,以字典形式保存"""
        result: Dict[str, Any] = {}
        for row in fundbjson['rows']:
            cell: Dict[str, Any] = row['cell']
            fundb_id: str = cell['fund_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatjisilujson(data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for row in data['rows']:
            cell: Dict[str, Any] = row['cell']
            id_: str = row['id']
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

    def funda(self, fields: Optional[List[str]] = None, min_volume: int = 0, min_discount: int = 0, ignore_nodown: bool = False, forever: bool = False) -> Dict[str, Any]:
        """以字典形式返回分级A数据
        :param fields:利率范围，形如['+3.0%', '6.0%']
        :param min_volume:最小交易量，单位万元
        :param min_discount:最小折价率, 单位%
        :param ignore_nodown:是否忽略无下折品种,默认 False
        :param forever: 是否选择永续品种,默认 False
        """
        if fields is None:
            fields = []
        self.__funda_url = self.__funda_url.format(ctime=int(time.time()))
        rep: requests.Response = requests.get(self.__funda_url)
        fundajson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[str, Any] = self.formatfundajson(fundajson)
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

    def fundm(self) -> Dict[str, Any]:
        """以字典形式返回分级母基数据
        """
        self.__fundm_url = self.__fundm_url.format(ctime=int(time.time()))
        rep: requests.Response = requests.get(self.__fundm_url)
        fundmjson: Dict[str, Any] = json.loads(rep.text)
        data: Dict[str, Any] = self.formatfundajson(fundmjson)
        self.__fundm = data
        return self.__fundm

    def fundb(self, fields: Optional[List[str]] = None, min_volume: int = 0, min_discount: int = 0, forever: bool = False) -> Dict[str, Any]:
        """以字典形式返回分级B数据
        :param fields:利率范围，形如['+3.0%', '6.0%']
        :param min_volume:最小交易量，单位万元
        :param min_discount:最小折价率, 单位%
        :param forever: 是否选择永续品种,默认 False
        """
        if fields is None:
            fields = []
        self.__fundb_url = self.__fundb_url.format(ctime=int(time.time()))
        rep