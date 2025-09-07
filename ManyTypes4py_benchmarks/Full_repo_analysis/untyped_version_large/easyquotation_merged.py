# Merged Python files from easyquotation
# Total files: 17


# File: easyquotation\__init__.py

from .api import *
from .helpers import get_stock_codes, update_stock_codes
__version__ = '0.7.5'
__author__ = 'shidenggui'

# File: easyquotation\api.py

from . import boc, daykline, hkquote, jsl, sina, tencent, timekline

def use(source):
    if source in ['sina']:
        return sina.Sina()
    if source in ['jsl']:
        return jsl.Jsl()
    if source in ['qq', 'tencent']:
        return tencent.Tencent()
    if source in ['boc']:
        return boc.Boc()
    if source in ['timekline']:
        return timekline.TimeKline()
    if source in ['daykline']:
        return daykline.DayKline()
    if source in ['hkquote']:
        return hkquote.HKQuote()
    raise NotImplementedError

# File: easyquotation\basequotation.py

import abc
import json
import multiprocessing.pool
import warnings
import requests
from . import helpers

class BaseQuotation(metaclass=abc.ABCMeta):
    """行情获取基类"""
    max_num = 800

    @property
    @abc.abstractmethod
    def stock_api(self):
        """
        行情 api 地址
        """
        pass

    def __init__(self):
        self._session = requests.session()
        stock_codes = self.load_stock_codes()
        self.stock_list = self.gen_stock_list(stock_codes)

    def gen_stock_list(self, stock_codes):
        stock_with_exchange_list = self._gen_stock_prefix(stock_codes)
        if self.max_num > len(stock_with_exchange_list):
            request_list = ','.join(stock_with_exchange_list)
            return [request_list]
        stock_list = []
        for i in range(0, len(stock_codes), self.max_num):
            request_list = ','.join(stock_with_exchange_list[i:i + self.max_num])
            stock_list.append(request_list)
        return stock_list

    def _gen_stock_prefix(self, stock_codes):
        return [helpers.get_stock_type(code) + code[-6:] for code in stock_codes]

    @staticmethod
    def load_stock_codes():
        with open(helpers.STOCK_CODE_PATH) as f:
            return json.load(f)['stock']

    @property
    def all(self):
        warnings.warn('use market_snapshot instead', DeprecationWarning)
        return self.get_stock_data(self.stock_list)

    @property
    def all_market(self):
        """return quotation with stock_code prefix key"""
        return self.get_stock_data(self.stock_list, prefix=True)

    def stocks(self, stock_codes, prefix=False):
        return self.real(stock_codes, prefix)

    def real(self, stock_codes, prefix=False):
        """return specific stocks real quotation
        :param stock_codes: stock code or list of stock code,
                when prefix is True, stock code must start with sh/sz
        :param prefix: if prefix i True, stock_codes must contain sh/sz market
            flag. If prefix is False, index quotation can't return
        :return quotation dict, key is stock_code, value is real quotation.
            If prefix with True, key start with sh/sz market flag

        """
        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]
        stock_list = self.gen_stock_list(stock_codes)
        return self.get_stock_data(stock_list, prefix=prefix)

    def market_snapshot(self, prefix=False):
        """return all market quotation snapshot
        :param prefix: if prefix is True, return quotation dict's  stock_code
             key start with sh/sz market flag
        """
        return self.get_stock_data(self.stock_list, prefix=prefix)

    def get_stocks_by_range(self, params):
        headers = self._get_headers()
        r = self._session.get(self.stock_api + params, headers=headers)
        return r.text

    def _get_headers(self):
        return {'Accept-Encoding': 'gzip, deflate, sdch', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.100 Safari/537.36'}

    def get_stock_data(self, stock_list, **kwargs):
        """获取并格式化股票信息"""
        res = self._fetch_stock_data(stock_list)
        return self.format_response_data(res, **kwargs)

    def _fetch_stock_data(self, stock_list):
        """获取股票信息"""
        pool = multiprocessing.pool.ThreadPool(len(stock_list))
        try:
            res = pool.map(self.get_stocks_by_range, stock_list)
        finally:
            pool.close()
        return [d for d in res if d is not None]

    def format_response_data(self, rep_data, **kwargs):
        pass

# File: easyquotation\boc.py

import re
import requests

class Boc:
    """中行美元最新汇率"""
    url = 'http://www.boc.cn/sourcedb/whpj/'

    def get_exchange_rate(self, currency='usa'):
        rep = requests.get(self.url)
        data = re.findall('<td>(.*?)</td>', rep.text)
        if currency == 'usa':
            return {'sell': data[-13], 'buy': data[-15]}
        return {}

# File: easyquotation\boc_gpt4o.py

import re
import requests
from typing import Dict

class Boc:
    """中行美元最新汇率"""
    url = 'http://www.boc.cn/sourcedb/whpj/'

    def get_exchange_rate(self, currency='usa'):
        rep = requests.get(self.url)
        data = re.findall('<td>(.*?)</td>', rep.text)
        if currency == 'usa':
            return {'sell': data[-13], 'buy': data[-15]}
        return {}

# File: easyquotation\daykline.py

"""
# pylint: disable=line-too-long
url = "http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?_var=kline_dayqfq&param=hk00001,day,,,660,qfq&r=0.7773272375526847"

url 参数改动
股票代码 :hk00001
日k线天数：660

更改为需要获取的股票代码和天数例如：

# pylint: disable=line-too-long
url = "http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?_var=kline_dayqfq&param=hk00700,day,,,350,qfq&r=0.7773272375526847"

"""
import json
import re
from . import basequotation

class DayKline(basequotation.BaseQuotation):
    """腾讯免费行情获取"""
    max_num = 1

    @property
    def stock_api(self):
        return 'http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?_var=kline_dayqfq&param='

    def _gen_stock_prefix(self, stock_codes, day=1500):
        return ['hk{},day,,,{},qfq'.format(code, day) for code in stock_codes]

    def format_response_data(self, rep_data, **kwargs):
        stock_dict = {}
        for raw_quotation in rep_data:
            raw_stocks_detail = re.search('=(.*)', raw_quotation).group(1)
            stock_details = json.loads(raw_stocks_detail)
            for (stock, value) in stock_details['data'].items():
                stock_code = stock[2:]
                if 'qfqday' in value:
                    stock_detail = value['qfqday']
                else:
                    stock_detail = value.get('day')
                if stock_detail is None:
                    print('stock code data not find %s' % stock_code)
                    continue
                stock_dict[stock_code] = stock_detail
                break
        return stock_dict
if __name__ == '__main__':
    pass

# File: easyquotation\helpers.py

import json
import os
import re
import requests
STOCK_CODE_PATH = os.path.join(os.path.dirname(__file__), 'stock_codes.conf')

def update_stock_codes():
    """获取所有股票 ID 到 all_stock_code 目录下"""
    response = requests.get('http://www.shdjt.com/js/lib/astock.js')
    stock_codes = re.findall('~([a-z0-9]*)`', response.text)
    with open(STOCK_CODE_PATH, 'w') as f:
        f.write(json.dumps(dict(stock=stock_codes)))
    return stock_codes

def get_stock_codes(realtime=False):
    """获取所有股票 ID 到 all_stock_code 目录下"""
    if realtime:
        return update_stock_codes()
    with open(STOCK_CODE_PATH) as f:
        return json.load(f)['stock']

def get_stock_type(stock_code):
    """判断股票ID对应的证券市场
    匹配规则
    ['50', '51', '60', '90', '110'] 为 sh
    ['00', '13', '18', '15', '16', '18', '20', '30', '39', '115'] 为 sz
    ['5', '6', '9'] 开头的为 sh， 其余为 sz
    :param stock_code:股票ID, 若以 'sz', 'sh' 开头直接返回对应类型，否则使用内置规则判断
    :return 'sh' or 'sz'"""
    assert type(stock_code) is str, 'stock code need str type'
    sh_head = ('50', '51', '60', '90', '110', '113', '118', '132', '204', '5', '6', '9', '7')
    if stock_code.startswith(('sh', 'sz', 'zz')):
        return stock_code[:2]
    else:
        return 'sh' if stock_code.startswith(sh_head) else 'sz'

# File: easyquotation\hkquote.py

"""
url = "http://sqt.gtimg.cn/utf8/q=r_hk00981"

url 参数改动
股票代码 q=r_hk00981
"""
import re
from . import basequotation

class HKQuote(basequotation.BaseQuotation):
    """腾讯免费行情获取"""

    @property
    def stock_api(self):
        return 'http://sqt.gtimg.cn/utf8/q='

    def _gen_stock_prefix(self, stock_codes):
        return ['r_hk{}'.format(code) for code in stock_codes]

    def format_response_data(self, rep_data, **kwargs):
        stocks_detail = ''.join(rep_data)
        stock_dict = {}
        for raw_quotation in re.findall('v_r_hk\\d+=".*?"', stocks_detail):
            quotation = re.search('"(.*?)"', raw_quotation).group(1).split('~')
            stock_dict[quotation[2]] = dict(lotSize=float(quotation[0]), name=quotation[1], price=float(quotation[3]), lastPrice=float(quotation[4]), openPrice=float(quotation[5]), amount=float(quotation[6]), time=quotation[30], dtd=float(quotation[32]), high=float(quotation[33]), low=float(quotation[34]))
        return stock_dict

# File: easyquotation\jsl.py

"""
获取集思路的数据
"""
import json
import time
from typing import Optional
import requests

class Jsl:
    """
    抓取集思路的分级A数据
    """
    __funda_url = 'http://www.jisilu.cn/data/sfnew/funda_list/?___t={ctime:d}'
    __fundb_url = 'http://www.jisilu.cn/data/sfnew/fundb_list/?___t={ctime:d}'
    __fundm_url = 'https://www.jisilu.cn/data/sfnew/fundm_list/?___t={ctime:d}'
    __fundarb_url = 'http://www.jisilu.cn/data/sfnew/arbitrage_vip_list/?___t={ctime:d}'
    __jsl_login_url = 'https://www.jisilu.cn/account/ajax/login_process/'
    __etf_index_url = 'https://www.jisilu.cn/data/etf/etf_list/?___jsl=LST___t={ctime:d}&rp=25&page=1'
    __etf_gold_url = 'https://www.jisilu.cn/jisiludata/etf.php?qtype=pmetf&___t={ctime:d}'
    __etf_money_url = 'https://www.jisilu.cn/data/money_fund/list/?___t={ctime:d}'
    __qdii_url = 'https://www.jisilu.cn/data/qdii/qdii_list/?___t={ctime:d}'
    __cb_url = 'https://www.jisilu.cn/data/cbnew/cb_list/?___t={ctime:d}'

    def __init__(self):
        self.__funda = None
        self.__fundm = None
        self.__fundb = None
        self.__fundarb = None
        self.__etfindex = None
        self.__qdii = None
        self.__cb = None
        self._cookie = None

    def set_cookie(self, cookie):
        self._cookie = cookie

    def _get_headers(self):
        default = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'}
        if self._cookie:
            default = {**default, 'Cookie': self._cookie}
        return default

    @staticmethod
    def formatfundajson(fundajson):
        """格式化集思录返回的json数据,以字典形式保存"""
        result = {}
        for row in fundajson['rows']:
            funda_id = row['id']
            cell = row['cell']
            result[funda_id] = cell
        return result

    @staticmethod
    def formatfundbjson(fundbjson):
        """格式化集思录返回的json数据,以字典形式保存"""
        result = {}
        for row in fundbjson['rows']:
            cell = row['cell']
            fundb_id = cell['fundb_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatetfindexjson(fundbjson):
        """格式化集思录返回 指数ETF 的json数据,以字典形式保存"""
        result = {}
        for row in fundbjson['rows']:
            cell = row['cell']
            fundb_id = cell['fund_id']
            result[fundb_id] = cell
        return result

    @staticmethod
    def formatjisilujson(data):
        result = {}
        for row in data['rows']:
            cell = row['cell']
            id_ = row['id']
            result[id_] = cell
        return result

    @staticmethod
    def percentage2float(per):
        """
        将字符串的百分数转化为浮点数
        :param per:
        :return:
        """
        return float(per.strip('%')) / 100.0

    def funda(self, fields=None, min_volume=0, min_discount=0, ignore_nodown=False, forever=False):
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
        rep = requests.get(self.__funda_url)
        fundajson = json.loads(rep.text)
        data = self.formatfundajson(fundajson)
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

    def fundm(self):
        """以字典形式返回分级母基数据
        """
        self.__fundm_url = self.__fundm_url.format(ctime=int(time.time()))
        rep = requests.get(self.__fundm_url)
        fundmjson = json.loads(rep.text)
        data = self.formatfundajson(fundmjson)
        self.__fundm = data
        return self.__fundm

    def fundb(self, fields=None, min_volume=0, min_discount=0, forever=False):
        """以字典形式返回分级B数据
        :param fields:利率范围，形如['+3.0%', '6.0%']
        :param min_volume:最小交易量，单位万元
        :param min_discount:最小折价率, 单位%
        :param forever: 是否选择永续品种,默认 False
        """
        if fields is None:
            fields = []
        self.__fundb_url = self.__fundb_url.format(ctime=int(time.time()))
        rep = requests.get(self.__fundb_url)
        fundbjson = json.loads(rep.text)
        data = self.formatfundbjson(fundbjson)
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

    def fundarb(self, jsl_username, jsl_password, avolume=100, bvolume=100, ptype='price'):
        """以字典形式返回分级A数据
        :param jsl_username: 集思录用户名
        :param jsl_password: 集思路登录密码
        :param avolume: A成交额，单位百万
        :param bvolume: B成交额，单位百万
        :param ptype: 溢价计算方式，price=现价，buy=买一，sell=卖一
        """
        session = requests.session()
        session.headers.update(self._get_headers())
        logindata = dict(return_url='http://www.jisilu.cn/', user_name=jsl_username, password=jsl_password, net_auto_login='1', _post_type='ajax')
        rep = session.post(self.__jsl_login_url, data=logindata)
        if rep.json()['err'] is not None:
            return rep.json()
        fundarb_url = self.__fundarb_url.format(ctime=int(time.time()))
        pdata = dict(avolume=avolume, bvolume=bvolume, ptype=ptype, is_search='1', market=['sh', 'sz'], rp='50')
        rep = session.post(fundarb_url, data=pdata)
        fundajson = json.loads(rep.text)
        data = self.formatfundajson(fundajson)
        self.__fundarb = data
        return self.__fundarb

    def etfindex(self, index_id='', min_volume=0, max_discount=None, min_discount=None):
        """
        以字典形式返回 指数ETF 数据
        :param index_id: 获取指定的指数
        :param min_volume: 最小成交量
        :param min_discount: 最低溢价率, 适用于溢价套利, 格式 "-1.2%", "-1.2", -0.012 三种均可
        :param max_discount: 最高溢价率, 适用于折价套利, 格式 "-1.2%", "-1.2", -0.012 三种均可
        :return: {"fund_id":{}}
        """
        etf_index_url = self.__etf_index_url.format(ctime=int(time.time()))
        etf_json = requests.get(etf_index_url).json()
        data = self.formatetfindexjson(etf_json)
        if index_id:
            data = {fund_id: cell for (fund_id, cell) in data.items() if cell['index_id'] == index_id}
        if min_volume:
            data = {fund_id: cell for (fund_id, cell) in data.items() if float(cell['volume']) >= min_volume}
        if min_discount is not None:
            if isinstance(min_discount, str):
                if min_discount.endswith('%'):
                    min_discount = self.percentage2float(min_discount)
                else:
                    min_discount = float(min_discount) / 100.0
            data = {fund_id: cell for (fund_id, cell) in data.items() if self.percentage2float(cell['discount_rt']) >= min_discount}
        if max_discount is not None:
            if isinstance(max_discount, str):
                if max_discount.endswith('%'):
                    max_discount = self.percentage2float(max_discount)
                else:
                    max_discount = float(max_discount) / 100.0
            data = {fund_id: cell for (fund_id, cell) in data.items() if self.percentage2float(cell['discount_rt']) <= max_discount}
        self.__etfindex = data
        return self.__etfindex

    def qdii(self, min_volume=0):
        """以字典形式返回QDII数据
        :param min_volume:最小交易量，单位万元
        """
        self.__qdii_url = self.__qdii_url.format(ctime=int(time.time()))
        rep = requests.get(self.__qdii_url)
        fundjson = json.loads(rep.text)
        data = self.formatjisilujson(fundjson)
        data = {x: y for (x, y) in data.items() if y['notes'] != '估值有问题'}
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['volume']) > min_volume}
        self.__qdii = data
        return self.__qdii

    def cb(self, min_volume=0, cookie=None):
        """以字典形式返回可转债数据
        :param min_volume:最小交易量，单位万元
        :cookie: 登陆凭证，可以从浏览器获取的对应 Cookie
        """
        self.__cb_url = self.__cb_url.format(ctime=int(time.time()))
        session = requests.Session()
        rep = session.get(self.__cb_url, headers=self._get_headers())
        fundjson = json.loads(rep.text)
        data = self.formatjisilujson(fundjson)
        if min_volume:
            data = {k: data[k] for k in data if float(data[k]['volume']) > min_volume}
        self.__cb = data
        return self.__cb
if __name__ == '__main__':
    Jsl().etfindex(index_id='000016', min_volume=0, max_discount='-0.4', min_discount='-1.3%')

# File: easyquotation\sina.py

import re
import time
from . import basequotation

class Sina(basequotation.BaseQuotation):
    """新浪免费行情获取"""
    max_num = 800
    grep_detail = re.compile('(\\d+)=[^\\s]([^\\s,]+?)%s%s' % (',([\\.\\d]+)' * 29, ',([-\\.\\d:]+)' * 2))
    grep_detail_with_prefix = re.compile('(\\w{2}\\d+)=[^\\s]([^\\s,]+?)%s%s' % (',([\\.\\d]+)' * 29, ',([-\\.\\d:]+)' * 2))
    del_null_data_stock = re.compile('(\\w{2}\\d+)=\\"\\";')

    @property
    def stock_api(self):
        return f'http://hq.sinajs.cn/rn={int(time.time() * 1000)}&list='

    def _get_headers(self):
        headers = super()._get_headers()
        return {**headers, 'Referer': 'http://finance.sina.com.cn/'}

    def format_response_data(self, rep_data, prefix=False):
        stocks_detail = ''.join(rep_data)
        stocks_detail = self.del_null_data_stock.sub('', stocks_detail)
        stocks_detail = stocks_detail.replace(' ', '')
        grep_str = self.grep_detail_with_prefix if prefix else self.grep_detail
        result = grep_str.finditer(stocks_detail)
        stock_dict = dict()
        for stock_match_object in result:
            stock = stock_match_object.groups()
            stock_dict[stock[0]] = dict(name=stock[1], open=float(stock[2]), close=float(stock[3]), now=float(stock[4]), high=float(stock[5]), low=float(stock[6]), buy=float(stock[7]), sell=float(stock[8]), turnover=int(stock[9]), volume=float(stock[10]), bid1_volume=int(stock[11]), bid1=float(stock[12]), bid2_volume=int(stock[13]), bid2=float(stock[14]), bid3_volume=int(stock[15]), bid3=float(stock[16]), bid4_volume=int(stock[17]), bid4=float(stock[18]), bid5_volume=int(stock[19]), bid5=float(stock[20]), ask1_volume=int(stock[21]), ask1=float(stock[22]), ask2_volume=int(stock[23]), ask2=float(stock[24]), ask3_volume=int(stock[25]), ask3=float(stock[26]), ask4_volume=int(stock[27]), ask4=float(stock[28]), ask5_volume=int(stock[29]), ask5=float(stock[30]), date=stock[31], time=stock[32])
        return stock_dict

# File: easyquotation\tencent.py

import re
from datetime import datetime
from typing import Optional
from . import basequotation

class Tencent(basequotation.BaseQuotation):
    """腾讯免费行情获取"""
    grep_stock_code = re.compile('(?<=_)\\w+')
    max_num = 60

    @property
    def stock_api(self):
        return 'http://qt.gtimg.cn/q='

    def format_response_data(self, rep_data, prefix=False):
        stocks_detail = ''.join(rep_data)
        stock_details = stocks_detail.split(';')
        stock_dict = dict()
        for stock_detail in stock_details:
            stock = stock_detail.split('~')
            if len(stock) <= 49:
                continue
            stock_code = self.grep_stock_code.search(stock[0]).group() if prefix else stock[2]
            stock_dict[stock_code] = {'name': stock[1], 'code': stock_code, 'now': float(stock[3]), 'close': float(stock[4]), 'open': float(stock[5]), 'volume': float(stock[6]) * 100, 'bid_volume': int(stock[7]) * 100, 'ask_volume': float(stock[8]) * 100, 'bid1': float(stock[9]), 'bid1_volume': int(stock[10]) * 100, 'bid2': float(stock[11]), 'bid2_volume': int(stock[12]) * 100, 'bid3': float(stock[13]), 'bid3_volume': int(stock[14]) * 100, 'bid4': float(stock[15]), 'bid4_volume': int(stock[16]) * 100, 'bid5': float(stock[17]), 'bid5_volume': int(stock[18]) * 100, 'ask1': float(stock[19]), 'ask1_volume': int(stock[20]) * 100, 'ask2': float(stock[21]), 'ask2_volume': int(stock[22]) * 100, 'ask3': float(stock[23]), 'ask3_volume': int(stock[24]) * 100, 'ask4': float(stock[25]), 'ask4_volume': int(stock[26]) * 100, 'ask5': float(stock[27]), 'ask5_volume': int(stock[28]) * 100, '最近逐笔成交': stock[29], 'datetime': datetime.strptime(stock[30], '%Y%m%d%H%M%S'), '涨跌': float(stock[31]), '涨跌(%)': float(stock[32]), 'high': float(stock[33]), 'low': float(stock[34]), '价格/成交量(手)/成交额': stock[35], '成交量(手)': int(stock[36]) * 100, '成交额(万)': float(stock[37]) * 10000, 'turnover': self._safe_float(stock[38]), 'PE': self._safe_float(stock[39]), 'unknown': stock[40], 'high_2': float(stock[41]), 'low_2': float(stock[42]), '振幅': float(stock[43]), '流通市值': self._safe_float(stock[44]), '总市值': self._safe_float(stock[45]), 'PB': float(stock[46]), '涨停价': float(stock[47]), '跌停价': float(stock[48]), '量比': self._safe_float(stock[49]), '委差': self._safe_acquire_float(stock, 50), '均价': self._safe_acquire_float(stock, 51), '市盈(动)': self._safe_acquire_float(stock, 52), '市盈(静)': self._safe_acquire_float(stock, 53)}
        return stock_dict

    def _safe_acquire_float(self, stock, idx):
        """
        There are some securities that only have 50 fields. See example below:
        ['
v_sh518801="1',
        '国泰申赎',
        '518801',
        '2.229',
        ......
         '', '0.000', '2.452', '2.006', '"']
        """
        try:
            return self._safe_float(stock[idx])
        except IndexError:
            return None

    def _safe_float(self, s):
        try:
            return float(s)
        except ValueError:
            return None

# File: easyquotation\timekline.py

"""
# pylint: disable=line-too-long
url = "http://data.gtimg.cn/flashdata/hushen/minute/sz000001.js?maxage=110&0.28163905744440854"
"""
import re
from . import basequotation, helpers

class TimeKline(basequotation.BaseQuotation):
    """腾讯免费行情获取"""
    max_num = 1

    @property
    def stock_api(self):
        return 'http://data.gtimg.cn/flashdata/hushen/minute/'

    def _gen_stock_prefix(self, stock_codes):
        return [helpers.get_stock_type(code) + code[-6:] + '.js' for code in stock_codes]

    def _fetch_stock_data(self, stock_list):
        """因为 timekline 的返回没有带对应的股票代码，所以要手动带上"""
        res = super()._fetch_stock_data(stock_list)
        with_stock = []
        for (stock, resp) in zip(stock_list, res):
            if resp is not None:
                with_stock.append((stock, resp))
        return with_stock

    def format_response_data(self, rep_data, **kwargs):
        stock_dict = dict()
        for (stock_code, stock_detail) in rep_data:
            res = re.split('\\\\n\\\\\\n', stock_detail)
            date = '20{}'.format(res[1][-6:])
            time_data = list((d.split() for d in res[2:] if re.match('\\d{4}', d)))
            stock_dict[stock_code] = {'date': date, 'time_data': time_data}
        return stock_dict

# File: setup.py

from os import path
from setuptools import setup
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(name='easyquotation', version='0.7.5', description='A utility for Fetch China Stock Info', long_description=long_description, long_description_content_type='text/markdown', author='shidenggui', author_email='longlyshidenggui@gmail.com', license='BSD', url='https://github.com/shidenggui/easyquotation', keywords='China stock trade', install_requires=['requests', 'six', 'easyutils'], classifiers=['Development Status :: 4 - Beta', 'Programming Language :: Python :: 3.5', 'License :: OSI Approved :: BSD License'], packages=['easyquotation'], package_data={'': ['*.conf']})

# File: tests\__init__.py



# File: tests\test_easyquotation.py

import unittest
import easyquotation

class TestEasyquotation(unittest.TestCase):

    def test_stock_code_with_prefix(self):
        cases = ['sina', 'qq']
        for src in cases:
            q = easyquotation.use(src)
            data = q.market_snapshot(prefix=True)
            for k in data.keys():
                self.assertRegex(k, '(sh|sz)\\d{6}')

    def test_all(self):
        cases = ['sina', 'qq']
        for src in cases:
            q = easyquotation.use(src)
            data = q.market_snapshot()
            for k in data.keys():
                self.assertRegex(k, '\\d{6}')

class TestHqouteQuotatin(unittest.TestCase):
    MOCK_RESPONSE_DATA = 'v_r_hk00700="100~腾讯控股~00700~409.600~412.200~414.000~41115421.0~0~0~409.600~0~0~0~0~0~0~0~0~0~409.600~0~0~0~0~0~0~0~0~0~41115421.0~2018/03/29 16:08:11~-2.600~-0.63~417.000~405.200~409.600~41115421.0~16899465578.300~0~44.97~~0~0~2.86~38909.443~38909.443~TENCENT~0.21~476.600~222.400~0.40~0~0~0~0~0~0~42.25~12.70~"; v_r_hk00980="100~联华超市~00980~2.390~2.380~2.380~825000.0~0~0~2.390~0~0~0~0~0~0~0~0~0~2.390~0~0~0~0~0~0~0~0~0~825000.0~2018/03/29 16:08:11~0.010~0.42~2.440~2.330~2.390~825000.0~1949820.000~0~-5.38~~0~0~4.62~8.905~26.758~LIANHUA~0.00~4.530~2.330~1.94~0~0~0~0~0~0~-0.01~0.94~";'

    def setUp(self):
        self._obj = easyquotation.use('hkquote')

    def test_format_response_data(self):
        excepted = {'00700': {'amount': 41115421.0, 'dtd': -0.63, 'high': 417.0, 'lastPrice': 412.2, 'lotSize': 100.0, 'low': 405.2, 'name': '腾讯控股', 'openPrice': 414.0, 'price': 409.6, 'time': '2018/03/29 16:08:11'}, '00980': {'amount': 825000.0, 'dtd': 0.42, 'high': 2.44, 'lastPrice': 2.38, 'lotSize': 100.0, 'low': 2.33, 'name': '联华超市', 'openPrice': 2.38, 'price': 2.39, 'time': '2018/03/29 16:08:11'}}
        result = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, excepted)

class TestDayklineQuotatin(unittest.TestCase):
    MOCK_RESPONSE_DATA = ['kline_dayqfq={"code":0,"msg":"","data":{"hk00001":{"qfqday":[["2018-04-09","91.00","91.85","93.50","91.00","8497462.00"]],"qt":{"hk00001":["100","长和","00001","91.850","91.500","91.000","8497462.0","0","0","91.850","0","0","0","0","0","0","0","0","0","91.850","0","0","0","0","0","0","0","0","0","8497462.0","2018\\/04\\/09 16:08:10","0.350","0.38","93.500","91.000","91.850","8497462.0","781628889.560","0","10.09","","0","0","2.73","3543.278","3543.278","CKH HOLDINGS","3.10","108.900","91.000","1.89","0","0","0","0","0","0","7.67","0.10",""],"market":["2018-04-09 22:36:01|HK_close_已收盘|SH_close_已收盘|SZ_close_已收盘|US_open_交易中|SQ_close_已休市|DS_close_已休市|ZS_close_已休市"]},"prec":"91.50","vcm":"","version":"4"}}}']

    def setUp(self):
        self._obj = easyquotation.use('daykline')

    def test_format_response_data(self):
        excepted = {'00001': [['2018-04-09', '91.00', '91.85', '93.50', '91.00', '8497462.00']]}
        result = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, excepted)
if __name__ == '__main__':
    unittest.main()

# File: tests\test_sina.py

import unittest
import easyquotation

class TestSina(unittest.TestCase):

    def setUp(self):
        self._sina = easyquotation.use('sina')

    def test_extract_stock_name(self):
        """
        fix https://github.com/shidenggui/easyquotation/issues/51
        """
        stock_name = self._sina.format_response_data(MOCK_DATA)['162411']['name']
        self.assertEqual(stock_name, '华宝油气')

    def test_skip_empty_quotation_stock(self):
        """
        有些股票的行情返回为 var hq_str_sz160923="";
        这时候应该跳过解析
        :return:
        """
        excepted = {'160922': {'ask1': 1.11, 'ask1_volume': 3100, 'ask2': 1.13, 'ask2_volume': 100, 'ask3': 1.169, 'ask3_volume': 2900, 'ask4': 1.17, 'ask4_volume': 12000, 'ask5': 0.0, 'ask5_volume': 0, 'bid1': 1.053, 'bid1_volume': 42000, 'bid2': 1.05, 'bid2_volume': 7500, 'bid3': 0.0, 'bid3_volume': 0, 'bid4': 0.0, 'bid4_volume': 0, 'bid5': 0.0, 'bid5_volume': 0, 'buy': 1.053, 'close': 1.074, 'date': '2019-04-08', 'high': 0.0, 'low': 0.0, 'name': '恒生中小', 'now': 0.0, 'open': 0.0, 'sell': 1.11, 'time': '09:41:45', 'turnover': 0, 'volume': 0.0}, '160924': {'ask1': 1.077, 'ask1_volume': 400, 'ask2': 1.134, 'ask2_volume': 900, 'ask3': 1.16, 'ask3_volume': 9300, 'ask4': 1.196, 'ask4_volume': 1000, 'ask5': 0.0, 'ask5_volume': 0, 'bid1': 1.034, 'bid1_volume': 42000, 'bid2': 1.031, 'bid2_volume': 300, 'bid3': 1.009, 'bid3_volume': 700, 'bid4': 0.992, 'bid4_volume': 500, 'bid5': 0.99, 'bid5_volume': 8000, 'buy': 1.034, 'close': 1.095, 'date': '2019-04-08', 'high': 0.0, 'low': 0.0, 'name': '恒指LOF', 'now': 0.0, 'open': 0.0, 'sell': 1.077, 'time': '09:41:36', 'turnover': 0, 'volume': 0.0}}
        result = self._sina.format_response_data([MOCK_EMPTY_STOCK_DATA])
        self.maxDiff = None
        self.assertDictEqual(result, excepted)
MOCK_DATA = 'var hq_str_sz162411="华宝油气,0.489,0.488,0.491,0.492,0.488,0.490,0.491,133819867,65623147.285,2422992,0.490,4814611,0.489,2663142,0.488,1071900,0.487,357900,0.486,5386166,0.491,8094689,0.492,6087538,0.493,2132373,0.494,5180900,0.495,2019-03-12,15:00:03,00";\n'
MOCK_EMPTY_STOCK_DATA = 'var hq_str_sz160922="恒生中小,0.000,1.074,0.000,0.000,0.000,1.053,1.110,0,0.000,42000,1.053,7500,1.050,0,0.000,0,0.000,0,0.000,3100,1.110,100,1.130,2900,1.169,12000,1.170,0,0.000,2019-04-08,09:41:45,00";\nvar hq_str_sz160923="";\nvar hq_str_sz160924="恒指LOF,0.000,1.095,0.000,0.000,0.000,1.034,1.077,0,0.000,42000,1.034,300,1.031,700,1.009,500,0.992,8000,0.990,400,1.077,900,1.134,9300,1.160,1000,1.196,0,0.000,2019-04-08,09:41:36,00";\n'

# File: tests\test_timekline.py

import unittest
from unittest import mock
import easyquotation

class TestTimeklineQuotation(unittest.TestCase):
    MOCK_RESPONSE_DATA = [('000001', 'min_data="\\n\\\ndate:180413\\n\\\n0930 11.64 29727\\n\\\n0931 11.65 52410\\n\\\n";')]

    def setUp(self):
        self._obj = easyquotation.use('timekline')

    @mock.patch('easyquotation.timekline.basequotation.BaseQuotation._fetch_stock_data')
    def test_fetch_stock_data(self, mock_super_fetch):
        test_cases = [(['000001'], ['test_data'], [('000001', 'test_data')]), (['000001', '000002'], ['test_data', None], [('000001', 'test_data')]), ([], [], [])]
        for (stock_list, resp_data, expected) in test_cases:
            mock_super_fetch.return_value = resp_data
            res = self._obj._fetch_stock_data(stock_list)
            self.assertListEqual(res, expected)

    def test_format_response_data(self):
        excepted = {'000001': {'date': '20180413', 'time_data': [['0930', '11.64', '29727'], ['0931', '11.65', '52410']]}}
        result = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, excepted)
if __name__ == '__main__':
    unittest.main()