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

    def qdii(self, min_volume=0):
        """以字典形式返回QDII数据
        :param min_volume:最小交易量，单位万元
        """
        self.__qdii_url = self.__qdii_url.format(ctime=int(time.time()))
        rep = requests.get(self.__qdii_url)
        fundjson = json.loads(rep.text)
        data = self.formatjisilujson(fundjson)
        data = {x: y for x, y in data.items() if y['notes'] != '估值有问题'}
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