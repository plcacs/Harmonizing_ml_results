import abc
import json
import multiprocessing.pool
import warnings
from typing import Dict, List, Optional, Union, Any, Set, TypeVar, ClassVar
import requests
from . import helpers

class BaseQuotation(metaclass=abc.ABCMeta):
    """行情获取基类"""
    max_num: ClassVar[int] = 800

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
            request_list: str = ','.join(stock_with_exchange_list[i:i + self.max_num])
            stock_list.append(request_list)
        return stock_list

    def _gen_stock_prefix(self, stock_codes: List[str]) -> List[str]:
        return [helpers.get_stock_type(code) + code[-6:] for code in stock_codes]

    @staticmethod
    def load_stock_codes() -> List[str]:
        with open(helpers.STOCK_CODE_PATH) as f:
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
            stock_codes = [stock_codes]
        stock_list: List[str] = self.gen_stock_list(stock_codes)
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
            res: List[Optional[str]] = pool.map(self.get_stocks_by_range, stock_list)
        finally:
            pool.close()
        return [d for d in res if d is not None]

    def format_response_data(self, rep_data: List[str], **kwargs: Any) -> Dict[str, Any]:
        pass
