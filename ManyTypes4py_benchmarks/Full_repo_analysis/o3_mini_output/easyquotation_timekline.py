import re
from typing import List, Tuple, Dict, Optional, Union, Any
from . import basequotation, helpers

class TimeKline(basequotation.BaseQuotation):
    """腾讯免费行情获取"""
    max_num: int = 1

    @property
    def stock_api(self) -> str:
        return 'http://data.gtimg.cn/flashdata/hushen/minute/'

    def _gen_stock_prefix(self, stock_codes: List[str]) -> List[str]:
        return [helpers.get_stock_type(code) + code[-6:] + '.js' for code in stock_codes]

    def _fetch_stock_data(self, stock_list: List[str]) -> List[Tuple[str, str]]:
        """因为 timekline 的返回没有带对应的股票代码，所以要手动带上"""
        res: List[Optional[str]] = super()._fetch_stock_data(stock_list)
        with_stock: List[Tuple[str, str]] = []
        for stock, resp in zip(stock_list, res):
            if resp is not None:
                with_stock.append((stock, resp))
        return with_stock

    def format_response_data(self, rep_data: List[Tuple[str, str]], **kwargs: Any) -> Dict[str, Dict[str, Union[str, List[List[str]]]]]:
        stock_dict: Dict[str, Dict[str, Union[str, List[List[str]]]]] = {}
        for stock_code, stock_detail in rep_data:
            res = re.split('\\\\n\\\\\\n', stock_detail)
            date: str = '20{}'.format(res[1][-6:])
            time_data: List[List[str]] = [d.split() for d in res[2:] if re.match(r'\d{4}', d)]
            stock_dict[stock_code] = {'date': date, 'time_data': time_data}
        return stock_dict