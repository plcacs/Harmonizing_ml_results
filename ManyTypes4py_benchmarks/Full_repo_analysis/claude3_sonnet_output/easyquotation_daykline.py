# pylint: disable=line-too-long
import json
import re
from typing import Dict, List, Optional, Any, Union
from . import basequotation

class DayKline(basequotation.BaseQuotation):
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
            raw_stocks_detail: str = re.search('=(.*)', raw_quotation).group(1)
            stock_details: Dict[str, Any] = json.loads(raw_stocks_detail)
            for (stock, value) in stock_details['data'].items():
                stock_code: str = stock[2:]
                if 'qfqday' in value:
                    stock_detail: Optional[List[Any]] = value['qfqday']
                else:
                    stock_detail: Optional[List[Any]] = value.get('day')
                if stock_detail is None:
                    print('stock code data not find %s' % stock_code)
                    continue
                stock_dict[stock_code] = stock_detail
                break
        return stock_dict

if __name__ == '__main__':
    pass
