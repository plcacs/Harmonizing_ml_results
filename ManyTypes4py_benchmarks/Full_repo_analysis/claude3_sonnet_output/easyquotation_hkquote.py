import re
from typing import Dict, List, Any
from . import basequotation

class HKQuote(basequotation.BaseQuotation):
    """腾讯免费行情获取"""

    @property
    def stock_api(self) -> str:
        return 'http://sqt.gtimg.cn/utf8/q='

    def _gen_stock_prefix(self, stock_codes: List[str]) -> List[str]:
        return ['r_hk{}'.format(code) for code in stock_codes]

    def format_response_data(self, rep_data: List[str], **kwargs: Any) -> Dict[str, Dict[str, float | str]]:
        stocks_detail = ''.join(rep_data)
        stock_dict: Dict[str, Dict[str, float | str]] = {}
        for raw_quotation in re.findall('v_r_hk\\d+=".*?"', stocks_detail):
            quotation = re.search('"(.*?)"', raw_quotation).group(1).split('~')
            stock_dict[quotation[2]] = dict(lotSize=float(quotation[0]), name=quotation[1], price=float(quotation[3]), lastPrice=float(quotation[4]), openPrice=float(quotation[5]), amount=float(quotation[6]), time=quotation[30], dtd=float(quotation[32]), high=float(quotation[33]), low=float(quotation[34]))
        return stock_dict
