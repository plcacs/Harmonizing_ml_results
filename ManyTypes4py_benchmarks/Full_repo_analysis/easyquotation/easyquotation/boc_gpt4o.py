# coding:utf8
import re
import requests
from typing import Dict

class Boc:
    """中行美元最新汇率"""

    url: str = "http://www.boc.cn/sourcedb/whpj/"

    def get_exchange_rate(self, currency: str = "usa") -> Dict[str, str]:
        rep = requests.get(self.url)
        data = re.findall(r"<td>(.*?)</td>", rep.text)

        if currency == "usa":
            return {"sell": data[-13], "buy": data[-15]}
        return {}
