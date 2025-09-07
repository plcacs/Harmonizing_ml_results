import unittest
import easyquotation
from typing import Dict, List, Any, Pattern, Union

class TestEasyquotation(unittest.TestCase):

    def test_stock_code_with_prefix(self) -> None:
        cases: List[str] = ['sina', 'qq']
        for src in cases:
            q: Any = easyquotation.use(src)
            data: Dict[str, Any] = q.market_snapshot(prefix=True)
            for k in data.keys():
                self.assertRegex(k, '(sh|sz)\\d{6}')

    def test_all(self) -> None:
        cases: List[str] = ['sina', 'qq']
        for src in cases:
            q: Any = easyquotation.use(src)
            data: Dict[str, Any] = q.market_snapshot()
            for k in data.keys():
                self.assertRegex(k, '\\d{6}')

class TestHqouteQuotatin(unittest.TestCase):
    MOCK_RESPONSE_DATA: str = 'v_r_hk00700="100~腾讯控股~00700~409.600~412.200~414.000~41115421.0~0~0~409.600~0~0~0~0~0~0~0~0~0~409.600~0~0~0~0~0~0~0~0~0~41115421.0~2018/03/29 16:08:11~-2.600~-0.63~417.000~405.200~409.600~41115421.0~16899465578.300~0~44.97~~0~0~2.86~38909.443~38909.443~TENCENT~0.21~476.600~222.400~0.40~0~0~0~0~0~0~42.25~12.70~"; v_r_hk00980="100~联华超市~00980~2.390~2.380~2.380~825000.0~0~0~2.390~0~0~0~0~0~0~0~0~0~2.390~0~0~0~0~0~0~0~0~0~825000.0~2018/03/29 16:08:11~0.010~0.42~2.440~2.330~2.390~825000.0~1949820.000~0~-5.38~~0~0~4.62~8.905~26.758~LIANHUA~0.00~4.530~2.330~1.94~0~0~0~0~0~0~-0.01~0.94~";'

    def setUp(self) -> None:
        self._obj: Any = easyquotation.use('hkquote')

    def test_format_response_data(self) -> None:
        excepted: Dict[str, Dict[str, Union[float, str]]] = {'00700': {'amount': 41115421.0, 'dtd': -0.63, 'high': 417.0, 'lastPrice': 412.2, 'lotSize': 100.0, 'low': 405.2, 'name': '腾讯控股', 'openPrice': 414.0, 'price': 409.6, 'time': '2018/03/29 16:08:11'}, '00980': {'amount': 825000.0, 'dtd': 0.42, 'high': 2.44, 'lastPrice': 2.38, 'lotSize': 100.0, 'low': 2.33, 'name': '联华超市', 'openPrice': 2.38, 'price': 2.39, 'time': '2018/03/29 16:08:11'}}
        result: Dict[str, Dict[str, Union[float, str]]] = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, excepted)

class TestDayklineQuotatin(unittest.TestCase):
    MOCK_RESPONSE_DATA: List[str] = ['kline_dayqfq={"code":0,"msg":"","data":{"hk00001":{"qfqday":[["2018-04-09","91.00","91.85","93.50","91.00","8497462.00"]],"qt":{"hk00001":["100","长和","00001","91.850","91.500","91.000","8497462.0","0","0","91.850","0","0","0","0","0","0","0","0","0","91.850","0","0","0","0","0","0","0","0","0","8497462.0","2018\\/04\\/09 16:08:10","0.350","0.38","93.500","91.000","91.850","8497462.0","781628889.560","0","10.09","","0","0","2.73","3543.278","3543.278","CKH HOLDINGS","3.10","108.900","91.000","1.89","0","0","0","0","0","0","7.67","0.10",""],"market":["2018-04-09 22:36:01|HK_close_已收盘|SH_close_已收盘|SZ_close_已收盘|US_open_交易中|SQ_close_已休市|DS_close_已休市|ZS_close_已休市"]},"prec":"91.50","vcm":"","version":"4"}}}']

    def setUp(self) -> None:
        self._obj: Any = easyquotation.use('daykline')

    def test_format_response_data(self) -> None:
        excepted: Dict[str, List[List[Union[str, float]]]] = {'00001': [['2018-04-09', '91.00', '91.85', '93.50', '91.00', '8497462.00']]}
        result: Dict[str, List[List[Union[str, float]]]] = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, excepted)

if __name__ == '__main__':
    unittest.main()
