#!/usr/bin/env python3
import unittest
from unittest import mock
import easyquotation
from typing import List, Tuple, Optional, Dict, Any


class TestTimeklineQuotation(unittest.TestCase):
    MOCK_RESPONSE_DATA: List[Tuple[str, str]] = [
        ('000001', 'min_data="\\n\\\ndate:180413\\n\\\n0930 11.64 29727\\n\\\n0931 11.65 52410\\n\\\n";')
    ]

    def setUp(self) -> None:
        self._obj: Any = easyquotation.use('timekline')

    @mock.patch('easyquotation.timekline.basequotation.BaseQuotation._fetch_stock_data')
    def test_fetch_stock_data(self, mock_super_fetch: mock.MagicMock) -> None:
        test_cases: List[
            Tuple[List[str], List[Optional[str]], List[Tuple[str, str]]]
        ] = [
            (['000001'], ['test_data'], [('000001', 'test_data')]),
            (['000001', '000002'], ['test_data', None], [('000001', 'test_data')]),
            ([], [], [])
        ]
        for (stock_list, resp_data, expected) in test_cases:
            mock_super_fetch.return_value = resp_data
            res: List[Tuple[str, str]] = self._obj._fetch_stock_data(stock_list)
            self.assertListEqual(res, expected)

    def test_format_response_data(self) -> None:
        expected: Dict[str, Dict[str, object]] = {
            '000001': {
                'date': '20180413',
                'time_data': [['0930', '11.64', '29727'], ['0931', '11.65', '52410']]
            }
        }
        result: Dict[str, Dict[str, object]] = self._obj.format_response_data(self.MOCK_RESPONSE_DATA)
        self.assertDictEqual(result, expected)


if __name__ == '__main__':
    unittest.main()