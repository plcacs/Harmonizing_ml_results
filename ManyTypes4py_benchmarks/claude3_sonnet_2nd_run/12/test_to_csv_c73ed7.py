from datetime import datetime
from io import StringIO
from typing import Any, Dict, Optional, Union

import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.io.common import get_handle

class TestSeriesToCSV:

    def read_csv(self, path: Union[str, StringIO], **kwargs: Any) -> Series:
        params: Dict[str, Any] = {'index_col': 0, 'header': None}
        params.update(**kwargs)
        header = params.get('header')
        out = pd.read_csv(path, **params).squeeze('columns')
        if header is None:
            out.name = out.index.name = None
        return out

    def test_from_csv(self, datetime_series: Series, string_series: Series, temp_file: str) -> None:
        datetime_series.index = datetime_series.index._with_freq(None)
        path = temp_file
        datetime_series.to_csv(path, header=False)
        ts = self.read_csv(path, parse_dates=True)
        expected = datetime_series.copy()
        expected.index = expected.index.as_unit('s')
        tm.assert_series_equal(expected, ts, check_names=False)
        assert ts.name is None
        assert ts.index.name is None
        datetime_series.to_csv(path, header=True)
        ts_h = self.read_csv(path, header=0)
        assert ts_h.name == 'ts'
        string_series.to_csv(path, header=False)
        series = self.read_csv(path)
        tm.assert_series_equal(string_series, series, check_names=False)
        assert series.name is None
        assert series.index.name is None
        string_series.to_csv(path, header=True)
        series_h = self.read_csv(path, header=0)
        assert series_h.name == 'series'
        with open(path, 'w', encoding='utf-8') as outfile:
            outfile.write('1998-01-01|1.0\n1999-01-01|2.0')
        series = self.read_csv(path, sep='|', parse_dates=True)
        check_series = Series({datetime(1998, 1, 1): 1.0, datetime(1999, 1, 1): 2.0})
        check_series.index = check_series.index.as_unit('s')
        tm.assert_series_equal(check_series, series)
        series = self.read_csv(path, sep='|', parse_dates=False)
        check_series = Series({'1998-01-01': 1.0, '1999-01-01': 2.0})
        tm.assert_series_equal(check_series, series)

    def test_to_csv(self, datetime_series: Series, temp_file: str) -> None:
        datetime_series.to_csv(temp_file, header=False)
        with open(temp_file, newline=None, encoding='utf-8') as f:
            lines = f.readlines()
        assert lines[1] != '\n'
        datetime_series.to_csv(temp_file, index=False, header=False)
        arr = np.loadtxt(temp_file)
        tm.assert_almost_equal(arr, datetime_series.values)

    def test_to_csv_unicode_index(self) -> None:
        buf = StringIO()
        s = Series(['א', 'd2'], index=['א', 'ב'])
        s.to_csv(buf, encoding='UTF-8', header=False)
        buf.seek(0)
        s2 = self.read_csv(buf, index_col=0, encoding='UTF-8')
        tm.assert_series_equal(s, s2)

    def test_to_csv_float_format(self, temp_file: str) -> None:
        ser = Series([0.123456, 0.234567, 0.567567])
        ser.to_csv(temp_file, float_format='%.2f', header=False)
        rs = self.read_csv(temp_file)
        xp = Series([0.12, 0.23, 0.57])
        tm.assert_series_equal(rs, xp)

    def test_to_csv_list_entries(self) -> None:
        s = Series(['jack and jill', 'jesse and frank'])
        split = s.str.split('\\s+and\\s+')
        buf = StringIO()
        split.to_csv(buf, header=False)

    def test_to_csv_path_is_none(self) -> None:
        s = Series([1, 2, 3])
        csv_str = s.to_csv(path_or_buf=None, header=False)
        assert isinstance(csv_str, str)

    @pytest.mark.parametrize('s,encoding', [(Series([0.123456, 0.234567, 0.567567], index=['A', 'B', 'C'], name='X'), None), (Series(['abc', 'def', 'ghi'], name='X'), 'ascii'), (Series(['123', '你好', '世界'], name='中文'), 'gb2312'), (Series(['123', 'Γειά σου', 'Κόσμε'], name='Ελληνικά'), 'cp737')])
    def test_to_csv_compression(self, s: Series, encoding: Optional[str], compression: Any, temp_file: str) -> None:
        filename = temp_file
        s.to_csv(filename, compression=compression, encoding=encoding, header=True)
        result = pd.read_csv(filename, compression=compression, encoding=encoding, index_col=0).squeeze('columns')
        tm.assert_series_equal(s, result)
        with get_handle(filename, 'w', compression=compression, encoding=encoding) as handles:
            s.to_csv(handles.handle, encoding=encoding, header=True)
        result = pd.read_csv(filename, compression=compression, encoding=encoding, index_col=0).squeeze('columns')
        tm.assert_series_equal(s, result)
        with tm.decompress_file(filename, compression) as fh:
            text = fh.read().decode(encoding or 'utf8')
            assert s.name in text
        with tm.decompress_file(filename, compression) as fh:
            tm.assert_series_equal(s, pd.read_csv(fh, index_col=0, encoding=encoding).squeeze('columns'))

    def test_to_csv_interval_index(self, using_infer_string: Any, temp_file: str) -> None:
        s = Series(['foo', 'bar', 'baz'], index=pd.interval_range(0, 3))
        s.to_csv(temp_file, header=False)
        result = self.read_csv(temp_file, index_col=0)
        expected = s
        expected.index = expected.index.astype('str')
        tm.assert_series_equal(result, expected)
