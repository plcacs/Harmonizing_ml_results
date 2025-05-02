from collections.abc import Iterator
from functools import partial
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import NA, DataFrame, MultiIndex, Series, Timestamp, date_range, read_csv, read_html, to_datetime
import pandas._testing as tm
from pandas.io.common import file_path_to_url

@pytest.fixture(params=['chinese_utf-16.html', 'chinese_utf-32.html', 'chinese_utf-8.html', 'letz_latin1.html'])
def html_encoding_file(request: pytest.FixtureRequest, datapath: Any) -> str:
    """Parametrized fixture for HTML encoding test filenames."""
    return datapath('io', 'data', 'html_encoding', request.param)

def assert_framelist_equal(list1: List[DataFrame], list2: List[DataFrame], *args: Any, **kwargs: Any) -> None:
    assert len(list1) == len(list2), f'lists are not of equal size len(list1) == {len(list1)}, len(list2) == {len(list2)}'
    msg = 'not all list elements are DataFrames'
    both_frames = all(map(lambda x, y: isinstance(x, DataFrame) and isinstance(y, DataFrame), list1, list2))
    assert both_frames, msg
    for frame_i, frame_j in zip(list1, list2):
        tm.assert_frame_equal(frame_i, frame_j, *args, **kwargs)
        assert not frame_i.empty, 'frames are both empty'

def test_bs4_version_fails(monkeypatch: pytest.MonkeyPatch, datapath: Any) -> None:
    bs4 = pytest.importorskip('bs4')
    pytest.importorskip('html5lib')
    monkeypatch.setattr(bs4, '__version__', '4.2')
    with pytest.raises(ImportError, match='Pandas requires version'):
        read_html(datapath('io', 'data', 'html', 'spam.html'), flavor='bs4')

def test_invalid_flavor() -> None:
    url = 'google.com'
    flavor = 'invalid flavor'
    msg = '\\{' + flavor + '\\} is not a valid set of flavors'
    with pytest.raises(ValueError, match=msg):
        read_html(StringIO(url), match='google', flavor=flavor)

def test_same_ordering(datapath: Any) -> None:
    pytest.importorskip('bs4')
    pytest.importorskip('lxml')
    pytest.importorskip('html5lib')
    filename = datapath('io', 'data', 'html', 'valid_markup.html')
    dfs_lxml = read_html(filename, index_col=0, flavor=['lxml'])
    dfs_bs4 = read_html(filename, index_col=0, flavor=['bs4'])
    assert_framelist_equal(dfs_lxml, dfs_bs4)

@pytest.fixture(params=[pytest.param('bs4', marks=[td.skip_if_no('bs4'), td.skip_if_no('html5lib')]), pytest.param('lxml', marks=td.skip_if_no('lxml'))])
def flavor_read_html(request: pytest.FixtureRequest) -> partial:
    return partial(read_html, flavor=request.param)

class TestReadHtml:

    def test_literal_html_deprecation(self, flavor_read_html: partial) -> None:
        msg = '\\[Errno 2\\] No such file or director'
        with pytest.raises(FileNotFoundError, match=msg):
            flavor_read_html('<table>\n                <thead>\n                    <tr>\n                        <th>A</th>\n                        <th>B</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td>1</td>\n                        <td>2</td>\n                    </tr>\n                </tbody>\n                <tbody>\n                    <tr>\n                        <td>3</td>\n                        <td>4</td>\n                    </tr>\n                </tbody>\n            </table>')

    @pytest.fixture
    def spam_data(self, datapath: Any) -> str:
        return datapath('io', 'data', 'html', 'spam.html')

    @pytest.fixture
    def banklist_data(self, datapath: Any) -> str:
        return datapath('io', 'data', 'html', 'banklist.html')

    def test_to_html_compat(self, flavor_read_html: partial) -> None:
        df = DataFrame(np.random.default_rng(2).random((4, 3)), columns=pd.Index(list('abc'))).map('{:.3f}'.format).astype(float)
        out = df.to_html()
        res = flavor_read_html(StringIO(out), attrs={'class': 'dataframe'}, index_col=0)[0]
        tm.assert_frame_equal(res, df)

    def test_dtype_backend(self, string_storage: str, dtype_backend: str, flavor_read_html: partial) -> None:
        df = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': [True, False, None], 'f': [True, False, True], 'g': ['a', 'b', 'c'], 'h': ['a', 'b', None]})
        out = df.to_html(index=False)
        with pd.option_context('mode.string_storage', string_storage):
            result = flavor_read_html(StringIO(out), dtype_backend=dtype_backend)[0]
        if dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)
        expected = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': Series([True, False, NA], dtype='boolean'), 'f': Series([True, False, True], dtype='boolean'), 'g': Series(['a', 'b', 'c'], dtype=string_dtype), 'h': Series(['a', 'b', None], dtype=string_dtype)})
        if dtype_backend == 'pyarrow':
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_banklist_url(self, httpserver: Any, banklist_data: str, flavor_read_html: partial) -> None:
        with open(banklist_data, encoding='utf-8') as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(httpserver.url, match='First Federal Bank of Florida')
            df2 = flavor_read_html(httpserver.url, match='Metcalf Bank')
        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_spam_url(self, httpserver: Any, spam_data: str, flavor_read_html: partial) -> None:
        with open(spam_data, encoding='utf-8') as f:
            httpserver.serve_content(content=f.read())
            df1 = flavor_read_html(httpserver.url, match='.*Water.*')
            df2 = flavor_read_html(httpserver.url, match='Unit')
        assert_framelist_equal(df1, df2)

    @pytest.mark.slow
    def test_banklist(self, banklist_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(banklist_data, match='.*Florida.*', attrs={'id': 'table'})
        df2 = flavor_read_html(banklist_data, match='Metcalf Bank', attrs={'id': 'table'})
        assert_framelist_equal(df1, df2)

    def test_spam(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*')
        df2 = flavor_read_html(spam_data, match='Unit')
        assert_framelist_equal(df1, df2)
        assert df1[0].iloc[0, 0] == 'Proximates'
        assert df1[0].columns[0] == 'Nutrient'

    def test_spam_no_match(self, spam_data: str, flavor_read_html: partial) -> None:
        dfs = flavor_read_html(spam_data)
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_banklist_no_match(self, banklist_data: str, flavor_read_html: partial) -> None:
        dfs = flavor_read_html(banklist_data, attrs={'id': 'table'})
        for df in dfs:
            assert isinstance(df, DataFrame)

    def test_spam_header(self, spam_data: str, flavor_read_html: partial) -> None:
        df = flavor_read_html(spam_data, match='.*Water.*', header=2)[0]
        assert df.columns[0] == 'Proximates'
        assert not df.empty

    def test_skiprows_int(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=1)
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=1)
        assert_framelist_equal(df1, df2)

    def test_skiprows_range(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=range(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=range(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_list(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=[1, 2])
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=[2, 1])
        assert_framelist_equal(df1, df2)

    def test_skiprows_set(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows={1, 2})
        df2 = flavor_read_html(spam_data, match='Unit', skiprows={2, 1})
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=1)
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=1)
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_short(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=slice(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=slice(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_slice_long(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=slice(2, 5))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=slice(4, 1, -1))
        assert_framelist_equal(df1, df2)

    def test_skiprows_ndarray(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', skiprows=np.arange(2))
        df2 = flavor_read_html(spam_data, match='Unit', skiprows=np.arange(2))
        assert_framelist_equal(df1, df2)

    def test_skiprows_invalid(self, spam_data: str, flavor_read_html: partial) -> None:
        with pytest.raises(TypeError, match='is not a valid type for skipping rows'):
            flavor_read_html(spam_data, match='.*Water.*', skiprows='asdf')

    def test_index(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_no_types(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_header_and_index_with_types(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    def test_infer_types(self, spam_data: str, flavor_read_html: partial) -> None:
        df1 = flavor_read_html(spam_data, match='.*Water.*', index_col=0)
        df2 = flavor_read_html(spam_data, match='Unit', index_col=0)
        assert_framelist_equal(df1, df2)

    def test_string_io(self, spam_data: str, flavor_read_html: partial) -> None:
        with open(spam_data, encoding='UTF-8') as f:
            data1 = StringIO(f.read())
        with open(spam_data, encoding='UTF-8') as f:
            data2 = StringIO(f.read())
        df1 = flavor_read_html(data1, match='.*Water.*')
        df2 = flavor_read_html(data2, match='Unit')
        assert_framelist_equal(df1, df2)

    def test_string(self, spam_data: str, flavor_read_html: partial) -> None:
        with open(spam_data, encoding='UTF-8') as f:
            data = f.read()
        df1 = flavor_read_html(StringIO(data), match='.*Water.*')
        df2 = flavor_read_html(StringIO(data), match='Unit')
        assert_framelist_equal(df1, df2)

    def test_file_like(self, spam_data: str, flavor_read_html: partial) -> None:
        with open(spam_data, encoding='UTF-8') as f:
            df1 = flavor_read_html(f, match='.*Water.*')
        with open(spam_data, encoding='UTF-8') as f:
            df2 = flavor_read_html(f, match='Unit')
        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_bad_url_protocol(self, httpserver: Any, flavor_read_html: partial) -> None:
        httpserver.serve_content('urlopen error unknown url type: git', code=404)
        with pytest.raises(URLError, match='urlopen error unknown url type: git'):
            flavor_read_html('git://github.com', match='.*Water.*')

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_invalid_url(self, httpserver: Any, flavor_read_html: partial) -> None:
        httpserver.serve_content('Name or service not known', code=404)
        with pytest.raises((URLError, ValueError), match='HTTP Error 404: NOT FOUND'):
            flavor_read_html(httpserver.url, match='.*Water.*')

    @pytest.mark.slow
    def test_file_url(self, banklist_data: str, flavor_read_html: partial) -> None:
        url = banklist_data
        dfs = flavor_read_html(file_path_to_url(os.path.abspath(url)), match='First', attrs={'id': 'table'})
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    @pytest.mark.slow
    def test_invalid_table_attrs(self, banklist_data: str, flavor_read_html: partial) -> None:
        url = banklist_data
        with pytest.raises(ValueError, match='No tables found'):
            flavor_read_html(url, match='First Federal Bank of Florida', attrs={'id': 'tasdfable'})

    @pytest.mark.slow
    def test_multiindex_header(self, banklist_data: str, flavor_read_html: partial) -> None:
        df = flavor_read_html(banklist_data, match='Metcalf', attrs={'id': 'table'}, header=[0, 1])[0]
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_index(self, banklist