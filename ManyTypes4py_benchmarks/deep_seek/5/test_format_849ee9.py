from datetime import datetime
from io import StringIO
import re
from shutil import get_terminal_size
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas._config import using_string_dtype
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, NaT, Series, Timestamp, date_range, get_option, option_context, read_csv, reset_option
from pandas.io.formats import printing
import pandas.io.formats.format as fmt


def has_info_repr(df: DataFrame) -> bool:
    r = repr(df)
    c1 = r.split('\n')[0].startswith('<class')
    c2 = r.split('\n')[0].startswith('&lt;class')
    return c1 or c2


def has_non_verbose_info_repr(df: DataFrame) -> bool:
    has_info = has_info_repr(df)
    r = repr(df)
    nv = len(r.split('\n')) == 6
    return has_info and nv


def has_horizontally_truncated_repr(df: DataFrame) -> bool:
    try:
        fst_line = np.array(repr(df).splitlines()[0].split())
        cand_col = np.where(fst_line == '...')[0][0]
    except IndexError:
        return False
    r = repr(df)
    for ix, _ in enumerate(r.splitlines()):
        if not r.split()[cand_col] == '...':
            return False
    return True


def has_vertically_truncated_repr(df: DataFrame) -> bool:
    r = repr(df)
    only_dot_row = False
    for row in r.splitlines():
        if re.match('^[\\.\\ ]+$', row):
            only_dot_row = True
    return only_dot_row


def has_truncated_repr(df: DataFrame) -> bool:
    return has_horizontally_truncated_repr(df) or has_vertically_truncated_repr(df)


def has_doubly_truncated_repr(df: DataFrame) -> bool:
    return has_horizontally_truncated_repr(df) and has_vertically_truncated_repr(df)


def has_expanded_repr(df: DataFrame) -> bool:
    r = repr(df)
    for line in r.split('\n'):
        if line.endswith('\\'):
            return True
    return False


class TestDataFrameFormatting:

    def test_repr_truncation(self) -> None:
        max_len = 20
        with option_context('display.max_colwidth', max_len):
            df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ['a' * np.random.default_rng(2).integers(max_len - 1, max_len + 1) for _ in range(10)]})
            r = repr(df)
            r = r[r.find('\n') + 1:]
            adj = printing.get_adjustment()
            for line, value in zip(r.split('\n'), df['B']):
                if adj.len(value) + 1 > max_len:
                    assert '...' in line
                else:
                    assert '...' not in line
        with option_context('display.max_colwidth', 999999):
            assert '...' not in repr(df)
        with option_context('display.max_colwidth', max_len + 2):
            assert '...' not in repr(df)

    def test_repr_truncation_preserves_na(self) -> None:
        df = DataFrame({'a': [pd.NA for _ in range(10)]})
        with option_context('display.max_rows', 2, 'display.show_dimensions', False):
            assert repr(df) == '       a\n0   <NA>\n..   ...\n9   <NA>'

    def test_repr_truncation_dataframe_attrs(self) -> None:
        df = DataFrame([[0] * 10])
        df.attrs['b'] = DataFrame([])
        with option_context('display.max_columns', 2, 'display.show_dimensions', False):
            assert repr(df) == '   0  ...  9\n0  0  ...  0'

    def test_repr_truncation_series_with_dataframe_attrs(self) -> None:
        ser = Series([0] * 10)
        ser.attrs['b'] = DataFrame([])
        with option_context('display.max_rows', 2, 'display.show_dimensions', False):
            assert repr(ser) == '0    0\n    ..\n9    0\ndtype: int64'

    def test_max_colwidth_negative_int_raises(self) -> None:
        with pytest.raises(ValueError, match='Value must be a nonnegative integer or None'):
            with option_context('display.max_colwidth', -1):
                pass

    def test_repr_chop_threshold(self) -> None:
        df = DataFrame([[0.1, 0.5], [0.5, -0.1]])
        reset_option('display.chop_threshold')
        assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'
        with option_context('display.chop_threshold', 0.2):
            assert repr(df) == '     0    1\n0  0.0  0.5\n1  0.5  0.0'
        with option_context('display.chop_threshold', 0.6):
            assert repr(df) == '     0    1\n0  0.0  0.0\n1  0.0  0.0'
        with option_context('display.chop_threshold', None):
            assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'

    def test_repr_chop_threshold_column_below(self) -> None:
        df = DataFrame([[10, 20, 30, 40], [8e-10, -1e-11, 2e-09, -2e-11]]).T
        with option_context('display.chop_threshold', 0):
            assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0 -1.000000e-11\n2  30.0  2.000000e-09\n3  40.0 -2.000000e-11'
        with option_context('display.chop_threshold', 1e-08):
            assert repr(df) == '      0             1\n0  10.0  0.000000e+00\n1  20.0  0.000000e+00\n2  30.0  0.000000e+00\n3  40.0  0.000000e+00'
        with option_context('display.chop_threshold', 5e-11):
            assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0  0.000000e+00\n2  30.0  2.000000e-09\n3  40.0  0.000000e+00'

    def test_repr_no_backslash(self) -> None:
        with option_context('mode.sim_interactive', True):
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
            assert '\\' not in repr(df)

    def test_expand_frame_repr(self) -> None:
        df_small = DataFrame('hello', index=[0], columns=[0])
        df_wide = DataFrame('hello', index=[0], columns=range(10))
        df_tall = DataFrame('hello', index=range(30), columns=range(5))
        with option_context('mode.sim_interactive', True):
            with option_context('display.max_columns', 10, 'display.width', 20, 'display.max_rows', 20, 'display.show_dimensions', True):
                with option_context('display.expand_frame_repr', True):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_truncated_repr(df_wide)
                    assert has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert has_expanded_repr(df_tall)
                with option_context('display.expand_frame_repr', False):
                    assert not has_truncated_repr(df_small)
                    assert not has_expanded_repr(df_small)
                    assert not has_horizontally_truncated_repr(df_wide)
                    assert not has_expanded_repr(df_wide)
                    assert has_vertically_truncated_repr(df_tall)
                    assert not has_expanded_repr(df_tall)

    def test_repr_non_interactive(self) -> None:
        df = DataFrame('hello', index=range(1000), columns=range(5))
        with option_context('mode.sim_interactive', False, 'display.width', 0, 'display.max_rows', 5000):
            assert not has_truncated_repr(df)
            assert not has_expanded_repr(df)

    def test_repr_truncates_terminal_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        terminal_size = (118, 96)
        monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
        index = range(5)
        columns = MultiIndex.from_tuples([('This is a long title with > 37 chars.', 'cat'), ('This is a loooooonger title with > 43 chars.', 'dog')])
        df = DataFrame(1, index=index, columns=columns)
        result = repr(df)
        h1, h2 = result.split('\n')[:2]
        assert 'long' in h1
        assert 'loooooonger' in h1
        assert 'cat' in h2
        assert 'dog' in h2
        df2 = DataFrame({'A' * 41: [1, 2], 'B' * 41: [1, 2]})
        result = repr(df2)
        assert df2.columns[0] in result.split('\n')[0]

    def test_repr_truncates_terminal_size_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        terminal_size = (80, 24)
        df = DataFrame(np.random.default_rng(2).random((1, 7)))
        monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
        assert '...' not in str(df)

    def test_repr_truncation_column_size(self) -> None:
        df = DataFrame({'a': [108480, 30830], 'b': [12345, 12345], 'c': [12345, 12345], 'd': [12345, 12345], 'e': ['a' * 50] * 2})
        assert '...' in str(df)
        assert '    ...    ' not in str(df)

    def test_repr_max_columns_max_rows(self) -> None:
        term_width, term_height = get_terminal_size()
        if term_width < 10 or term_height < 10:
            pytest.skip(f'terminal size too small, {term_width} x {term_height}')

        def mkframe(n: int) -> DataFrame:
            index = [f'{i:05d}' for i in range(n)]
            return DataFrame(0, index, index)
        df6 = mkframe(6)
        df10 = mkframe(10)
        with option_context('mode.sim_interactive', True):
            with option_context('display.width', term_width * 2):
                with option_context('display.max_rows', 5, 'display.max_columns', 5):
                    assert not has_expanded_repr(mkframe(4))
                    assert not has_expanded_repr(mkframe(5))
                    assert not has_expanded_repr(df6)
                    assert has_doubly_truncated_repr(df6)
                with option_context('display.max_rows', 20, 'display.max_columns', 10):
                    assert not has_expanded_repr(df6)
                    assert not has_truncated_repr(df6)
                with option_context('display.max_rows', 9, 'display.max_columns', 10):
                    assert not has_expanded_repr(df10)
                    assert has_vertically_truncated_repr(df10)
            with option_context('display.max_columns', 100, 'display.max_rows', term_width * 20, 'display.width', None):
                df = mkframe(term_width // 7 - 2)
                assert not has_expanded_repr(df)
                df = mkframe(term_width // 7 + 2)
                printing.pprint_thing(df._repr_fits_horizontal_())
                assert has_expanded_repr(df)

    def test_repr_min_rows(self) -> None:
        df = DataFrame({'a': range(20)})
        assert '..' not in repr(df)
        assert '..' not in df._repr_html_()
        df = DataFrame({'a': range(61)})
        assert '..' in repr(df)
        assert '..' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.min_rows', 4):
            assert '..' in repr(df)
            assert '2  ' not in repr(df)
            assert '...' in df._repr_html_()
            assert '<td>2</td>' not in df._repr_html_()
        with option_context('display.max_rows', 12, 'display.min_rows', None):
            assert '5    5' in repr(df)
            assert '<td>5</td>' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.min_rows', 12):
            assert '5    5' not in repr(df)
            assert '<td>5</td>' not in df._repr_html_()
        with option_context('display.max_rows', None, 'display.min_rows', 12):
            assert '..' not in repr(df)
            assert '..' not in df._repr_html_()

    @pytest.mark.parametrize('data, format_option, expected_values', [(12345.6789, '{:12.3f}', '12345.679'), (None, '{:.3f}', 'None'), ('', '{:.2f}', ''), (112345.6789, '{:6.3f}', '112345.679'), ('foo      foo', None, 'foo&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;foo'), (' foo', None, 'foo'), ('foo foo       foo', None, 'foo foo&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; foo'), ('foo foo    foo', None, 'foo foo&nbsp;&nbsp;&nbsp;&nbsp;foo')])
    def test_repr_float_formatting_html_output(self, data: Any, format_option: Optional[str], expected_values: str) -> None:
        if format_option is not None:
            with option_context('display.float_format', format_option.format):
                df = DataFrame({'A': [data]})
                html_output = df._repr_html_()
                assert expected_values in html_output
        else:
            df = DataFrame({'A': [data]})
            html_output = df._repr_html_()
            assert expected_values in html_output

    def test_str_max_colwidth(self) -> None:
        df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': 'uncomfortably long line with lots of stuff', 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
        df.set_index(['a', 'b', 'c'])
        assert str(df) == '     a    b                                           c  d\n0  foo  bar  uncomfortably long line with lots of stuff  1\n1  foo  bar                                       stuff  1'
        with option_context('max_colwidth', 20):
            assert str(df) == '     a    b                    c  d\n0  foo  bar  uncomfortably lo...  1\n1  foo  bar                stuff  1'

    def test_auto_detect(self) -> None:
        term_width, term_height = get_terminal_size()
        fac = 1.05
        cols = range(int(term_width * fac))
        index = range(10)
        df = DataFrame(index=index, columns=cols)
        with option_context('mode.sim_interactive', True):
            with option_context('display.max_rows', None):
                with option_context('display.max_columns', None):
                    assert has_expanded_repr(df)
            with option_context('display.max_rows', 0):
                with option_context('display.max_columns', 0):
                    assert has_horizontally_truncated_repr(df)
            index = range(int(term_height * fac))
            df = DataFrame(index=index, columns=cols)
            with option_context('display.max_rows', 0):
                with option_context('display.max_columns', None):
                    assert has_expanded_repr(df)
                    assert has_vertically_truncated_repr(df)
            with option_context('display.max_rows', None):
                with option_context('display.max_columns', 0):
                    assert has_horizontally_truncated_repr(df)

    def test_to_string_repr_unicode2(self) -> None:
        idx = Index(['abc', 'σa', 'aegdvg'])
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        rs = repr(ser).split('\n')
        line_len = len(rs[0])
        for line in rs[1:]:
            try:
                line = line.decode(get_option('display.encoding'))
            except AttributeError:
                pass
            if not line.startswith('dtype:'):
                assert len(line) == line_len

    def test_east_asian_unicode_false(self) -> None:
        df = DataFrame({'a': ['あ', 'いいい', 'う', 'ええええええ'], 'b': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '          a      b\na         あ      1\nbb      いいい    222\nc         う  33333\nddd  ええええええ      4'
