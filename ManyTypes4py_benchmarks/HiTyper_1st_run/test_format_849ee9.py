"""
Tests for the file pandas.io.formats.format, *not* tests for general formatting
of pandas objects.
"""
from datetime import datetime
from io import StringIO
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_string_dtype
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, NaT, Series, Timestamp, date_range, get_option, option_context, read_csv, reset_option
from pandas.io.formats import printing
import pandas.io.formats.format as fmt

def has_info_repr(df: Any) -> bool:
    r = repr(df)
    c1 = r.split('\n')[0].startswith('<class')
    c2 = r.split('\n')[0].startswith('&lt;class')
    return c1 or c2

def has_non_verbose_info_repr(df: Any) -> bool:
    has_info = has_info_repr(df)
    r = repr(df)
    nv = len(r.split('\n')) == 6
    return has_info and nv

def has_horizontally_truncated_repr(df: Any) -> bool:
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

def has_vertically_truncated_repr(df: Any) -> bool:
    r = repr(df)
    only_dot_row = False
    for row in r.splitlines():
        if re.match('^[\\.\\ ]+$', row):
            only_dot_row = True
    return only_dot_row

def has_truncated_repr(df: Any):
    return has_horizontally_truncated_repr(df) or has_vertically_truncated_repr(df)

def has_doubly_truncated_repr(df: Any):
    return has_horizontally_truncated_repr(df) and has_vertically_truncated_repr(df)

def has_expanded_repr(df: Any) -> bool:
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

    def test_repr_truncates_terminal_size(self, monkeypatch: Any) -> None:
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

    def test_repr_truncates_terminal_size_full(self, monkeypatch: Any) -> None:
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

        def mkframe(n: Any) -> DataFrame:
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
    def test_repr_float_formatting_html_output(self, data: Any, format_option: Any, expected_values: Any) -> None:
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
        assert repr(df) == expected
        df = DataFrame({'a': [1, 222, 33333, 4], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '         a       b\na        1       あ\nbb     222     いいい\nc    33333       う\nddd      4  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
        expected = '         a       b\na    あああああ       あ\nbb       い     いいい\nc        う       う\nddd    えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'b': ['あ', 'いいい', 'う', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
        expected = '          b  あああああ\na         あ      1\nbb      いいい    222\nc         う  33333\nddd  ええええええ      4'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['あああ', 'いいいいいい', 'うう', 'え'])
        expected = '            a       b\nあああ     あああああ       あ\nいいいいいい      い     いいい\nうう          う       う\nえ         えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=Index(['あ', 'い', 'うう', 'え'], name='おおおお'))
        expected = '          a       b\nおおおお               \nあ     あああああ       あ\nい         い     いいい\nうう        う       う\nえ       えええ  ええええええ'
        assert repr(df) == expected
        df = DataFrame({'あああ': ['あああ', 'い', 'う', 'えええええ'], 'いいいいい': ['あ', 'いいい', 'う', 'ええ']}, index=Index(['あ', 'いいい', 'うう', 'え'], name='お'))
        expected = '       あああ いいいいい\nお               \nあ      あああ     あ\nいいい      い   いいい\nうう       う     う\nえ    えええええ    ええ'
        assert repr(df) == expected
        idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
        df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=idx)
        expected = '              a       b\nあ   いい    あああああ       あ\nう   え         い     いいい\nおおお かかかか      う       う\nき   くく      えええ  ええええええ'
        assert repr(df) == expected
        with option_context('display.max_rows', 3, 'display.max_columns', 3):
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ'], 'c': ['お', 'か', 'ききき', 'くくくくくく'], 'ああああ': ['さ', 'し', 'す', 'せ']}, columns=['a', 'b', 'c', 'ああああ'])
            expected = '        a  ... ああああ\n0   あああああ  ...    さ\n..    ...  ...  ...\n3     えええ  ...    せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected
            df.index = ['あああ', 'いいいい', 'う', 'aaa']
            expected = '         a  ... ああああ\nあああ  あああああ  ...    さ\n..     ...  ...  ...\naaa    えええ  ...    せ\n\n[4 rows x 4 columns]'
            assert repr(df) == expected

    def test_east_asian_unicode_true(self) -> None:
        with option_context('display.unicode.east_asian_width', True):
            df = DataFrame({'a': ['あ', 'いいい', 'う', 'ええええええ'], 'b': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
            expected = '                a      b\na              あ      1\nbb         いいい    222\nc              う  33333\nddd  ええええええ      4'
            assert repr(df) == expected
            df = DataFrame({'a': [1, 222, 33333, 4], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
            expected = '         a             b\na        1            あ\nbb     222        いいい\nc    33333            う\nddd      4  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['a', 'bb', 'c', 'ddd'])
            expected = '              a             b\na    あああああ            あ\nbb           い        いいい\nc            う            う\nddd      えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'b': ['あ', 'いいい', 'う', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', 'ddd'])
            expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              う       33333\nddd  ええええええ           4'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=['あああ', 'いいいいいい', 'うう', 'え'])
            expected = '                       a             b\nあああ        あああああ            あ\nいいいいいい          い        いいい\nうう                  う            う\nえ                えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=Index(['あ', 'い', 'うう', 'え'], name='おおおお'))
            expected = '                   a             b\nおおおお                          \nあ        あああああ            あ\nい                い        いいい\nうう              う            う\nえ            えええ  ええええええ'
            assert repr(df) == expected
            df = DataFrame({'あああ': ['あああ', 'い', 'う', 'えええええ'], 'いいいいい': ['あ', 'いいい', 'う', 'ええ']}, index=Index(['あ', 'いいい', 'うう', 'え'], name='お'))
            expected = '            あああ いいいいい\nお                           \nあ          あああ         あ\nいいい          い     いいい\nうう            う         う\nえ      えええええ       ええ'
            assert repr(df) == expected
            idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
            df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ']}, index=idx)
            expected = '                          a             b\nあ     いい      あああああ            あ\nう     え                い        いいい\nおおお かかかか          う            う\nき     くく          えええ  ええええええ'
            assert repr(df) == expected
            with option_context('display.max_rows', 3, 'display.max_columns', 3):
                df = DataFrame({'a': ['あああああ', 'い', 'う', 'えええ'], 'b': ['あ', 'いいい', 'う', 'ええええええ'], 'c': ['お', 'か', 'ききき', 'くくくくくく'], 'ああああ': ['さ', 'し', 'す', 'せ']}, columns=['a', 'b', 'c', 'ああああ'])
                expected = '             a  ... ああああ\n0   あああああ  ...       さ\n..         ...  ...      ...\n3       えええ  ...       せ\n\n[4 rows x 4 columns]'
                assert repr(df) == expected
                df.index = ['あああ', 'いいいい', 'う', 'aaa']
                expected = '                 a  ... ああああ\nあああ  あああああ  ...       さ\n...            ...  ...      ...\naaa         えええ  ...       せ\n\n[4 rows x 4 columns]'
                assert repr(df) == expected
            df = DataFrame({'b': ['あ', 'いいい', '¡¡', 'ええええええ'], 'あああああ': [1, 222, 33333, 4]}, index=['a', 'bb', 'c', '¡¡¡'])
            expected = '                b  あああああ\na              あ           1\nbb         いいい         222\nc              ¡¡       33333\n¡¡¡  ええええええ           4'
            assert repr(df) == expected

    def test_to_string_buffer_all_unicode(self) -> None:
        buf = StringIO()
        empty = DataFrame({'c/σ': Series(dtype=object)})
        nonempty = DataFrame({'c/σ': Series([1, 2, 3])})
        print(empty, file=buf)
        print(nonempty, file=buf)
        buf.getvalue()

    @pytest.mark.parametrize('index_scalar', ['a' * 10, 1, Timestamp(2020, 1, 1), pd.Period('2020-01-01')])
    @pytest.mark.parametrize('h', [10, 20])
    @pytest.mark.parametrize('w', [10, 20])
    def test_to_string_truncate_indices(self, index_scalar: Any, h: Any, w: Any) -> None:
        with option_context('display.expand_frame_repr', False):
            df = DataFrame(index=[index_scalar] * h, columns=[str(i) * 10 for i in range(w)])
            with option_context('display.max_rows', 15):
                if h == 20:
                    assert has_vertically_truncated_repr(df)
                else:
                    assert not has_vertically_truncated_repr(df)
            with option_context('display.max_columns', 15):
                if w == 20:
                    assert has_horizontally_truncated_repr(df)
                else:
                    assert not has_horizontally_truncated_repr(df)
            with option_context('display.max_rows', 15, 'display.max_columns', 15):
                if h == 20 and w == 20:
                    assert has_doubly_truncated_repr(df)
                else:
                    assert not has_doubly_truncated_repr(df)

    def test_to_string_truncate_multilevel(self) -> None:
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        df = DataFrame(index=arrays, columns=arrays)
        with option_context('display.max_rows', 7, 'display.max_columns', 7):
            assert has_doubly_truncated_repr(df)

    @pytest.mark.parametrize('dtype', ['object', 'datetime64[us]'])
    def test_truncate_with_different_dtypes(self, dtype: Any) -> None:
        ser = Series([datetime(2012, 1, 1)] * 10 + [datetime(1012, 1, 2)] + [datetime(2012, 1, 3)] * 10, dtype=dtype)
        with option_context('display.max_rows', 8):
            result = str(ser)
        assert dtype in result

    def test_truncate_with_different_dtypes2(self) -> None:
        df = DataFrame({'text': ['some words'] + [None] * 9}, dtype=object)
        with option_context('display.max_rows', 8, 'display.max_columns', 3):
            result = str(df)
            assert 'None' in result
            assert 'NaN' not in result

    def test_truncate_with_different_dtypes_multiindex(self) -> None:
        df = DataFrame({'Vals': range(100)})
        frame = pd.concat([df], keys=['Sweep'], names=['Sweep', 'Index'])
        result = repr(frame)
        result2 = repr(frame.iloc[:5])
        assert result.startswith(result2)

    def test_datetimelike_frame(self) -> None:
        df = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC')] + [NaT] * 5})
        with option_context('display.max_rows', 5):
            result = str(df)
            assert '2013-01-01 00:00:00+00:00' in result
            assert 'NaT' in result
            assert '...' in result
            assert '[6 rows x 1 columns]' in result
        dts = [Timestamp('2011-01-01', tz='US/Eastern')] * 5 + [NaT] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                          dt   x\n0  2011-01-01 00:00:00-05:00   1\n1  2011-01-01 00:00:00-05:00   2\n..                       ...  ..\n8                        NaT   9\n9                        NaT  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected
        dts = [NaT] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                          dt   x\n0                        NaT   1\n1                        NaT   2\n..                       ...  ..\n8  2011-01-01 00:00:00-05:00   9\n9  2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected
        dts = [Timestamp('2011-01-01', tz='Asia/Tokyo')] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
        df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        with option_context('display.max_rows', 5):
            expected = '                           dt   x\n0   2011-01-01 00:00:00+09:00   1\n1   2011-01-01 00:00:00+09:00   2\n..                        ...  ..\n8   2011-01-01 00:00:00-05:00   9\n9   2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
            assert repr(df) == expected

    @pytest.mark.parametrize('start_date', ['2017-01-01 23:59:59.999999999', '2017-01-01 23:59:59.99999999', '2017-01-01 23:59:59.9999999', '2017-01-01 23:59:59.999999', '2017-01-01 23:59:59.99999', '2017-01-01 23:59:59.9999'])
    def test_datetimeindex_highprecision(self, start_date: Any) -> None:
        df = DataFrame({'A': date_range(start=start_date, freq='D', periods=5)})
        result = str(df)
        assert start_date in result
        dti = date_range(start=start_date, freq='D', periods=5)
        df = DataFrame({'A': range(5)}, index=dti)
        result = str(df.index)
        assert start_date in result

    def test_string_repr_encoding(self, datapath: Any) -> None:
        filepath = datapath('io', 'parser', 'data', 'unicode_series.csv')
        df = read_csv(filepath, header=None, encoding='latin1')
        repr(df)
        repr(df[1])

    def test_repr_corner(self) -> None:
        df = DataFrame({'foo': [-np.inf, np.inf]})
        repr(df)

    def test_frame_info_encoding(self) -> None:
        index = ["'Til There Was You (1997)", 'ldum klaka (Cold Fever) (1994)']
        with option_context('display.max_rows', 1):
            df = DataFrame(columns=['a', 'b', 'c'], index=index)
            repr(df)
            repr(df.T)

    def test_wide_repr(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.show_dimensions', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            assert f'10 rows x {max_cols - 1} columns' in rep_str
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 120):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_columns(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a' * 90, 'b' * 90, 'c' * 90])
            rep_str = repr(df)
            assert len(rep_str.splitlines()) == 20

    def test_wide_repr_named(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
            df.index.name = 'DataFrame Index'
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)
            for line in wide_repr.splitlines()[1::13]:
                assert 'DataFrame Index' in line

    def test_wide_repr_multiindex(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            midx = MultiIndex.from_arrays([['a' * 5] * 10] * 2)
            max_cols = get_option('display.max_columns')
            df = DataFrame([['a' * 25] * (max_cols - 1)] * 10, index=midx)
            df.index.names = ['Level 0', 'Level 1']
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)
            for line in wide_repr.splitlines()[1::13]:
                assert 'Level 0 Level 1' in line

    def test_wide_repr_multiindex_cols(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = get_option('display.max_columns')
            midx = MultiIndex.from_arrays([['a' * 5] * 10] * 2)
            mcols = MultiIndex.from_arrays([['b' * 3] * (max_cols - 1)] * 2)
            df = DataFrame([['c' * 25] * (max_cols - 1)] * 10, index=midx, columns=mcols)
            df.index.names = ['Level 0', 'Level 1']
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
        with option_context('display.width', 150, 'display.max_columns', 20):
            wider_repr = repr(df)
            assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_unicode(self) -> None:
        with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
            max_cols = 20
            df = DataFrame([['a' * 25] * 10] * (max_cols - 1))
            with option_context('display.expand_frame_repr', False):
                rep_str = repr(df)
            with option_context('display.expand_frame_repr', True):
                wide_repr = repr(df)
            assert rep_str != wide_repr
            with option_context('display.width', 150):
                wider_repr = repr(df)
                assert len(wider_repr) < len(wide_repr)

    def test_wide_repr_wide_long_columns(self) -> None:
        with option_context('mode.sim_interactive', True):
            df = DataFrame({'a': ['a' * 30, 'b' * 30], 'b': ['c' * 70, 'd' * 80]})
            result = repr(df)
            assert 'ccccc' in result
            assert 'ddddd' in result

    def test_long_series(self) -> None:
        n = 1000
        s = Series(np.random.default_rng(2).integers(-50, 50, n), index=[f's{x:04d}' for x in range(n)], dtype='int64')
        str_rep = str(s)
        nmatches = len(re.findall('dtype', str_rep))
        assert nmatches == 1

    def test_to_string_ascii_error(self) -> None:
        data = [('0  ', '                        .gitignore ', '     5 ', ' â\x80¢â\x80¢â\x80¢â\x80¢â\x80¢')]
        df = DataFrame(data)
        repr(df)

    def test_show_dimensions(self) -> None:
        df = DataFrame(123, index=range(10, 15), columns=range(30))
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', True):
            assert '5 rows' in str(df)
            assert '5 rows' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', False):
            assert '5 rows' not in str(df)
            assert '5 rows' not in df._repr_html_()
        with option_context('display.max_rows', 2, 'display.max_columns', 2, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', 'truncate'):
            assert '5 rows' in str(df)
            assert '5 rows' in df._repr_html_()
        with option_context('display.max_rows', 10, 'display.max_columns', 40, 'display.width', 500, 'display.expand_frame_repr', 'info', 'display.show_dimensions', 'truncate'):
            assert '5 rows' not in str(df)
            assert '5 rows' not in df._repr_html_()

    def test_info_repr(self) -> None:
        term_width, term_height = get_terminal_size()
        max_rows = 60
        max_cols = 20 + (max(term_width, 80) - 80) // 4
        h, w = (max_rows + 1, max_cols - 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_vertically_truncated_repr(df)
        with option_context('display.large_repr', 'info'):
            assert has_info_repr(df)
        h, w = (max_rows - 1, max_cols + 1)
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        assert has_horizontally_truncated_repr(df)
        with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
            assert has_info_repr(df)

    def test_info_repr_max_cols(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 4):
            assert has_non_verbose_info_repr(df)
        with option_context('display.large_repr', 'info', 'display.max_columns', 1, 'display.max_info_columns', 5):
            assert not has_non_verbose_info_repr(df)

    def test_pprint_pathological_object(self) -> None:
        """
        If the test fails, it at least won't hang.
        """

        class A:

            def __getitem__(self, key) -> list[bool]:
                return 3
        df = DataFrame([A()])
        repr(df)

    def test_float_trim_zeros(self) -> None:
        vals = [20843091730.5, 35220501730.5, 23067481730.5, 20395421730.5, 55989781730.5]
        skip = True
        for line in repr(DataFrame({'A': vals})).split('\n')[:-2]:
            if line.startswith('dtype:'):
                continue
            if _three_digit_exp():
                assert '+010' in line or skip
            else:
                assert '+10' in line or skip
            skip = False

    @pytest.mark.parametrize('data, expected', [(['3.50'], '0    3.50\ndtype: object'), ([1.2, '1.00'], '0     1.2\n1    1.00\ndtype: object'), ([np.nan], '0   NaN\ndtype: float64'), ([None], '0    None\ndtype: object'), (['3.50', np.nan], '0    3.50\n1     NaN\ndtype: object'), ([3.5, np.nan], '0    3.5\n1    NaN\ndtype: float64'), ([3.5, np.nan, '3.50'], '0     3.5\n1     NaN\n2    3.50\ndtype: object'), ([3.5, None, '3.50'], '0     3.5\n1    None\n2    3.50\ndtype: object')])
    def test_repr_str_float_truncation(self, data: Any, expected: Any, using_infer_string: Any) -> None:
        series = Series(data, dtype=object if '3.50' in data else None)
        result = repr(series)
        assert result == expected

    @pytest.mark.parametrize('float_format,expected', [('{:,.0f}'.format, '0   1,000\n1    test\ndtype: object'), ('{:.4f}'.format, '0   1000.0000\n1        test\ndtype: object')])
    def test_repr_float_format_in_object_col(self, float_format: Any, expected: Any) -> None:
        df = Series([1000.0, 'test'])
        with option_context('display.float_format', float_format):
            result = repr(df)
        assert result == expected

    def test_period(self) -> None:
        df = DataFrame({'A': pd.period_range('2013-01', periods=4, freq='M'), 'B': [pd.Period('2011-01', freq='M'), pd.Period('2011-02-01', freq='D'), pd.Period('2011-03-01 09:00', freq='h'), pd.Period('2011-04', freq='M')], 'C': list('abcd')})
        exp = '         A                 B  C\n0  2013-01           2011-01  a\n1  2013-02        2011-02-01  b\n2  2013-03  2011-03-01 09:00  c\n3  2013-04           2011-04  d'
        assert str(df) == exp

    @pytest.mark.parametrize('length, max_rows, min_rows, expected', [(10, 10, 10, 10), (10, 10, None, 10), (10, 8, None, 8), (20, 30, 10, 30), (50, 30, 10, 10), (100, 60, 10, 10), (60, 60, 10, 60), (61, 60, 10, 10)])
    def test_max_rows_fitted(self, length: Any, min_rows: Any, max_rows: Any, expected: Any) -> None:
        """Check that display logic is correct.

        GH #37359

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options
        """
        formatter = fmt.DataFrameFormatter(DataFrame(np.random.default_rng(2).random((length, 3))), max_rows=max_rows, min_rows=min_rows)
        result = formatter.max_rows_fitted
        assert result == expected

def gen_series_formatting() -> dict[typing.Text, Series]:
    s1 = Series(['a'] * 100)
    s2 = Series(['ab'] * 100)
    s3 = Series(['a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef'])
    s4 = s3[::-1]
    test_sers = {'onel': s1, 'twol': s2, 'asc': s3, 'desc': s4}
    return test_sers

class TestSeriesFormatting:

    def test_freq_name_separation(self) -> None:
        s = Series(np.random.default_rng(2).standard_normal(10), index=date_range('1/1/2000', periods=10), name=0)
        result = repr(s)
        assert 'Freq: D, Name: 0' in result

    def test_unicode_name_in_footer(self) -> None:
        s = Series([1, 2], name='עברית')
        sf = fmt.SeriesFormatter(s, name='עברית')
        sf._get_footer()

    @pytest.mark.xfail(using_string_dtype(), reason='Fixup when arrow is default')
    def test_east_asian_unicode_series(self) -> None:
        s = Series(['a', 'bb', 'CCC', 'D'], index=['あ', 'いい', 'ううう', 'ええええ'])
        expected = ''.join(['あ         a\n', 'いい       bb\n', 'ううう     CCC\n', 'ええええ      D\ndtype: object'])
        assert repr(s) == expected
        s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['a', 'bb', 'c', 'ddd'])
        expected = ''.join(['a         あ\n', 'bb       いい\n', 'c       ううう\n', 'ddd    ええええ\n', 'dtype: object'])
        assert repr(s) == expected
        s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['ああ', 'いいいい', 'う', 'えええ'])
        expected = ''.join(['ああ         あ\n', 'いいいい      いい\n', 'う        ううう\n', 'えええ     ええええ\n', 'dtype: object'])
        assert repr(s) == expected
        s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['ああ', 'いいいい', 'う', 'えええ'], name='おおおおおおお')
        expected = 'ああ         あ\nいいいい      いい\nう        ううう\nえええ     ええええ\nName: おおおおおおお, dtype: object'
        assert repr(s) == expected
        idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
        s = Series([1, 22, 3333, 44444], index=idx)
        expected = 'あ    いい          1\nう    え          22\nおおお  かかかか     3333\nき    くく      44444\ndtype: int64'
        assert repr(s) == expected
        s = Series([1, 22, 3333, 44444], index=[1, 'AB', np.nan, 'あああ'])
        expected = '1          1\nAB        22\nNaN     3333\nあああ    44444\ndtype: int64'
        assert repr(s) == expected
        s = Series([1, 22, 3333, 44444], index=[1, 'AB', Timestamp('2011-01-01'), 'あああ'])
        expected = '1                          1\nAB                        22\n2011-01-01 00:00:00     3333\nあああ                    44444\ndtype: int64'
        assert repr(s) == expected
        with option_context('display.max_rows', 3):
            s = Series(['あ', 'いい', 'ううう', 'ええええ'], name='おおおおおおお')
            expected = '0       あ\n     ... \n3    ええええ\nName: おおおおおおお, Length: 4, dtype: object'
            assert repr(s) == expected
            s.index = ['ああ', 'いいいい', 'う', 'えええ']
            expected = 'ああ        あ\n       ... \nえええ    ええええ\nName: おおおおおおお, Length: 4, dtype: object'
            assert repr(s) == expected
        with option_context('display.unicode.east_asian_width', True):
            s = Series(['a', 'bb', 'CCC', 'D'], index=['あ', 'いい', 'ううう', 'ええええ'])
            expected = 'あ            a\nいい         bb\nううう      CCC\nええええ      D\ndtype: object'
            assert repr(s) == expected
            s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['a', 'bb', 'c', 'ddd'])
            expected = 'a            あ\nbb         いい\nc        ううう\nddd    ええええ\ndtype: object'
            assert repr(s) == expected
            s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['ああ', 'いいいい', 'う', 'えええ'])
            expected = 'ああ              あ\nいいいい        いい\nう            ううう\nえええ      ええええ\ndtype: object'
            assert repr(s) == expected
            s = Series(['あ', 'いい', 'ううう', 'ええええ'], index=['ああ', 'いいいい', 'う', 'えええ'], name='おおおおおおお')
            expected = 'ああ              あ\nいいいい        いい\nう            ううう\nえええ      ええええ\nName: おおおおおおお, dtype: object'
            assert repr(s) == expected
            idx = MultiIndex.from_tuples([('あ', 'いい'), ('う', 'え'), ('おおお', 'かかかか'), ('き', 'くく')])
            s = Series([1, 22, 3333, 44444], index=idx)
            expected = 'あ      いい            1\nう      え             22\nおおお  かかかか     3333\nき      くく        44444\ndtype: int64'
            assert repr(s) == expected
            s = Series([1, 22, 3333, 44444], index=[1, 'AB', np.nan, 'あああ'])
            expected = '1             1\nAB           22\nNaN        3333\nあああ    44444\ndtype: int64'
            assert repr(s) == expected
            s = Series([1, 22, 3333, 44444], index=[1, 'AB', Timestamp('2011-01-01'), 'あああ'])
            expected = '1                          1\nAB                        22\n2011-01-01 00:00:00     3333\nあああ                 44444\ndtype: int64'
            assert repr(s) == expected
            with option_context('display.max_rows', 3):
                s = Series(['あ', 'いい', 'ううう', 'ええええ'], name='おおおおおおお')
                expected = '0          あ\n       ...   \n3    ええええ\nName: おおおおおおお, Length: 4, dtype: object'
                assert repr(s) == expected
                s.index = ['ああ', 'いいいい', 'う', 'えええ']
                expected = 'ああ            あ\n            ...   \nえええ    ええええ\nName: おおおおおおお, Length: 4, dtype: object'
                assert repr(s) == expected
            s = Series(['¡¡', 'い¡¡', 'ううう', 'ええええ'], index=['ああ', '¡¡¡¡いい', '¡¡', 'えええ'])
            expected = 'ああ              ¡¡\n¡¡¡¡いい        い¡¡\n¡¡            ううう\nえええ      ええええ\ndtype: object'
            assert repr(s) == expected

    def test_float_trim_zeros(self) -> None:
        vals = [20843091730.5, 35220501730.5, 23067481730.5, 20395421730.5, 55989781730.5]
        for line in repr(Series(vals)).split('\n'):
            if line.startswith('dtype:'):
                continue
            if _three_digit_exp():
                assert '+010' in line
            else:
                assert '+10' in line

    @pytest.mark.parametrize('start_date', ['2017-01-01 23:59:59.999999999', '2017-01-01 23:59:59.99999999', '2017-01-01 23:59:59.9999999', '2017-01-01 23:59:59.999999', '2017-01-01 23:59:59.99999', '2017-01-01 23:59:59.9999'])
    def test_datetimeindex_highprecision(self, start_date: Any) -> None:
        s1 = Series(date_range(start=start_date, freq='D', periods=5))
        result = str(s1)
        assert start_date in result
        dti = date_range(start=start_date, freq='D', periods=5)
        s2 = Series(3, index=dti)
        result = str(s2.index)
        assert start_date in result

    def test_mixed_datetime64(self) -> None:
        df = DataFrame({'A': [1, 2], 'B': ['2012-01-01', '2012-01-02']})
        df['B'] = pd.to_datetime(df.B)
        result = repr(df.loc[0])
        assert '2012-01-01' in result

    def test_period(self) -> None:
        index = pd.period_range('2013-01', periods=6, freq='M')
        s = Series(np.arange(6, dtype='int64'), index=index)
        exp = '2013-01    0\n2013-02    1\n2013-03    2\n2013-04    3\n2013-05    4\n2013-06    5\nFreq: M, dtype: int64'
        assert str(s) == exp
        s = Series(index)
        exp = '0    2013-01\n1    2013-02\n2    2013-03\n3    2013-04\n4    2013-05\n5    2013-06\ndtype: period[M]'
        assert str(s) == exp
        s = Series([pd.Period('2011-01', freq='M'), pd.Period('2011-02-01', freq='D'), pd.Period('2011-03-01 09:00', freq='h')])
        exp = '0             2011-01\n1          2011-02-01\n2    2011-03-01 09:00\ndtype: object'
        assert str(s) == exp

    def test_max_multi_index_display(self) -> None:
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
        s = Series(np.random.default_rng(2).standard_normal(8), index=index)
        with option_context('display.max_rows', 10):
            assert len(str(s).split('\n')) == 10
        with option_context('display.max_rows', 3):
            assert len(str(s).split('\n')) == 5
        with option_context('display.max_rows', 2):
            assert len(str(s).split('\n')) == 5
        with option_context('display.max_rows', 1):
            assert len(str(s).split('\n')) == 4
        with option_context('display.max_rows', 0):
            assert len(str(s).split('\n')) == 10
        s = Series(np.random.default_rng(2).standard_normal(8), None)
        with option_context('display.max_rows', 10):
            assert len(str(s).split('\n')) == 9
        with option_context('display.max_rows', 3):
            assert len(str(s).split('\n')) == 4
        with option_context('display.max_rows', 2):
            assert len(str(s).split('\n')) == 4
        with option_context('display.max_rows', 1):
            assert len(str(s).split('\n')) == 3
        with option_context('display.max_rows', 0):
            assert len(str(s).split('\n')) == 9

    def test_consistent_format(self) -> None:
        s = Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9999, 1, 1] * 10)
        with option_context('display.max_rows', 10, 'display.show_dimensions', False):
            res = repr(s)
        exp = '0      1.0000\n1      1.0000\n2      1.0000\n3      1.0000\n4      1.0000\n        ...  \n125    1.0000\n126    1.0000\n127    0.9999\n128    1.0000\n129    1.0000\ndtype: float64'
        assert res == exp

    def chck_ncols(self, s: Any) -> None:
        lines = [line for line in repr(s).split('\n') if not re.match('[^\\.]*\\.+', line)][:-1]
        ncolsizes = len({len(line.strip()) for line in lines})
        assert ncolsizes == 1

    @pytest.mark.xfail(using_string_dtype(), reason='change when arrow is default')
    def test_format_explicit(self) -> None:
        test_sers = gen_series_formatting()
        with option_context('display.max_rows', 4, 'display.show_dimensions', False):
            res = repr(test_sers['onel'])
            exp = '0     a\n1     a\n     ..\n98    a\n99    a\ndtype: object'
            assert exp == res
            res = repr(test_sers['twol'])
            exp = '0     ab\n1     ab\n      ..\n98    ab\n99    ab\ndtype: object'
            assert exp == res
            res = repr(test_sers['asc'])
            exp = '0         a\n1        ab\n      ...  \n4     abcde\n5    abcdef\ndtype: object'
            assert exp == res
            res = repr(test_sers['desc'])
            exp = '5    abcdef\n4     abcde\n      ...  \n1        ab\n0         a\ndtype: object'
            assert exp == res

    def test_ncols(self) -> None:
        test_sers = gen_series_formatting()
        for s in test_sers.values():
            self.chck_ncols(s)

    def test_max_rows_eq_one(self) -> None:
        s = Series(range(10), dtype='int64')
        with option_context('display.max_rows', 1):
            strrepr = repr(s).split('\n')
        exp1 = ['0', '0']
        res1 = strrepr[0].split()
        assert exp1 == res1
        exp2 = ['..']
        res2 = strrepr[1].split()
        assert exp2 == res2

    def test_truncate_ndots(self) -> None:

        def getndots(s: Any) -> int:
            return len(re.match('[^\\.]*(\\.*)', s).groups()[0])
        s = Series([0, 2, 3, 6])
        with option_context('display.max_rows', 2):
            strrepr = repr(s).replace('\n', '')
        assert getndots(strrepr) == 2
        s = Series([0, 100, 200, 400])
        with option_context('display.max_rows', 2):
            strrepr = repr(s).replace('\n', '')
        assert getndots(strrepr) == 3

    def test_show_dimensions(self) -> None:
        s = Series(range(5))
        assert 'Length' not in repr(s)
        with option_context('display.max_rows', 4):
            assert 'Length' in repr(s)
        with option_context('display.show_dimensions', True):
            assert 'Length' in repr(s)
        with option_context('display.max_rows', 4, 'display.show_dimensions', False):
            assert 'Length' not in repr(s)

    def test_repr_min_rows(self) -> None:
        s = Series(range(20))
        assert '..' not in repr(s)
        s = Series(range(61))
        assert '..' in repr(s)
        with option_context('display.max_rows', 10, 'display.min_rows', 4):
            assert '..' in repr(s)
            assert '2  ' not in repr(s)
        with option_context('display.max_rows', 12, 'display.min_rows', None):
            assert '5      5' in repr(s)
        with option_context('display.max_rows', 10, 'display.min_rows', 12):
            assert '5      5' not in repr(s)
        with option_context('display.max_rows', None, 'display.min_rows', 12):
            assert '..' not in repr(s)

class TestGenericArrayFormatter:

    def test_1d_array(self) -> None:
        obj = fmt._GenericArrayFormatter(np.array([True, False]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == '  True'
        assert res[1] == ' False'

    def test_2d_array(self) -> None:
        obj = fmt._GenericArrayFormatter(np.array([[True, False], [False, True]]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == ' [True, False]'
        assert res[1] == ' [False, True]'

    def test_3d_array(self) -> None:
        obj = fmt._GenericArrayFormatter(np.array([[[True, True], [False, False]], [[False, True], [True, False]]]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == ' [[True, True], [False, False]]'
        assert res[1] == ' [[False, True], [True, False]]'

    def test_2d_extension_type(self) -> None:

        class DtypeStub(pd.api.extensions.ExtensionDtype):

            @property
            def type(self):
                return np.ndarray

            @property
            def name(self) -> typing.Text:
                return 'DtypeStub'

        class ExtTypeStub(pd.api.extensions.ExtensionArray):

            def __len__(self) -> int:
                return 2

            def __getitem__(self, ix: Any) -> list[bool]:
                return [ix == 1, ix == 0]

            @property
            def dtype(self) -> DtypeStub:
                return DtypeStub()
        series = Series(ExtTypeStub(), copy=False)
        res = repr(series)
        expected = '\n'.join(['0    [False True]', '1    [True False]', 'dtype: DtypeStub'])
        assert res == expected

def _three_digit_exp() -> bool:
    return f'{170000000.0:.4g}' == '1.7e+008'

class TestFloatArrayFormatter:

    def test_misc(self) -> None:
        obj = fmt.FloatArrayFormatter(np.array([], dtype=np.float64))
        result = obj.get_result()
        assert len(result) == 0

    def test_format(self) -> None:
        obj = fmt.FloatArrayFormatter(np.array([12, 0], dtype=np.float64))
        result = obj.get_result()
        assert result[0] == ' 12.0'
        assert result[1] == '  0.0'

    def test_output_display_precision_trailing_zeroes(self) -> None:
        with option_context('display.precision', 0):
            s = Series([840.0, 4200.0])
            expected_output = '0     840\n1    4200\ndtype: float64'
            assert str(s) == expected_output

    @pytest.mark.parametrize('value,expected', [([9.4444], '   0\n0  9'), ([0.49], '       0\n0  5e-01'), ([10.9999], '    0\n0  11'), ([9.5444, 9.6], '    0\n0  10\n1  10'), ([0.46, 0.78, -9.9999], '       0\n0  5e-01\n1  8e-01\n2 -1e+01')])
    def test_set_option_precision(self, value: Any, expected: Any) -> None:
        with option_context('display.precision', 0):
            df_value = DataFrame(value)
            assert str(df_value) == expected

    def test_output_significant_digits(self) -> None:
        with option_context('display.precision', 6):
            d = DataFrame({'col1': [9.999e-08, 1e-07, 1.0001e-07, 2e-07, 4.999e-07, 5e-07, 5.0001e-07, 6e-07, 9.999e-07, 1e-06, 1.0001e-06, 2e-06, 4.999e-06, 5e-06, 5.0001e-06, 6e-06]})
            expected_output = {(0, 6): '           col1\n0  9.999000e-08\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07', (1, 6): '           col1\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07', (1, 8): '           col1\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07\n6  5.000100e-07\n7  6.000000e-07', (8, 16): '            col1\n8   9.999000e-07\n9   1.000000e-06\n10  1.000100e-06\n11  2.000000e-06\n12  4.999000e-06\n13  5.000000e-06\n14  5.000100e-06\n15  6.000000e-06', (9, 16): '        col1\n9   0.000001\n10  0.000001\n11  0.000002\n12  0.000005\n13  0.000005\n14  0.000005\n15  0.000006'}
            for (start, stop), v in expected_output.items():
                assert str(d[start:stop]) == v

    def test_too_long(self) -> None:
        with option_context('display.precision', 4):
            df = DataFrame({'x': [12345.6789]})
            assert str(df) == '            x\n0  12345.6789'
            df = DataFrame({'x': [2000000.0]})
            assert str(df) == '           x\n0  2000000.0'
            df = DataFrame({'x': [12345.6789, 2000000.0]})
            assert str(df) == '            x\n0  1.2346e+04\n1  2.0000e+06'

class TestTimedelta64Formatter:

    def test_days(self) -> None:
        x = pd.to_timedelta(list(range(5)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == '1 days'

    def test_days_neg(self) -> None:
        x = pd.to_timedelta(list(range(5)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(-x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '-1 days'

    def test_subdays(self) -> None:
        y = pd.to_timedelta(list(range(5)) + [NaT], unit='s')._values
        result = fmt._Timedelta64Formatter(y).get_result()
        assert result[0].strip() == '0 days 00:00:00'
        assert result[1].strip() == '0 days 00:00:01'

    def test_subdays_neg(self) -> None:
        y = pd.to_timedelta(list(range(5)) + [NaT], unit='s')._values
        result = fmt._Timedelta64Formatter(-y).get_result()
        assert result[0].strip() == '0 days 00:00:00'
        assert result[1].strip() == '-1 days +23:59:59'

    def test_zero(self) -> None:
        x = pd.to_timedelta(list(range(1)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        x = pd.to_timedelta(list(range(1)), unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'

class TestDatetime64Formatter:

    def test_mixed(self) -> None:
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 1, 12), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '2013-01-01 00:00:00'
        assert result[1].strip() == '2013-01-01 12:00:00'

    def test_dates(self) -> None:
        x = Series([datetime(2013, 1, 1), datetime(2013, 1, 2), NaT])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '2013-01-01'
        assert result[1].strip() == '2013-01-02'

    def test_date_nanos(self) -> None:
        x = Series([Timestamp(200)])._values
        result = fmt._Datetime64Formatter(x).get_result()
        assert result[0].strip() == '1970-01-01 00:00:00.000000200'

    def test_dates_display(self) -> None:
        x = Series(date_range('20130101 09:00:00', periods=5, freq='D'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-05 09:00:00'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='s'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:04'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='ms'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.004'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='us'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.000004'
        x = Series(date_range('20130101 09:00:00', periods=5, freq='ns'))
        x.iloc[1] = np.nan
        result = fmt._Datetime64Formatter(x._values).get_result()
        assert result[0].strip() == '2013-01-01 09:00:00.000000000'
        assert result[1].strip() == 'NaT'
        assert result[4].strip() == '2013-01-01 09:00:00.000000004'

    def test_datetime64formatter_yearmonth(self) -> None:
        x = Series([datetime(2016, 1, 1), datetime(2016, 2, 2)])._values

        def format_func(x: Any):
            return x.strftime('%Y-%m')
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ['2016-01', '2016-02']

    def test_datetime64formatter_hoursecond(self) -> None:
        x = Series(pd.to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f'))._values

        def format_func(x: Any):
            return x.strftime('%H:%M')
        formatter = fmt._Datetime64Formatter(x, formatter=format_func)
        result = formatter.get_result()
        assert result == ['10:10', '12:12']

    def test_datetime64formatter_tz_ms(self) -> None:
        x = Series(np.array(['2999-01-01', '2999-01-02', 'NaT'], dtype='datetime64[ms]')).dt.tz_localize('US/Pacific')._values
        result = fmt._Datetime64TZFormatter(x).get_result()
        assert result[0].strip() == '2999-01-01 00:00:00-08:00'
        assert result[1].strip() == '2999-01-02 00:00:00-08:00'

class TestFormatPercentiles:

    @pytest.mark.parametrize('percentiles, expected', [([0.01999, 0.02001, 0.5, 0.666666, 0.9999], ['1.999%', '2.001%', '50%', '66.667%', '99.99%']), ([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999], ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']), ([0.281, 0.29, 0.57, 0.58], ['28.1%', '29%', '57%', '58%']), ([0.28, 0.29, 0.57, 0.58], ['28%', '29%', '57%', '58%']), ([0.9, 0.99, 0.999, 0.9999, 0.99999], ['90%', '99%', '99.9%', '99.99%', '99.999%'])])
    def test_format_percentiles(self, percentiles: Any, expected: Any) -> None:
        result = fmt.format_percentiles(percentiles)
        assert result == expected

    @pytest.mark.parametrize('percentiles', [[0.1, np.nan, 0.5], [-0.001, 0.1, 0.5], [2, 0.1, 0.5], [0.1, 0.5, 'a']])
    def test_error_format_percentiles(self, percentiles: Any) -> None:
        msg = 'percentiles should all be in the interval \\[0,1\\]'
        with pytest.raises(ValueError, match=msg):
            fmt.format_percentiles(percentiles)

    def test_format_percentiles_integer_idx(self) -> None:
        result = fmt.format_percentiles(np.linspace(0, 1, 10 + 1))
        expected = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        assert result == expected

@pytest.mark.parametrize('method', ['to_string', 'to_html', 'to_latex'])
@pytest.mark.parametrize('encoding, data', [(None, 'abc'), ('utf-8', 'abc'), ('gbk', '造成输出中文显示乱码'), ('foo', 'abc')])
@pytest.mark.parametrize('filepath_or_buffer_id', ['string', 'pathlike', 'buffer'])
def test_filepath_or_buffer_arg(method: Any, tmp_path: Any, encoding: Any, data: Any, filepath_or_buffer_id: Any) -> None:
    if filepath_or_buffer_id == 'buffer':
        filepath_or_buffer = StringIO()
    elif filepath_or_buffer_id == 'pathlike':
        filepath_or_buffer = tmp_path / 'foo'
    else:
        filepath_or_buffer = str(tmp_path / 'foo')
    df = DataFrame([data])
    if method in ['to_latex']:
        pytest.importorskip('jinja2')
    if filepath_or_buffer_id not in ['string', 'pathlike'] and encoding is not None:
        with pytest.raises(ValueError, match='buf is not a file name and encoding is specified.'):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    elif encoding == 'foo':
        with pytest.raises(LookupError, match='unknown encoding'):
            getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
    else:
        expected = getattr(df, method)()
        getattr(df, method)(buf=filepath_or_buffer, encoding=encoding)
        encoding = encoding or 'utf-8'
        if filepath_or_buffer_id == 'string':
            with open(filepath_or_buffer, encoding=encoding) as f:
                result = f.read()
        elif filepath_or_buffer_id == 'pathlike':
            result = filepath_or_buffer.read_text(encoding=encoding)
        elif filepath_or_buffer_id == 'buffer':
            result = filepath_or_buffer.getvalue()
        assert result == expected
    if filepath_or_buffer_id == 'buffer':
        assert not filepath_or_buffer.closed

@pytest.mark.parametrize('method', ['to_string', 'to_html', 'to_latex'])
def test_filepath_or_buffer_bad_arg_raises(float_frame: Any, method: Any) -> None:
    if method in ['to_latex']:
        pytest.importorskip('jinja2')
    msg = 'buf is not a file name and it has no write method'
    with pytest.raises(TypeError, match=msg):
        getattr(float_frame, method)(buf=object())