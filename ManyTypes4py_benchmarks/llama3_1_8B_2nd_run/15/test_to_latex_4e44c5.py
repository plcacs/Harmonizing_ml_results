import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import DataFrame, Series
import pandas._testing as tm
pytest.importorskip('jinja2')

def _dedent(string: str) -> str:
    """Dedent without new line in the beginning.

    Built-in textwrap.dedent would keep new line character in the beginning
    of multi-line string starting from the new line.
    This version drops the leading new line character.
    """
    return dedent(string).lstrip()

@pytest.fixture
def df_short() -> DataFrame:
    """Short dataframe for testing table/tabular/longtable LaTeX env."""
    return DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})

class TestToLatex:
    def test_to_latex_to_file(self, float_frame: DataFrame) -> None:
        with tm.ensure_clean('test.tex') as path:
            float_frame.to_latex(path)
            with open(path, encoding='utf-8') as f:
                assert float_frame.to_latex() == f.read()

    def test_to_latex_to_file_utf8_with_encoding(self) -> None:
        df = DataFrame([['außgangen']])
        with tm.ensure_clean('test.tex') as path:
            df.to_latex(path, encoding='utf-8')
            with codecs.open(path, 'r', encoding='utf-8') as f:
                assert df.to_latex() == f.read()

    def test_to_latex_to_file_utf8_without_encoding(self) -> None:
        df = DataFrame([['außgangen']])
        with tm.ensure_clean('test.tex') as path:
            df.to_latex(path)
            with codecs.open(path, 'r', encoding='utf-8') as f:
                assert df.to_latex() == f.read()

    def test_to_latex_tabular_with_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex()
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_tabular_without_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(index=False)
        expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            a & b \\\\\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('bad_column_format', [5, 1.2, ['l', 'r'], ('r', 'c'), {'r', 'c', 'l'}, {'a': 'r', 'b': 'l'}])
    def test_to_latex_bad_column_format(self, bad_column_format: any) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        msg = '`column_format` must be str or unicode'
        with pytest.raises(ValueError, match=msg):
            df.to_latex(column_format=bad_column_format)

    def test_to_latex_column_format_just_works(self, float_frame: DataFrame) -> None:
        float_frame.to_latex(column_format='lcr')

    def test_to_latex_column_format(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(column_format='lcr')
        expected = _dedent('\n            \\begin{tabular}{lcr}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_float_format_object_col(self) -> None:
        ser = Series([1000.0, 'test'])
        result = ser.to_latex(float_format='{:,.0f}'.format)
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & 1,000 \\\\\n            1 & test \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_empty_tabular(self) -> None:
        df = DataFrame()
        result = df.to_latex()
        expected = _dedent('\n            \\begin{tabular}{l}\n            \\toprule\n            \\midrule\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_series(self) -> None:
        s = Series(['a', 'b', 'c'])
        result = s.to_latex()
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & a \\\\\n            1 & b \\\\\n            2 & c \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_midrule_location(self) -> None:
        df = DataFrame({'a': [1, 2]})
        df.index.name = 'foo'
        result = df.to_latex(index_names=False)
        expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & a \\\\\n            \\midrule\n            0 & 1 \\\\\n            1 & 2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

class TestToLatexLongtable:
    def test_to_latex_empty_longtable(self) -> None:
        df = DataFrame()
        result = df.to_latex(longtable=True)
        expected = _dedent('\n            \\begin{longtable}{l}\n            \\toprule\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{0}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_with_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(longtable=True)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_without_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(index=False, longtable=True)
        expected = _dedent('\n            \\begin{longtable}{rl}\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n            a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{2}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    @pytest.mark.parametrize('df_data, expected_number', [({'a': [1, 2]}, 1), ({'a': [1, 2], 'b': [3, 4]}, 2), ({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, 3)])
    def test_to_latex_longtable_continued_on_next_page(self, df_data: dict, expected_number: int) -> None:
        df = DataFrame(df_data)
        result = df.to_latex(index=False, longtable=True)
        assert f'\\multicolumn{{{expected_number}}}' in result

class TestToLatexHeader:
    def test_to_latex_no_header_with_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=False)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_no_header_without_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(index=False, header=False)
        expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_with_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['AA', 'BB'])
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & AA & BB \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_without_index(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['AA', 'BB'], index=False)
        expected = _dedent('\n            \\begin{tabular}{rl}\n            \\toprule\n            AA & BB \\\\\n            \\midrule\n            1 & b1 \\\\\n            2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('header, num_aliases', [(['A'], 1), (('B',), 1), (('Col1', 'Col2', 'Col3'), 3), (('Col1', 'Col2', 'Col3', 'Col4'), 4)])
    def test_to_latex_number_of_items_in_header_missmatch_raises(self, header: list, num_aliases: int) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        msg = f'Writing 2 cols but got {num_aliases} aliases'
        with pytest.raises(ValueError, match=msg):
            df.to_latex(header=header)

    def test_to_latex_decimal(self) -> None:
        df = DataFrame({'a': [1.0, 2.1], 'b': ['b1', 'b2']})
        result = df.to_latex(decimal=',')
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1,000000 & b1 \\\\\n            1 & 2,100000 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

class TestToLatexBold:
    def test_to_latex_bold_rows(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(bold_rows=True)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\textbf{0} & 1 & b1 \\\\\n            \\textbf{1} & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_no_bold_rows(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(bold_rows=False)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

class TestToLatexCaptionLabel:
    @pytest.fixture
    def caption_table(self) -> str:
        """Caption for table/tabular LaTeX environment."""
        return 'a table in a \\texttt{table/tabular} environment'

    @pytest.fixture
    def short_caption(self) -> str:
        """Short caption for testing \\caption[short_caption]{full_caption}."""
        return 'a table'

    @pytest.fixture
    def label_table(self) -> str:
        """Label for table/tabular LaTeX environment."""
        return 'tab:table_tabular'

    @pytest.fixture
    def caption_longtable(self) -> str:
        """Caption for longtable LaTeX environment."""
        return 'a table in a \\texttt{longtable} environment'

    @pytest.fixture
    def label_longtable(self) -> str:
        """Label for longtable LaTeX environment."""
        return 'tab:longtable'

    def test_to_latex_caption_only(self, df_short: DataFrame, caption_table: str) -> None:
        result = df_short.to_latex(caption=caption_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption{a table in a \\texttt{table/tabular} environment}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_label_only(self, df_short: DataFrame, label_table: str) -> None:
        result = df_short.to_latex(label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_label(self, df_short: DataFrame, caption_table: str, label_table: str) -> None:
        result = df_short.to_latex(caption=caption_table, label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption{a table in a \\texttt{table/tabular} environment}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_shortcaption(self, df_short: DataFrame, caption_table: str, short_caption: str) -> None:
        result = df_short.to_latex(caption=(caption_table, short_caption))
        expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short: DataFrame) -> None:
        caption = ('Long-long-caption', 'Short')
        result_tuple = df_short.to_latex(caption=caption)
        result_list = df_short.to_latex(caption=list(caption))
        assert result_tuple == result_list

    def test_to_latex_caption_shortcaption_and_label(self, df_short: DataFrame, caption_table: str, short_caption: str, label_table: str) -> None:
        result = df_short.to_latex(caption=(caption_table, short_caption), label=label_table)
        expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\label{tab:table_tabular}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    @pytest.mark.parametrize('bad_caption', [('full_caption', 'short_caption', 'extra_string'), ('full_caption', 'short_caption', 1), ('full_caption', 'short_caption', None), ('full_caption',), (None,)])
    def test_to_latex_bad_caption_raises(self, bad_caption: any) -> None:
        df = DataFrame({'a': [1]})
        msg = '`caption` must be either a string or 2-tuple of strings'
        with pytest.raises(ValueError, match=msg):
            df.to_latex(caption=bad_caption)

    def test_to_latex_two_chars_caption(self, df_short: DataFrame) -> None:
        result = df_short.to_latex(caption='xy')
        expected = _dedent('\n            \\begin{table}\n            \\caption{xy}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_longtable_caption_only(self, df_short: DataFrame, caption_longtable: str) -> None:
        result = df_short.to_latex(longtable=True, caption=caption_longtable)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\caption{a table in a \\texttt{longtable} environment} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\caption[]{a table in a \\texttt{longtable} environment} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_label_only(self, df_short: DataFrame, label_longtable: str) -> None:
        result = df_short.to_latex(longtable=True, label=label_longtable)
        expected = _dedent('\n            \\begin{longtable}{lrl}\n            \\label{tab:longtable} \\\\\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

    def test_to_latex_longtable_caption_and_label(self, df_short: DataFrame, caption_longtable: str, label_longtable: str) -> None:
        result = df_short.to_latex(longtable=True, caption=caption_longtable, label=label_longtable)
        expected = _dedent('\n        \\begin{longtable}{lrl}\n        \\caption{a table in a \\texttt{longtable} environment} \\label{tab:longtable} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endfirsthead\n        \\caption[]{a table in a \\texttt{longtable} environment} \\\\\n        \\toprule\n         & a & b \\\\\n        \\midrule\n        \\endhead\n        \\midrule\n        \\multicolumn{3}{r}{Continued on next page} \\\\\n        \\midrule\n        \\endfoot\n        \\bottomrule\n        \\endlastfoot\n        0 & 1 & b1 \\\\\n        1 & 2 & b2 \\\\\n        \\end{longtable}\n        ')
        assert result == expected

    def test_to_latex_longtable_caption_shortcaption_and_label(self, df_short: DataFrame, caption_longtable: str, short_caption: str, label_longtable: str) -> None:
        result = df_short.to_latex(longtable=True, caption=(caption_longtable, short_caption), label=label_longtable)
        expected = _dedent('\n\\begin{longtable}{lrl}\n\\caption[a table]{a table in a \\texttt{longtable} environment} \\label{tab:longtable} \\\\\n\\toprule\n & a & b \\\\\n\\midrule\n\\endfirsthead\n\\caption[]{a table in a \\texttt{longtable} environment} \\\\\n\\toprule\n & a & b \\\\\n\\midrule\n\\endhead\n\\midrule\n\\multicolumn{3}{r}{Continued on next page} \\\\\n\\midrule\n\\endfoot\n\\bottomrule\n\\endlastfoot\n0 & 1 & b1 \\\\\n1 & 2 & b2 \\\\\n\\end{longtable}\n')
        assert result == expected

class TestToLatexEscape:
    @pytest.fixture
    def df_with_symbols(self) -> DataFrame:
        """Dataframe with special characters for testing chars escaping."""
        a = 'a'
        b = 'b'
        return DataFrame({'co$e^x$': {a: 'a', b: 'b'}, 'co^l1': {a: 'a', b: 'b'}})

    def test_to_latex_escape_false(self, df_with_symbols: DataFrame) -> None:
        result = df_with_symbols.to_latex(escape=False)
        expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             & co$e^x$ & co^l1 \\\\\n            \\midrule\n            a & a & a \\\\\n            b & b & b \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_escape_default(self, df_with_symbols: DataFrame) -> None:
        default = df_with_symbols.to_latex()
        specified_true = df_with_symbols.to_latex(escape=True)
        assert default != specified_true

    def test_to_latex_special_escape(self) -> None:
        df = DataFrame(['a\\b\\c', '^a^b^c', '~a~b~c'])
        result = df.to_latex(escape=True)
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & a\\textbackslash b\\textbackslash c \\\\\n            1 & \\textasciicircum a\\textasciicircum b\\textasciicircum c \\\\\n            2 & \\textasciitilde a\\textasciitilde b\\textasciitilde c \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_escape_special_chars(self) -> None:
        special_characters = ['&', '%', '$', '#', '_', '{', '}', '~', '^', '\\']
        df = DataFrame(data=special_characters)
        result = df.to_latex(escape=True)
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & \\& \\\\\n            1 & \\% \\\\\n            2 & \\$ \\\\\n            3 & \\# \\\\\n            4 & \\_ \\\\\n            5 & \\{ \\\\\n            6 & \\} \\\\\n            7 & \\textasciitilde  \\\\\n            8 & \\textasciicircum  \\\\\n            9 & \\textbackslash  \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_specified_header_special_chars_without_escape(self) -> None:
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(header=['$A$', '$B$'], escape=False)
        expected = _dedent('\n            \\begin{tabular}{lrl}\n            \\toprule\n             & $A$ & $B$ \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

class TestToLatexPosition:
    def test_to_latex_position(self) -> None:
        the_position = 'h'
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(position=the_position)
        expected = _dedent('\n            \\begin{table}[h]\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\n            ')
        assert result == expected

    def test_to_latex_longtable_position(self) -> None:
        the_position = 't'
        df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
        result = df.to_latex(longtable=True, position=the_position)
        expected = _dedent('\n            \\begin{longtable}[t]{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endfirsthead\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            \\endhead\n            \\midrule\n            \\multicolumn{3}{r}{Continued on next page} \\\\\n            \\midrule\n            \\endfoot\n            \\bottomrule\n            \\endlastfoot\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\end{longtable}\n            ')
        assert result == expected

class TestToLatexFormatters:
    def test_to_latex_with_formatters(self) -> None:
        df = DataFrame({'datetime64': [datetime(2016, 1, 1), datetime(2016, 2, 5), datetime(2016, 3, 3)], 'float': [1.0, 2.0, 3.0], 'int': [1, 2, 3], 'object': [(1, 2), True, False]})
        formatters = {'datetime64': lambda x: x.strftime('%Y-%m'), 'float': lambda x: f'[{x: 4.1f}]', 'int': lambda x: f'0x{x:x}', 'object': lambda x: f'-{x!s}-', '__index__': lambda x: f'index: {x}'}
        result = df.to_latex(formatters=dict(formatters))
        expected = _dedent('\n            \\begin{tabular}{llrrl}\n            \\toprule\n             & datetime64 & float & int & object \\\\\n            \\midrule\n            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\\\\n            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\\\\n            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_3decimals(self) -> None:
        df = DataFrame({'x': [0.19999]})
        result = df.to_latex(float_format='%.3f')
        expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & x \\\\\n            \\midrule\n            0 & 0.200 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_float_format_no_fixed_width_integer(self) -> None:
        df = DataFrame({'x': [100.0]})
        result = df.to_latex(float_format='%.0f')
        expected = _dedent('\n            \\begin{tabular}{lr}\n            \\toprule\n             & x \\\\\n            \\midrule\n            0 & 100 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
    def test_to_latex_na_rep_and_float_format(self, na_rep: str) -> None:
        df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
        result = df.to_latex(na_rep=na_rep, float_format='{:.2f}'.format)
        expected = _dedent(f'\n            \\begin{{tabular}}{{llr}}\n            \\toprule\n             & Group & Data \\\\\n            \\midrule\n            0 & A & 1.22 \\\\\n            1 & A & {na_rep} \\\\\n            \\bottomrule\n            \\end{{tabular}}\n            ')
        assert result == expected

class TestToLatexMultiindex:
    @pytest.fixture
    def multiindex_frame(self) -> DataFrame:
        """Multiindex dataframe for testing multirow LaTeX macros."""
        return DataFrame.from_dict({('c1', 0): Series({x: x for x in range(4)}), ('c1', 1): Series({x: x + 4 for x in range(4)}), ('c2', 0): Series({x: x for x in range(4)}), ('c2', 1): Series({x: x + 4 for x in range(4)}), ('c3', 0): Series({x: x for x in range(4)})}).T

    @pytest.fixture
    def multicolumn_frame(self) -> DataFrame:
        """Multicolumn dataframe for testing multicolumn LaTeX macros."""
        return DataFrame({('c1', 0): {x: x for x in range(5)}, ('c1', 1): {x: x + 5 for x in range(5)}, ('c2', 0): {x: x for x in range(5)}, ('c2', 1): {x: x + 5 for x in range(5)}, ('c3', 0): {x: x for x in range(5)}})

    def test_to_latex_multindex_header(self) -> None:
        df = DataFrame({'a': [0], 'b': [1], 'c': [2], 'd': [3]})
        df = df.set_index(['a', 'b'])
        observed = df.to_latex(header=['r1', 'r2'], multirow=False)
        expected = _dedent('\n            \\begin{tabular}{llrr}\n            \\toprule\n             &  & r1 & r2 \\\\\n            a & b &  &  \\\\\n            \\midrule\n            0 & 1 & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert observed == expected

    def test_to_latex_multiindex_empty_name(self) -> None:
        mi = pd.MultiIndex.from_product([[1, 2]], names=[''])
        df = DataFrame(-1, index=mi, columns=range(4))
        observed = df.to_latex()
        expected = _dedent('\n            \\begin{tabular}{lrrrr}\n            \\toprule\n             & 0 & 1 & 2 & 3 \\\\\n             &  &  &  &  \\\\\n            \\midrule\n            1 & -1 & -1 & -1 & -1 \\\\\n            2 & -1 & -1 & -1 & -1 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert observed == expected

    def test_to_latex_multiindex_column_tabular(self) -> None:
        df = DataFrame({('x', 'y'): ['a']})
        result = df.to_latex()
        expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & x \\\\\n             & y \\\\\n            \\midrule\n            0 & a \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_small_tabular(self) -> None:
        df = DataFrame({('x', 'y'): ['a']}).T
        result = df.to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             &  & 0 \\\\\n            \\midrule\n            x & y & a \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_tabular(self, multiindex_frame: DataFrame) -> None:
        result = multiindex_frame.to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{llrrrr}\n            \\toprule\n             &  & 0 & 1 & 2 & 3 \\\\\n            \\midrule\n            c1 & 0 & 0 & 1 & 2 & 3 \\\\\n             & 1 & 4 & 5 & 6 & 7 \\\\\n            c2 & 0 & 0 & 1 & 2 & 3 \\\\\n             & 1 & 4 & 5 & 6 & 7 \\\\\n            c3 & 0 & 0 & 1 & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multicolumn_tabular(self, multiindex_frame: DataFrame) -> None:
        df = multiindex_frame.T
        df.columns.names = ['a', 'b']
        result = df.to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{lrrrrr}\n            \\toprule\n            a & \\multicolumn{2}{r}{c1} & \\multicolumn{2}{r}{c2} & c3 \\\\\n            b & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            0 & 0 & 4 & 0 & 4 & 0 \\\\\n            1 & 1 & 5 & 1 & 5 & 1 \\\\\n            2 & 2 & 6 & 2 & 6 & 2 \\\\\n            3 & 3 & 7 & 3 & 7 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_index_has_name_tabular(self) -> None:
        df = DataFrame({'a': [0, 0, 1, 1], 'b': list('abab'), 'c': [1, 2, 3, 4]})
        result = df.set_index(['a', 'b']).to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & c \\\\\n            a & b &  \\\\\n            \\midrule\n            0 & a & 1 \\\\\n             & b & 2 \\\\\n            1 & a & 3 \\\\\n             & b & 4 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_groupby_tabular(self) -> None:
        df = DataFrame({'a': [0, 0, 1, 1], 'b': list('abab'), 'c': [1, 2, 3, 4]})
        result = df.groupby('a').describe().to_latex(float_format='{:.1f}'.format, escape=True)
        expected = _dedent('\n            \\begin{tabular}{lrrrrrrrr}\n            \\toprule\n             & \\multicolumn{8}{r}{c} \\\\\n             & count & mean & std & min & 25\\% & 50\\% & 75\\% & max \\\\\n            a &  &  &  &  &  &  &  &  \\\\\n            \\midrule\n            0 & 2.0 & 1.5 & 0.7 & 1.0 & 1.2 & 1.5 & 1.8 & 2.0 \\\\\n            1 & 2.0 & 3.5 & 0.7 & 3.0 & 3.2 & 3.5 & 3.8 & 4.0 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_dupe_level(self) -> None:
        df = DataFrame(index=pd.MultiIndex.from_tuples([('A', 'c'), ('B', 'c')]), columns=['col'])
        result = df.to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n             &  & col \\\\\n            \\midrule\n            A & c & NaN \\\\\n            B & c & NaN \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multicolumn_default(self, multicolumn_frame: DataFrame) -> None:
        result = multicolumn_frame.to_latex()
        expected = _dedent('\n            \\begin{tabular}{lrrrrr}\n            \\toprule\n             & \\multicolumn{2}{r}{c1} & \\multicolumn{2}{r}{c2} & c3 \\\\\n             & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            0 & 0 & 5 & 0 & 5 & 0 \\\\\n            1 & 1 & 6 & 1 & 6 & 1 \\\\\n            2 & 2 & 7 & 2 & 7 & 2 \\\\\n            3 & 3 & 8 & 3 & 8 & 3 \\\\\n            4 & 4 & 9 & 4 & 9 & 4 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multicolumn_false(self, multicolumn_frame: DataFrame) -> None:
        result = multicolumn_frame.to_latex(multicolumn=False, multicolumn_format='l')
        expected = _dedent('\n            \\begin{tabular}{lrrrrr}\n            \\toprule\n             & c1 & & c2 & & c3 \\\\\n             & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            0 & 0 & 5 & 0 & 5 & 0 \\\\\n            1 & 1 & 6 & 1 & 6 & 1 \\\\\n            2 & 2 & 7 & 2 & 7 & 2 \\\\\n            3 & 3 & 8 & 3 & 8 & 3 \\\\\n            4 & 4 & 9 & 4 & 9 & 4 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multirow_true(self, multicolumn_frame: DataFrame) -> None:
        result = multicolumn_frame.T.to_latex(multirow=True)
        expected = _dedent('\n            \\begin{tabular}{llrrrrr}\n            \\toprule\n             &  & 0 & 1 & 2 & 3 & 4 \\\\\n            \\midrule\n            \\multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            \\multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n            \\cline{1-7}\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multicolumnrow_with_multicol_format(self, multicolumn_frame: DataFrame) -> None:
        multicolumn_frame.index = multicolumn_frame.T.index
        result = multicolumn_frame.T.to_latex(multirow=True, multicolumn=True, multicolumn_format='c')
        expected = _dedent('\n            \\begin{tabular}{llrrrrr}\n            \\toprule\n             &  & \\multicolumn{2}{c}{c1} & \\multicolumn{2}{c}{c2} & c3 \\\\\n             &  & 0 & 1 & 0 & 1 & 0 \\\\\n            \\midrule\n            \\multirow[t]{2}{*}{c1} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            \\multirow[t]{2}{*}{c2} & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n             & 1 & 5 & 6 & 7 & 8 & 9 \\\\\n            \\cline{1-7}\n            c3 & 0 & 0 & 1 & 2 & 3 & 4 \\\\\n            \\cline{1-7}\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    @pytest.mark.parametrize('name0', [None, 'named0'])
    @pytest.mark.parametrize('name1', [None, 'named1'])
    @pytest.mark.parametrize('axes', [[0], [1], [0, 1]])
    def test_to_latex_multiindex_names(self, name0: str, name1: str, axes: list) -> None:
        names = [name0, name1]
        mi = pd.MultiIndex.from_product([[1, 2], [3, 4]])
        df = DataFrame(-1, index=mi.copy(), columns=mi.copy())
        for idx in axes:
            df.axes[idx].names = names
        idx_names = tuple((n or '' for n in names))
        idx_names_row = f'{idx_names[0]} & {idx_names[1]} &  &  &  &  \\\\\n' if 0 in axes and any(names) else ''
        col_names = [n if bool(n) and 1 in axes else '' for n in names]
        observed = df.to_latex(multirow=False)
        expected = '\\begin{tabular}{llrrrr}\n\\toprule\n & %s & \\multicolumn{2}{r}{1} & \\multicolumn{2}{r}{2} \\\\\n & %s & 3 & 4 & 3 & 4 \\\\\n%s\\midrule\n1 & 3 & -1 & -1 & -1 & -1 \\\\\n & 4 & -1 & -1 & -1 & -1 \\\\\n2 & 3 & -1 & -1 & -1 & -1 \\\\\n & 4 & -1 & -1 & -1 & -1 \\\\\n\\bottomrule\n\\end{tabular}\n' % tuple(list(col_names) + [idx_names_row])
        assert observed == expected

    @pytest.mark.parametrize('one_row', [True, False])
    def test_to_latex_multiindex_nans(self, one_row: bool) -> None:
        df = DataFrame({'a': [None, 1], 'b': [2, 3], 'c': [4, 5]})
        if one_row:
            df = df.iloc[[0]]
        observed = df.set_index(['a', 'b']).to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & c \\\\\n            a & b &  \\\\\n            \\midrule\n            NaN & 2 & 4 \\\\\n            ')
        if not one_row:
            expected += '1.000000 & 3 & 5 \\\\\n'
        expected += '\\bottomrule\n\\end{tabular}\n'
        assert observed == expected

    def test_to_latex_non_string_index(self) -> None:
        df = DataFrame([[1, 2, 3]] * 2).set_index([0, 1])
        result = df.to_latex(multirow=False)
        expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & 2 \\\\\n            0 & 1 &  \\\\\n            \\midrule\n            1 & 2 & 3 \\\\\n             & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_multirow(self) -> None:
        mi = pd.MultiIndex.from_product([[0.0, 1.0], [3.0, 2.0, 1.0], ['0', '1']], names=['i', 'val0', 'val1'])
        df = DataFrame(index=mi)
        result = df.to_latex(multirow=True, escape=False)
        expected = _dedent('\n            \\begin{tabular}{lll}\n            \\toprule\n            i & val0 & val1 \\\\\n            \\midrule\n            \\multirow[t]{6}{*}{0.000000} & \\multirow[t]{2}{*}{3.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{2.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{1.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\multirow[t]{6}{*}{1.000000} & \\multirow[t]{2}{*}{3.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{2.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{2-3}\n             & \\multirow[t]{2}{*}{1.000000} & 0 \\\\\n             &  & 1 \\\\\n            \\cline{1-3} \\cline{2-3}\n            \\bottomrule\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_format_single_index_hidden(self) -> None:
        df = DataFrame({'A': [1, 2], 'B': [4, 5]})
        result = df.style.hide(axis='index').map_index(lambda v: 'textbf:--rwrap;', axis='columns').to_latex()
        expected = _dedent('\n            \\begin{tabular}{rr}\n            \\textbf{A} & \\textbf{B} \\\\\n            1 & 4 \\\\\n            2 & 5 \\\\\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_format_triple_index_two_hidden(self) -> None:
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two'], ['x', 'x', 'y', 'y']]
        index = pd.MultiIndex.from_arrays(arrays, names=['Level 0', 'Level 1', 'Level 2'])
        df = DataFrame([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], index=index, columns=['C1', 'C2', 'C3'])
        result = df.style.hide(axis='index', level=[0, 1]).map_index(lambda v: 'textbf:--rwrap;', axis='columns').to_latex()
        expected = _dedent('\n            \\begin{tabular}{lrrr}\n             & \\textbf{C1} & \\textbf{C2} & \\textbf{C3} \\\\\n            Level 2 &  &  &  \\\\\n            x & 0 & 0 & 0 \\\\\n            x & 0 & 0 & 0 \\\\\n            y & 0 & 0 & 0 \\\\\n            y & 0 & 0 & 0 \\\\\n            \\end{tabular}\n            ')
        assert result == expected

    def test_to_latex_multiindex_format_triple_index_all_hidden(self) -> None:
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two'], ['x', 'x', 'y', 'y']]
        index = pd.MultiIndex.from_arrays(arrays, names=['Level 0', 'Level 1', 'Level 2'])
        df = DataFrame([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], index=index, columns=['C1', 'C2', 'C3'])
        result = df.style.hide(axis='index', level=[0, 1, 2]).map_index(lambda v: 'textbf:--rwrap;', axis='columns').to_latex()
        expected = _dedent('\n            \\begin{tabular}{rrr}\n            \\textbf{C1} & \\textbf{C2} & \\textbf{C3} \\\\\n            0 & 0 & 0 \\\\\n            0 & 0 & 0 \\\\\n            0 & 0 & 0 \\\\\n            0 & 0 & 0 \\\\\n            \\end{tabular}\n            ')
        assert result == expected
