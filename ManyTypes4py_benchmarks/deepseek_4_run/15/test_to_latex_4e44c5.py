import codecs
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
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
    def test_to_latex_bad_column_format(self, bad_column_format: Any) -> None:
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
    def test_to_latex_longtable_continued_on_next_page(self, df_data: Dict[str, List[int]], expected_number: int) -> None:
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
    def test_to_latex_number_of_items_in_header_missmatch_raises(self, header: Union[List[str], Tuple[str, ...]], num_aliases: int) -> None:
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
        expected = _dedent('\n            \\begin{table}\n            \\caption[a table]{a table in a \\texttt{table/tabular} environment}\n            \\begin{tabular}{lrl}\n            \\toprule\n             & a & b \\\\\n            \\midrule\n            0 & 1 & b1 \\\\\n            1 & 2 & b2 \\\\\n            \\bottomrule\n            \\end{tabular}\n            \\end{table}\