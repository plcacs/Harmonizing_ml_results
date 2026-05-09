import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import DataFrame, Series

@pytest.fixture
def df_short() -> DataFrame:
    ...

class TestToLatex:
    def test_to_latex_to_file(self, float_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_to_file_utf8_with_encoding(self) -> None:
        ...

    def test_to_latex_to_file_utf8_without_encoding(self) -> None:
        ...

    def test_to_latex_tabular_with_index(self) -> None:
        ...

    def test_to_latex_tabular_without_index(self) -> None:
        ...

    @pytest.mark.parametrize('bad_column_format', [5, 1.2, ['l', 'r'], ('r', 'c'), {'r', 'c', 'l'}, {'a': 'r', 'b': 'l'}])
    def test_to_latex_bad_column_format(self, bad_column_format: object) -> None:
        ...

    def test_to_latex_column_format_just_works(self, float_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_column_format(self) -> None:
        ...

    def test_to_latex_float_format_object_col(self) -> None:
        ...

    def test_to_latex_empty_tabular(self) -> None:
        ...

    def test_to_latex_series(self) -> None:
        ...

    def test_to_latex_midrule_location(self) -> None:
        ...

class TestToLatexLongtable:
    def test_to_latex_empty_longtable(self) -> None:
        ...

    def test_to_latex_longtable_with_index(self) -> None:
        ...

    def test_to_latex_longtable_without_index(self) -> None:
        ...

    @pytest.mark.parametrize('df_data, expected_number', [({'a': [1, 2]}, 1), ({'a': [1, 2], 'b': [3, 4]}, 2), ({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, 3)])
    def test_to_latex_longtable_continued_on_next_page(self, df_data: dict, expected_number: int) -> None:
        ...

class TestToLatexHeader:
    def test_to_latex_no_header_with_index(self) -> None:
        ...

    def test_to_latex_no_header_without_index(self) -> None:
        ...

    def test_to_latex_specified_header_with_index(self) -> None:
        ...

    def test_to_latex_specified_header_without_index(self) -> None:
        ...

    @pytest.mark.parametrize('header, num_aliases', [(['A'], 1), (('B',), 1), (('Col1', 'Col2', 'Col3'), 3), (('Col1', 'Col2', 'Col3', 'Col4'), 4)])
    def test_to_latex_number_of_items_in_header_missmatch_raises(self, header: object, num_aliases: int) -> None:
        ...

    def test_to_latex_decimal(self) -> None:
        ...

class TestToLatexBold:
    def test_to_latex_bold_rows(self) -> None:
        ...

    def test_to_latex_no_bold_rows(self) -> None:
        ...

class TestToLatexCaptionLabel:
    def test_to_latex_caption_only(self, df_short: pytest.fixture, caption_table: pytest.fixture) -> None:
        ...

    def test_to_latex_label_only(self, df_short: pytest.fixture, label_table: pytest.fixture) -> None:
        ...

    def test_to_latex_caption_and_label(self, df_short: pytest.fixture, caption_table: pytest.fixture, label_table: pytest.fixture) -> None:
        ...

    def test_to_latex_caption_and_shortcaption(self, df_short: pytest.fixture, caption_table: pytest.fixture, short_caption: pytest.fixture) -> None:
        ...

    def test_to_latex_caption_and_shortcaption_list_is_ok(self, df_short: pytest.fixture) -> None:
        ...

    def test_to_latex_caption_shortcaption_and_label(self, df_short: pytest.fixture, caption_table: pytest.fixture, short_caption: pytest.fixture, label_table: pytest.fixture) -> None:
        ...

    @pytest.mark.parametrize('bad_caption', [('full_caption', 'short_caption', 'extra_string'), ('full_caption', 'short_caption', 1), ('full_caption', 'short_caption', None), ('full_caption',), (None,)])
    def test_to_latex_bad_caption_raises(self, bad_caption: object) -> None:
        ...

    def test_to_latex_two_chars_caption(self, df_short: pytest.fixture) -> None:
        ...

    def test_to_latex_longtable_caption_only(self, df_short: pytest.fixture, caption_longtable: pytest.fixture) -> None:
        ...

    def test_to_latex_longtable_label_only(self, df_short: pytest.fixture, label_longtable: pytest.fixture) -> None:
        ...

    def test_to_latex_longtable_caption_and_label(self, df_short: pytest.fixture, caption_longtable: pytest.fixture, label_longtable: pytest.fixture) -> None:
        ...

    def test_to_latex_longtable_caption_shortcaption_and_label(self, df_short: pytest.fixture, caption_longtable: pytest.fixture, short_caption: pytest.fixture, label_longtable: pytest.fixture) -> None:
        ...

class TestToLatexEscape:
    def test_to_latex_escape_false(self, df_with_symbols: pytest.fixture) -> None:
        ...

    def test_to_latex_escape_default(self, df_with_symbols: pytest.fixture) -> None:
        ...

    def test_to_latex_special_escape(self) -> None:
        ...

    def test_to_latex_escape_special_chars(self) -> None:
        ...

    def test_to_latex_specified_header_special_chars_without_escape(self) -> None:
        ...

class TestToLatexPosition:
    def test_to_latex_position(self) -> None:
        ...

    def test_to_latex_longtable_position(self) -> None:
        ...

class TestToLatexFormatters:
    def test_to_latex_with_formatters(self) -> None:
        ...

    def test_to_latex_float_format_no_fixed_width_3decimals(self) -> None:
        ...

    def test_to_latex_float_format_no_fixed_width_integer(self) -> None:
        ...

    @pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
    def test_to_latex_na_rep_and_float_format(self, na_rep: str) -> None:
        ...

class TestToLatexMultiindex:
    def test_to_latex_multindex_header(self) -> None:
        ...

    def test_to_latex_multiindex_empty_name(self) -> None:
        ...

    def test_to_latex_multiindex_column_tabular(self) -> None:
        ...

    def test_to_latex_multiindex_small_tabular(self) -> None:
        ...

    def test_to_latex_multiindex_tabular(self, multiindex_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_multicolumn_tabular(self, multiindex_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_index_has_name_tabular(self) -> None:
        ...

    def test_to_latex_groupby_tabular(self) -> None:
        ...

    def test_to_latex_multiindex_dupe_level(self) -> None:
        ...

    def test_to_latex_multicolumn_default(self, multicolumn_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_multicolumn_false(self, multicolumn_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_multirow_true(self, multicolumn_frame: pytest.fixture) -> None:
        ...

    def test_to_latex_multicolumnrow_with_multicol_format(self, multicolumn_frame: pytest.fixture) -> None:
        ...

    @pytest.mark.parametrize('name0', [None, 'named0'])
    @pytest.mark.parametrize('name1', [None, 'named1'])
    @pytest.mark.parametrize('axes', [[0], [1], [0, 1]])
    def test_to_latex_multiindex_names(self, name0: object, name1: object, axes: object) -> None:
        ...

    @pytest.mark.parametrize('one_row', [True, False])
    def test_to_latex_multiindex_nans(self, one_row: bool) -> None:
        ...

    def test_to_latex_non_string_index(self) -> None:
        ...

    def test_to_latex_multiindex_multirow(self) -> None:
        ...

    def test_to_latex_multiindex_format_single_index_hidden(self) -> None:
        ...

    def test_to_latex_multiindex_format_triple_index_two_hidden(self) -> None:
        ...

    def test_to_latex_multiindex_format_triple_index_all_hidden(self) -> None:
        ...