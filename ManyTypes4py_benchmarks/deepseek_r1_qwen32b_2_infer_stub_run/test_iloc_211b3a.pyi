"""test positional based indexing with iloc"""
from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    Interval,
    NaT,
    Series,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

_slice_iloc_msg = str

class TestiLoc:
    @pytest.mark.parametrize('key', [int, list[int]])
    @pytest.mark.parametrize('index', [Index])
    def test_iloc_getitem_int_and_list_int(self, key: Union[int, list[int]], frame_or_series: Union[DataFrame, Series], index: Index, request) -> None:
        ...

class TestiLocBaseIndependent:
    @pytest.mark.parametrize('key', [slice, range, list[int], Index, np.ndarray])
    def test_iloc_setitem_fullcol_categorical(self, indexer_li: Callable, key: Union[slice, range, list[int], Index, np.ndarray]) -> None:
        ...

    def test_iloc_setitem_ea_inplace(self, frame_or_series: Union[DataFrame, Series], index_or_series_or_array: Union[Index, Series, np.ndarray]) -> None:
        ...

    def test_is_scalar_access(self) -> None:
        ...

    def test_iloc_exceeds_bounds(self) -> None:
        ...

    @pytest.mark.parametrize('index,columns', [(np.ndarray, list[str])])
    @pytest.mark.parametrize('index_vals,column_vals', [[slice, list[str]], (list[str], slice), (list[datetime], slice)])
    def test_iloc_non_integer_raises(self, index: np.ndarray, columns: list[str], index_vals: Union[slice, list[str], list[datetime]], column_vals: Union[slice, list[str]]) -> None:
        ...

    def test_iloc_getitem_invalid_scalar(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_iloc_array_not_mutating_negative_indices(self) -> None:
        ...

    def test_iloc_getitem_neg_int_can_reach_first_index(self) -> None:
        ...

    def test_iloc_getitem_dups(self) -> None:
        ...

    def test_iloc_getitem_array(self) -> None:
        ...

    def test_iloc_getitem_bool(self) -> None:
        ...

    @pytest.mark.parametrize('index', [list[bool], list[bool]])
    def test_iloc_getitem_bool_diff_len(self, index: list[bool]) -> None:
        ...

    def test_iloc_getitem_slice(self) -> None:
        ...

    def test_iloc_getitem_slice_dups(self) -> None:
        ...

    def test_iloc_setitem(self) -> None:
        ...

    def test_iloc_setitem_axis_argument(self) -> None:
        ...

    def test_iloc_setitem_list(self) -> None:
        ...

    def test_iloc_setitem_pandas_object(self) -> None:
        ...

    def test_iloc_setitem_dups(self) -> None:
        ...

    def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(self) -> None:
        ...

    def test_iloc_getitem_frame(self) -> None:
        ...

    def test_iloc_getitem_labelled_frame(self) -> None:
        ...

    def test_iloc_getitem_doc_issue(self) -> None:
        ...

    def test_iloc_setitem_series(self) -> None:
        ...

    def test_iloc_setitem_list_of_lists(self) -> None:
        ...

    @pytest.mark.parametrize('indexer', [list[int], slice, np.ndarray])
    @pytest.mark.parametrize('value', [list[str], np.ndarray])
    def test_iloc_setitem_with_scalar_index(self, indexer: Union[list[int], slice, np.ndarray], value: Union[list[str], np.ndarray]) -> None:
        ...

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_iloc_mask(self) -> None:
        ...

    def test_iloc_non_unique_indexing(self) -> None:
        ...

    def test_iloc_empty_list_indexer_is_ok(self) -> None:
        ...

    def test_identity_slice_returns_new_object(self) -> None:
        ...

    def test_indexing_zerodim_np_array(self) -> None:
        ...

    def test_series_indexing_zerodim_np_array(self) -> None:
        ...

    def test_iloc_setitem_categorical_updates_inplace(self) -> None:
        ...

    def test_iloc_with_boolean_operation(self) -> None:
        ...

    def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self) -> None:
        ...

    def test_iloc_getitem_categorical_values(self) -> None:
        ...

    @pytest.mark.parametrize('value', [None, NaT, float])
    def test_iloc_setitem_td64_values_cast_na(self, value: Union[None, NaT, float]) -> None:
        ...

    @pytest.mark.parametrize('not_na', [Interval, str, float])
    def test_setitem_mix_of_nan_and_interval(self, not_na: Union[Interval, str, float], nulls_fixture: Any) -> None:
        ...

    def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self) -> None:
        ...

    def test_iloc_getitem_read_only_values(self, indexer_li: Callable) -> None:
        ...

    def test_iloc_getitem_readonly_key(self) -> None:
        ...

    def test_iloc_assign_series_to_df_cell(self) -> None:
        ...

    @pytest.mark.parametrize('klass', [list, np.ndarray])
    def test_iloc_setitem_bool_indexer(self, klass: type) -> None:
        ...

    @pytest.mark.parametrize('indexer', [list[int], slice])
    def test_iloc_setitem_pure_position_based(self, indexer: Union[list[int], slice]) -> None:
        ...

    def test_iloc_setitem_dictionary_value(self) -> None:
        ...

    def test_iloc_getitem_float_duplicates(self) -> None:
        ...

    def test_iloc_setitem_custom_object(self) -> None:
        ...

    def test_iloc_getitem_with_duplicates(self) -> None:
        ...

    def test_iloc_getitem_with_duplicates2(self) -> None:
        ...

    def test_iloc_interval(self) -> None:
        ...

    @pytest.mark.parametrize('indexing_func', [list, np.ndarray])
    @pytest.mark.parametrize('rhs_func', [list, np.ndarray])
    def test_loc_setitem_boolean_list(self, rhs_func: type, indexing_func: type) -> None:
        ...

    def test_iloc_getitem_slice_negative_step_ea_block(self) -> None:
        ...

    def test_iloc_setitem_2d_ndarray_into_ea_block(self) -> None:
        ...

    def test_iloc_getitem_int_single_ea_block_view(self) -> None:
        ...

    def test_iloc_setitem_multicolumn_to_datetime(self, using_infer_string: bool) -> None:
        ...

class TestILocErrors:
    def test_iloc_float_raises(self, series_with_simple_index: Series, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame: DataFrame) -> None:
        ...

    def test_iloc_frame_indexer(self) -> None:
        ...

class TestILocSetItemDuplicateColumns:
    def test_iloc_setitem_scalar_duplicate_columns(self) -> None:
        ...

    def test_iloc_setitem_list_duplicate_columns(self) -> None:
        ...

    def test_iloc_setitem_series_duplicate_columns(self) -> None:
        ...

    @pytest.mark.parametrize(['dtypes', 'init_value', 'expected_value'], [('int64', '0', 0), ('float', '1.2', 1.2)])
    def test_iloc_setitem_dtypes_duplicate_columns(self, dtypes: str, init_value: str, expected_value: Union[int, float]) -> None:
        ...

class TestILocCallable:
    def test_frame_iloc_getitem_callable(self) -> None:
        ...

    def test_frame_iloc_setitem_callable(self) -> None:
        ...

class TestILocSeries:
    def test_iloc(self) -> None:
        ...

    def test_iloc_getitem_nonunique(self) -> None:
        ...

    def test_iloc_setitem_pure_position_based(self) -> None:
        ...

    def test_iloc_nullable_int64_size_1_nan(self) -> None:
        ...