import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series
from pandas._testing import assert_frame_equal, assert_series_equal, assert_almost_equal
import pytest
from typing import Any, Callable, Optional, Union, Tuple, List, Sequence, overload
from numpy.typing import NDArray

def assert_equal(a: Any, b: Any) -> None: ...

class TestMultiIndexSetItem:
    def check(
        self,
        target: DataFrame,
        indexers: Union[
            Tuple[Any, Any],
            Tuple[Any, List[str]],
            Tuple[MultiIndex, MultiIndex],
            Tuple[MultiIndex, slice],
            Tuple[Any, str],
            Tuple[pd.Index, pd.Index],
            Tuple[Series, List[str]]
        ],
        value: Union[int, float, NDArray[np.float64], DataFrame, Series],
        compare_fn: Callable[..., Any] = ...,
        expected: Optional[Union[int, float, DataFrame]] = None
    ) -> None: ...
    
    def test_setitem_multiindex(self) -> None: ...
    
    def test_setitem_multiindex2(self) -> None: ...
    
    def test_setitem_multiindex3(self) -> None: ...
    
    def test_multiindex_setitem(self) -> None: ...
    
    def test_multiindex_setitem2(self) -> None: ...
    
    def test_multiindex_assignment(self) -> None: ...
    
    def test_multiindex_assignment_single_dtype(self) -> None: ...
    
    def test_groupby_example(self) -> None: ...
    
    def test_series_setitem(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None: ...
    
    def test_frame_getitem_setitem_boolean(
        self,
        multiindex_dataframe_random_data: DataFrame
    ) -> None: ...
    
    def test_frame_getitem_setitem_multislice(self) -> None: ...
    
    def test_frame_setitem_multi_column(self) -> None: ...
    
    def test_frame_setitem_multi_column2(self) -> None: ...
    
    def test_loc_getitem_tuple_plus_columns(
        self,
        multiindex_year_month_day_dataframe_random_data: DataFrame
    ) -> None: ...
    
    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_loc_getitem_setitem_slice_integers(
        self,
        frame_or_series: Union[DataFrame, Series]
    ) -> None: ...
    
    def test_setitem_change_dtype(
        self,
        multiindex_dataframe_random_data: DataFrame
    ) -> None: ...
    
    def test_set_column_scalar_with_loc(
        self,
        multiindex_dataframe_random_data: DataFrame
    ) -> None: ...
    
    def test_nonunique_assignment_1750(self) -> None: ...
    
    def test_astype_assignment_with_dups(self) -> None: ...
    
    def test_setitem_nonmonotonic(self) -> None: ...

class TestSetitemWithExpansionMultiIndex:
    def test_setitem_new_column_mixed_depth(self) -> None: ...
    
    def test_setitem_new_column_all_na(self) -> None: ...
    
    def test_setitem_enlargement_keep_index_names(self) -> None: ...

def test_frame_setitem_view_direct(
    multiindex_dataframe_random_data: DataFrame
) -> None: ...

def test_frame_setitem_copy_raises(
    multiindex_dataframe_random_data: DataFrame
) -> None: ...

def test_frame_setitem_copy_no_write(
    multiindex_dataframe_random_data: DataFrame
) -> None: ...

def test_frame_setitem_partial_multiindex() -> None: ...