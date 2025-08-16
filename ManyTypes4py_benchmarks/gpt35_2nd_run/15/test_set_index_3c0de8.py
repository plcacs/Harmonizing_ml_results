from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas import Categorical, CategoricalIndex, DataFrame, DatetimeIndex, Index, MultiIndex, Series, date_range, period_range, to_datetime
import pandas._testing as tm

@pytest.fixture
def frame_of_index_cols() -> DataFrame:
    ...

class TestSetIndex:

    def test_set_index_multiindex(self) -> None:
        ...

    def test_set_index_empty_column(self) -> None:
        ...

    def test_set_index_empty_dataframe(self) -> None:
        ...

    def test_set_index_multiindexcolumns(self) -> None:
        ...

    def test_set_index_timezone(self) -> None:
        ...

    def test_set_index_cast_datetimeindex(self) -> None:
        ...

    def test_set_index_dst(self) -> None:
        ...

    def test_set_index(self, float_string_frame: DataFrame) -> None:
        ...

    def test_set_index_names(self) -> None:
        ...

    def test_set_index_drop_inplace(self, frame_of_index_cols: DataFrame, drop: bool, inplace: bool, keys: str) -> None:
        ...

    def test_set_index_append(self, frame_of_index_cols: DataFrame, drop: bool, keys: str) -> None:
        ...

    def test_set_index_append_to_multiindex(self, frame_of_index_cols: DataFrame, drop: bool, keys: str) -> None:
        ...

    def test_set_index_after_mutation(self) -> None:
        ...

    def test_set_index_pass_single_array(self, frame_of_index_cols: DataFrame, drop: bool, append: bool, index_name: str, box: callable) -> None:
        ...

    def test_set_index_pass_arrays(self, frame_of_index_cols: DataFrame, drop: bool, append: bool, index_name: str, box: callable) -> None:
        ...

    def test_set_index_pass_arrays_duplicate(self, frame_of_index_cols: DataFrame, drop: bool, append: bool, index_name: str, box1: callable, box2: callable) -> None:
        ...

    def test_set_index_pass_multiindex(self, frame_of_index_cols: DataFrame, drop: bool, append: bool) -> None:
        ...

    def test_construction_with_categorical_index(self) -> None:
        ...

    def test_set_index_preserve_categorical_dtype(self) -> None:
        ...

    def test_set_index_datetime(self) -> None:
        ...

    def test_set_index_period(self) -> None:
        ...

class TestSetIndexInvalid:

    def test_set_index_verify_integrity(self, frame_of_index_cols: DataFrame) -> None:
        ...

    def test_set_index_raise_keys(self, frame_of_index_cols: DataFrame, drop: bool, append: bool) -> None:
        ...

    def test_set_index_raise_on_type(self, frame_of_index_cols: DataFrame, drop: bool, append: bool) -> None:
        ...

    def test_set_index_raise_on_len(self, frame_of_index_cols: DataFrame, box: callable, length: int, drop: bool, append: bool) -> None:
        ...

class TestSetIndexCustomLabelType:

    def test_set_index_custom_label_type(self) -> None:
        ...

    def test_set_index_custom_label_hashable_iterable(self) -> None:
        ...

    def test_set_index_custom_label_type_raises(self) -> None:
        ...

    def test_set_index_periodindex(self) -> None:
        ...
