from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    isna,
)
from pandas._libs.tslibs.timezones import dateutil_gettz
from pandas.api.types import CategoricalDtype

class TestReindexSetIndex:
    def test_dti_set_index_reindex_datetimeindex(self) -> None:
        ...

    def test_dti_set_index_reindex_freq_with_tz(self) -> None:
        ...

    def test_set_reset_index_intervalindex(self) -> None:
        ...

    def test_setitem_reset_index_dtypes(self) -> None:
        ...

    @pytest.mark.parametrize('timezone, year, month, day, hour', [[str, int, int, int, int], [str, int, int, int, int]])
    def test_reindex_timestamp_with_fold(self, timezone: str, year: int, month: int, day: int, hour: int) -> None:
        ...

class TestDataFrameSelectReindex:
    @pytest.mark.xfail(not IS64 or (is_platform_windows() and (not np_version_gt2)), reason: str)
    def test_reindex_tzaware_fill_value(self) -> None:
        ...

    def test_reindex_date_fill_value(self) -> None:
        ...

    def test_reindex_with_multi_index(self) -> None:
        ...

    @pytest.mark.parametrize('method, expected_values', [[str, List[int]], [str, List[int]], [str, List[int]]])
    def test_reindex_methods(self, method: str, expected_values: List[int]) -> None:
        ...

    @pytest.mark.parametrize('method, exp_values', [[str, List[int]], [str, List[int]]])
    def test_reindex_frame_tz_ffill_bfill(self, method: str, exp_values: List[int]) -> None:
        ...

    @pytest.mark.parametrize('method, tolerance', [[str, timedelta], [str, List[timedelta]]])
    def test_reindex_methods_nearest_special(self, method: str, tolerance: Union[timedelta, List[timedelta]]) -> None:
        ...

    def test_reindex_nearest_tz(self, tz_aware_fixture: Any) -> None:
        ...

    def test_reindex_nearest_tz_empty_frame(self) -> None:
        ...

    def test_reindex_frame_add_nat(self) -> None:
        ...

    def test_reindex_limit(self) -> None:
        ...

    @pytest.mark.parametrize('idx, check_index_type', [[List[str], bool], [List[str], bool], [List[str], bool]])
    def test_reindex_level_verify_first_level(self, idx: List[str], check_index_type: bool) -> None:
        ...

    @pytest.mark.parametrize('idx', [List[str], List[str], List[str]])
    def test_reindex_level_verify_first_level_repeats(self, idx: List[str]) -> None:
        ...

    @pytest.mark.parametrize('idx, indexer', [[List[str], List[int]], [List[str], List[int]], [List[str], List[int]]])
    def test_reindex_level_verify_repeats(self, idx: List[str], indexer: List[int]) -> None:
        ...

    @pytest.mark.parametrize('idx, indexer, check_index_type', [[List[str], List[int], bool], [List[str], List[int], bool]])
    def test_reindex_level_verify(self, idx: List[str], indexer: List[int], check_index_type: bool) -> None:
        ...

    def test_non_monotonic_reindex_methods(self) -> None:
        ...

    def test_reindex_sparse(self) -> None:
        ...

    def test_reindex(self, float_frame: DataFrame) -> None:
        ...

    def test_reindex_nan(self) -> None:
        ...

    def test_reindex_name_remains(self) -> None:
        ...

    def test_reindex_int(self, int_frame: DataFrame) -> None:
        ...

    def test_reindex_columns(self, float_frame: DataFrame) -> None:
        ...

    def test_reindex_columns_method(self) -> None:
        ...

    def test_reindex_axes(self) -> None:
        ...

    def test_reindex_fill_value(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]'])
    def test_reindex_datetimelike_to_object(self, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('src_idx', [Index, CategoricalIndex])
    @pytest.mark.parametrize('cat_idx', [Index, CategoricalIndex])
    def test_reindex_empty(self, src_idx: Any, cat_idx: Any) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]'])
    def test_reindex_datetimelike_to_object(self, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('klass', [Index, CategoricalIndex])
    @pytest.mark.parametrize('data', ['A', 'B'])
    def test_reindex_not_category(self, klass: Any, data: str) -> None:
        ...

    def test_invalid_method(self) -> None:
        ...