import numpy as np
import pytest
from pandas import DataFrame, Index, RangeIndex, Series
from pandas._testing import tm

def gen_obj(klass: type, index: Index) -> DataFrame | Series:
    ...

class TestFloatIndexers:
    def check(self, result: Series | DataFrame, original: Series | DataFrame, indexer: int | slice, getitem: bool) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, Index, date_range, timedelta_range, period_range])
    def test_scalar_non_numeric(self, index: Index, frame_or_series: DataFrame | Series, indexer_sl: Callable) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, Index, date_range, timedelta_range, period_range])
    def test_scalar_non_numeric_series_fallback(self, index: Index) -> None:
        ...

    def test_scalar_with_mixed(self, indexer_sl: Callable) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, RangeIndex])
    def test_scalar_integer(self, index: Index, frame_or_series: DataFrame | Series, indexer_sl: Callable) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, RangeIndex])
    def test_scalar_integer_contains_float(self, index: Index, frame_or_series: DataFrame | Series) -> None:
        ...

    def test_scalar_float(self, frame_or_series: DataFrame | Series) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, date_range, timedelta_range, period_range])
    @pytest.mark.parametrize('idx', [slice, slice, slice])
    def test_slice_non_numeric(self, index: Index, idx: slice, frame_or_series: DataFrame | Series, indexer_sli: Callable) -> None:
        ...

    def test_slice_integer(self) -> None:
        ...

    def test_integer_positional_indexing(self, idx: slice) -> None:
        ...

    @pytest.mark.parametrize('index', [Index, RangeIndex])
    def test_slice_integer_frame_getitem(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('idx', [slice, slice, slice])
    @pytest.mark.parametrize('index', [Index, RangeIndex])
    def test_float_slice_getitem_with_integer_index_raises(self, idx: slice, index: Index) -> None:
        ...

    @pytest.mark.parametrize('idx', [slice, slice, slice])
    def test_slice_float(self, idx: slice, frame_or_series: DataFrame | Series, indexer_sl: Callable) -> None:
        ...

    def test_floating_index_doc_example(self) -> None:
        ...

    def test_floating_misc(self, indexer_sl: Callable) -> None:
        ...

    def test_floatindex_slicing_bug(self, float_numpy_dtype: type) -> None:
        ...