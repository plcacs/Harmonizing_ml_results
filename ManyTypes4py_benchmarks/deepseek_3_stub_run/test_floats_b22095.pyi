import numpy as np
import pytest
from pandas import DataFrame, Index, RangeIndex, Series
from pandas._testing import _LocIndexer, _IlocIndexer, _SetitemIndexer
from typing import Any, Literal, Type, TypeVar, Union, overload

_T = TypeVar("_T")
_FrameOrSeries = Union[DataFrame, Series]

def gen_obj(klass: Type[_T], index: Index) -> _T: ...

class TestFloatIndexers:
    def check(
        self,
        result: Any,
        original: Union[Series, DataFrame],
        indexer: Union[int, slice],
        getitem: bool,
    ) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    def test_scalar_non_numeric(
        self,
        index: Index,
        frame_or_series: Type[_FrameOrSeries],
        indexer_sl: Union[_LocIndexer, _SetitemIndexer],
    ) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    def test_scalar_non_numeric_series_fallback(self, index: Index) -> None: ...
    
    def test_scalar_with_mixed(
        self, indexer_sl: Union[_LocIndexer, _SetitemIndexer]
    ) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    def test_scalar_integer(
        self,
        index: Union[Index, RangeIndex],
        frame_or_series: Type[_FrameOrSeries],
        indexer_sl: Union[_LocIndexer, _SetitemIndexer],
    ) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    def test_scalar_integer_contains_float(
        self,
        index: Union[Index, RangeIndex],
        frame_or_series: Type[_FrameOrSeries],
    ) -> None: ...
    
    def test_scalar_float(self, frame_or_series: Type[_FrameOrSeries]) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    @pytest.mark.parametrize("idx", ...)
    def test_slice_non_numeric(
        self,
        index: Index,
        idx: slice,
        frame_or_series: Type[_FrameOrSeries],
        indexer_sli: Union[_LocIndexer, _IlocIndexer, _SetitemIndexer],
    ) -> None: ...
    
    def test_slice_integer(self) -> None: ...
    
    @pytest.mark.parametrize("idx", ...)
    def test_integer_positional_indexing(self, idx: slice) -> None: ...
    
    @pytest.mark.parametrize("index", ...)
    def test_slice_integer_frame_getitem(
        self, index: Union[Index, RangeIndex]
    ) -> None: ...
    
    @pytest.mark.parametrize("idx", ...)
    @pytest.mark.parametrize("index", ...)
    def test_float_slice_getitem_with_integer_index_raises(
        self, idx: slice, index: Union[Index, RangeIndex]
    ) -> None: ...
    
    @pytest.mark.parametrize("idx", ...)
    def test_slice_float(
        self,
        idx: slice,
        frame_or_series: Type[_FrameOrSeries],
        indexer_sl: Union[_LocIndexer, _SetitemIndexer],
    ) -> None: ...
    
    def test_floating_index_doc_example(self) -> None: ...
    
    def test_floating_misc(
        self, indexer_sl: Union[_LocIndexer, _SetitemIndexer]
    ) -> None: ...
    
    def test_floatindex_slicing_bug(self, float_numpy_dtype: np.dtype) -> None: ...