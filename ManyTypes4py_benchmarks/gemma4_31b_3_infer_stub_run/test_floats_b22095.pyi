import numpy as np
from pandas import DataFrame, Index, RangeIndex, Series
from typing import Any, Union, Callable, overload

def gen_obj(klass: Union[type[Series], type[DataFrame]], index: Index) -> Union[Series, DataFrame]: ...

class TestFloatIndexers:
    def check(
        self, 
        result: Any, 
        original: Union[Series, DataFrame], 
        indexer: Any, 
        getitem: bool
    ) -> None: ...

    def test_scalar_non_numeric(
        self, 
        index: Index, 
        frame_or_series: Union[type[Series], type[DataFrame]], 
        indexer_sl: Callable[[Any], Any]
    ) -> None: ...

    def test_scalar_non_numeric_series_fallback(self, index: Index) -> None: ...

    def test_scalar_with_mixed(self, indexer_sl: Callable[[Any], Any]) -> None: ...

    def test_scalar_integer(
        self, 
        index: Union[Index, RangeIndex], 
        frame_or_series: Union[type[Series], type[DataFrame]], 
        indexer_sl: Callable[[Any], Any]
    ) -> None: ...

    def test_scalar_integer_contains_float(
        self, 
        index: Union[Index, RangeIndex], 
        frame_or_series: Union[type[Series], type[DataFrame]]
    ) -> None: ...

    def test_scalar_float(self, frame_or_series: Union[type[Series], type[DataFrame]]) -> None: ...

    def test_slice_non_numeric(
        self, 
        index: Index, 
        idx: slice, 
        frame_or_series: Union[type[Series], type[DataFrame]], 
        indexer_sli: Callable[[Any], Any]
    ) -> None: ...

    def test_slice_integer(self) -> None: ...

    def test_integer_positional_indexing(self, idx: slice) -> None: ...

    def test_slice_integer_frame_getitem(self, index: Union[Index, RangeIndex]) -> None: ...

    def test_float_slice_getitem_with_integer_index_raises(self, idx: slice, index: Union[Index, RangeIndex]) -> None: ...

    def test_slice_float(
        self, 
        idx: slice, 
        frame_or_series: Union[type[Series], type[DataFrame]], 
        indexer_sl: Callable[[Any], Any]
    ) -> None: ...

    def test_floating_index_doc_example(self) -> None: ...

    def test_floating_misc(self, indexer_sl: Callable[[Any], Any]) -> None: ...

    def test_floatindex_slicing_bug(self, float_numpy_dtype: np.dtype) -> None: ...