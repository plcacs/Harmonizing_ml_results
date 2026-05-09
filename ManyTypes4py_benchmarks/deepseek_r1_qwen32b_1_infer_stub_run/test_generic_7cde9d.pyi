from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Callable, Iterable, Iterator
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index, date_range
from pandas._testing import tm
import pytest

T = TypeVar('T')

def construct(
    box: type,
    shape: Union[int, Tuple[int, ...]],
    value: Optional[Union[np.ndarray, Any]] = None,
    dtype: Optional[np.dtype] = None,
    **kwargs: Any
) -> Any:
    ...

class TestGeneric:
    def test_rename(
        self,
        frame_or_series: Union[DataFrame, Series],
        func: Union[Callable[[str], str], Dict[str, str], Series]
    ) -> None:
        ...

    def test_get_numeric_data(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_get_bool_data_empty_preserve_index(self) -> None:
        ...

    def test_nonzero(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_frame_or_series_compound_dtypes(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_metadata_propagation(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_size_compat(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_split_compat(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_stat_unexpected_keyword(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_api_compat(
        self,
        func: str,
        frame_or_series: Union[DataFrame, Series]
    ) -> None:
        ...

    def test_stat_non_defaults_args(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_truncate_out_of_bounds(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_copy_and_deepcopy(
        self,
        frame_or_series: Union[DataFrame, Series],
        shape: int,
        func: Callable[[Any], Any]
    ) -> None:
        ...

class TestNDFrame:
    def test_squeeze_series_noop(self, ser: Series) -> None:
        ...

    def test_squeeze_frame_noop(self) -> None:
        ...

    def test_squeeze_frame_reindex(self) -> None:
        ...

    def test_squeeze_0_len_dim(self) -> None:
        ...

    def test_squeeze_axis(self) -> None:
        ...

    def test_squeeze_axis_len_3(self) -> None:
        ...

    def test_numpy_squeeze(self) -> None:
        ...

    def test_transpose_series(self, ser: Series) -> None:
        ...

    def test_transpose_frame(self) -> None:
        ...

    def test_numpy_transpose(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_take_series(self, ser: Series) -> None:
        ...

    def test_take_frame(self) -> None:
        ...

    def test_take_invalid_kwargs(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_axis_classmethods(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_flags_identity(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...