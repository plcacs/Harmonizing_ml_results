from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal
import numpy as np
import pandas as pd
from pandas import Categorical, CategoricalIndex, Index, Interval, IntervalIndex
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas.core.arrays import IntervalArray
import pytest

_T = TypeVar("_T")

class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """
    def test_constructor(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks_and_expected_subtype: Tuple[
            Union[
                List[int],
                np.ndarray,
                Index,
                pd.DatetimeIndex,
                pd.TimedeltaIndex
            ],
            Union[np.dtype, str]
        ],
        closed: Literal["right", "left", "both", "neither"],
        name: Optional[str]
    ) -> None: ...
    
    def test_constructor_dtype(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Index,
        subtype: str
    ) -> None: ...
    
    def test_constructor_pass_closed(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Union[
            Index,
            pd.DatetimeIndex,
            pd.TimedeltaIndex
        ]
    ) -> None: ...
    
    def test_constructor_nan(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: List[float],
        closed: Literal["right", "left", "both", "neither"]
    ) -> None: ...
    
    def test_constructor_empty(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Union[List[Any], np.ndarray],
        closed: Literal["right", "left", "both", "neither"]
    ) -> None: ...
    
    def test_constructor_string(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Union[
            Tuple[str, ...],
            List[str],
            np.ndarray
        ]
    ) -> None: ...
    
    def test_constructor_categorical_valid(
        self,
        constructor: Callable[..., IntervalIndex],
        cat_constructor: Union[Type[Categorical], Type[CategoricalIndex]]
    ) -> None: ...
    
    def test_generic_errors(
        self,
        constructor: Callable[..., IntervalIndex]
    ) -> None: ...
    
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> dict: ...

class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...
    
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> dict: ...
    
    def test_constructor_errors(self) -> None: ...
    
    def test_mixed_float_int(
        self,
        left_subtype: np.dtype,
        right_subtype: np.dtype
    ) -> None: ...
    
    def test_from_arrays_mismatched_datetimelike_resos(
        self,
        interval_cls: Union[Type[IntervalArray], Type[IntervalIndex]]
    ) -> None: ...

class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...
    
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> dict: ...
    
    def test_constructor_errors(self) -> None: ...
    
    def test_length_one(self) -> None: ...
    
    def test_left_right_dont_share_data(self) -> None: ...

class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...
    
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> dict: ...
    
    def test_constructor_errors(self) -> None: ...
    
    def test_na_tuples(self) -> None: ...

class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...
    
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> dict: ...
    
    def test_generic_errors(
        self,
        constructor: Callable[..., IntervalIndex]
    ) -> None: ...
    
    def test_constructor_string(self) -> None: ...
    
    def test_constructor_errors(
        self,
        klass: Union[Type[IntervalIndex], Callable[..., Index]]
    ) -> None: ...
    
    def test_override_inferred_closed(
        self,
        constructor: Callable[..., IntervalIndex],
        data: Union[
            List[Union[Interval, float]],
            IntervalIndex
        ],
        closed: Literal["right", "left", "both", "neither"]
    ) -> None: ...
    
    def test_index_object_dtype(
        self,
        values_constructor: Union[
            Type[List],
            Type[np.ndarray],
            Type[IntervalIndex],
            Type[IntervalArray]
        ]
    ) -> None: ...
    
    def test_index_mixed_closed(self) -> None: ...

def test_interval_index_subtype(
    timezone: str,
    inclusive_endpoints_fixture: Literal["right", "left", "both", "neither"]
) -> None: ...

def test_dtype_closed_mismatch() -> None: ...

def test_ea_dtype(dtype: str) -> None: ...