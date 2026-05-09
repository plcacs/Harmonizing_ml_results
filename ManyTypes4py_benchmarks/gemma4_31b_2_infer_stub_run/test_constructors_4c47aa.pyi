import numpy as np
import pandas as pd
from pandas import Categorical, CategoricalDtype, CategoricalIndex, Index, Interval, IntervalIndex
from pandas.core.arrays import IntervalArray
from pandas.core.dtypes.dtypes import IntervalDtype
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    def test_constructor(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks_and_expected_subtype: Tuple[Any, Any],
        closed: str,
        name: Optional[str],
    ) -> None: ...

    def test_constructor_dtype(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Index,
        subtype: str,
    ) -> None: ...

    def test_constructor_pass_closed(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Index,
    ) -> None: ...

    def test_constructor_nan(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: List[float],
        closed: str,
    ) -> None: ...

    def test_constructor_empty(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Any,
        closed: str,
    ) -> None: ...

    def test_constructor_string(
        self,
        constructor: Callable[..., IntervalIndex],
        breaks: Any,
    ) -> None: ...

    def test_constructor_categorical_valid(
        self,
        constructor: Callable[..., IntervalIndex],
        cat_constructor: Callable[..., Union[Categorical, CategoricalIndex]],
    ) -> None: ...

    def test_generic_errors(self, constructor: Callable[..., IntervalIndex]) -> None: ...

class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = 'right') -> Dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    def test_mixed_float_int(self, left_subtype: Any, right_subtype: Any) -> None: ...

    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls: Union[IntervalArray, type[IntervalIndex]]) -> None: ...

class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = 'right') -> Dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    def test_length_one(self) -> None: ...

    def test_left_right_dont_share_data(self) -> None: ...

class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = 'right') -> Dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    def test_na_tuples(self) -> None: ...

class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = 'right') -> Dict[str, Any]: ...

    def test_generic_errors(self, constructor: Callable[..., IntervalIndex]) -> None: ...

    def test_constructor_string(self) -> None: ...

    def test_constructor_errors(self, klass: Callable[..., IntervalIndex]) -> None: ...

    def test_override_inferred_closed(
        self,
        constructor: Callable[..., IntervalIndex],
        data: Any,
        closed: str,
    ) -> None: ...

    def test_index_object_dtype(self, values_constructor: Callable[..., Any]) -> None: ...

    def test_index_mixed_closed(self) -> None: ...

def test_interval_index_subtype(timezone: str, inclusive_endpoints_fixture: str) -> None: ...

def test_dtype_closed_mismatch() -> None: ...

def test_ea_dtype(dtype: str) -> None: ...