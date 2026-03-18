```python
from typing import Any, Optional, Union, overload
from pandas import Categorical, CategoricalIndex, Index, Interval, IntervalIndex
from pandas.core.dtypes.dtypes import IntervalDtype
import numpy as np
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com

class ConstructorTests:
    def test_constructor(
        self,
        constructor: Any,
        breaks_and_expected_subtype: Any,
        closed: Any,
        name: Optional[str]
    ) -> None: ...
    def test_constructor_dtype(
        self,
        constructor: Any,
        breaks: Any,
        subtype: Any
    ) -> None: ...
    def test_constructor_pass_closed(
        self,
        constructor: Any,
        breaks: Any
    ) -> None: ...
    def test_constructor_nan(
        self,
        constructor: Any,
        breaks: Any,
        closed: Any
    ) -> None: ...
    def test_constructor_empty(
        self,
        constructor: Any,
        breaks: Any,
        closed: Any
    ) -> None: ...
    def test_constructor_string(
        self,
        constructor: Any,
        breaks: Any
    ) -> None: ...
    def test_constructor_categorical_valid(
        self,
        constructor: Any,
        cat_constructor: Any
    ) -> None: ...
    def test_generic_errors(
        self,
        constructor: Any
    ) -> None: ...

class TestFromArrays(ConstructorTests):
    @property
    def constructor(self) -> Any: ...
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: str = "right"
    ) -> dict[str, Any]: ...
    def test_constructor_errors(self) -> None: ...
    def test_mixed_float_int(
        self,
        left_subtype: Any,
        right_subtype: Any
    ) -> None: ...
    def test_from_arrays_mismatched_datetimelike_resos(
        self,
        interval_cls: Any
    ) -> None: ...

class TestFromBreaks(ConstructorTests):
    @property
    def constructor(self) -> Any: ...
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: str = "right"
    ) -> dict[str, Any]: ...
    def test_constructor_errors(self) -> None: ...
    def test_length_one(self) -> None: ...
    def test_left_right_dont_share_data(self) -> None: ...

class TestFromTuples(ConstructorTests):
    @property
    def constructor(self) -> Any: ...
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: str = "right"
    ) -> dict[str, Any]: ...
    def test_constructor_errors(self) -> None: ...
    def test_na_tuples(self) -> None: ...

class TestClassConstructors(ConstructorTests):
    @property
    def constructor(self) -> Any: ...
    def get_kwargs_from_breaks(
        self,
        breaks: Any,
        closed: str = "right"
    ) -> dict[str, Any]: ...
    def test_generic_errors(
        self,
        constructor: Any
    ) -> None: ...
    def test_constructor_string(self) -> None: ...
    def test_constructor_errors(
        self,
        klass: Any
    ) -> None: ...
    def test_override_inferred_closed(
        self,
        constructor: Any,
        data: Any,
        closed: Any
    ) -> None: ...
    def test_index_object_dtype(
        self,
        values_constructor: Any
    ) -> None: ...
    def test_index_mixed_closed(self) -> None: ...

def test_interval_index_subtype(
    timezone: str,
    inclusive_endpoints_fixture: Any
) -> None: ...
def test_dtype_closed_mismatch() -> None: ...
def test_ea_dtype(dtype: Any) -> None: ...
```