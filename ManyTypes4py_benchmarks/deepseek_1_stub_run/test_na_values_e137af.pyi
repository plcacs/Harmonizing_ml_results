```python
import _pytest.fixtures
import _pytest.mark.structures
import io
import numpy as np
import pandas
import pandas._testing as tm
from _typeshed import Incomplete
from pandas import DataFrame, Index, MultiIndex
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

pytestmark: Any = ...

def test_string_nas(all_parsers: Any) -> None: ...

def test_detect_string_na(all_parsers: Any) -> None: ...

def test_non_string_na_values(
    all_parsers: Any,
    data: str,
    na_values: Union[List[Union[str, float]], List[str], List[float], str, float],
    request: _pytest.fixtures.SubRequest
) -> None: ...

def test_default_na_values(all_parsers: Any) -> None: ...

def test_custom_na_values(all_parsers: Any, na_values: Union[str, List[str]]) -> None: ...

def test_bool_na_values(all_parsers: Any) -> None: ...

def test_na_value_dict(all_parsers: Any) -> None: ...

def test_na_value_dict_multi_index(
    all_parsers: Any,
    index_col: Union[List[int], List[str]],
    expected: DataFrame
) -> None: ...

def test_na_values_keep_default(
    all_parsers: Any,
    kwargs: Dict[str, Any],
    expected: Dict[str, List[Any]],
    request: _pytest.fixtures.SubRequest,
    using_infer_string: Any
) -> None: ...

def test_no_na_values_no_keep_default(all_parsers: Any) -> None: ...

def test_no_keep_default_na_dict_na_values(all_parsers: Any) -> None: ...

def test_no_keep_default_na_dict_na_scalar_values(all_parsers: Any) -> None: ...

def test_no_keep_default_na_dict_na_values_diff_reprs(
    all_parsers: Any,
    col_zero_na_values: Union[int, str]
) -> None: ...

def test_na_values_na_filter_override(
    request: _pytest.fixtures.SubRequest,
    all_parsers: Any,
    na_filter: bool,
    row_data: List[List[Any]],
    using_infer_string: Any
) -> None: ...

def test_na_trailing_columns(all_parsers: Any) -> None: ...

def test_na_values_scalar(
    all_parsers: Any,
    na_values: Union[int, Dict[str, int]],
    row_data: List[List[float]]
) -> None: ...

def test_na_values_dict_aliasing(all_parsers: Any) -> None: ...

def test_na_values_dict_null_column_name(all_parsers: Any) -> None: ...

def test_na_values_dict_col_index(all_parsers: Any) -> None: ...

def test_na_values_uint64(
    all_parsers: Any,
    data: str,
    kwargs: Dict[str, Any],
    expected: List[Any],
    request: _pytest.fixtures.SubRequest
) -> None: ...

def test_empty_na_values_no_default_with_index(all_parsers: Any) -> None: ...

def test_no_na_filter_on_index(
    all_parsers: Any,
    na_filter: bool,
    index_data: List[Any],
    request: _pytest.fixtures.SubRequest
) -> None: ...

def test_inf_na_values_with_int_index(all_parsers: Any) -> None: ...

def test_na_values_with_dtype_str_and_na_filter(
    all_parsers: Any,
    na_filter: bool
) -> None: ...

def test_cast_NA_to_bool_raises_error(
    all_parsers: Any,
    data: str,
    na_values: Optional[Union[str, List[str], Dict[str, str]]]
) -> None: ...

def test_str_nan_dropped(all_parsers: Any) -> None: ...

def test_nan_multi_index(all_parsers: Any) -> None: ...

def test_bool_and_nan_to_bool(all_parsers: Any) -> None: ...

def test_bool_and_nan_to_int(all_parsers: Any) -> None: ...

def test_bool_and_nan_to_float(all_parsers: Any) -> None: ...

def test_na_values_dict_without_dtype(
    all_parsers: Any,
    na_values: List[Union[float, int]]
) -> None: ...
```