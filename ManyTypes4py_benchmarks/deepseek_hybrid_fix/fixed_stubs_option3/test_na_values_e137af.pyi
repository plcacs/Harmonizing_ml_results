from io import StringIO
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex
from pandas._libs.parsers import STR_NA_VALUES
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

def test_string_nas(all_parsers: Any) -> None: ...
def test_detect_string_na(all_parsers: Any) -> None: ...
@pytest.mark.parametrize(
    "na_values",
    [
        ["-999.0", "-999"],
        [-999, -999.0],
        [-999.0, -999],
        ["-999.0"],
        ["-999"],
        [-999.0],
        [-999],
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        "A,B\n-999,1.2\n2,-999\n3,4.5\n",
        "A,B\n-999,1.200\n2,-999.000\n3,4.500\n",
    ],
)
def test_non_string_na_values(
    all_parsers: Any,
    data: str,
    na_values: Any,
    request: Any,
) -> None: ...
def test_default_na_values(all_parsers: Any) -> None: ...
@pytest.mark.parametrize("na_values", ["baz", ["baz"]])
def test_custom_na_values(
    all_parsers: Any, na_values: Union[str, List[str]]
) -> None: ...
def test_bool_na_values(all_parsers: Any) -> None: ...
def test_na_value_dict(all_parsers: Any) -> None: ...
@pytest.mark.parametrize(
    "index_col,expected",
    [
        (
            [0],
            DataFrame(
                {"b": [np.nan], "c": [1], "d": [5]},
                index=Index([0], name="a"),
            ),
        ),
        (
            [0, 2],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
        (
            ["a", "c"],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
    ],
)
def test_na_value_dict_multi_index(
    all_parsers: Any, index_col: Union[List[int], List[str]], expected: DataFrame
) -> None: ...
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}, "keep_default_na": False},
            {
                "A": ["a", "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": ["a"], "keep_default_na": False},
            {
                "A": [np.nan, "b", "", "d", "e", "nan", "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", "nan", "five", "", "seven"],
            },
        ),
        (
            {"na_values": {"A": [], "C": []}},
            {
                "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                "B": [1, 2, 3, 4, 5, 6, 7],
                "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
            },
        ),
    ],
)
def test_na_values_keep_default(
    all_parsers: Any,
    kwargs: Dict[str, Any],
    expected: Dict[str, List[Any]],
    request: Any,
    using_infer_string: bool,
) -> None: ...
def test_no_na_values_no_keep_default(all_parsers: Any) -> None: ...
def test_no_keep_default_na_dict_na_values(all_parsers: Any) -> None: ...
def test_no_keep_default_na_dict_na_scalar_values(all_parsers: Any) -> None: ...
@pytest.mark.parametrize("col_zero_na_values", [113125, "113125"])
def test_no_keep_default_na_dict_na_values_diff_reprs(
    all_parsers: Any, col_zero_na_values: Union[int, str]
) -> None: ...
@pytest.mark.parametrize(
    "na_filter,row_data",
    [
        (True, [[1, "A"], [np.nan, np.nan], [3, "C"]]),
        (False, [["1", "A"], ["nan", "B"], ["3", "C"]]),
    ],
)
def test_na_values_na_filter_override(
    request: Any,
    all_parsers: Any,
    na_filter: bool,
    row_data: List[List[Any]],
    using_infer_string: bool,
) -> None: ...
@skip_pyarrow
def test_na_trailing_columns(all_parsers: Any) -> None: ...
@pytest.mark.parametrize(
    "na_values,row_data",
    [
        (1, [[np.nan, 2.0], [2.0, np.nan]]),
        ({"a": 2, "b": 1}, [[1.0, 2.0], [np.nan, np.nan]]),
    ],
)
def test_na_values_scalar(
    all_parsers: Any,
    na_values: Union[int, Dict[str, int]],
    row_data: List[List[float]],
) -> None: ...
def test_na_values_dict_aliasing(all_parsers: Any) -> None: ...
def test_na_values_dict_null_column_name(all_parsers: Any) -> None: ...
def test_na_values_dict_col_index(all_parsers: Any) -> None: ...
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            str(2**63) + "\n" + str(2**63 + 1),
            {"na_values": [2**63]},
            [str(2**63), str(2**63 + 1)],
        ),
        (
            str(2**63) + ",1" + "\n,2",
            {},
            [[str(2**63), 1], ["", 2]],
        ),
        (str(2**63) + "\n1", {"na_values": [2**63]}, [np.nan, 1]),
    ],
)
def test_na_values_uint64(
    all_parsers: Any,
    data: str,
    kwargs: Dict[str, Any],
    expected: List[Any],
    request: Any,
) -> None: ...
def test_empty_na_values_no_default_with_index(all_parsers: Any) -> None: ...
@pytest.mark.parametrize(
    "na_filter,index_data",
    [(False, ["", "5"]), (True, [np.nan, 5.0])],
)
def test_no_na_filter_on_index(
    all_parsers: Any,
    na_filter: bool,
    index_data: List[Any],
    request: Any,
) -> None: ...
def test_inf_na_values_with_int_index(all_parsers: Any) -> None: ...
@xfail_pyarrow
@pytest.mark.parametrize("na_filter", [True, False])
def test_na_values_with_dtype_str_and_na_filter(
    all_parsers: Any, na_filter: bool
) -> None: ...
@xfail_pyarrow
@pytest.mark.parametrize(
    "data, na_values",
    [
        ("false,1\n,1\ntrue", None),
        ("false,1\nnull,1\ntrue", None),
        ("false,1\nnan,1\ntrue", None),
        ("false,1\nfoo,1\ntrue", "foo"),
        ("false,1\nfoo,1\ntrue", ["foo"]),
        ("false,1\nfoo,1\ntrue", {"a": "foo"}),
    ],
)
def test_cast_NA_to_bool_raises_error(
    all_parsers: Any,
    data: str,
    na_values: Optional[Union[str, List[str], Dict[str, str]]],
) -> None: ...
@xfail_pyarrow
def test_str_nan_dropped(all_parsers: Any) -> None: ...
def test_nan_multi_index(all_parsers: Any) -> None: ...
@xfail_pyarrow
def test_bool_and_nan_to_bool(all_parsers: Any) -> None: ...
def test_bool_and_nan_to_int(all_parsers: Any) -> None: ...
def test_bool_and_nan_to_float(all_parsers: Any) -> None: ...
@xfail_pyarrow
@pytest.mark.parametrize("na_values", [[-99.0, -99], [-99, -99.0]])
def test_na_values_dict_without_dtype(
    all_parsers: Any, na_values: List[Union[float, int]]
) -> None: ...