from __future__ import annotations

import weakref
from typing import Any, Callable, Optional, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pandas import CategoricalDtype, DataFrame, Index, MultiIndex, Series, _testing as tm, option_context
from pandas.core.strings.accessor import StringMethods

_any_allowed_skipna_inferred_dtype: list[tuple[str, list[object]]] = [
    ("string", ["a", np.nan, "c"]),
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
]
ids, _ = zip(*_any_allowed_skipna_inferred_dtype)
ids = cast(tuple[str, ...], ids)


@pytest.fixture(params=_any_allowed_skipna_inferred_dtype, ids=ids)
def any_allowed_skipna_inferred_dtype(request: pytest.FixtureRequest) -> tuple[str, NDArray[object]]:
    """
    Fixture for all (inferred) dtypes allowed in StringMethods.__init__

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_allowed_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_allowed_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    ...
    ...     # constructor for .str-accessor will also pass
    ...     Series(values).str
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)
    values = cast(NDArray[object], values)
    return (inferred_dtype, values)


def test_api(any_string_dtype: object) -> None:
    assert Series.str is StringMethods
    assert isinstance(Series([""], dtype=any_string_dtype).str, StringMethods)


def test_no_circular_reference(any_string_dtype: object) -> None:
    ser = Series([""], dtype=any_string_dtype)
    ref: weakref.ReferenceType[Series] = weakref.ref(ser)
    ser.str
    del ser
    assert ref() is None


def test_api_mi_raises() -> None:
    mi = MultiIndex.from_arrays([["a", "b", "c"]])
    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        mi.str
    assert not hasattr(mi, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_dtype(
    index_or_series: type[Index] | type[Series],
    dtype: type | str,
    any_skipna_inferred_dtype: tuple[str, NDArray[object]],
) -> None:
    box: type[Index] | type[Series] = index_or_series
    inferred_dtype, values = any_skipna_inferred_dtype
    t: Index | Series = box(values, dtype=dtype)
    types_passing_constructor: list[str] = ["string", "unicode", "empty", "bytes", "mixed", "mixed-integer"]
    if inferred_dtype in types_passing_constructor:
        assert isinstance(t.str, StringMethods)
    else:
        msg = "Can only use .str accessor with string values.*"
        with pytest.raises(AttributeError, match=msg):
            t.str
        assert not hasattr(t, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_method(
    index_or_series: type[Index] | type[Series],
    dtype: type | str,
    any_allowed_skipna_inferred_dtype: tuple[str, NDArray[object]],
    any_string_method: tuple[str, tuple[Any, ...], dict[str, Any]],
    request: pytest.FixtureRequest,
    using_infer_string: bool,
) -> None:
    box: type[Index] | type[Series] = index_or_series
    inferred_dtype, values = any_allowed_skipna_inferred_dtype
    method_name, args, kwargs = any_string_method
    reason: Optional[str] = None
    raises: Optional[type[BaseException]] = None
    if box is Index and values.size == 0:
        if method_name in ["partition", "rpartition"] and kwargs.get("expand", True):
            raises = TypeError
            reason = "Method cannot deal with empty Index"
        elif method_name == "split" and kwargs.get("expand", None):
            raises = TypeError
            reason = "Split fails on empty Series when expand=True"
        elif method_name == "get_dummies":
            raises = ValueError
            reason = "Need to fortify get_dummies corner cases"
    elif box is Index and inferred_dtype == "empty" and (dtype == object) and (method_name == "get_dummies"):
        raises = ValueError
        reason = "Need to fortify get_dummies corner cases"
    if reason is not None and raises is not None:
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.applymarker(mark)
    t: Index | Series = box(values, dtype=dtype)
    method: Callable[..., Any] = getattr(t.str, method_name)
    if using_infer_string and dtype == "category":
        string_allowed = method_name not in ["decode"]
    else:
        string_allowed = True
    bytes_allowed = method_name in ["decode", "get", "len", "slice"]
    mixed_allowed = method_name not in ["cat"]
    allowed_types: list[str] = ["empty"] + ["string", "unicode"] * int(string_allowed) + ["bytes"] * int(bytes_allowed) + [
        "mixed",
        "mixed-integer",
    ] * int(mixed_allowed)
    if inferred_dtype in allowed_types:
        with option_context("future.no_silent_downcasting", True):
            method(*args, **kwargs)
    else:
        msg = (
            f"Cannot use .str.{method_name} with values of inferred dtype {inferred_dtype!r}.|"
            "a bytes-like object is required, not 'str'"
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)


def test_api_for_categorical(
    any_string_method: tuple[str, tuple[Any, ...], dict[str, Any]],
    any_string_dtype: object,
) -> None:
    s = Series(list("aabb"), dtype=any_string_dtype)
    s = s + " " + s
    c = s.astype("category")
    c = c.astype(CategoricalDtype(c.dtype.categories.astype("object")))
    assert isinstance(c.str, StringMethods)
    method_name, args, kwargs = any_string_method
    result = getattr(c.str, method_name)(*args, **kwargs)
    expected = getattr(s.astype("object").str, method_name)(*args, **kwargs)
    if isinstance(result, DataFrame):
        tm.assert_frame_equal(result, expected)
    elif isinstance(result, Series):
        tm.assert_series_equal(result, expected)
    else:
        assert result == expected