# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import importlib
import math
import types
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
    get_args,
    get_origin,
)

import numpy as np

from hypothesis import strategies as st
from hypothesis._settings import note_deprecation
from hypothesis.errors import HypothesisException, InvalidArgument
from hypothesis.extra._array_helpers import (
    NDIM_MAX,
    BasicIndex,
    BasicIndexStrategy,
    BroadcastableShapes,
    Shape,
    array_shapes,
    broadcastable_shapes,
    check_argument,
    check_valid_dims,
    mutually_broadcastable_shapes as _mutually_broadcastable_shapes,
    order_check,
    valid_tuple_axes as _valid_tuple_axes,
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.numbers import Real
from hypothesis.strategies._internal.strategies import (
    Ex,
    MappedStrategy,
    T,
    check_strategy,
)
from hypothesis.strategies._internal.utils import defines_strategy


def _try_import(mod_name: str, attr_name: str) -> Any:
    assert "." not in attr_name
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name, None)
    except ImportError:
        return None


if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
else:
    NDArray = _try_import("numpy.typing", "NDArray")

ArrayLike = _try_import("numpy.typing", "ArrayLike")
_NestedSequence = _try_import("numpy._typing._nested_sequence", "_NestedSequence")
_SupportsArray = _try_import("numpy._typing._array_like", "_SupportsArray")

__all__ = [
    "BroadcastableShapes",
    "array_dtypes",
    "array_shapes",
    "arrays",
    "basic_indices",
    "boolean_dtypes",
    "broadcastable_shapes",
    "byte_string_dtypes",
    "complex_number_dtypes",
    "datetime64_dtypes",
    "floating_dtypes",
    "from_dtype",
    "integer_array_indices",
    "integer_dtypes",
    "mutually_broadcastable_shapes",
    "nested_dtypes",
    "scalar_dtypes",
    "timedelta64_dtypes",
    "unicode_string_dtypes",
    "unsigned_integer_dtypes",
    "valid_tuple_axes",
]

TIME_RESOLUTIONS: Tuple[str, ...] = tuple("Y  M  D  h  m  s  ms  us  ns  ps  fs  as".split())

# See https://github.com/HypothesisWorks/hypothesis/pull/3394 and linked discussion.
NP_FIXED_UNICODE: bool = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 19)


@defines_strategy(force_reusable_values=True)
def from_dtype(
    dtype: np.dtype,
    *,
    alphabet: Optional[st.SearchStrategy[str]] = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    min_value: Union[int, float, None] = None,
    max_value: Union[int, float, None] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
    min_magnitude: Real = 0,
    max_magnitude: Optional[Real] = None,
) -> st.SearchStrategy[Any]:
    """Creates a strategy which can generate any value of the given dtype."""
    check_type(np.dtype, dtype, "dtype")
    kwargs = {k: v for k, v in locals().items() if k != "dtype" and v is not None}

    # Compound datatypes, eg 'f4,f4,f4'
    if dtype.names is not None and dtype.fields is not None:
        # mapping np.void.type over a strategy is nonsense, so return now.
        subs = [from_dtype(dtype.fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)

    # Subarray datatypes, eg '(2, 3)i4'
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args: str, **kw: Any) -> Dict[str, Any]:
        """Update default args to the strategy with user-supplied keyword args."""
        assert {"min_value", "max_value", "max_size"}.issuperset(kw)
        for key in set(kwargs).intersection(kw):
            msg = f"dtype {dtype!r} requires {key}={kwargs[key]!r} to be %s {kw[key]!r}"
            if kw[key] is not None:
                if key.startswith("min_") and kw[key] > kwargs[key]:
                    raise InvalidArgument(msg % ("at least",))
                elif key.startswith("max_") and kw[key] < kwargs[key]:
                    raise InvalidArgument(msg % ("at most",))
        kw.update({k: v for k, v in kwargs.items() if k in args or k in kw})
        return kw

    # Scalar datatypes
    if dtype.kind == "b":
        result: st.SearchStrategy[Any] = st.booleans()
    elif dtype.kind == "f":
        result = st.floats(
            width=cast(Literal[16, 32, 64], min(8 * dtype.itemsize, 64)),
            **compat_kw(
                "min_value",
                "max_value",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
                "exclude_min",
                "exclude_max",
            ),
        )
    elif dtype.kind == "c":
        result = st.complex_numbers(
            width=cast(
                Literal[32, 64, 128], min(8 * dtype.itemsize, 128)
            ),  # convert from bytes to bits
            **compat_kw(
                "min_magnitude",
                "max_magnitude",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
            ),
        )
    elif dtype.kind in ("S", "a"):
        # Numpy strings are null-terminated; only allow round-trippable values.
        # `itemsize == 0` means 'fixed length determined at array creation'
        max_size = dtype.itemsize or None
        result = st.binary(**compat_kw("min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != b"\0"
        )
    elif dtype.kind == "u":
        kw = compat_kw(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
        result = st.integers(**kw)
    elif dtype.kind == "i":
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(**compat_kw(min_value=-overflow, max_value=overflow - 1))
    elif dtype.kind == "U":
        # Encoded in UTF-32 (four bytes/codepoint) and null-terminated
        max_size = (dtype.itemsize or 0) // 4 or None
        if NP_FIXED_UNICODE and "alphabet" not in kwargs:
            kwargs["alphabet"] = st.characters()
        result = st.text(**compat_kw("alphabet", "min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != "\0"
        )
    elif dtype.kind in ("m", "M"):
        if "[" in dtype.str:
            res = st.just(dtype.str.split("[")[-1][:-1])
        else:
            # Note that this case isn't valid to pass to arrays(), but we support
            # it here because we'd have to guard against equivalents in arrays()
            # regardless and drawing scalars is a valid use-case.
            res = st.sampled_from(TIME_RESOLUTIONS)
        if allow_nan is not False:
            elems = st.integers(-(2**63), 2**63 - 1) | st.just("NaT")
        else:  # NEP-7 defines the NaT value as integer -(2**63)
            elems = st.integers(-(2**63) + 1, 2**63 - 1)
        result = st.builds(dtype.type, elems, res)
    else:
        raise InvalidArgument(f"No strategy inference for {dtype}")
    return result.map(dtype.type)


class ArrayStrategy(st.SearchStrategy[NDArray[Any]]):
    def __init__(
        self,
        element_strategy: st.SearchStrategy[Any],
        shape: Shape,
        dtype: np.dtype,
        fill: st.SearchStrategy[Any],
        unique: bool,
    ):
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique
        self._check_elements = dtype.kind not in ("O", "V")

    def __repr__(self) -> str:
        return (
            f"ArrayStrategy({self.element_strategy!r}, shape={self.shape}, "
            f"dtype={self.dtype!r}, fill={self.fill!r}, unique={self.unique!r})"
        )

    def set_element(self, val: Any, result: NDArray[Any], idx: int, *, fill: bool = False) -> None:
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f"Could not add element={val!r} of {val.dtype!r} to array of "
                f"{result.dtype!r} - possible mismatch of time units in dtypes?"
            ) from err
        try:
            elem_changed = self._check_elements and val != result[idx] and val == val
        except Exception as err:  # pragma: no cover
            # This branch only exists to help debug weird behaviour in Numpy,
            # such as the string problems we had a while back.
            raise HypothesisException(
                "Internal error when checking element=%r of %r to array of %r"
                % (val, val.dtype, result.dtype)
            ) from err
        if elem_changed:
            strategy = self.fill if fill else self.element_strategy
            if self.dtype.kind == "f":  # pragma: no cover
                # This logic doesn't trigger in our coverage tests under Numpy 1.24+,
                # with built-in checks for overflow, but we keep it for good error
                # messages and compatibility with older versions of Numpy.
                try:
                    is_subnormal = 0 < abs(val) < np.finfo(self.dtype).tiny
                except Exception:
                    # val may be a non-float that does not support the
                    # operations __lt__ and __abs__
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(
                        f"Generated subnormal float {val} from strategy "
                        f"{strategy} resulted in {result[idx]!r}, probably "
                        "as a result of NumPy being built with flush-to-zero "
                        "compiler options. Consider passing "
                        "allow_subnormal=False."
                    )
            raise InvalidArgument(
                "Generated array element %r from %r cannot be represented as "
                "dtype %r - instead it becomes %r (type %r).  Consider using a more "
                "precise strategy, for example passing the `width` argument to "
                "`floats()`."
                % (val, strategy, self.dtype, result[idx], type(result[idx]))
            )

    def do_draw(self, data: Any) -> NDArray[Any]:
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        # Because Numpy allocates memory for strings at array creation, if we have
        # an unsized string dtype we'll fill an object array and then cast it back.
        unsized_string_dtype = (
            self.dtype.kind in ("S", "a", "U") and self.dtype.itemsize == 0
        )

        # This could legitimately be a np.empty, but the performance gains for
        # that would be so marginal that there's really not much point risking
        # undefined behaviour shenanigans.
        result = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.
            if self.unique:
                elems = st.lists(
                    self.element_strategy,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    unique=True,
                )
                for i, v in enumerate(data.draw(elems)):
                    self.set_element(v, result, i)
            else:
                for i in range(len(result)):
                    self.set_element(data.draw(self.element_strategy), result, i)
        else:
            # We draw numpy arrays as "sparse with an offset". We draw a
            # collection of index assignments within the array and assign
            # fresh values from our elements strategy to those indices. If at
            # the end we have not assigned every element then we draw a single
            # value from our fill strategy and use that to populate the
            # remaining positions with that strategy.

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It
                # just grows reasonably quickly but sublinearly, and for small
                # arrays it represents a decent fraction of the array size.
                average_size=min(
                    0.9 * self.array_size,  # ensure small arrays sometimes use fill
                    max(10, math.sqrt(self.array_size)),  # ...but *only* sometimes
                ),
            )

            needs_fill = np.full(self.array_size, True)
            seen = set()

            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data.draw(self.element_strategy), result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])
                needs_fill[i] = False
            if needs_fill.any():
                # We didn't fill all of the indices in the early loop, so we
                # put a fill value into the rest.

                # We have to do this hilarious little song and dance to work
                # around numpy's special handling of iterable values. If the
                # value here were e.g. a tuple then neither array creation
                # nor putmask would do the right thing. But by creating an
                # array of size one and then assigning the fill value as a
                # single element, we both get an array with the right value in
                # it and putmask will do the right thing by repeating the
                # values of the array across the mask.
                one_element = np.zeros(
                    shape=1, dtype=object if unsized_string_dtype else self.dtype
                )
                self.set_element(data.draw(self.fill), one_element, 0, fill=True)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False

                    if not is_nan:
                        raise InvalidArgument(
                            f"Cannot fill unique array with non-NaN value {fill_value!r}"
                        )

                np.putmask(result, needs_fill, one_element)

        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                raise InvalidArgument(
                    "Array elements %r cannot be represented as dtype %r - instead "
                    "they become %r.  Use a more precise strategy, e.g. without "
                    "trailing null bytes, as this will be an error future versions."
                    % (result[mismatch], self.dtype, out[mismatch])
                )
            result = out

        result = result.reshape(self.shape).copy()

        assert result.base is None

        return result


def fill_for(
    elements: st.SearchStrategy[Any],
    unique: bool,
    fill: Optional[st.SearchStrategy[Any]],
    name: str = "",
) -> st.SearchStrategy[Any]:
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        check_strategy(fill, f"{name}.fill" if name else "fill")
    return fill


D = TypeVar("D", bound="DTypeLike")
G = TypeVar("G", bound="np.generic")


@