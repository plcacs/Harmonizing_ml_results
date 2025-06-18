from typing import Any, Optional, Sequence, Tuple, TypeVar, Union, overload, Mapping, cast, Literal
import types
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
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

D = TypeVar("D", bound="DTypeLike")
G = TypeVar("G", bound="np.generic")
I = TypeVar("I", bound=np.integer)

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

TIME_RESOLUTIONS = tuple("Y  M  D  h  m  s  ms  us  ns  ps  fs  as".split())

NP_FIXED_UNICODE = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 19)

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
    check_type(np.dtype, dtype, "dtype")
    kwargs = {k: v for k, v in locals().items() if k != "dtype" and v is not None}

    if dtype.names is not None and dtype.fields is not None:
        subs = [from_dtype(dtype.fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)

    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args, **kw):
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
            width=cast(Literal[32, 64, 128], min(8 * dtype.itemsize, 128)),
            **compat_kw(
                "min_magnitude",
                "max_magnitude",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
            ),
        )
    elif dtype.kind in ("S", "a"):
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
            res = st.sampled_from(TIME_RESOLUTIONS)
        if allow_nan is not False:
            elems = st.integers(-(2**63), 2**63 - 1) | st.just("NaT")
        else:
            elems = st.integers(-(2**63) + 1, 2**63 - 1)
        result = st.builds(dtype.type, elems, res)
    else:
        raise InvalidArgument(f"No strategy inference for {dtype}")
    return result.map(dtype.type)

class ArrayStrategy(st.SearchStrategy):
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

    def set_element(self, val: Any, result: np.ndarray, idx: int, *, fill: bool = False) -> None:
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f"Could not add element={val!r} of {val.dtype!r} to array of "
                f"{result.dtype!r} - possible mismatch of time units in dtypes?"
            ) from err
        try:
            elem_changed = self._check_elements and val != result[idx] and val == val
        except Exception as err:
            raise HypothesisException(
                "Internal error when checking element=%r of %r to array of %r"
                % (val, val.dtype, result.dtype)
            ) from err
        if elem_changed:
            strategy = self.fill if fill else self.element_strategy
            if self.dtype.kind == "f":
                try:
                    is_subnormal = 0 < abs(val) < np.finfo(self.dtype).tiny
                except Exception:
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

    def do_draw(self, data: cu.DataObject) -> np.ndarray:
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        unsized_string_dtype = (
            self.dtype.kind in ("S", "a", "U") and self.dtype.itemsize == 0
        )

        result = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
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
            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                average_size=min(
                    0.9 * self.array_size,
                    max(10, math.sqrt(self.array_size)),
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

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union["np.dtype[G]", st.SearchStrategy["np.dtype[G]"]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[G]]": ...

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union[D, st.SearchStrategy[D]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[Any]]": ...

@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union[D, st.SearchStrategy[D]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[Any]]":
    if type(dtype) in (getattr(types, "UnionType", object()), Union):
        dtype = st.one_of(*(from_dtype(np.dtype(d)) for d in dtype.__args__))  # type: ignore

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique)
        )
    dtype = np.dtype(dtype)  # type: ignore[arg-type]
    assert isinstance(dtype, np.dtype)
    if elements is None or isinstance(elements, Mapping):
        if dtype.kind in ("m", "M") and "[" not in dtype.str:
            return (
                st.sampled_from(TIME_RESOLUTIONS)
                .map((dtype.str + "[{}]").format)
                .flatmap(lambda d: arrays(d, shape=shape, fill=fill, unique=unique))
            )
        elements = from_dtype(dtype, **(elements or {}))
    check_strategy(elements, "elements")
    unwrapped = unwrap_strategies(elements)
    if isinstance(unwrapped, MappedStrategy) and unwrapped.pack == dtype.type:
        elements = unwrapped.mapped_strategy
        if getattr(unwrapped, "force_has_reusable_values", False):
            elements.force_has_reusable_values = True  # type: ignore
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all(isinstance(s, int) for s in shape),
        "Array shape must be integer in each dimension, provided shape was {}",
        shape,
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)

@defines_strategy()
def scalar_dtypes() -> st.SearchStrategy[np.dtype]:
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(),
        complex_number_dtypes(),
        datetime64_dtypes(),
        timedelta64_dtypes(),
    )

def defines_dtype_strategy(strat: T) -> T:
    @defines_strategy()
    @proxies(strat)
    def inner(*args, **kwargs):
        return strat(*args, **kwargs).map(np.dtype)

    return inner

@defines_dtype_strategy
def boolean_dtypes() -> st.SearchStrategy["np.dtype[np.bool_]"]:
    return st.just("?")  # type: ignore[arg-type]

def dtype_factory(kind: str, sizes: Sequence[int], valid_sizes: Optional[Sequence[int]], endianness: str) -> st.SearchStrategy[str]:
    valid_endian = ("?", "<", "=", ">")
    check_argument(
        endianness in valid_endian,
        "Unknown endianness: was {}, must be in {}",
        endianness,
        valid_endian,
    )
    if valid_sizes is not None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        check_argument(sizes, "Dtype must have at least one possible size.")
        check_argument(
            all(s in valid_sizes for s in sizes),
            "Invalid sizes: was {} must be an item or sequence in {}",
            sizes,
            valid_sizes,
        )
        if all(isinstance(s, int) for s in sizes):
            sizes = sorted({s // 8 for s in sizes})
    strat = st.sampled_from(sizes)
    if "{}" not in kind:
        kind += "{}"
    if endianness == "?":
        return strat.map(("<" + kind).format) | strat.map((">" + kind).format)
    return strat.map((endianness + kind).format)

@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8],
) -> st.SearchStrategy["np.dtype[np.uint8]"]: ...

@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.uint16]"]: ...

@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.uint32]"]: ...

@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.uint64]"]: ...

@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.unsignedinteger[Any]]"]: ...

@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[8, 16, 32, 64], Sequence[Literal[8, 16, 32, 64]]] = (
        8,
        16,
        32,
        64,
    ),
) -> st.SearchStrategy["np.dtype[np.unsignedinteger[Any]]"]:
    return dtype_factory("u", sizes, (8, 16, 32, 64), endianness)

@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8],
) -> st.SearchStrategy["np.dtype[np.int8]"]: ...

@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.int16]"]: ...

@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.int32]"]: ...

@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.int64]"]: ...

@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.signedinteger[Any]]"]: ...

@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[8, 16, 32, 64], Sequence[Literal[8, 16, 32, 64]]] = (
        8,
        16,
        32,
        64,
    ),
) -> st.SearchStrategy["np.dtype[np.signedinteger[Any]]"]:
    return dtype_factory("i", sizes, (8, 16, 32, 64), endianness)

@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.float16]"]: ...

@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.float32]"]: ...

@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.float64]"]: ...

@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[128],
) -> st.SearchStrategy["np.dtype[np.float128]"]: ...

@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[16, 32, 64, 96, 128]] = (16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.floating[Any]]"]: ...

@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[16, 32, 64, 96, 128], Sequence[Literal[16, 32, 64, 96, 128]]] = (16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.floating[Any]]"]:
    return dtype_factory("f", sizes, (16, 32, 64, 96, 128), endianness)

@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.complex64]"]: ...

@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[128],
) -> st.SearchStrategy["np.dtype[np.complex128]"]: ...

@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[256],
) -> st.SearchStrategy["np.dtype[np.complex256]"]: ...

@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[64, 128, 192, 256]] = (64, 128),
) -> st.SearchStrategy["np.dtype[np.complexfloating[Any, Any]]"]: ...

@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[64, 128, 192, 256], Sequence[Literal[64, 128, 192, 256]]] = (
        64,
        128,
    ),
) -> st.SearchStrategy["np.dtype[np.complexfloating[Any, Any]]"]:
    return dtype_factory("c", sizes, (64, 128, 192, 256), endianness)

@check_function
def validate_time_slice(max_period: str, min_period: str) -> Sequence[str]:
    check_argument(
        max_period in TIME_RESOLUTIONS,
        "max_period {} must be a valid resolution in {}",
        max_period,
        TIME_RESOLUTIONS,
    )
    check_argument(
        min_period in TIME_RESOLUTIONS,
        "min_period {} must be a valid resolution in {}",
        min_period,
        TIME_RESOLUTIONS,
    )
    start = TIME_RESOLUTIONS.index(max_period)
    end = TIME_RESOLUTIONS.index(min_period) + 1
    check_argument(
        start < end,
        "max_period {} must be earlier in sequence {} than min_period {}",
        max_period,
        TIME_RESOLUTIONS,
        min_period,
    )
    return TIME_RESOLUTIONS[start:end]

@defines_dtype_strategy
def datetime64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy["np.dtype[np.datetime64]"]:
    return dtype_factory(
        "datetime64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )

@defines_dtype_strategy
def timedelta64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy["np.dtype[np.timedelta64]"]:
    return dtype_factory(
        "timedelta64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )

@defines_dtype_strategy
def byte_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy["np.dtype[np.bytes_]"]:
    order_check("len", 1, min_len, max_len)
    return dtype_factory("S", list(range(min_len, max_len + 1)), None, endianness)

@defines_dtype_strategy
def unicode_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy["np.dtype[np.str_]"]:
    order_check("len", 1, min_len, max_len)
    return dtype_factory("U", list(range(min_len, max_len + 1)), None, endianness)

def _no_title_is_name_of_a_titled_field(ls: Sequence[Tuple[Union[str, Tuple[str, str]], ...]]) -> bool:
    seen = set()
    for title_and_name, *_ in ls:
        if isinstance(title_and_name, tuple):
            if seen.intersection(title_and_name):
                return False
            seen.update(title_and_name)
    return True

@defines_dtype_strategy
def array_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    min_size: int = 1,
    max_size: int = 5,
    allow_subarrays: bool = False,
) -> st.SearchStrategy[np.dtype]:
    order_check("size", 0, min_size, max_size)
    field_names = st.integers(0, 127).map("f{}".format) | st.text(min_size=1)
    name_titles = st.one_of(
        field_names,
        st.tuples(field_names, field_names).filter(lambda ns: ns[0] != ns[1]),
    )
    elements: st.SearchStrategy[tuple] = st.tuples(name_titles, subtype_strategy)
    if allow_subarrays:
        elements |= st.tuples(
            name_titles, subtype_strategy, array_shapes(max_dims=2, max_side=2)
        )
    return st.lists(
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=(
            lambda d: d[0] if isinstance(d[0], str) else d[0][0],
            lambda d: d[0] if isinstance(d[0], str) else d[0][1],
        ),
    ).filter(_no_title_is_name_of_a_titled_field)

@defines_strategy()
def nested_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    max_leaves: int = 10,
    max_itemsize: Optional[int] = None,
) -> st.SearchStrategy[np.dtype]:
    return st.recursive(
        subtype_strategy,
        lambda x: array_dtypes(x, allow_subarrays=True),
        max_leaves=max_leaves,
    ).filter(lambda d: max_itemsize is None or d.itemsize <= max_itemsize)

@proxies(_valid_tuple_axes)
def valid_tuple_axes(*args, **kwargs):
    return _valid_tuple_axes(*args, **kwargs)

valid_tuple_axes.__doc__ = f"""
    Return a strategy for generating permissible tuple-values for the
    ``axis`` argument for a numpy sequential function (e.g.
    :func:`numpy:numpy.sum`), given an array of the specified
    dimensionality.

    {_valid_tuple_axes.__doc__}
    """

@proxies(_mutually_broadcastable_shapes)
def mutually_broadcastable_shapes(*args, **kwargs):
    return _mutually_broadcastable_shapes(*args, **kwargs)

mutually_broadcastable_shapes.__doc__ = f"""
    {_mutually_broadcastable_shapes.__doc__}

    **Use with Generalised Universal Function signatures**

    A :doc:`universal function <numpy:reference/ufuncs>` (or ufunc for short) is a function
    that operates on ndarrays in an element-by-element fashion, supporting array
    broadcasting, type casting, and several other standard features.
    A :doc:`generalised ufunc <numpy:reference/c-api/generalized-ufuncs>` operates on
    sub-arrays rather than elements, based on the "signature" of the function.
    Compare e.g. :obj:`numpy.add() <numpy:numpy.add>` (ufunc) to
    :obj:`numpy.matmul() <numpy:numpy.matmul>` (gufunc).

    To generate shapes for a gufunc, you can pass the ``signature`` argument instead of
    ``num_shapes``.  This must be a gufunc signature string; which you can write by
    hand or access as e.g. ``np.matmul.signature`` on generalised ufuncs.

    In this case, the ``side`` arguments are applied to the 'core dimensions' as well,
    ignoring any frozen dimensions.  ``base_shape``  and the ``dims`` arguments are
    applied to the 'loop dimensions', and if necessary, the dimensionality of each
    shape is silently capped to respect the 32-dimension limit.

    The generated ``result_shape`` is the real result shape of applying the gufunc
    to arrays of the generated ``input_shapes``, even where this is different to
    broadcasting the loop dimensions.

    gufunc-compatible shapes shrink their loop dimensions as above, towards omitting
    optional core dimensions, and smaller-size core dimensions.

    .. code-block:: pycon

        >>> # np.matmul.signature == "(m?,n),(n,p?)->(m?,p?)"
        >>> for _ in range(3):
        ...     mutually_broadcastable_shapes(signature=np.matmul.signature).example()
        BroadcastableShapes(input_shapes=((2,), (2,)), result_shape=())
        BroadcastableShapes(input_shapes=((3, 4, 2), (1, 2)), result_shape=(3, 4))
        BroadcastableShapes(input_shapes=((4, 2), (1, 2, 3)), result_shape=(4, 3))

    """

@defines_strategy()
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    allow_newaxis: bool = False,
    allow_ellipsis: bool = True,
) -> st.SearchStrategy[BasicIndex]:
    check_type(tuple, shape, "shape")
    check_argument(
        all(isinstance(x, int) and x >= 0 for x in shape),
        f"{shape=}, but all dimensions must be non-negative integers.",
    )
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(bool, allow_newaxis, "allow_newaxis")
    check_type(int, min_dims, "min_dims")
    if min_dims > len(shape) and not allow_newaxis:
        note_deprecation(
            f"min_dims={min_dims} is larger than len