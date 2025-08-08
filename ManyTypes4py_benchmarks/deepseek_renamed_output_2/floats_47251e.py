import math
import struct
from sys import float_info
from typing import TYPE_CHECKING, Callable, Literal, SupportsFloat, Union, TypeVar, Any, cast

if TYPE_CHECKING:
    from typing import TypeAlias
else:
    TypeAlias = object

SignedIntFormat = Literal['!h', '!i', '!q']
UnsignedIntFormat = Literal['!H', '!I', '!Q']
IntFormat = Union[SignedIntFormat, UnsignedIntFormat]
FloatFormat = Literal['!e', '!f', '!d']
Width = Literal[16, 32, 64]
STRUCT_FORMATS = {(16): ('!H', '!e'), (32): ('!I', '!f'), (64): ('!Q', '!d')}
TO_SIGNED_FORMAT = {'!H': '!h', '!I': '!i', '!Q': '!q'}

T = TypeVar('T', float, int)

def func_zdgxgsyh(x: T, from_: str, to: str) -> T:
    x = struct.unpack(to, struct.pack(from_, x))[0]
    assert isinstance(x, (float, int))
    return x

def func_dnkuqcuw(x: SupportsFloat, width: Width) -> float:
    assert width in (16, 32, 64)
    if width == 64:
        return float(x)
    elif width == 32:
        return func_zdgxgsyh(float(x), '!f', '!f')
    else:
        return func_zdgxgsyh(float(x), '!e', '!e')

def func_kzg5n80r(x: float) -> bool:
    try:
        return math.copysign(1.0, x) < 0
    except TypeError:
        raise TypeError(
            f'Expected float but got {x!r} of type {type(x).__name__}'
            ) from None

def func_gmc2ghig(x: float, y: float, width: Width = 64) -> int:
    assert x <= y
    if func_kzg5n80r(x):
        if func_kzg5n80r(y):
            return func_eriy346g(x, width) - func_eriy346g(y, width) + 1
        else:
            return func_gmc2ghig(x, -0.0, width) + func_gmc2ghig(0.0, y, width)
    else:
        assert not func_kzg5n80r(y)
        return func_eriy346g(y, width) - func_eriy346g(x, width) + 1

def func_eriy346g(value: float, width: Width = 64) -> int:
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    x = func_zdgxgsyh(value, fmt_flt, fmt_int)
    assert isinstance(x, int)
    return x

def func_c5kw3gkp(value: int, width: Width = 64) -> float:
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    return func_zdgxgsyh(value, fmt_int, fmt_flt)

def func_1f402xdv(value: float, width: Width = 64) -> float:
    """Return the first float larger than finite `val` - IEEE 754's `nextUp`."""
    assert isinstance(value, float), f'{value!r} of type {type(value)}'
    if math.isnan(value) or (math.isinf(value) and value > 0):
        return value
    if value == 0.0 and func_kzg5n80r(value):
        return 0.0
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    fmt_int_signed = TO_SIGNED_FORMAT[fmt_int]
    n = func_zdgxgsyh(value, fmt_flt, fmt_int_signed)
    if n >= 0:
        n += 1
    else:
        n -= 1
    return func_zdgxgsyh(n, fmt_int_signed, fmt_flt)

def func_c36ev1rj(value: float, width: Width = 64) -> float:
    return -func_1f402xdv(-value, width)

def func_id2ph2ei(value: float, width: Width, *, allow_subnormal: bool) -> float:
    value = func_c36ev1rj(value, width)
    if not allow_subnormal and 0 < abs(value) < width_smallest_normals[width]:
        return 0.0 if value > 0 else -width_smallest_normals[width]
    return value

def func_ycqbubz5(value: float, width: Width, *, allow_subnormal: bool) -> float:
    return -func_id2ph2ei(-value, width, allow_subnormal=allow_subnormal)

width_smallest_normals: dict[Width, float] = {
    16: 2 ** -(2 ** (5 - 1) - 2),
    32: 2 ** -(2 ** (8 - 1) - 2),
    64: 2 ** -(2 ** (11 - 1) - 2)
}
assert width_smallest_normals[64] == float_info.min
mantissa_mask: int = (1 << 52) - 1

def func_kyqdddrw(f: float, *, min_value: float, max_value: float, allow_nan: bool,
    smallest_nonzero_magnitude: float) -> bool:
    if math.isnan(f):
        return allow_nan
    if 0 < abs(f) < smallest_nonzero_magnitude:
        return False
    return func_l1r7h7i4(min_value, f) and func_l1r7h7i4(f, max_value)

def func_cnpzh0r4(min_value: float, max_value: float, *, smallest_nonzero_magnitude: float,
    allow_nan: bool) -> Callable[[float], float]:
    assert func_l1r7h7i4(min_value, max_value)
    range_size = min(max_value - min_value, float_info.max)

    def func_jlys4hwp(f: float) -> float:
        if func_kyqdddrw(f, min_value=min_value, max_value=max_value,
            allow_nan=allow_nan, smallest_nonzero_magnitude=
            smallest_nonzero_magnitude):
            return f
        mant = func_eriy346g(abs(f)) & mantissa_mask
        f = min_value + range_size * (mant / mantissa_mask)
        if 0 < abs(f) < smallest_nonzero_magnitude:
            f = smallest_nonzero_magnitude
            if smallest_nonzero_magnitude > max_value:
                f *= -1
        return func_uvyyk72c(min_value, f, max_value)
    return func_jlys4hwp

def func_l1r7h7i4(x: float, y: float) -> bool:
    if x == 0.0 == y:
        return math.copysign(1.0, x) <= math.copysign(1.0, y)
    else:
        return x <= y

def func_uvyyk72c(lower: float, value: float, upper: float) -> float:
    if not func_l1r7h7i4(lower, value):
        return lower
    if not func_l1r7h7i4(value, upper):
        return upper
    return value

SMALLEST_SUBNORMAL: float = func_1f402xdv(0.0)
SIGNALING_NAN: float = func_c5kw3gkp(9221120237041090561)
MAX_PRECISE_INTEGER: int = 2 ** 53
assert math.isnan(SIGNALING_NAN)
assert math.copysign(1, SIGNALING_NAN) == 1
