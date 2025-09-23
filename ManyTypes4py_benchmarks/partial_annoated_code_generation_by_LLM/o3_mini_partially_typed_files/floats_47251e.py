#!/usr/bin/env python3
import math
import struct
from sys import float_info
from typing import TYPE_CHECKING, Callable, Literal, SupportsFloat, Union

if TYPE_CHECKING:
    from typing import TypeAlias
else:
    TypeAlias = object

SignedIntFormat: TypeAlias = Literal['!h', '!i', '!q']
UnsignedIntFormat: TypeAlias = Literal['!H', '!I', '!Q']
IntFormat: TypeAlias = Union[SignedIntFormat, UnsignedIntFormat]
FloatFormat: TypeAlias = Literal['!e', '!f', '!d']
Width: TypeAlias = Literal[16, 32, 64]

STRUCT_FORMATS: dict[Width, tuple[IntFormat, FloatFormat]] = {
    16: ('!H', '!e'),
    32: ('!I', '!f'),
    64: ('!Q', '!d')
}
TO_SIGNED_FORMAT: dict[UnsignedIntFormat, SignedIntFormat] = {'!H': '!h', '!I': '!i', '!Q': '!q'}

def reinterpret_bits(x: Union[int, float], from_: str, to: str) -> Union[int, float]:
    x = struct.unpack(to, struct.pack(from_, x))[0]
    assert isinstance(x, (float, int))
    return x

def float_of(x: float, width: Width) -> float:
    assert width in (16, 32, 64)
    if width == 64:
        return float(x)
    elif width == 32:
        return reinterpret_bits(float(x), '!f', '!f')
    else:
        return reinterpret_bits(float(x), '!e', '!e')

def is_negative(x: SupportsFloat) -> bool:
    try:
        return math.copysign(1.0, x) < 0
    except TypeError:
        raise TypeError(f'Expected float but got {x!r} of type {type(x).__name__}') from None

def float_to_int(value: float, width: Width = 64) -> int:
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    x = reinterpret_bits(value, fmt_flt, fmt_int)
    assert isinstance(x, int)
    return x

def int_to_float(value: int, width: Width = 64) -> float:
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    return reinterpret_bits(value, fmt_int, fmt_flt)

def next_up(value: float, width: Width = 64) -> float:
    """Return the first float larger than finite `value` - IEEE 754's `nextUp`.

    From https://stackoverflow.com/a/10426033, with thanks to Mark Dickinson.
    """
    assert isinstance(value, float), f'{value!r} of type {type(value)}'
    if math.isnan(value) or (math.isinf(value) and value > 0):
        return value
    if value == 0.0 and is_negative(value):
        return 0.0
    fmt_int, fmt_flt = STRUCT_FORMATS[width]
    fmt_int_signed = TO_SIGNED_FORMAT[fmt_int]  # type: ignore
    n = reinterpret_bits(value, fmt_flt, fmt_int_signed)
    if n >= 0:
        n += 1
    else:
        n -= 1
    return reinterpret_bits(n, fmt_int_signed, fmt_flt)

def next_down(value: float, width: Width = 64) -> float:
    return -next_up(-value, width)

width_smallest_normals: dict[Width, float] = {
    16: 2 ** (-(2 ** (5 - 1) - 2)),
    32: 2 ** (-(2 ** (8 - 1) - 2)),
    64: 2 ** (-(2 ** (11 - 1) - 2))
}
assert width_smallest_normals[64] == float_info.min

mantissa_mask: int = (1 << 52) - 1

def next_down_normal(value: float, width: Width, *, allow_subnormal: bool) -> float:
    value = next_down(value, width)
    if not allow_subnormal and 0 < abs(value) < width_smallest_normals[width]:
        return 0.0 if value > 0 else -width_smallest_normals[width]
    return value

def next_up_normal(value: float, width: Width, *, allow_subnormal: bool) -> float:
    return -next_down_normal(-value, width, allow_subnormal=allow_subnormal)

def float_permitted(
    f: float, *,
    min_value: float,
    max_value: float,
    allow_nan: bool,
    smallest_nonzero_magnitude: float
) -> bool:
    if math.isnan(f):
        return allow_nan
    if 0 < abs(f) < smallest_nonzero_magnitude:
        return False
    return sign_aware_lte(min_value, f) and sign_aware_lte(f, max_value)

def make_float_clamper(
    min_value: float,
    max_value: float,
    *,
    smallest_nonzero_magnitude: float,
    allow_nan: bool
) -> Callable[[float], float]:
    """
    Return a function that clamps positive floats into the given bounds.
    """
    assert sign_aware_lte(min_value, max_value)
    range_size: float = min(max_value - min_value, float_info.max)

    def float_clamper(f: float) -> float:
        if float_permitted(
            f,
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            smallest_nonzero_magnitude=smallest_nonzero_magnitude
        ):
            return f
        mant: int = float_to_int(abs(f)) & mantissa_mask
        f = min_value + range_size * (mant / mantissa_mask)
        if 0 < abs(f) < smallest_nonzero_magnitude:
            f = smallest_nonzero_magnitude
            if smallest_nonzero_magnitude > max_value:
                f *= -1
        return clamp(min_value, f, max_value)
    return float_clamper

def sign_aware_lte(x: float, y: float) -> bool:
    """Less-than-or-equals, but strictly orders -0.0 and 0.0"""
    if x == 0.0 == y:
        return math.copysign(1.0, x) <= math.copysign(1.0, y)
    else:
        return x <= y

def clamp(lower: float, value: float, upper: float) -> float:
    """Given a value and lower/upper bounds, 'clamp' the value so that
    it satisfies lower <= value <= upper.  NaN is mapped to lower."""
    if not sign_aware_lte(lower, value):
        return lower
    if not sign_aware_lte(value, upper):
        return upper
    return value

SMALLEST_SUBNORMAL: float = next_up(0.0)
SIGNALING_NAN: float = int_to_float(9221120237041090561)
MAX_PRECISE_INTEGER: int = 2 ** 53

assert math.isnan(SIGNALING_NAN)
assert math.copysign(1, SIGNALING_NAN) == 1