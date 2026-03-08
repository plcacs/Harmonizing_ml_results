import math
from decimal import Decimal
from fractions import Fraction
from typing import Literal, Optional, Union, Dict, Any, Callable, Tuple, TypeVar, cast
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.filtering import get_float_predicate_bounds, get_integer_predicate_bounds
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, float_of, float_to_int, int_to_float, is_negative, next_down, next_down_normal, next_up, next_up_normal, width_smallest_normals
from hypothesis.internal.validation import check_type, check_valid_bound, check_valid_interval
from hypothesis.strategies._internal.misc import nothing
from hypothesis.strategies._internal.strategies import SampledFromStrategy, SearchStrategy
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

T = TypeVar('T')
Real = Union[int, float, Fraction, Decimal]
Predicate = Callable[[T], bool]
FloatFilter = Callable[[float], bool]
IntegerFilter = Callable[[int], bool]

class IntegersStrategy(SearchStrategy[int]):
    def __init__(self, start: Optional[int], end: Optional[int]) -> None:
        assert isinstance(start, int) or start is None
        assert isinstance(end, int) or end is None
        assert start is None or end is None or start <= end
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        if self.start is None and self.end is None:
            return 'integers()'
        if self.end is None:
            return f'integers(min_value={self.start})'
        if self.start is None:
            return f'integers(max_value={self.end})'
        return f'integers({self.start}, {self.end})'

    def do_draw(self, data: Any) -> int:
        weights: Optional[Dict[int, float]] = None
        if self.end is not None and self.start is not None and (self.end - self.start > 127):
            weights = {self.start: 2 / 128, self.start + 1: 1 / 128, self.end - 1: 1 / 128, self.end: 2 / 128}
        return data.draw_integer(min_value=self.start, max_value=self.end, weights=weights)

    def filter(self, condition: IntegerFilter) -> SearchStrategy[int]:
        if condition is math.isfinite:
            return self
        if condition in [math.isinf, math.isnan]:
            return nothing()
        kwargs, pred = get_integer_predicate_bounds(condition)
        start, end = (self.start, self.end)
        if 'min_value' in kwargs:
            start = max(kwargs['min_value'], -math.inf if start is None else start)
        if 'max_value' in kwargs:
            end = min(kwargs['max_value'], math.inf if end is None else end)
        if start != self.start or end != self.end:
            if start is not None and end is not None and (start > end):
                return nothing()
            self = type(self)(start, end)
        if pred is None:
            return self
        return super().filter(pred)

@cacheable
@defines_strategy(force_reusable_values=True)
def integers(min_value: Optional[Real] = None, max_value: Optional[Real] = None) -> SearchStrategy[int]:
    check_valid_bound(min_value, 'min_value')
    check_valid_bound(max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    min_int: Optional[int] = None
    max_int: Optional[int] = None
    if min_value is not None:
        if min_value != int(min_value):
            raise InvalidArgument('min_value=%r of type %r cannot be exactly represented as an integer.' % (min_value, type(min_value)))
        min_int = int(min_value)
    if max_value is not None:
        if max_value != int(max_value):
            raise InvalidArgument('max_value=%r of type %r cannot be exactly represented as an integer.' % (max_value, type(max_value)))
        max_int = int(max_value)
    return IntegersStrategy(min_int, max_int)

class FloatStrategy(SearchStrategy[float]):
    def __init__(
        self,
        *,
        min_value: float,
        max_value: float,
        allow_nan: bool,
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL
    ) -> None:
        super().__init__()
        assert isinstance(allow_nan, bool)
        assert smallest_nonzero_magnitude >= 0.0, 'programmer error if this is negative'
        if smallest_nonzero_magnitude == 0.0:
            raise FloatingPointError("Got allow_subnormal=True, but we can't represent subnormal floats right now, in violation of the IEEE-754 floating-point specification.  This is usually because something was compiled with -ffast-math or a similar option, which sets global processor state.  See https://simonbyrne.github.io/notes/fastmath/ for a more detailed writeup - and good luck!")
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.smallest_nonzero_magnitude = smallest_nonzero_magnitude

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(self.min_value={self.min_value!r}, self.max_value={self.max_value!r}, self.allow_nan={self.allow_nan!r}, self.smallest_nonzero_magnitude={self.smallest_nonzero_magnitude!r})'.replace('self.', '')

    def do_draw(self, data: Any) -> float:
        return data.draw_float(
            min_value=self.min_value,
            max_value=self.max_value,
            allow_nan=self.allow_nan,
            smallest_nonzero_magnitude=self.smallest_nonzero_magnitude
        )

    def filter(self, condition: FloatFilter) -> SearchStrategy[float]:
        if condition is math.isfinite:
            return FloatStrategy(
                min_value=max(self.min_value, next_up(float('-inf'))),
                max_value=min(self.max_value, next_down(float('inf'))),
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude
            )
        if condition is math.isinf:
            if (permitted_infs := [x for x in (-math.inf, math.inf) if self.min_value <= x <= self.max_value]):
                return SampledFromStrategy(permitted_infs)
            return nothing()
        if condition is math.isnan:
            if not self.allow_nan:
                return nothing()
            return NanStrategy()
        kwargs, pred = get_float_predicate_bounds(condition)
        if not kwargs:
            return super().filter(pred)
        min_bound = max(kwargs.get('min_value', -math.inf), self.min_value)
        max_bound = min(kwargs.get('max_value', math.inf), self.max_value)
        if -self.smallest_nonzero_magnitude < min_bound < 0:
            min_bound = -0.0
        elif 0 < min_bound < self.smallest_nonzero_magnitude:
            min_bound = self.smallest_nonzero_magnitude
        if -self.smallest_nonzero_magnitude < max_bound < 0:
            max_bound = -self.smallest_nonzero_magnitude
        elif 0 < max_bound < self.smallest_nonzero_magnitude:
            max_bound = 0.0
        if min_bound > max_bound:
            return nothing()
        if min_bound > self.min_value or self.max_value > max_bound or (self.allow_nan and (-math.inf < min_bound or max_bound < math.inf)):
            self = type(self)(
                min_value=min_bound,
                max_value=max_bound,
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude
            )
        if pred is None:
            return self
        return super().filter(pred)

@cacheable
@defines_strategy(force_reusable_values=True)
def floats(
    min_value: Optional[Real] = None,
    max_value: Optional[Real] = None,
    *,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    width: Literal[16, 32, 64] = 64,
    exclude_min: bool = False,
    exclude_max: bool = False
) -> SearchStrategy[float]:
    check_type(bool, exclude_min, 'exclude_min')
    check_type(bool, exclude_max, 'exclude_max')
    if allow_nan is None:
        allow_nan = bool(min_value is None and max_value is None)
    elif allow_nan and (min_value is not None or max_value is not None):
        raise InvalidArgument(f'Cannot have allow_nan={allow_nan!r}, with min_value or max_value')
    if width not in (16, 32, 64):
        raise InvalidArgument(f'Got width={width!r}, but the only valid values are the integers 16, 32, and 64.')
    check_valid_bound(min_value, 'min_value')
    check_valid_bound(max_value, 'max_value')
    if math.copysign(1.0, -0.0) == 1.0:
        raise FloatingPointError("Your Python install can't represent -0.0, which is required by the IEEE-754 floating-point specification.  This is probably because it was compiled with an unsafe option like -ffast-math; for a more detailed explanation see https://simonbyrne.github.io/notes/fastmath/")
    if allow_subnormal and next_up(0.0, width=width) == 0:
        from _hypothesis_ftz_detector import identify_ftz_culprits
        try:
            ftz_pkg = identify_ftz_culprits()
        except Exception:
            ftz_pkg = None
        if ftz_pkg:
            ftz_msg = f"This seems to be because the `{ftz_pkg}` package was compiled with -ffast-math or a similar option, which sets global processor state - see https://simonbyrne.github.io/notes/fastmath/ for details.  If you don't know why {ftz_pkg} is installed, `pipdeptree -rp {ftz_pkg}` will show which packages depend on it."
        else:
            ftz_msg = 'This is usually because something was compiled with -ffast-math or a similar option, which sets global processor state.  See https://simonbyrne.github.io/notes/fastmath/ for a more detailed writeup - and good luck!'
        raise FloatingPointError(f"Got allow_subnormal={allow_subnormal!r}, but we can't represent subnormal floats right now, in violation of the IEEE-754 floating-point specification.  {ftz_msg}")
    min_arg, max_arg = (min_value, max_value)
    min_float: Optional[float] = None
    max_float: Optional[float] = None
    if min_value is not None:
        min_value_float = float_of(min_value, width)
        assert isinstance(min_value_float, float)
        min_float = min_value_float
    if max_value is not None:
        max_value_float = float_of(max_value, width)
        assert isinstance(max_value_float, float)
        max_float = max_value_float
    if min_float is not None and min_arg is not None and min_float != min_arg:
        raise InvalidArgument(f'min_value={min_arg!r} cannot be exactly represented as a float of width {width} - use min_value={min_float!r} instead.')
    if max_float is not None and max_arg is not None and max_float != max_arg:
        raise InvalidArgument(f'max_value={max_arg!r} cannot be exactly represented as a float of width {width} - use max_value={max_float!r} instead.')
    if exclude_min and (min_float is None or min_float == math.inf):
        raise InvalidArgument(f'Cannot exclude min_value={min_float!r}')
    if exclude_max and (max_float is None or max_float == -math.inf):
        raise InvalidArgument(f'Cannot exclude max_value={max_float!r}')
    assumed_allow_subnormal = allow_subnormal is None or allow_subnormal
    if min_float is not None and (exclude_min or (min_arg is not None and min_float < min_arg)):
        min_float = next_up_normal(min_float, width, allow_subnormal=assumed_allow_subnormal)
        if min_float == min_arg:
            assert min_float == min_arg == 0
            assert is_negative(min_arg)
            assert not is_negative(min_float)
            min_float = next_up_normal(min_float, width, allow_subnormal=assumed_allow_subnormal)
        assert min_float > min_arg
    if max_float is not None and (exclude_max or (max_arg is not None and max_float > max_arg)):
        max_float = next_down_normal(max_float, width, allow_subnormal=assumed_allow_subnormal)
        if max_float == max_arg:
            assert max_float == max_arg == 0
            assert is_negative(max_float)
            assert not is_negative(max_arg)
            max_float = next_down_normal(max_float, width, allow_subnormal=assumed_allow_subnormal)
        assert max_float < max_arg
    if min_float == -math.inf:
        min_float = None
    if max_float == math.inf:
        max_float = None
    bad_zero_bounds = min_float == max_float == 0 and is_negative(max_float) and (not is_negative(min_float))
    if min_float is not None and max_float is not None and (min_float > max_float or bad_zero_bounds):
        msg = 'There are no %s-bit floating-point values between min_value=%r and max_value=%r' % (width, min_arg, max_arg)
        if exclude_min or exclude_max:
            msg += f', exclude_min={exclude_min!r} and exclude_max={exclude_max!r}'
        raise InvalidArgument(msg)
    if allow_infinity is None:
        allow_infinity = bool(min_float is None or max_float is None)
    elif allow_infinity:
        if min_float is not None and max_float is not None:
            raise InvalidArgument(f'Cannot have allow_infinity={allow_infinity!r}, with both min_value and max_value')
    elif min_float == math.inf:
        if min_arg == math.inf:
            raise InvalidArgument('allow_infinity=False excludes min_value=inf')
        raise InvalidArgument(f'exclude_min=True turns min_value={min_arg!r} into inf, but allow_infinity=False')
    elif max_float == -math.inf:
        if max_arg == -math.inf:
            raise InvalidArgument('allow_infinity=False excludes max_value=-inf')
        raise InvalidArgument(f'exclude_max=True turns max_value={max_arg!r} into -inf, but allow_infinity=False')
    smallest_normal = width_smallest_normals[width]
    if allow_subnormal is None:
        if min_float is not None and max_float is not None:
            if min_float == max_float:
                allow_subnormal = -smallest_normal < min_float < smallest_normal
            else:
                allow_subnormal = min_float < smallest_normal and max_float > -smallest_normal
        elif min_float is not None:
            allow_subnormal = min_float < smallest_normal
        elif max_float is not None:
            allow_subnormal = max_float > -smallest_normal
        else:
            allow_subnormal = True
    if allow_subnormal:
        if min_float is not None and min_float >= smallest_normal:
            raise InvalidArgument(f"allow_subnormal=True, but minimum value {min_float} excludes values below float{width}'s smallest positive normal {smallest_normal}")
        if max_float is not None and max_float <= -smallest_normal:
            raise InvalidArgument(f"allow_subnormal=True, but maximum value {max_float} excludes values above float{width}'s smallest negative normal {-smallest_normal}")
    final_min: float = float('-inf')
    final_max: float = float('inf')
    if min_float is not None:
        final_min = min_float
    if max_float is not None:
        final_max = max_float
    if not allow_infinity:
        final_min = max(final_min, next_up(float('-inf')))
        final_max = min(final_max, next_down(float('inf')))
    assert isinstance(final_min, float)
    assert isinstance(final_max, float)
    smallest_nonzero_magnitude = SMALLEST_SUBNORMAL if allow_subnormal else smallest_normal
    result = FloatStrategy(
        min_value=final_min,
        max_value=final_max,
        allow_nan=allow_nan,
        smallest_nonzero_magnitude=smallest_nonzero_magnitude
    )
    if width < 64:
        def downcast(x: float) -> float:
            try:
                return float_of(x, width)
            except OverflowError:
                reject()
        result = result.map(downcast)
    return result

class NanStrategy(SearchStrategy[float]):
    def do_draw(self, data: Any) -> float:
        sign_bit = int(data.draw_boolean()) << 63
        nan_bits = float_to_int(math.nan)
        mantissa_bits = data.draw_integer(0, 2 ** 52 - 1)
        return int_to_float(sign_bit | nan_bits | mantissa_bits)
    
    def filter(self, condition: FloatFilter) -> SearchStrategy[float]:
        return super().filter(condition)