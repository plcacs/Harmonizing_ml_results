import math
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Union
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.filtering import get_float_predicate_bounds, get_integer_predicate_bounds
from hypothesis.internal.floats import (
    SMALLEST_SUBNORMAL,
    float_of,
    float_to_int,
    int_to_float,
    is_negative,
    next_down,
    next_down_normal,
    next_up,
    next_up_normal,
    width_smallest_normals,
)
from hypothesis.internal.validation import check_type, check_valid_bound, check_valid_interval
from hypothesis.strategies._internal.misc import nothing
from hypothesis.strategies._internal.strategies import SampledFromStrategy, SearchStrategy
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

Real = Union[int, float, Fraction, Decimal]


class IntegersStrategy(SearchStrategy[int]):
    def __init__(self, start: Optional[int], end: Optional[int]) -> None:
        assert isinstance(start, int) or start is None
        assert isinstance(end, int) or end is None
        assert start is None or end is None or start <= end
        self.start: Optional[int] = start
        self.end: Optional[int] = end

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
        if (
            self.end is not None
            and self.start is not None
            and (self.end - self.start > 127)
        ):
            weights = {
                self.start: 2 / 128,
                self.start + 1: 1 / 128,
                self.end - 1: 1 / 128,
                self.end: 2 / 128,
            }
        return data.draw_integer(min_value=self.start, max_value=self.end, weights=weights)

    def filter(self, condition: Callable[[int], bool]) -> SearchStrategy[int]:
        if condition is math.isfinite:
            return self
        if condition in [math.isinf, math.isnan]:
            return nothing()
        kwargs, pred = get_integer_predicate_bounds(condition)
        start: Optional[int] = self.start
        end: Optional[int] = self.end
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
    """Returns a strategy which generates integers.

    If min_value is not None then all values will be >= min_value. If
    max_value is not None then all values will be <= max_value

    Examples from this strategy will shrink towards zero, and negative values
    will also shrink towards positive (i.e. -n may be replaced by +n).
    """
    check_valid_bound(min_value, 'min_value')
    check_valid_bound(max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if min_value is not None:
        if min_value != int(min_value):
            raise InvalidArgument(
                'min_value=%r of type %r cannot be exactly represented as an integer.'
                % (min_value, type(min_value))
            )
        min_value = int(min_value)
    if max_value is not None:
        if max_value != int(max_value):
            raise InvalidArgument(
                'max_value=%r of type %r cannot be exactly represented as an integer.'
                % (max_value, type(max_value))
            )
        max_value = int(max_value)
    return IntegersStrategy(min_value, max_value)


class FloatStrategy(SearchStrategy[float]):
    """A strategy for floating point numbers."""

    def __init__(
        self,
        *,
        min_value: float,
        max_value: float,
        allow_nan: bool,
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
    ) -> None:
        super().__init__()
        assert isinstance(allow_nan, bool)
        assert smallest_nonzero_magnitude >= 0.0, 'programmer error if this is negative'
        if smallest_nonzero_magnitude == 0.0:
            raise FloatingPointError(
                "Got allow_subnormal=True, but we can't represent subnormal floats right now, in violation of the IEEE-754 floating-point specification.  This is usually because something was compiled with -ffast-math or a similar option, which sets global processor state.  See https://simonbyrne.github.io/notes/fastmath/ for a more detailed writeup - and good luck!"
            )
        self.min_value: float = min_value
        self.max_value: float = max_value
        self.allow_nan: bool = allow_nan
        self.smallest_nonzero_magnitude: float = smallest_nonzero_magnitude

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(min_value={self.min_value!r}, max_value={self.max_value!r}, allow_nan={self.allow_nan!r}, smallest_nonzero_magnitude={self.smallest_nonzero_magnitude!r})'.replace('self.', '')

    def do_draw(self, data: Any) -> float:
        return data.draw_float(
            min_value=self.min_value,
            max_value=self.max_value,
            allow_nan=self.allow_nan,
            smallest_nonzero_magnitude=self.smallest_nonzero_magnitude,
        )

    def filter(self, condition: Callable[[float], bool]) -> SearchStrategy[float]:
        if condition is math.isfinite:
            return FloatStrategy(
                min_value=max(self.min_value, next_up(float('-inf'))),
                max_value=min(self.max_value, next_down(float('inf'))),
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude,
            )
        if condition is math.isinf:
            permitted_infs: List[float] = [x for x in (-math.inf, math.inf) if self.min_value <= x <= self.max_value]
            if permitted_infs:
                return SampledFromStrategy(permitted_infs)
            return nothing()
        if condition is math.isnan:
            if not self.allow_nan:
                return nothing()
            return NanStrategy()
        kwargs, pred = get_float_predicate_bounds(condition)
        if not kwargs:
            return super().filter(pred)
        min_bound: float = max(kwargs.get('min_value', -math.inf), self.min_value)
        max_bound: float = min(kwargs.get('max_value', math.inf), self.max_value)
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
        if (
            min_bound > self.min_value
            or self.max_value > max_bound
            or (self.allow_nan and (-math.inf < min_bound or max_bound < math.inf))
        ):
            self = type(self)(
                min_value=min_bound,
                max_value=max_bound,
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude,
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
    width: int = 64,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> SearchStrategy[float]:
    """Returns a strategy which generates floats.

    - If min_value is not None, all values will be ``>= min_value``
      (or ``> min_value`` if ``exclude_min``).
    - If max_value is not None, all values will be ``<= max_value``
      (or ``< max_value`` if ``exclude_max``).
    - If min_value or max_value is not None, it is an error to enable
      allow_nan.
    - If both min_value and max_value are not None, it is an error to enable
      allow_infinity.
    - If inferred values range does not include subnormal values, it is an error
      to enable allow_subnormal.

    Where not explicitly ruled out by the bounds,
    :wikipedia:`subnormals <Subnormal_number>`, infinities, and NaNs are possible
    values generated by this strategy.

    The width argument specifies the maximum number of bits of precision
    required to represent the generated float. Valid values are 16, 32, or 64.
    Passing ``width=32`` will still use the builtin 64-bit :class:`~python:float` class,
    but always for values which can be exactly represented as a 32-bit float.

    The exclude_min and exclude_max argument can be used to generate numbers
    from open or half-open intervals, by excluding the respective endpoints.
    Excluding either signed zero will also exclude the other.
    Attempting to exclude an endpoint which is None will raise an error;
    use ``allow_infinity=False`` to generate finite floats.  You can however
    use e.g. ``min_value=-math.inf, exclude_min=True`` to exclude only
    one infinite endpoint.

    Examples from this strategy have a complicated and hard to explain
    shrinking behaviour, but it tries to improve "human readability". Finite
    numbers will be preferred to infinity and infinity will be preferred to
    NaN.
    """
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
        raise FloatingPointError(
            "Your Python install can't represent -0.0, which is required by the IEEE-754 floating-point specification.  This is probably because it was compiled with an unsafe option like -ffast-math; for a more detailed explanation see https://simonbyrne.github.io/notes/fastmath/"
        )
    if allow_subnormal and next_up(0.0, width=width) == 0:
        from _hypothesis_ftz_detector import identify_ftz_culprits
        try:
            ftz_pkg = identify_ftz_culprits()
        except Exception:
            ftz_pkg = None
        if ftz_pkg:
            ftz_msg = (
                f"This seems to be because the `{ftz_pkg}` package was compiled with -ffast-math or a similar option, "
                f"which sets global processor state - see https://simonbyrne.github.io/notes/fastmath/ for details.  "
                f"If you don't know why {ftz_pkg} is installed, `pipdeptree -rp {ftz_pkg}` will show which packages depend on it."
            )
        else:
            ftz_msg = (
                "This is usually because something was compiled with -ffast-math or a similar option, "
                "which sets global processor state.  See https://simonbyrne.github.io/notes/fastmath/ for a more detailed writeup - and good luck!"
            )
        raise FloatingPointError(
            f"Got allow_subnormal={allow_subnormal!r}, but we can't represent subnormal floats right now, "
            f"in violation of the IEEE-754 floating-point specification.  {ftz_msg}"
        )
    min_arg: Optional[Real] = min_value
    max_arg: Optional[Real] = max_value
    if min_value is not None:
        min_value = float_of(min_value, width)
        assert isinstance(min_value, float)
    if max_value is not None:
        max_value = float_of(max_value, width)
        assert isinstance(max_value, float)
    if min_value != min_arg:
        raise InvalidArgument(
            f'min_value={min_arg!r} cannot be exactly represented as a float of width {width} - use min_value={min_value!r} instead.'
        )
    if max_value != max_arg:
        raise InvalidArgument(
            f'max_value={max_arg!r} cannot be exactly represented as a float of width {width} - use max_value={max_value!r} instead.'
        )
    if exclude_min and (min_value is None or min_value == math.inf):
        raise InvalidArgument(f'Cannot exclude min_value={min_value!r}')
    if exclude_max and (max_value is None or max_value == -math.inf):
        raise InvalidArgument(f'Cannot exclude max_value={max_value!r}')
    assumed_allow_subnormal: bool = allow_subnormal is None or allow_subnormal
    if min_value is not None and (exclude_min or (min_arg is not None and min_value < float(min_arg))):
        min_value = next_up_normal(min_value, width, allow_subnormal=assumed_allow_subnormal)
        if min_value == float(min_arg):
            assert min_value == float(min_arg) == 0
            assert is_negative(float(min_arg))
            assert not is_negative(min_value)
            min_value = next_up_normal(min_value, width, allow_subnormal=assumed_allow_subnormal)
        assert min_value > float(min_arg)  # type: ignore
    if max_value is not None and (exclude_max or (max_arg is not None and max_value > float(max_arg))):
        max_value = next_down_normal(max_value, width, allow_subnormal=assumed_allow_subnormal)
        if max_value == float(max_arg):
            assert max_value == float(max_arg) == 0
            assert is_negative(max_value)
            assert not is_negative(float(max_arg))
            max_value = next_down_normal(max_value, width, allow_subnormal=assumed_allow_subnormal)
        assert max_value < float(max_arg)  # type: ignore
    if min_value == -math.inf:
        min_value = None  # type: ignore
    if max_value == math.inf:
        max_value = None  # type: ignore
    bad_zero_bounds: bool = (
        min_value == max_value == 0 and is_negative(max_value) and (not is_negative(min_value))
    )
    if min_value is not None and max_value is not None and (min_value > max_value or bad_zero_bounds):
        msg = 'There are no %s-bit floating-point values between min_value=%r and max_value=%r' % (width, min_arg, max_arg)
        if exclude_min or exclude_max:
            msg += f', exclude_min={exclude_min!r} and exclude_max={exclude_max!r}'
        raise InvalidArgument(msg)
    if allow_infinity is None:
        allow_infinity = bool(min_value is None or max_value is None)
    elif allow_infinity:
        if min_value is not None and max_value is not None:
            raise InvalidArgument(f'Cannot have allow_infinity={allow_infinity!r}, with both min_value and max_value')
    elif min_value == math.inf:
        if min_arg == math.inf:
            raise InvalidArgument('allow_infinity=False excludes min_value=inf')
        raise InvalidArgument(f'exclude_min=True turns min_value={min_arg!r} into inf, but allow_infinity=False')
    elif max_value == -math.inf:
        if max_arg == -math.inf:
            raise InvalidArgument('allow_infinity=False excludes max_value=-inf')
        raise InvalidArgument(f'exclude_max=True turns max_value={max_arg!r} into -inf, but allow_infinity=False')
    smallest_normal: float = width_smallest_normals[width]
    if allow_subnormal is None:
        if min_value is not None and max_value is not None:
            if min_value == max_value:
                allow_subnormal = -smallest_normal < min_value < smallest_normal
            else:
                allow_subnormal = min_value < smallest_normal and max_value > -smallest_normal
        elif min_value is not None:
            allow_subnormal = min_value < smallest_normal
        elif max_value is not None:
            allow_subnormal = max_value > -smallest_normal
        else:
            allow_subnormal = True
    if allow_subnormal:
        if min_value is not None and min_value >= smallest_normal:
            raise InvalidArgument(
                f"allow_subnormal=True, but minimum value {min_value} excludes values below float{width}'s smallest positive normal {smallest_normal}"
            )
        if max_value is not None and max_value <= -smallest_normal:
            raise InvalidArgument(
                f"allow_subnormal=True, but maximum value {max_value} excludes values above float{width}'s smallest negative normal {-smallest_normal}"
            )
    if min_value is None:
        min_value = float('-inf')
    if max_value is None:
        max_value = float('inf')
    if not allow_infinity:
        min_value = max(min_value, next_up(float('-inf')))
        max_value = min(max_value, next_down(float('inf')))
    assert isinstance(min_value, float)
    assert isinstance(max_value, float)
    smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL if allow_subnormal else smallest_normal
    result: SearchStrategy[float] = FloatStrategy(
        min_value=min_value,
        max_value=max_value,
        allow_nan=allow_nan,
        smallest_nonzero_magnitude=smallest_nonzero_magnitude,
    )
    if width < 64:
        def downcast(x: float) -> float:
            try:
                return float_of(x, width)
            except OverflowError:
                reject()
            # Unreachable; added to satisfy type checkers.
            return x
        result = result.map(downcast)
    return result


class NanStrategy(SearchStrategy[float]):
    """Strategy for sampling the space of nan float values."""

    def do_draw(self, data: Any) -> float:
        sign_bit: int = int(data.draw_boolean()) << 63
        nan_bits: int = float_to_int(math.nan)
        mantissa_bits: int = data.draw_integer(0, 2 ** 52 - 1)
        return int_to_float(sign_bit | nan_bits | mantissa_bits)