import enum
import hashlib
import heapq
import math
import sys
from collections import OrderedDict, abc
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, TypeVar, Union, Any, Tuple, List

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import int_from_bytes
from hypothesis.internal.floats import next_up

if TYPE_CHECKING:
    from hypothesis.internal.conjecture.data import ConjectureData

LABEL_MASK: int = 2 ** 64 - 1

def calc_label_from_name(name: str) -> int:
    hashed: bytes = hashlib.sha384(name.encode()).digest()
    return int_from_bytes(hashed[:8])

def calc_label_from_cls(cls: type) -> int:
    return calc_label_from_name(cls.__qualname__)

def combine_labels(*labels: int) -> int:
    label: int = 0
    for l in labels:
        label = (label << 1) & LABEL_MASK
        label ^= l
    return label

SAMPLE_IN_SAMPLER_LABEL: int = calc_label_from_name('a sample() in Sampler')
ONE_FROM_MANY_LABEL: int = calc_label_from_name('one more from many()')
T = TypeVar('T')

def identity(v: T) -> T:
    return v

def check_sample(values: Any, strategy_name: str) -> Union[range, Tuple[Any, ...]]:
    if 'numpy' in sys.modules and isinstance(values, sys.modules['numpy'].ndarray):
        if values.ndim != 1:
            raise InvalidArgument(
                f'Only one-dimensional arrays are supported for sampling, and the given value has {values.ndim} dimensions (shape {values.shape}).  This array would give samples of array slices instead of elements!  Use np.ravel(values) to convert to a one-dimensional array, or tuple(values) if you want to sample slices.'
            )
    elif not isinstance(values, (OrderedDict, abc.Sequence, enum.EnumMeta)):
        raise InvalidArgument(
            f'Cannot sample from {values!r} because it is not an ordered collection. Hypothesis goes to some length to ensure that the {strategy_name} strategy has stable results between runs. To replay a saved example, the sampled values must have the same iteration order on every run - ruling out sets, dicts, etc due to hash randomization. Most cases can simply use `sorted(values)`, but mixed types or special values such as math.nan require careful handling - and note that when simplifying an example, Hypothesis treats earlier values as simpler.'
        )
    if isinstance(values, range):
        return values
    return tuple(values)

@lru_cache(64)
def compute_sampler_table(weights: Tuple[float, ...]) -> List[Tuple[int, int, float]]:
    n: int = len(weights)
    table: List[List[Union[int, None, float]]] = [[i, None, None] for i in range(n)]
    total = sum(weights)
    num_type = type(total)
    zero = num_type(0)
    one = num_type(1)
    small: List[int] = []
    large: List[int] = []
    probabilities: List[float] = [w / total for w in weights]
    scaled_probabilities: List[float] = []
    for i, alternate_chance in enumerate(probabilities):
        scaled = alternate_chance * n
        scaled_probabilities.append(scaled)
        if scaled == 1:
            table[i][2] = zero  # type: ignore
        elif scaled < 1:
            small.append(i)
        else:
            large.append(i)
    heapq.heapify(small)
    heapq.heapify(large)
    while small and large:
        lo = heapq.heappop(small)
        hi = heapq.heappop(large)
        assert lo != hi
        assert scaled_probabilities[hi] > one
        assert table[lo][1] is None
        table[lo][1] = hi
        table[lo][2] = one - scaled_probabilities[lo]  # type: ignore
        scaled_probabilities[hi] = scaled_probabilities[hi] + scaled_probabilities[lo] - one
        if scaled_probabilities[hi] < 1:
            heapq.heappush(small, hi)
        elif scaled_probabilities[hi] == 1:
            table[hi][2] = zero
        else:
            heapq.heappush(large, hi)
    while large:
        idx = large.pop()
        table[idx][2] = zero
    while small:
        idx = small.pop()
        table[idx][2] = zero
    new_table: List[Tuple[int, int, float]] = []
    for base, alternate, alternate_chance in table:
        assert isinstance(base, int)
        assert isinstance(alternate, int) or alternate is None
        assert alternate_chance is not None
        if alternate is None:
            new_table.append((base, base, alternate_chance))
        elif alternate < base:
            new_table.append((alternate, base, one - alternate_chance))
        else:
            new_table.append((base, alternate, alternate_chance))
    new_table.sort()
    return new_table

class Sampler:
    """
    Sampler based on Vose's algorithm for the alias method.
    """
    def __init__(self, weights: Sequence[float], *, observe: bool = True) -> None:
        self.observe: bool = observe
        self.table: List[Tuple[int, int, float]] = compute_sampler_table(tuple(weights))

    def sample(self, data: "ConjectureData", *, forced: Optional[int] = None) -> int:
        if self.observe:
            data.start_example(SAMPLE_IN_SAMPLER_LABEL)
        forced_choice: Optional[Tuple[int, int, float]] = None
        if forced is not None:
            for triple in self.table:
                base, alternate, alternate_chance = triple
                if forced == base or (forced == alternate and alternate_chance > 0):
                    forced_choice = triple
                    break
        base, alternate, alternate_chance = data.choice(self.table, forced=forced_choice, observe=self.observe)  # type: ignore
        forced_use_alternate: Optional[bool] = None
        if forced is not None:
            forced_use_alternate = (forced == alternate and alternate_chance > 0)
            assert forced == base or forced_use_alternate
        use_alternate: bool = data.draw_boolean(alternate_chance, forced=forced_use_alternate, observe=self.observe)  # type: ignore
        if self.observe:
            data.stop_example()
        if use_alternate:
            assert forced is None or alternate == forced, (forced, alternate)
            return alternate
        else:
            assert forced is None or base == forced, (forced, base)
            return base

INT_SIZES: Tuple[int, ...] = (8, 16, 32, 64, 128)
INT_SIZES_SAMPLER: Sampler = Sampler((4.0, 8.0, 1.0, 1.0, 0.5), observe=False)

class many:
    """
    Utility class for collections. Bundles up the logic we use for "should I
    keep drawing more values?" and handles starting and stopping examples in
    the right place.
    """
    def __init__(
        self,
        data: "ConjectureData",
        min_size: int,
        max_size: int,
        average_size: float,
        *,
        forced: Optional[int] = None,
        observe: bool = True
    ) -> None:
        assert 0 <= min_size <= average_size <= max_size
        assert forced is None or min_size <= forced <= max_size
        self.min_size: int = min_size
        self.max_size: int = max_size
        self.data: "ConjectureData" = data
        self.forced_size: Optional[int] = forced
        self.p_continue: float = _calc_p_continue(average_size - min_size, max_size - min_size)
        self.count: int = 0
        self.rejections: int = 0
        self.drawn: bool = False
        self.force_stop: bool = False
        self.rejected: bool = False
        self.observe: bool = observe

    def stop_example(self) -> None:
        if self.observe:
            self.data.stop_example()

    def start_example(self, label: int) -> None:
        if self.observe:
            self.data.start_example(label)

    def more(self) -> bool:
        """Should I draw another element to add to the collection?"""
        if self.drawn:
            self.stop_example()
        self.drawn = True
        self.rejected = False
        self.start_example(ONE_FROM_MANY_LABEL)
        if self.min_size == self.max_size:
            should_continue: bool = self.count < self.min_size
        else:
            forced_result: Optional[bool] = None
            if self.force_stop:
                assert self.forced_size is None or self.count == self.forced_size
                forced_result = False
            elif self.count < self.min_size:
                forced_result = True
            elif self.count >= self.max_size:
                forced_result = False
            elif self.forced_size is not None:
                forced_result = self.count < self.forced_size
            should_continue = self.data.draw_boolean(self.p_continue, forced=forced_result, observe=self.observe)  # type: ignore
        if should_continue:
            self.count += 1
            return True
        else:
            self.stop_example()
            return False

    def reject(self, why: Optional[Any] = None) -> None:
        """Reject the last example."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        self.rejected = True
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.data.mark_invalid(why)  # type: ignore
            else:
                self.force_stop = True

SMALLEST_POSITIVE_FLOAT: float = next_up(0.0) or sys.float_info.min

@lru_cache(maxsize=None)
def _calc_p_continue(desired_avg: float, max_size: float) -> float:
    """Return the p_continue which will generate the desired average size."""
    assert desired_avg <= max_size, (desired_avg, max_size)
    if desired_avg == max_size:
        return 1.0
    p_continue: float = 1 - 1.0 / (1 + desired_avg)
    if p_continue == 0 or max_size == math.inf:
        assert 0 <= p_continue < 1, p_continue
        return p_continue
    assert 0 < p_continue < 1, p_continue
    while _p_continue_to_avg(p_continue, max_size) > desired_avg:
        p_continue -= 0.0001
        if p_continue < SMALLEST_POSITIVE_FLOAT:
            p_continue = SMALLEST_POSITIVE_FLOAT
            break
    hi: float = 1.0
    while desired_avg - _p_continue_to_avg(p_continue, max_size) > 0.01:
        assert 0 < p_continue < hi, (p_continue, hi)
        mid: float = (p_continue + hi) / 2
        if _p_continue_to_avg(mid, max_size) <= desired_avg:
            p_continue = mid
        else:
            hi = mid
    assert 0 < p_continue < 1, p_continue
    assert _p_continue_to_avg(p_continue, max_size) <= desired_avg
    return p_continue

def _p_continue_to_avg(p_continue: float, max_size: float) -> float:
    """Return the average size generated by this p_continue and max_size."""
    if p_continue >= 1:
        return max_size
    return (1.0 / (1 - p_continue) - 1) * (1 - p_continue ** max_size)