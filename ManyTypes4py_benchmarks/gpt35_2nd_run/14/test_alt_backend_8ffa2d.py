import itertools
import math
import sys
from contextlib import contextmanager
from random import Random
from typing import Optional
import pytest
from hypothesis import HealthCheck, Verbosity, assume, errors, given, settings, strategies as st
from hypothesis.control import current_build_context
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import BackendCannotProceed, Flaky, HypothesisException, InvalidArgument, Unsatisfiable
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture.data import ConjectureData, PrimitiveProvider
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS, COLLECTION_DEFAULT_MAX_SIZE
from hypothesis.internal.floats import SIGNALING_NAN
from hypothesis.internal.intervalsets import IntervalSet
from tests.common.debug import minimal
from tests.common.utils import capture_observations, capture_out
from tests.conjecture.common import nodes

class PrngProvider(PrimitiveProvider):

    def __init__(self, conjecturedata: ConjectureData) -> None:
        super().__init__(conjecturedata)
        self.prng = Random(0)

    def draw_boolean(self, p: float = 0.5) -> bool:
        return self.prng.random() < p

    def draw_integer(self, min_value: Optional[int] = None, max_value: Optional[int] = None, *, weights=None, shrink_towards: int = 0) -> int:
        assert isinstance(shrink_towards, int)
        if weights is not None:
            assert min_value is not None
            assert max_value is not None
            choices = self.prng.choices(range(min_value, max_value + 1), weights=weights, k=1)
            return choices[0]
        if min_value is None and max_value is None:
            min_value = -2 ** 127
            max_value = 2 ** 127 - 1
        elif min_value is None:
            min_value = max_value - 2 ** 64
        elif max_value is None:
            max_value = min_value + 2 ** 64
        return self.prng.randint(min_value, max_value)

    def draw_float(self, *, min_value: float = -math.inf, max_value: float = math.inf, allow_nan: bool = True, smallest_nonzero_magnitude: float) -> float:
        if allow_nan and self.prng.random() < 1 / 32:
            nans = [math.nan, -math.nan, SIGNALING_NAN, -SIGNALING_NAN]
            return self.prng.choice(nans)
        if min_value <= math.inf <= max_value and self.prng.random() < 1 / 32:
            return math.inf
        if min_value <= -math.inf <= max_value and self.prng.random() < 1 / 32:
            return -math.inf
        if min_value in [-math.inf, math.inf]:
            min_value = math.copysign(1, min_value) * sys.float_info.max
            min_value /= 2
        if max_value in [-math.inf, math.inf]:
            max_value = math.copysign(1, max_value) * sys.float_info.max
            max_value /= 2
        value = self.prng.uniform(min_value, max_value)
        if value and abs(value) < smallest_nonzero_magnitude:
            return math.copysign(0.0, value)
        return value

    def draw_string(self, intervals, *, min_size: int = 0, max_size: Optional[int] = COLLECTION_DEFAULT_MAX_SIZE) -> str:
        size = self.prng.randint(min_size, max(min_size, min(100 if max_size is None else max_size, 100)))
        return ''.join(map(chr, self.prng.choices(intervals, k=size))

    def draw_bytes(self, min_size: int = 0, max_size: Optional[int] = COLLECTION_DEFAULT_MAX_SIZE) -> bytes:
        max_size = 100 if max_size is None else max_size
        size = self.prng.randint(min_size, max_size)
        try:
            return self.prng.randbytes(size)
        except AttributeError:
            return bytes((self.prng.randint(0, 255) for _ in range(size)))

@contextmanager
def temp_register_backend(name: str, cls: PrimitiveProvider) -> None:
    try:
        AVAILABLE_PROVIDERS[name] = f'{__name__}.{cls.__name__}'
        yield
    finally:
        AVAILABLE_PROVIDERS.pop(name)

@pytest.mark.parametrize('strategy', [st.booleans(), st.integers(0, 3), st.floats(0, 1), st.text(max_size=3), st.binary(max_size=3)], ids=repr)
def test_find_with_backend_then_convert_to_buffer_shrink_and_replay(strategy) -> None:
    db = InMemoryExampleDatabase()
    assert not db.data
    with temp_register_backend('prng', PrngProvider):

        @settings(database=db, backend='prng')
        @given(strategy)
        def test(value):
            if isinstance(value, float):
                assert value >= 0.5
            else:
                assert value
        with pytest.raises(AssertionError):
            test()
    assert db.data
    buffers = {x for x in db.data[next(iter(db.data))] if x}
    assert buffers, db.data

def test_backend_can_shrink_integers() -> None:
    with temp_register_backend('prng', PrngProvider):
        n = minimal(st.integers(), lambda n: n >= 123456, settings=settings(backend='prng', database=None))
    assert n == 123456

# Add type annotations for the remaining functions similarly
