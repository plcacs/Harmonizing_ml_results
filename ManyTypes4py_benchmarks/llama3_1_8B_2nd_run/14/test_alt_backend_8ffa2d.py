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
    """A random number generator provider."""

    def __init__(self, conjecturedata: ConjectureData, /) -> None:
        super().__init__(conjecturedata)
        self.prng = Random(0)

    def draw_boolean(self, p: float = 0.5) -> bool:
        """Draw a boolean value with the given probability."""
        return self.prng.random() < p

    def draw_integer(self, min_value: Optional[int] = None, max_value: Optional[int] = None, *, weights: Optional[list[float]] = None, shrink_towards: int = 0) -> int:
        """Draw an integer value within the given range."""
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
        """Draw a floating point value within the given range."""
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

    def draw_string(self, intervals: list[int], *, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> str:
        """Draw a string value of the given length."""
        size = self.prng.randint(min_size, min(100 if max_size is None else max_size, 100))
        return ''.join(map(chr, self.prng.choices(intervals, k=size)))

    def draw_bytes(self, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> bytes:
        """Draw a bytes value of the given length."""
        max_size = 100 if max_size is None else max_size
        size = self.prng.randint(min_size, max_size)
        try:
            return self.prng.randbytes(size)
        except AttributeError:
            return bytes((self.prng.randint(0, 255) for _ in range(size)))

@contextmanager
def temp_register_backend(name: str, cls: type) -> None:
    """Temporarily register a provider backend."""
    try:
        AVAILABLE_PROVIDERS[name] = f'{__name__}.{cls.__name__}'
        yield
    finally:
        AVAILABLE_PROVIDERS.pop(name)

@pytest.mark.parametrize('strategy', [st.booleans(), st.integers(0, 3), st.floats(0, 1), st.text(max_size=3), st.binary(max_size=3)], ids=repr)
def test_find_with_backend_then_convert_to_buffer_shrink_and_replay(strategy: st.Strategy) -> None:
    """Test finding a strategy with a backend then converting to a buffer."""
    db = InMemoryExampleDatabase()
    assert not db.data
    with temp_register_backend('prng', PrngProvider):

        @settings(database=db, backend='prng')
        @given(strategy)
        def test(value: str) -> None:
            """Test function."""
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
    """Test that the backend can shrink integers."""
    with temp_register_backend('prng', PrngProvider):
        n = minimal(st.integers(), lambda n: n >= 123456, settings=settings(backend='prng', database=None))
    assert n == 123456

def test_backend_can_shrink_bytes() -> None:
    """Test that the backend can shrink bytes."""
    with temp_register_backend('prng', PrngProvider):
        b = minimal(st.binary(min_size=2, max_size=2), lambda b: len(b) >= 2 and b[1] >= 10, settings=settings(backend='prng', database=None))
    assert b == int_to_bytes(10, size=2)

def test_backend_can_shrink_strings() -> None:
    """Test that the backend can shrink strings."""
    with temp_register_backend('prng', PrngProvider):
        s = minimal(st.text(), lambda s: len(s) >= 10, settings=settings(backend='prng', database=None))
    assert len(s) == 10

def test_backend_can_shrink_booleans() -> None:
    """Test that the backend can shrink booleans."""
    with temp_register_backend('prng', PrngProvider):
        b = minimal(st.booleans(), lambda b: b, settings=settings(backend='prng', database=None))
    assert b

def test_backend_can_shrink_floats() -> None:
    """Test that the backend can shrink floats."""
    with temp_register_backend('prng', PrngProvider):
        f = minimal(st.floats(), lambda f: f >= 100.5, settings=settings(backend='prng', database=None))
    assert f == 101.0

@given(nodes())
def test_new_conjecture_data_with_backend(node: nodes.Node) -> None:
    """Test new conjecture data with a backend."""
    def test(data: ConjectureData) -> None:
        """Test function."""
        getattr(data, f'draw_{node.type}')(**node.kwargs)
    with temp_register_backend('prng', PrngProvider):
        runner = ConjectureRunner(test, settings=settings(backend='prng'))
        runner.cached_test_function_ir([node.value])

class TrivialProvider(PrimitiveProvider):
    """A trivial provider."""

    def draw_integer(self, *args, **kwargs) -> int:
        """Draw an integer value."""
        return 1

    def draw_boolean(self, *args, **kwargs) -> bool:
        """Draw a boolean value."""
        return True

    def draw_float(self, *args, **kwargs) -> float:
        """Draw a float value."""
        return 1.0

    def draw_bytes(self, *args, **kwargs) -> bytes:
        """Draw a bytes value."""
        return b''

    def draw_string(self, *args, **kwargs) -> str:
        """Draw a string value."""
        return ''

class InvalidLifetime(TrivialProvider):
    """A provider with an invalid lifetime."""
    lifetime = 'forever and a day!'

def test_invalid_lifetime() -> None:
    """Test an invalid lifetime."""
    with temp_register_backend('invalid_lifetime', InvalidLifetime):
        with pytest.raises(InvalidArgument):
            ConjectureRunner(lambda: True, settings=settings(backend='invalid_lifetime'))
function_lifetime_init_count = 0

class LifetimeTestFunction(TrivialProvider):
    """A provider with a test function lifetime."""
    lifetime = 'test_function'

    def __init__(self, conjecturedata: ConjectureData) -> None:
        super().__init__(conjecturedata)
        global function_lifetime_init_count
        function_lifetime_init_count += 1

def test_function_lifetime() -> None:
    """Test a function lifetime."""
    with temp_register_backend('lifetime_function', LifetimeTestFunction):

        @given(st.integers())
        @settings(backend='lifetime_function')
        def test_function(n: int) -> None:
            """Test function."""
            pass
        assert function_lifetime_init_count == 0
        test_function()
        assert function_lifetime_init_count == 1
        test_function()
        assert function_lifetime_init_count == 2
test_case_lifetime_init_count = 0

class LifetimeTestCase(TrivialProvider):
    """A provider with a test case lifetime."""
    lifetime = 'test_case'

    def __init__(self, conjecturedata: ConjectureData) -> None:
        super().__init__(conjecturedata)
        global test_case_lifetime_init_count
        test_case_lifetime_init_count += 1

def test_case_lifetime() -> None:
    """Test a test case lifetime."""
    test_function_count = 0
    with temp_register_backend('lifetime_case', LifetimeTestCase):

        @given(st.integers())
        @settings(backend='lifetime_case', database=InMemoryExampleDatabase())
        def test_function(n: int) -> None:
            """Test function."""
            nonlocal test_function_count
            test_function_count += 1
        assert test_case_lifetime_init_count == 0
        test_function()
        assert test_function_count - 10 <= test_case_lifetime_init_count <= test_function_count + 10

def test_flaky_with_backend() -> None:
    """Test a flaky test with a backend."""
    with temp_register_backend('trivial', TrivialProvider), capture_observations():
        calls = 0

        @given(st.integers())
        @settings(backend='trivial', database=None)
        def test_function(n: int) -> None:
            """Test function."""
            nonlocal calls
            calls += 1
            assert n != calls % 2
        with pytest.raises(Flaky):
            test_function()

class BadRealizeProvider(TrivialProvider):
    """A provider that raises an exception on realize."""
    def realize(self, value: object) -> None:
        """Realize the value."""
        return None

def test_bad_realize() -> None:
    """Test a bad realize."""
    with temp_register_backend('bad_realize', BadRealizeProvider):

        @given(st.integers())
        @settings(backend='bad_realize')
        def test_function(n: int) -> None:
            """Test function."""
            pass
        with pytest.raises(HypothesisException, match='expected .* from BadRealizeProvider.realize'):
            test_function()

class RealizeProvider(TrivialProvider):
    """A provider that realizes values."""
    avoid_realization = True

    def realize(self, value: object) -> object:
        """Realize the value."""
        if isinstance(value, int):
            return 42
        return value

def test_realize() -> None:
    """Test realizing values."""
    with temp_register_backend('realize', RealizeProvider):
        values = []

        @given(st.integers())
        @settings(backend='realize')
        def test_function(n: int) -> None:
            """Test function."""
            values.append(current_build_context().data.provider.realize(n))
        test_function()
        assert all((n == 42 for n in values))

def test_realize_dependent_draw() -> None:
    """Test realizing dependent draw."""
    with temp_register_backend('realize', RealizeProvider):

        @given(st.data())
        @settings(backend='realize')
        def test_function(data: st.DataObject) -> None:
            """Test function."""
            n1 = data.draw(st.integers())
            n2 = data.draw(st.integers(n1, n1 + 10))
            assert n1 <= n2
        test_function()

@pytest.mark.parametrize('verbosity', [Verbosity.verbose, Verbosity.debug])
def test_realization_with_verbosity(verbosity: Verbosity) -> None:
    """Test realization with verbosity."""
    with temp_register_backend('realize', RealizeProvider):

        @given(st.floats())
        @settings(backend='realize', verbosity=verbosity)
        def test_function(f: float) -> None:
            """Test function."""
            pass
        with capture_out() as out:
            test_function()
        assert 'Trying example: <symbolics>' in out.getvalue()

@pytest.mark.parametrize('verbosity', [Verbosity.verbose, Verbosity.debug])
def test_realization_with_verbosity_draw(verbosity: Verbosity) -> None:
    """Test realization with verbosity draw."""
    with temp_register_backend('realize', RealizeProvider):

        @given(st.data())
        @settings(backend='realize', verbosity=verbosity)
        def test_function(data: st.DataObject) -> None:
            """Test function."""
            data.draw(st.integers())
        with capture_out() as out:
            test_function()
        assert 'Draw 1: <symbolic>' in out.getvalue()

class ObservableProvider(TrivialProvider):
    """A provider that observes test cases."""
    def observe_test_case(self) -> dict:
        """Observe the test case."""
        return {'msg_key': 'some message', 'data_key': [1, '2', {}]}

    def observe_information_messages(self, *, lifetime: str) -> Iterable[dict]:
        """Observe information messages."""
        if lifetime == 'test_case':
            yield {'type': 'info', 'title': 'trivial-data', 'content': {'k2': 'v2'}}
        else:
            assert lifetime == 'test_function'
            yield {'type': 'alert', 'title': 'Trivial alert', 'content': 'message here'}
            yield {'type': 'info', 'title': 'trivial-data', 'content': {'k2': 'v2'}}

    def realize(self, value: object) -> None:
        """Realize the value."""
        raise BackendCannotProceed

def test_custom_observations_from_backend() -> None:
    """Test custom observations from a backend."""
    with temp_register_backend('observable', ObservableProvider):

        @given(st.none())
        @settings(backend='observable', database=None)
        def test_function(_: object) -> None:
            """Test function."""
            pass
        with capture_observations() as ls:
            test_function()
    assert len(ls) >= 3
    cases = [t['metadata']['backend'] for t in ls if t['type'] == 'test_case']
    assert {'msg_key': 'some message', 'data_key': [1, '2', {}]} in cases
    assert '<backend failed to realize symbolic arguments>' in repr(ls)
    infos = [{k: v for k, v in t.items() if k in ('title', 'content')} for t in ls if t['type'] != 'test_case']
    assert {'title': 'Trivial alert', 'content': 'message here'} in infos
    assert {'title': 'trivial-data', 'content': {'k2': 'v2'}} in infos

class FallibleProvider(TrivialProvider):
    """A provider that raises exceptions."""
    def __init__(self, conjecturedata: ConjectureData, /) -> None:
        super().__init__(conjecturedata)
        self._it = itertools.cycle([1, 1, 1, 'discard_test_case', 'other'])

    def draw_integer(self, *args, **kwargs) -> int:
        """Draw an integer value."""
        x = next(self._it)
        if isinstance(x, str):
            raise BackendCannotProceed(x)
        return x

def test_falls_back_to_default_backend() -> None:
    """Test falling back to a default backend."""
    with temp_register_backend('fallible', FallibleProvider):
        seen_other_ints = False

        @given(st.integers())
        @settings(backend='fallible', database=None, max_examples=100)
        def test_function(x: int) -> None:
            """Test function."""
            nonlocal seen_other_ints
            seen_other_ints |= x != 1
        test_function()
        assert seen_other_ints

def test_can_raise_unsatisfiable_after_falling_back() -> None:
    """Test raising unsatisfiable after falling back."""
    with temp_register_backend('fallible', FallibleProvider):

        @given(st.integers())
        @settings(backend='fallible', database=None, max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
        def test_function(x: int) -> None:
            """Test function."""
            assume(x == 'unsatisfiable')
        with pytest.raises(Unsatisfiable):
            test_function()

class ExhaustibleProvider(TrivialProvider):
    """A provider that raises exceptions."""
    scope = 'exhausted'

    def __init__(self, conjecturedata: ConjectureData, /) -> None:
        super().__init__(conjecturedata)
        self._calls = 0

    def draw_integer(self, *args, **kwargs) -> int:
        """Draw an integer value."""
        self._calls += 1
        if self._calls > 20:
            raise BackendCannotProceed(self.scope)
        return 1

class UnsoundVerifierProvider(ExhaustibleProvider):
    """A provider that raises exceptions."""
    scope = 'verified'

@pytest.mark.parametrize('provider', [ExhaustibleProvider, UnsoundVerifierProvider])
def test_notes_incorrect_verification(provider: type) -> None:
    """Test notes for incorrect verification."""
    msg = "backend='p' claimed to verify this test passes - please send them a bug report!"
    with temp_register_backend('p', provider):

        @given(st.integers())
        @settings(backend='p', database=None, max_examples=100)
        def test_function(x: int) -> None:
            """Test function."""
            assert x == 1
        with pytest.raises(AssertionError) as ctx:
            test_function()
        assert (msg in ctx.value.__notes__) == (provider is UnsoundVerifierProvider)

def test_invalid_provider_kw() -> None:
    """Test invalid provider kw."""
    with pytest.raises(InvalidArgument, match='got an instance instead'):
        ConjectureData(random=None, provider=TrivialProvider(None), provider_kw={'one': 'two'})

def test_available_providers_deprecation() -> None:
    """Test available providers deprecation."""
    with pytest.warns(errors.HypothesisDeprecationWarning):
        from hypothesis.internal.conjecture.data import AVAILABLE_PROVIDERS
    with pytest.raises(ImportError):
        from hypothesis.internal.conjecture.data import does_not_exist
