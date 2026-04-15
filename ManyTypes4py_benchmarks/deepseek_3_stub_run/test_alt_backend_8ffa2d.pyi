import sys
from contextlib import contextmanager
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from typing_extensions import Literal

import pytest
from hypothesis import HealthCheck, Verbosity, settings
from hypothesis import strategies as st
from hypothesis.control import BuildContext
from hypothesis.database import ExampleDatabase
from hypothesis.errors import (
    BackendCannotProceed,
    Flaky,
    HypothesisException,
    InvalidArgument,
    Unsatisfiable,
)
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.providers import PrimitiveProvider
from tests.common.debug import minimal as minimal_debug
from tests.common.utils import capture_observations, capture_out
from tests.conjecture.common import Node

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

# Module-level variables
function_lifetime_init_count: int = ...
test_case_lifetime_init_count: int = ...

class PrngProvider(PrimitiveProvider):
    prng: Random

    def __init__(self, conjecturedata: ConjectureData, /) -> None: ...
    def draw_boolean(self, p: float = 0.5) -> bool: ...
    def draw_integer(
        self,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        *,
        weights: Optional[List[float]] = None,
        shrink_towards: int = 0,
    ) -> int: ...
    def draw_float(
        self,
        *,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
        allow_nan: bool = True,
        smallest_nonzero_magnitude: float,
    ) -> float: ...
    def draw_string(
        self,
        intervals: Iterable[int],
        *,
        min_size: int = 0,
        max_size: Optional[int] = ...,
    ) -> str: ...
    def draw_bytes(
        self,
        min_size: int = 0,
        max_size: Optional[int] = ...,
    ) -> bytes: ...

@contextmanager
def temp_register_backend(
    name: str,
    cls: type,
) -> Generator[None, None, None]: ...

@pytest.mark.parametrize("strategy", [
    st.booleans(),
    st.integers(0, 3),
    st.floats(0, 1),
    st.text(max_size=3),
    st.binary(max_size=3),
], ids=repr)
def test_find_with_backend_then_convert_to_buffer_shrink_and_replay(
    strategy: st.SearchStrategy[Any],
) -> None: ...

def test_backend_can_shrink_integers() -> None: ...
def test_backend_can_shrink_bytes() -> None: ...
def test_backend_can_shrink_strings() -> None: ...
def test_backend_can_shrink_booleans() -> None: ...
def test_backend_can_shrink_floats() -> None: ...

@given(nodes())
def test_new_conjecture_data_with_backend(node: Node) -> None: ...

class TrivialProvider(PrimitiveProvider):
    def draw_integer(self, *args: Any, **kwargs: Any) -> int: ...
    def draw_boolean(self, *args: Any, **kwargs: Any) -> bool: ...
    def draw_float(self, *args: Any, **kwargs: Any) -> float: ...
    def draw_bytes(self, *args: Any, **kwargs: Any) -> bytes: ...
    def draw_string(self, *args: Any, **kwargs: Any) -> str: ...

class InvalidLifetime(TrivialProvider):
    lifetime: str = ...

def test_invalid_lifetime() -> None: ...

class LifetimeTestFunction(TrivialProvider):
    lifetime: str = ...

    def __init__(self, conjecturedata: ConjectureData) -> None: ...

def test_function_lifetime() -> None: ...

class LifetimeTestCase(TrivialProvider):
    lifetime: str = ...

    def __init__(self, conjecturedata: ConjectureData) -> None: ...

def test_case_lifetime() -> None: ...

def test_flaky_with_backend() -> None: ...

class BadRealizeProvider(TrivialProvider):
    def realize(self, value: Any) -> Optional[Any]: ...

def test_bad_realize() -> None: ...

class RealizeProvider(TrivialProvider):
    avoid_realization: bool = ...

    def realize(self, value: Any) -> Any: ...

def test_realize() -> None: ...

def test_realize_dependent_draw() -> None: ...

@pytest.mark.parametrize("verbosity", [Verbosity.verbose, Verbosity.debug])
def test_realization_with_verbosity(verbosity: Verbosity) -> None: ...

@pytest.mark.parametrize("verbosity", [Verbosity.verbose, Verbosity.debug])
def test_realization_with_verbosity_draw(verbosity: Verbosity) -> None: ...

class ObservableProvider(TrivialProvider):
    def observe_test_case(self) -> Dict[str, Any]: ...
    def observe_information_messages(
        self,
        *,
        lifetime: Literal["test_case", "test_function"],
    ) -> Iterator[Dict[str, Any]]: ...
    def realize(self, value: Any) -> Any: ...

def test_custom_observations_from_backend() -> None: ...

class FallibleProvider(TrivialProvider):
    def __init__(self, conjecturedata: ConjectureData, /) -> None: ...
    def draw_integer(self, *args: Any, **kwargs: Any) -> Union[int, str]: ...

def test_falls_back_to_default_backend() -> None: ...

def test_can_raise_unsatisfiable_after_falling_back() -> None: ...

class ExhaustibleProvider(TrivialProvider):
    scope: str = ...

    def __init__(self, conjecturedata: ConjectureData, /) -> None: ...
    def draw_integer(self, *args: Any, **kwargs: Any) -> int: ...

class UnsoundVerifierProvider(ExhaustibleProvider):
    scope: str = ...

@pytest.mark.parametrize("provider", [ExhaustibleProvider, UnsoundVerifierProvider])
def test_notes_incorrect_verification(provider: type) -> None: ...

def test_invalid_provider_kw() -> None: ...

def test_available_providers_deprecation() -> None: ...