from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypedDict, TypeVar, Union

T = TypeVar('T')
_Lifetime = Literal['test_case', 'test_function']
COLLECTION_DEFAULT_MAX_SIZE: int = 10 ** 10
AVAILABLE_PROVIDERS: dict[str, str] = {'hypothesis': 'hypothesis.internal.conjecture.providers.HypothesisProvider'}

class _BackendInfoMsg(TypedDict):
    pass

class PrimitiveProvider(abc.ABC):
    lifetime: _Lifetime = 'test_function'
    avoid_realization: bool = False

    def __init__(self, conjecturedata: 'ConjectureData', /) -> None:
        self._cd = conjecturedata

    def per_test_case_context_manager(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def realize(self, value: Any) -> Any:
        ...

    def observe_test_case(self) -> dict[str, Any]:
        ...

    def observe_information_messages(self, *, lifetime: _Lifetime) -> Iterable[dict[str, Union[str, dict[str, str]]]:
        ...

    @abc.abstractmethod
    def draw_boolean(self, p: float = 0.5) -> bool:
        ...

    @abc.abstractmethod
    def draw_integer(self, min_value: Optional[int] = None, max_value: Optional[int] = None, *, weights: Optional[dict[int, float]] = None, shrink_towards: int = 0) -> int:
        ...

    @abc.abstractmethod
    def draw_float(self, *, min_value: float = -math.inf, max_value: float = math.inf, allow_nan: bool = True, smallest_nonzero_magnitude: float) -> float:
        ...

    @abc.abstractmethod
    def draw_string(self, intervals: IntervalSet, *, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> str:
        ...

    @abc.abstractmethod
    def draw_bytes(self, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> bytes:
        ...

    def span_start(self, label: int, /) -> None:
        ...

    def span_end(self, discard: bool, /) -> None:
        ...

class HypothesisProvider(PrimitiveProvider):
    lifetime: _Lifetime = 'test_case'

    def __init__(self, conjecturedata: 'ConjectureData', /) -> None:
        super().__init__(conjecturedata)

    def draw_boolean(self, p: float = 0.5) -> bool:
        ...

    def draw_integer(self, min_value: Optional[int] = None, max_value: Optional[int] = None, *, weights: Optional[dict[int, float]] = None, shrink_towards: int = 0) -> int:
        ...

    def draw_float(self, *, min_value: float = -math.inf, max_value: float = math.inf, allow_nan: bool = True, smallest_nonzero_magnitude: float) -> float:
        ...

    def draw_string(self, intervals: IntervalSet, *, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> str:
        ...

    def draw_bytes(self, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> bytes:
        ...

    @classmethod
    def _draw_float_init_logic(cls, *, min_value: float, max_value: float, allow_nan: bool, smallest_nonzero_magnitude: float) -> tuple[Optional[Sampler], Callable[[float], float], list[float]]:
        ...

class BytestringProvider(PrimitiveProvider):
    lifetime: _Lifetime = 'test_case'

    def __init__(self, conjecturedata: 'ConjectureData', /, *, bytestring: bytes) -> None:
        ...

    def _draw_bits(self, n: int) -> int:
        ...

    def draw_boolean(self, p: float = 0.5) -> bool:
        ...

    def draw_integer(self, min_value: Optional[int] = None, max_value: Optional[int] = None, *, weights: Optional[dict[int, float]] = None, shrink_towards: int = 0) -> int:
        ...

    def draw_float(self, *, min_value: float = -math.inf, max_value: float = math.inf, allow_nan: bool = True, smallest_nonzero_magnitude: float) -> float:
        ...

    def draw_string(self, intervals: IntervalSet, *, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> str:
        ...

    def draw_bytes(self, min_size: int = 0, max_size: int = COLLECTION_DEFAULT_MAX_SIZE) -> bytes:
        ...
