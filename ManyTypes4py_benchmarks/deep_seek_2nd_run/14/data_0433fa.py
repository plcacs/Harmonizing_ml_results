import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum
from functools import cached_property
from random import Random
from typing import (
    TYPE_CHECKING, Any, Dict, FrozenSet, List, NoReturn, Optional, Set, Tuple, 
    TypeVar, Union, cast
)
import attr
from hypothesis.errors import ChoiceTooLarge, Frozen, InvalidArgument, StopTest
from hypothesis.internal.cache import LRUCache
from hypothesis.internal.compat import add_note
from hypothesis.internal.conjecture.choice import (
    BooleanKWargs, BytesKWargs, ChoiceKwargsT, ChoiceNode, ChoiceT, 
    ChoiceTemplate, ChoiceTypeT, FloatKWargs, IntegerKWargs, StringKWargs, 
    choice_from_index, choice_kwargs_key, choice_permitted, choices_size
)
from hypothesis.internal.conjecture.junkdrawer import IntList, gc_cumulative_time
from hypothesis.internal.conjecture.providers import (
    COLLECTION_DEFAULT_MAX_SIZE, HypothesisProvider, PrimitiveProvider
)
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, float_to_int, int_to_float, sign_aware_lte
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.reporting import debug_report

if TYPE_CHECKING:
    from typing import TypeAlias
    from hypothesis.strategies import SearchStrategy
    from hypothesis.strategies._internal.strategies import Ex

def __getattr__(name: str) -> Any:
    if name == 'AVAILABLE_PROVIDERS':
        from hypothesis._settings import note_deprecation
        from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS
        note_deprecation(
            'hypothesis.internal.conjecture.data.AVAILABLE_PROVIDERS has been moved to '
            'hypothesis.internal.conjecture.providers.AVAILABLE_PROVIDERS.',
            since='2025-01-25', has_codemod=False, stacklevel=1
        )
        return AVAILABLE_PROVIDERS
    raise AttributeError(f"Module 'hypothesis.internal.conjecture.data' has no attribute {name}")

T = TypeVar('T')
TargetObservations = Dict[str, Union[int, float]]
MisalignedAt = Tuple[int, ChoiceTypeT, ChoiceKwargsT, Optional[ChoiceT]]
TOP_LABEL = calc_label_from_name('top')

class ExtraInformation:
    """A class for holding shared state on a ``ConjectureData`` that should
    be added to the final ``ConjectureResult``."""

    def __repr__(self) -> str:
        return 'ExtraInformation({})'.format(', '.join((f'{k}={v!r}' for k, v in self.__dict__.items()))

    def has_information(self) -> bool:
        return bool(self.__dict__)

class Status(IntEnum):
    OVERRUN = 0
    INVALID = 1
    VALID = 2
    INTERESTING = 3

    def __repr__(self) -> str:
        return f'Status.{self.name}'

@attr.s(slots=True, frozen=True)
class StructuralCoverageTag:
    label: int = attr.ib()

STRUCTURAL_COVERAGE_CACHE: Dict[int, StructuralCoverageTag] = {}

def structural_coverage(label: int) -> StructuralCoverageTag:
    try:
        return STRUCTURAL_COVERAGE_CACHE[label]
    except KeyError:
        return STRUCTURAL_COVERAGE_CACHE.setdefault(label, StructuralCoverageTag(label))

POOLED_KWARGS_CACHE: LRUCache[Tuple[Any, ...], Dict[str, Any]] = LRUCache(4096)

class Example:
    """Examples track the hierarchical structure of draws from the byte stream,
    within a single test run."""
    __slots__ = ('index', 'owner')

    def __init__(self, owner: 'Examples', index: int):
        self.owner = owner
        self.index = index

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is other.owner and self.index == other.index

    def __ne__(self, other: Any) -> bool:
        if self is other:
            return False
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is not other.owner or self.index != other.index

    def __repr__(self) -> str:
        return f'examples[{self.index}]'

    @property
    def label(self) -> int:
        return self.owner.labels[self.owner.label_indices[self.index]]

    @property
    def parent(self) -> Optional['Example']:
        if self.index == 0:
            return None
        return self.owner[self.owner.parentage[self.index]]

    @property
    def start(self) -> int:
        return self.owner.starts[self.index]

    @property
    def end(self) -> int:
        return self.owner.ends[self.index]

    @property
    def depth(self) -> int:
        return self.owner.depths[self.index]

    @property
    def discarded(self) -> bool:
        return self.index in self.owner.discarded

    @property
    def choice_count(self) -> int:
        return self.end - self.start

    @property
    def children(self) -> List['Example']:
        return [self.owner[i] for i in self.owner.children[self.index]]

class ExampleProperty:
    def __init__(self, examples: 'Examples'):
        self.example_stack: List[int] = []
        self.examples = examples
        self.example_count = 0
        self.choice_count = 0

    def run(self) -> Any:
        for record in self.examples.trail:
            if record == CHOICE_RECORD:
                self.choice_count += 1
            elif record >= START_EXAMPLE_RECORD:
                self.__push(record - START_EXAMPLE_RECORD)
            else:
                assert record in (STOP_EXAMPLE_DISCARD_RECORD, STOP_EXAMPLE_NO_DISCARD_RECORD)
                self.__pop(discarded=record == STOP_EXAMPLE_DISCARD_RECORD)
        return self.finish()

    def __push(self, label_index: int) -> None:
        i = self.example_count
        assert i < len(self.examples)
        self.start_example(i, label_index=label_index)
        self.example_count += 1
        self.example_stack.append(i)

    def __pop(self, *, discarded: bool) -> None:
        i = self.example_stack.pop()
        self.stop_example(i, discarded=discarded)

    def start_example(self, i: int, label_index: int) -> None:
        pass

    def stop_example(self, i: int, *, discarded: bool) -> None:
        pass

    def finish(self) -> Any:
        raise NotImplementedError

STOP_EXAMPLE_DISCARD_RECORD = 1
STOP_EXAMPLE_NO_DISCARD_RECORD = 2
START_EXAMPLE_RECORD = 3
CHOICE_RECORD = calc_label_from_name('ir draw record')

class ExampleRecord:
    def __init__(self):
        self.labels: List[int] = []
        self.__index_of_labels: Optional[Dict[int, int]] = {}
        self.trail = IntList()
        self.nodes: List[Any] = []

    def freeze(self) -> None:
        self.__index_of_labels = None

    def record_choice(self) -> None:
        self.trail.append(CHOICE_RECORD)

    def start_example(self, label: int) -> None:
        assert self.__index_of_labels is not None
        try:
            i = self.__index_of_labels[label]
        except KeyError:
            i = self.__index_of_labels.setdefault(label, len(self.labels))
            self.labels.append(label)
        self.trail.append(START_EXAMPLE_RECORD + i)

    def stop_example(self, *, discard: bool) -> None:
        if discard:
            self.trail.append(STOP_EXAMPLE_DISCARD_RECORD)
        else:
            self.trail.append(STOP_EXAMPLE_NO_DISCARD_RECORD)

class _starts_and_ends(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.starts = IntList.of_length(len(self.examples))
        self.ends = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.starts[i] = self.choice_count

    def stop_example(self, i: int, *, discarded: bool) -> None:
        self.ends[i] = self.choice_count

    def finish(self) -> Tuple[IntList, IntList]:
        return (self.starts, self.ends)

class _discarded(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.result: Set[int] = set()

    def finish(self) -> FrozenSet[int]:
        return frozenset(self.result)

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if discarded:
            self.result.add(i)

class _parentage(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if i > 0:
            self.result[i] = self.example_stack[-1]

    def finish(self) -> IntList:
        return self.result

class _depths(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = len(self.example_stack)

    def finish(self) -> IntList:
        return self.result

class _label_indices(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = label_index

    def finish(self) -> IntList:
        return self.result

class _mutator_groups(ExampleProperty):
    def __init__(self, examples: 'Examples'):
        super().__init__(examples)
        self.groups: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

    def start_example(self, i: int, label_index: int) -> None:
        key = (self.examples[i].start, self.examples[i].end)
        self.groups[label_index].add(key)

    def finish(self) -> List[Set[Tuple[int, int]]]:
        return [g for g in self.groups.values() if len(g) >= 2]

class Examples:
    def __init__(self, record: ExampleRecord):
        self.trail = record.trail
        self.labels = record.labels
        self.__length = self.trail.count(STOP_EXAMPLE_DISCARD_RECORD) + record.trail.count(STOP_EXAMPLE_NO_DISCARD_RECORD)
        self.__children: Optional[List[IntList]] = None

    @cached_property
    def starts_and_ends(self) -> Tuple[IntList, IntList]:
        return _starts_and_ends(self).run()

    @property
    def starts(self) -> IntList:
        return self.starts_and_ends[0]

    @property
    def ends(self) -> IntList:
        return self.starts_and_ends[1]

    @cached_property
    def discarded(self) -> FrozenSet[int]:
        return _discarded(self).run()

    @cached_property
    def parentage(self) -> IntList:
        return _parentage(self).run()

    @cached_property
    def depths(self) -> IntList:
        return _depths(self).run()

    @cached_property
    def label_indices(self) -> IntList:
        return _label_indices(self).run()

    @cached_property
    def mutator_groups(self) -> List[Set[Tuple[int, int]]]:
        return _mutator_groups(self).run()

    @property
    def children(self) -> List[IntList]:
        if self.__children is None:
            children: List[IntList] = [IntList() for _ in range(len(self))]
            for i, p in enumerate(self.parentage):
                if i > 0:
                    children[p].append(i)
            for i, c in enumerate(children):
                if not c:
                    children[i] = ()
            self.__children = children
        return self.__children

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, i: int) -> Example:
        n = self.__length
        if i < -n or i >= n:
            raise IndexError(f'Index {i} out of range [-{n}, {n})')
        if i < 0:
            i += n
        return Example(self, i)

    def __iter__(self) -> Iterator[Example]:
        for i in range(len(self)):
            yield self[i]

class _Overrun:
    status = Status.OVERRUN

    def __repr__(self) -> str:
        return 'Overrun'

Overrun = _Overrun()
global_test_counter = 0
MAX_DEPTH = 100

class DataObserver:
    def conclude_test(self, status: Status, interesting_origin: Optional[InterestingOrigin]) -> None:
        pass

    def kill_branch(self) -> None:
        pass

    def draw_integer(self, value: int, *, kwargs: IntegerKWargs, was_forced: bool) -> None:
        pass

    def draw_float(self, value: float, *, kwargs: FloatKWargs, was_forced: bool) -> None:
        pass

    def draw_string(self, value: str, *, kwargs: StringKWargs, was_forced: bool) -> None:
        pass

    def draw_bytes(self, value: bytes, *, kwargs: BytesKWargs, was_forced: bool) -> None:
        pass

    def draw_boolean(self, value: bool, *, kwargs: BooleanKWargs, was_forced: bool) -> None:
        pass

@attr.s(slots=True)
class ConjectureResult:
    status: Status = attr.ib()
    interesting_origin: Optional[InterestingOrigin] = attr.ib()
    nodes: Tuple[ChoiceNode, ...] = attr.ib(eq=False, repr=False)
    length: int = attr.ib()
    output: str = attr.ib()
    extra_information: Optional[ExtraInformation] = attr.ib()
    expected_exception: Optional[BaseException] = attr.ib()
    expected_traceback: Optional[str] = attr.ib()
    has_discards: bool = attr.ib()
    target_observations: TargetObservations = attr.ib()
    tags: FrozenSet[Any] = attr.ib()
    examples: Examples = attr.ib(repr=False, eq=False)
    arg_slices: Set[Any] = attr.ib(repr=False)
    slice_comments: Dict[Any, Any] = attr.ib(repr=False)
    misaligned_at: Optional[MisalignedAt] = attr.ib(repr=False)

    def as_result(self) -> 'ConjectureResult':
        return self

    @property
    def choices(self) -> Tuple[ChoiceT, ...]:
        return tuple((node.value for node in self.nodes))

class ConjectureData:
    @classmethod
    def for_choices(
        cls, 
        choices: Sequence[ChoiceT], 
        *, 
        observer: Optional[DataObserver] = None, 
        provider: Union[Type[PrimitiveProvider], PrimitiveProvider] = HypothesisProvider, 
        random: Optional[Random] = None
    ) -> 'ConjectureData':
        from hypothesis.internal.conjecture.engine import choice_count
        return cls(
            max_choices=choice_count(choices), 
            random=random, 
            prefix=choices, 
            observer=observer, 
            provider=provider
        )

    def __init__(
        self, 
        *, 
        random: Optional[Random], 
        observer: Optional[DataObserver] = None, 
        provider: Union[Type[PrimitiveProvider], PrimitiveProvider] = HypothesisProvider, 
        prefix: Optional[Sequence[ChoiceT]] = None, 
        max_choices: Optional[int] = None, 
        provider_kw: Optional[Dict[str, Any]] = None
    ):
        from hypothesis.internal.conjecture.engine import BUFFER_SIZE
        if observer is None:
            observer = DataObserver()
        if provider_kw is None:
            provider_kw = {}
        elif not isinstance(provider, type):
            raise InvalidArgument(
                f'Expected provider={provider!r} to be a class since provider_kw={provider_kw!r} was passed, '
                'but got an instance instead.'
            )
        assert isinstance(observer, DataObserver)
        self.observer: DataObserver = observer
        self.max_choices: Optional[int] = max_choices
        self.max_length: int = BUFFER_SIZE
        self.is_find: bool = False
        self.overdraw: int = 0
        self._random: Optional[Random] = random
        self.length: int = 0
        self.index: int = 0
        self.output: str = ''
        self.status: Status = Status.VALID
        self.frozen: bool = False
        global global_test_counter
        self.testcounter: int = global_test_counter
        global_test_counter += 1
        self.start_time: float = time.perf_counter()
        self.gc_start_time: float = gc_cumulative_time()
        self.events: Dict[str, Any] = {}
        self.interesting_origin: Optional[InterestingOrigin] = None
        self.draw_times: Dict[str, float] = {}
        self._stateful_run_times: Dict[str, float] = defaultdict(float)
        self.max_depth: int = 0
        self.has_discards: bool = False
        self.provider: PrimitiveProvider