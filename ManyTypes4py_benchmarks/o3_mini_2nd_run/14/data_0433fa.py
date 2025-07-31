import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum
from functools import cached_property
from random import Random
from typing import Any, Dict, List, NoReturn, Optional, Sequence as Seq, Set, Tuple, Type, TypeVar, Union
import attr
from hypothesis.errors import ChoiceTooLarge, Frozen, InvalidArgument, StopTest
from hypothesis.internal.cache import LRUCache
from hypothesis.internal.compat import add_note
from hypothesis.internal.conjecture.choice import (
    BooleanKWargs,
    BytesKWargs,
    ChoiceKwargsT,
    ChoiceNode,
    ChoiceT,
    ChoiceTemplate,
    ChoiceTypeT,
    FloatKWargs,
    IntegerKWargs,
    StringKWargs,
    choice_from_index,
    choice_kwargs_key,
    choice_permitted,
    choices_size,
)
from hypothesis.internal.conjecture.junkdrawer import IntList, gc_cumulative_time
from hypothesis.internal.conjecture.providers import (
    COLLECTION_DEFAULT_MAX_SIZE,
    HypothesisProvider,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, float_to_int, int_to_float, sign_aware_lte
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.reporting import debug_report
if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from hypothesis.strategies._internal.strategies import Ex

def __getattr__(name: str) -> Any:
    if name == 'AVAILABLE_PROVIDERS':
        from hypothesis._settings import note_deprecation
        from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS
        note_deprecation(
            'hypothesis.internal.conjecture.data.AVAILABLE_PROVIDERS has been moved to hypothesis.internal.conjecture.providers.AVAILABLE_PROVIDERS.',
            since='2025-01-25',
            has_codemod=False,
            stacklevel=1,
        )
        return AVAILABLE_PROVIDERS
    raise AttributeError(f"Module 'hypothesis.internal.conjecture.data' has no attribute {name}")

T = TypeVar('T')
TargetObservations = Dict[str, Union[int, float]]
MisalignedAt = Tuple[int, ChoiceTypeT, ChoiceKwargsT, Optional[ChoiceT]]
TOP_LABEL: int = calc_label_from_name('top')

class ExtraInformation:
    """A class for holding shared state on a ``ConjectureData`` that should
    be added to the final ``ConjectureResult``."""
    def __repr__(self) -> str:
        return 'ExtraInformation({})'.format(', '.join((f'{k}={v!r}' for k, v in self.__dict__.items())))
    
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

POOLED_KWARGS_CACHE: LRUCache = LRUCache(4096)

class Example:
    __slots__ = ('index', 'owner')

    def __init__(self, owner: "Examples", index: int) -> None:
        self.owner: Examples = owner
        self.index: int = index

    def __eq__(self, other: Any) -> Union[bool, Any]:
        if self is other:
            return True
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is other.owner and self.index == other.index

    def __ne__(self, other: Any) -> Union[bool, Any]:
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
    def parent(self) -> Optional["Example"]:
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
    def children(self) -> List["Example"]:
        return [self.owner[i] for i in self.owner.children[self.index]]

class ExampleProperty:
    def __init__(self, examples: "Examples") -> None:
        self.example_stack: List[int] = []
        self.examples: Examples = examples
        self.example_count: int = 0
        self.choice_count: int = 0

    def run(self) -> Any:
        for record in self.examples.trail:
            if record == CHOICE_RECORD:
                self.choice_count += 1
            elif record >= START_EXAMPLE_RECORD:
                self.__push(record - START_EXAMPLE_RECORD)
            else:
                assert record in (STOP_EXAMPLE_DISCARD_RECORD, STOP_EXAMPLE_NO_DISCARD_RECORD)
                self.__pop(discarded=(record == STOP_EXAMPLE_DISCARD_RECORD))
        return self.finish()

    def __push(self, label_index: int) -> None:
        i: int = self.example_count
        assert i < len(self.examples)
        self.start_example(i, label_index=label_index)
        self.example_count += 1
        self.example_stack.append(i)

    def __pop(self, *, discarded: bool) -> None:
        i: int = self.example_stack.pop()
        self.stop_example(i, discarded=discarded)

    def start_example(self, i: int, label_index: int) -> None:
        pass

    def stop_example(self, i: int, *, discarded: bool) -> None:
        pass

    def finish(self) -> Any:
        raise NotImplementedError

STOP_EXAMPLE_DISCARD_RECORD: int = 1
STOP_EXAMPLE_NO_DISCARD_RECORD: int = 2
START_EXAMPLE_RECORD: int = 3
CHOICE_RECORD: int = calc_label_from_name('ir draw record')

class ExampleRecord:
    def __init__(self) -> None:
        self.labels: List[int] = []
        self.__index_of_labels: Optional[Dict[int, int]] = {}
        self.trail: IntList = IntList()
        self.nodes: List[Any] = []

    def freeze(self) -> None:
        self.__index_of_labels = None

    def record_choice(self) -> None:
        self.trail.append(CHOICE_RECORD)

    def start_example(self, label: int) -> None:
        assert self.__index_of_labels is not None
        try:
            i: int = self.__index_of_labels[label]
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
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.starts: IntList = IntList.of_length(len(self.examples))
        self.ends: IntList = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.starts[i] = self.choice_count

    def stop_example(self, i: int, *, discarded: bool) -> None:
        self.ends[i] = self.choice_count

    def finish(self) -> Tuple[IntList, IntList]:
        return (self.starts, self.ends)

class _discarded(ExampleProperty):
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.result: Set[int] = set()

    def finish(self) -> frozenset:
        return frozenset(self.result)

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if discarded:
            self.result.add(i)

class _parentage(ExampleProperty):
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.result: IntList = IntList.of_length(len(self.examples))

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if i > 0:
            self.result[i] = self.example_stack[-1]

    def finish(self) -> IntList:
        return self.result

class _depths(ExampleProperty):
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.result: IntList = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = len(self.example_stack)

    def finish(self) -> IntList:
        return self.result

class _label_indices(ExampleProperty):
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.result: IntList = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = label_index

    def finish(self) -> IntList:
        return self.result

class _mutator_groups(ExampleProperty):
    def __init__(self, examples: "Examples") -> None:
        super().__init__(examples)
        self.groups: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

    def start_example(self, i: int, label_index: int) -> None:
        key: Tuple[int, int] = (self.examples[i].start, self.examples[i].end)
        self.groups[label_index].add(key)

    def finish(self) -> List[Set[Tuple[int, int]]]:
        return [g for g in self.groups.values() if len(g) >= 2]

class Examples:
    def __init__(self, record: ExampleRecord) -> None:
        self.trail: IntList = record.trail
        self.labels: List[int] = record.labels
        self.__length: int = self.trail.count(STOP_EXAMPLE_DISCARD_RECORD) + record.trail.count(STOP_EXAMPLE_NO_DISCARD_RECORD)
        self.__children: Optional[List[Union[IntList, Tuple[int, ...]]]] = None
        self.__example_record: ExampleRecord = record

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
    def discarded(self) -> frozenset:
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
    def children(self) -> List[Union[IntList, Tuple[int, ...]]]:
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
        n: int = self.__length
        if i < -n or i >= n:
            raise IndexError(f'Index {i} out of range [-{n}, {n})')
        if i < 0:
            i += n
        return Example(self, i)

    def __iter__(self) -> Iterator[Example]:
        for i in range(len(self)):
            yield self[i]

class _Overrun:
    status: Status = Status.OVERRUN

    def __repr__(self) -> str:
        return 'Overrun'

Overrun: _Overrun = _Overrun()
global_test_counter: int = 0
MAX_DEPTH: int = 100

class DataObserver:
    def conclude_test(self, status: Status, interesting_origin: Optional[Any]) -> None:
        pass

    def kill_branch(self) -> None:
        pass

    def draw_integer(self, value: int, *, kwargs: Any, was_forced: bool) -> None:
        pass

    def draw_float(self, value: float, *, kwargs: Any, was_forced: bool) -> None:
        pass

    def draw_string(self, value: str, *, kwargs: Any, was_forced: bool) -> None:
        pass

    def draw_bytes(self, value: bytes, *, kwargs: Any, was_forced: bool) -> None:
        pass

    def draw_boolean(self, value: bool, *, kwargs: Any, was_forced: bool) -> None:
        pass

@attr.s(slots=True)
class ConjectureResult:
    status: Status = attr.ib()
    interesting_origin: Optional[Any] = attr.ib()
    nodes: Tuple[ChoiceNode, ...] = attr.ib(eq=False, repr=False)
    length: int = attr.ib()
    output: str = attr.ib()
    extra_information: Optional[ExtraInformation] = attr.ib()
    expected_exception: Optional[Exception] = attr.ib()
    expected_traceback: Optional[str] = attr.ib()
    has_discards: bool = attr.ib()
    target_observations: TargetObservations = attr.ib()
    tags: frozenset = attr.ib()
    examples: Examples = attr.ib(repr=False, eq=False)
    arg_slices: Set[Any] = attr.ib(repr=False)
    slice_comments: Any = attr.ib(repr=False)
    misaligned_at: Optional[MisalignedAt] = attr.ib(repr=False)

    def as_result(self) -> "ConjectureResult":
        return self

    @property
    def choices(self) -> Tuple[Any, ...]:
        return tuple((node.value for node in self.nodes))

class ConjectureData:
    @classmethod
    def for_choices(cls, choices: Any, *, observer: Optional[DataObserver] = None, provider: Type[HypothesisProvider] = HypothesisProvider, random: Optional[Random] = None) -> "ConjectureData":
        from hypothesis.internal.conjecture.engine import choice_count
        return cls(max_choices=choice_count(choices), random=random, prefix=choices, observer=observer, provider=provider)

    def __init__(
        self,
        *,
        random: Optional[Random],
        observer: Optional[DataObserver] = None,
        provider: Union[Type[HypothesisProvider], HypothesisProvider] = HypothesisProvider,
        prefix: Optional[Any] = None,
        max_choices: Optional[int] = None,
        provider_kw: Optional[Dict[str, Any]] = None,
    ) -> None:
        from hypothesis.internal.conjecture.engine import BUFFER_SIZE
        if observer is None:
            observer = DataObserver()
        if provider_kw is None:
            provider_kw = {}
        elif not isinstance(provider, type):
            raise InvalidArgument(f'Expected provider={provider!r} to be a class since provider_kw={provider_kw!r} was passed, but got an instance instead.')
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
        self.provider: PrimitiveProvider = provider(self, **provider_kw) if isinstance(provider, type) else provider  # type: ignore
        assert isinstance(self.provider, PrimitiveProvider)
        self.__result: Optional[ConjectureResult] = None
        self.target_observations: TargetObservations = {}
        self.tags: Set[Any] = set()
        self.labels_for_structure_stack: List[Set[int]] = []
        self.__examples: Optional[Examples] = None
        self.depth: int = -1
        self.__example_record: ExampleRecord = ExampleRecord()
        self.arg_slices: Set[Any] = set()
        self.slice_comments: Dict[Any, Any] = {}
        self._observability_args: Dict[str, Any] = {}
        self._observability_predicates: Dict[str, Dict[str, int]] = defaultdict(lambda: {'satisfied': 0, 'unsatisfied': 0})
        self._sampled_from_all_strategies_elements_message: Optional[Any] = None
        self.expected_exception: Optional[Exception] = None
        self.expected_traceback: Optional[str] = None
        self.extra_information: ExtraInformation = ExtraInformation()
        self.prefix: Optional[Any] = prefix
        self.nodes: Tuple[ChoiceNode, ...] = ()
        self.misaligned_at: Optional[MisalignedAt] = None
        self.start_example(TOP_LABEL)

    def __repr__(self) -> str:
        return 'ConjectureData(%s, %d choices%s)' % (self.status.name, len(self.nodes), ', frozen' if self.frozen else '')

    @property
    def choices(self) -> Tuple[Any, ...]:
        return tuple((node.value for node in self.nodes))

    def _draw(self, choice_type: str, kwargs: Dict[str, Any], *, observe: bool, forced: Optional[Any]) -> Any:
        if self.length == self.max_length:
            debug_report(f'overrun because hit self.max_length={self.max_length!r}')
            self.mark_overrun()
        if len(self.nodes) == (self.max_choices if self.max_choices is not None else 0):
            debug_report(f'overrun because hit self.max_choices={self.max_choices!r}')
            self.mark_overrun()
        if observe and self.prefix is not None and (self.index < len(self.prefix)):
            value: Any = self._pop_choice(choice_type, kwargs, forced=forced)
        elif forced is None:
            value = getattr(self.provider, f'draw_{choice_type}')(**kwargs)
        if forced is not None:
            value = forced
        if choice_type == 'float' and math.isnan(value):
            value = int_to_float(float_to_int(value))
        if observe:
            was_forced: bool = forced is not None
            getattr(self.observer, f'draw_{choice_type}')(value, kwargs=kwargs, was_forced=was_forced)
            size: int = 0 if self.provider.avoid_realization else choices_size([value])
            if self.length + size > self.max_length:
                debug_report(f'overrun because self.length={self.length!r} + size={size!r} > self.max_length={self.max_length!r}')
                self.mark_overrun()
            node = ChoiceNode(type=choice_type, value=value, kwargs=kwargs, was_forced=was_forced, index=len(self.nodes))
            self.__example_record.record_choice()
            self.nodes += (node,)
            self.length += size
        return value

    def draw_integer(
        self,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        *,
        weights: Optional[Any] = None,
        shrink_towards: int = 0,
        forced: Optional[int] = None,
        observe: bool = True
    ) -> int:
        if weights is not None:
            assert min_value is not None
            assert max_value is not None
            assert len(weights) <= 255
            assert sum(weights.values()) < 1
            assert all((w != 0 for w in weights.values()))
        if forced is not None and min_value is not None:
            assert min_value <= forced
        if forced is not None and max_value is not None:
            assert forced <= max_value
        kwargs: Dict[str, Any] = self._pooled_kwargs('integer', {'min_value': min_value, 'max_value': max_value, 'weights': weights, 'shrink_towards': shrink_towards})
        result: Any = self._draw('integer', kwargs, observe=observe, forced=forced)
        return result

    def draw_float(
        self,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        *,
        allow_nan: bool = True,
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
        forced: Optional[float] = None,
        observe: bool = True
    ) -> float:
        assert smallest_nonzero_magnitude > 0
        assert not math.isnan(min_value)
        assert not math.isnan(max_value)
        if forced is not None:
            assert allow_nan or not math.isnan(forced)
            assert math.isnan(forced) or (sign_aware_lte(min_value, forced) and sign_aware_lte(forced, max_value))
        kwargs: Dict[str, Any] = self._pooled_kwargs('float', {'min_value': min_value, 'max_value': max_value, 'allow_nan': allow_nan, 'smallest_nonzero_magnitude': smallest_nonzero_magnitude})
        result: Any = self._draw('float', kwargs, observe=observe, forced=forced)
        return result

    def draw_string(
        self,
        intervals: Any,
        *,
        min_size: int = 0,
        max_size: int = COLLECTION_DEFAULT_MAX_SIZE,
        forced: Optional[str] = None,
        observe: bool = True
    ) -> str:
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0
        if len(intervals) == 0:
            assert min_size == 0
        kwargs: Dict[str, Any] = self._pooled_kwargs('string', {'intervals': intervals, 'min_size': min_size, 'max_size': max_size})
        result: Any = self._draw('string', kwargs, observe=observe, forced=forced)
        return result

    def draw_bytes(
        self,
        min_size: int = 0,
        max_size: int = COLLECTION_DEFAULT_MAX_SIZE,
        *,
        forced: Optional[bytes] = None,
        observe: bool = True
    ) -> bytes:
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0
        kwargs: Dict[str, Any] = self._pooled_kwargs('bytes', {'min_size': min_size, 'max_size': max_size})
        result: Any = self._draw('bytes', kwargs, observe=observe, forced=forced)
        return result

    def draw_boolean(
        self,
        p: float = 0.5,
        *,
        forced: Optional[bool] = None,
        observe: bool = True
    ) -> bool:
        assert forced is not True or p > 0
        assert forced is not False or p < 1
        kwargs: Dict[str, Any] = self._pooled_kwargs('boolean', {'p': p})
        result: Any = self._draw('boolean', kwargs, observe=observe, forced=forced)
        return result

    def _pooled_kwargs(self, choice_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider.avoid_realization:
            return kwargs
        key: Tuple[Any, ...] = (choice_type, *choice_kwargs_key(choice_type, kwargs))
        try:
            return POOLED_KWARGS_CACHE[key]
        except KeyError:
            POOLED_KWARGS_CACHE[key] = kwargs
            return kwargs

    def _pop_choice(self, choice_type: str, kwargs: Dict[str, Any], *, forced: Optional[Any]) -> Any:
        assert self.prefix is not None
        assert self.index < len(self.prefix)
        value: Any = self.prefix[self.index]
        if isinstance(value, ChoiceTemplate):
            node: ChoiceTemplate = value
            if node.count is not None:
                assert node.count >= 0
            assert self.index == len(self.prefix) - 1
            if node.type == 'simplest':
                if isinstance(self.provider, HypothesisProvider):
                    try:
                        choice = choice_from_index(0, choice_type, kwargs)
                    except ChoiceTooLarge:
                        self.mark_overrun()
                else:
                    choice = getattr(self.provider, f'draw_{choice_type}')(**kwargs)
            else:
                raise NotImplementedError
            if node.count is not None:
                node.count -= 1
                if node.count < 0:
                    self.mark_overrun()
            return choice
        choice = value
        node_choice_type: str = {str: 'string', float: 'float', int: 'integer', bool: 'boolean', bytes: 'bytes'}[type(choice)]
        if node_choice_type != choice_type or not choice_permitted(choice, kwargs):
            if self.misaligned_at is None:
                self.misaligned_at = (self.index, choice_type, kwargs, forced)
            try:
                choice = choice_from_index(0, choice_type, kwargs)
            except ChoiceTooLarge:
                self.mark_overrun()
        self.index += 1
        return choice

    def as_result(self) -> Union[ConjectureResult, _Overrun]:
        assert self.frozen
        if self.status == Status.OVERRUN:
            return Overrun
        if self.__result is None:
            self.__result = ConjectureResult(
                status=self.status,
                interesting_origin=self.interesting_origin,
                examples=self.examples,
                nodes=self.nodes,
                length=self.length,
                output=self.output,
                expected_traceback=self.expected_traceback,
                expected_exception=self.expected_exception,
                extra_information=self.extra_information if self.extra_information.has_information() else None,
                has_discards=self.has_discards,
                target_observations=self.target_observations,
                tags=frozenset(self.tags),
                arg_slices=self.arg_slices,
                slice_comments=self.slice_comments,
                misaligned_at=self.misaligned_at,
            )
            assert self.__result is not None
        return self.__result

    def __assert_not_frozen(self, name: str) -> None:
        if self.frozen:
            raise Frozen(f'Cannot call {name} on frozen ConjectureData')

    def note(self, value: Any) -> None:
        self.__assert_not_frozen('note')
        if not isinstance(value, str):
            value = repr(value)
        self.output += value

    def draw(self, strategy: Any, label: Optional[int] = None, observe_as: Optional[str] = None) -> Any:
        from hypothesis.internal.observability import TESTCASE_CALLBACKS
        from hypothesis.strategies._internal.utils import to_jsonable
        if self.is_find and (not strategy.supports_find):
            raise InvalidArgument(f'Cannot use strategy {strategy!r} within a call to find (presumably because it would be invalid after the call had ended).')
        at_top_level: bool = self.depth == 0
        start_time: Optional[float] = None
        if at_top_level:
            start_time = time.perf_counter()
            _ = gc_cumulative_time()
        strategy.validate()
        if strategy.is_empty:
            self.mark_invalid(f'empty strategy {self!r}')
        if self.depth >= MAX_DEPTH:
            self.mark_invalid('max depth exceeded')
        if label is None:
            assert isinstance(strategy.label, int)
            label = strategy.label
        self.start_example(label=label)
        try:
            if not at_top_level:
                return strategy.do_draw(self)
            assert start_time is not None
            key: str = observe_as or f'generate:unlabeled_{len(self.draw_times)}'
            try:
                strategy.validate()
                try:
                    v = strategy.do_draw(self)
                finally:
                    in_gctime: float = gc_cumulative_time() - _
                    self.draw_times[key] = time.perf_counter() - start_time - in_gctime
            except Exception as err:
                add_note(err, f'while generating {key.removeprefix("generate:")!r} from {strategy!r}')
                raise
            if TESTCASE_CALLBACKS:
                self._observability_args[key] = to_jsonable(v)
            return v
        finally:
            self.stop_example()

    def start_example(self, label: int) -> None:
        self.provider.span_start(label)
        self.__assert_not_frozen('start_example')
        self.depth += 1
        if self.depth > self.max_depth:
            self.max_depth = self.depth
        self.__example_record.start_example(label)
        self.labels_for_structure_stack.append({label})

    def stop_example(self, *, discard: bool = False) -> None:
        self.provider.span_end(discard)
        if self.frozen:
            return
        if discard:
            self.has_discards = True
        self.depth -= 1
        assert self.depth >= -1
        self.__example_record.stop_example(discard=discard)
        labels_for_structure: Set[int] = self.labels_for_structure_stack.pop()
        if not discard:
            if self.labels_for_structure_stack:
                self.labels_for_structure_stack[-1].update(labels_for_structure)
            else:
                self.tags.update([structural_coverage(l) for l in labels_for_structure])
        if discard:
            self.observer.kill_branch()

    @property
    def examples(self) -> Examples:
        assert self.frozen
        if self.__examples is None:
            self.__examples = Examples(record=self.__example_record)
        return self.__examples

    def freeze(self) -> None:
        if self.frozen:
            return
        self.finish_time: float = time.perf_counter()
        self.gc_finish_time: float = gc_cumulative_time()
        while self.depth >= 0:
            self.stop_example()
        self.__example_record.freeze()
        self.frozen = True
        self.observer.conclude_test(self.status, self.interesting_origin)

    def choice(self, values: Seq[Any], *, forced: Optional[Any] = None, observe: bool = True) -> Any:
        forced_i: Optional[int] = None if forced is None else values.index(forced)
        i: int = self.draw_integer(0, len(values) - 1, forced=forced_i, observe=observe)
        return values[i]

    def conclude_test(self, status: Status, interesting_origin: Optional[Any] = None) -> NoReturn:
        assert interesting_origin is None or status == Status.INTERESTING
        self.__assert_not_frozen('conclude_test')
        self.interesting_origin = interesting_origin
        self.status = status
        self.freeze()
        raise StopTest(self.testcounter)

    def mark_interesting(self, interesting_origin: Optional[Any] = None) -> NoReturn:
        self.conclude_test(Status.INTERESTING, interesting_origin)

    def mark_invalid(self, why: Optional[str] = None) -> NoReturn:
        if why is not None:
            self.events['invalid because'] = why
        self.conclude_test(Status.INVALID)

    def mark_overrun(self) -> NoReturn:
        self.conclude_test(Status.OVERRUN)

def draw_choice(choice_type: str, kwargs: Dict[str, Any], *, random: Random) -> Any:
    cd: ConjectureData = ConjectureData(random=random)
    return getattr(cd.provider, f'draw_{choice_type}')(**kwargs)