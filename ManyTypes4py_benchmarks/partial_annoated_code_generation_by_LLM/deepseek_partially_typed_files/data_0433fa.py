import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum
from functools import cached_property
from random import Random
from typing import TYPE_CHECKING, Any, NoReturn, Optional, TypeVar, Union, cast
import attr
from hypothesis.errors import ChoiceTooLarge, Frozen, InvalidArgument, StopTest
from hypothesis.internal.cache import LRUCache
from hypothesis.internal.compat import add_note
from hypothesis.internal.conjecture.choice import BooleanKWargs, BytesKWargs, ChoiceKwargsT, ChoiceNode, ChoiceT, ChoiceTemplate, ChoiceTypeT, FloatKWargs, IntegerKWargs, StringKWargs, choice_from_index, choice_kwargs_key, choice_permitted, choices_size
from hypothesis.internal.conjecture.junkdrawer import IntList, gc_cumulative_time
from hypothesis.internal.conjecture.providers import COLLECTION_DEFAULT_MAX_SIZE, HypothesisProvider, PrimitiveProvider
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
        note_deprecation('hypothesis.internal.conjecture.data.AVAILABLE_PROVIDERS has been moved to hypothesis.internal.conjecture.providers.AVAILABLE_PROVIDERS.', since='2025-01-25', has_codemod=False, stacklevel=1)
        return AVAILABLE_PROVIDERS
    raise AttributeError(f"Module 'hypothesis.internal.conjecture.data' has no attribute {name}")
T = TypeVar('T')
TargetObservations = dict[str, Union[int, float]]
MisalignedAt = tuple[int, ChoiceTypeT, ChoiceKwargsT, Optional[ChoiceT]]
TOP_LABEL = calc_label_from_name('top')

class ExtraInformation:
    """A class for holding shared state on a ``ConjectureData`` that should
    be added to the final ``ConjectureResult``."""

    def __repr__(self) -> str:
        return 'ExtraInformation({})'.format(', '.join((f'{k}={v!r}' for (k, v) in self.__dict__.items())))

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
STRUCTURAL_COVERAGE_CACHE: dict[int, StructuralCoverageTag] = {}

def structural_coverage(label: int) -> StructuralCoverageTag:
    try:
        return STRUCTURAL_COVERAGE_CACHE[label]
    except KeyError:
        return STRUCTURAL_COVERAGE_CACHE.setdefault(label, StructuralCoverageTag(label))
POOLED_KWARGS_CACHE: LRUCache = LRUCache(4096)

class Example:
    """Examples track the hierarchical structure of draws from the byte stream,
    within a single test run.

    Examples are created to mark regions of the byte stream that might be
    useful to the shrinker, such as:
    - The bytes used by a single draw from a strategy.
    - Useful groupings within a strategy, such as individual list elements.
    - Strategy-like helper functions that aren't first-class strategies.
    - Each lowest-level draw of bits or bytes from the byte stream.
    - A single top-level example that spans the entire input.

    Example-tracking allows the shrinker to try "high-level" transformations,
    such as rearranging or deleting the elements of a list, without having
    to understand their exact representation in the byte stream.

    Rather than store each ``Example`` as a rich object, it is actually
    just an index into the ``Examples`` class defined below. This has two
    purposes: Firstly, for most properties of examples we will never need
    to allocate storage at all, because most properties are not used on
    most examples. Secondly, by storing the properties as compact lists
    of integers, we save a considerable amount of space compared to
    Python's normal object size.

    This does have the downside that it increases the amount of allocation
    we do, and slows things down as a result, in some usage patterns because
    we repeatedly allocate the same Example or int objects, but it will
    often dramatically reduce our memory usage, so is worth it.
    """
    __slots__ = ('index', 'owner')

    def __init__(self, owner: 'Examples', index: int) -> None:
        self.owner = owner
        self.index = index

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is other.owner and self.index == other.index

    def __ne__(self, other: object) -> bool:
        if self is other:
            return False
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is not other.owner or self.index != other.index

    def __repr__(self) -> str:
        return f'examples[{self.index}]'

    @property
    def label(self) -> int:
        """A label is an opaque value that associates each example with its
        approximate origin, such as a particular strategy class or a particular
        kind of draw."""
        return self.owner.labels[self.owner.label_indices[self.index]]

    @property
    def parent(self) -> Optional[int]:
        """The index of the example that this one is nested directly within."""
        if self.index == 0:
            return None
        return self.owner.parentage[self.index]

    @property
    def start(self) -> int:
        return self.owner.starts[self.index]

    @property
    def end(self) -> int:
        return self.owner.ends[self.index]

    @property
    def depth(self) -> int:
        """Depth of this example in the example tree. The top-level example has a
        depth of 0."""
        return self.owner.depths[self.index]

    @property
    def discarded(self) -> bool:
        """True if this is example's ``stop_example`` call had ``discard`` set to
        ``True``. This means we believe that the shrinker should be able to delete
        this example completely, without affecting the value produced by its enclosing
        strategy. Typically set when a rejection sampler decides to reject a
        generated value and try again."""
        return self.index in self.owner.discarded

    @property
    def choice_count(self) -> int:
        """The number of choices in this example."""
        return self.end - self.start

    @property
    def children(self) -> 'list[Example]':
        """The list of all examples with this as a parent, in increasing index
        order."""
        return [self.owner[i] for i in self.owner.children[self.index]]

class ExampleProperty:
    """There are many properties of examples that we calculate by
    essentially rerunning the test case multiple times based on the
    calls which we record in ExampleRecord.

    This class defines a visitor, subclasses of which can be used
    to calculate these properties.
    """

    def __init__(self, examples: 'Examples') -> None:
        self.example_stack: list[int] = []
        self.examples = examples
        self.example_count = 0
        self.choice_count = 0

    def run(self) -> Any:
        """Rerun the test case with this visitor and return the
        results of ``self.finish()``."""
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
        """Called at the start of each example, with ``i`` the
        index of the example and ``label_index`` the index of
        its label in ``self.examples.labels``."""

    def stop_example(self, i: int, *, discarded: bool) -> None:
        """Called at the end of each example, with ``i`` the
        index of the example and ``discarded`` being ``True`` if ``stop_example``
        was called with ``discard=True``."""

    def finish(self) -> Any:
        raise NotImplementedError
STOP_EXAMPLE_DISCARD_RECORD: int = 1
STOP_EXAMPLE_NO_DISCARD_RECORD: int = 2
START_EXAMPLE_RECORD: int = 3
CHOICE_RECORD: int = calc_label_from_name('ir draw record')

class ExampleRecord:
    """Records the series of ``start_example``, ``stop_example``, and
    ``draw_bits`` calls so that these may be stored in ``Examples`` and
    replayed when we need to know about the structure of individual
    ``Example`` objects.

    Note that there is significant similarity between this class and
    ``DataObserver``, and the plan is to eventually unify them, but
    they currently have slightly different functions and implementations.
    """

    def __init__(self) -> None:
        self.labels: list[int] = []
        self.__index_of_labels: Optional[dict[int, int]] = {}
        self.trail = IntList()
        self.nodes: list[ChoiceNode] = []

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

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.starts = IntList.of_length(len(self.examples))
        self.ends = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.starts[i] = self.choice_count

    def stop_example(self, i: int, *, discarded: bool) -> None:
        self.ends[i] = self.choice_count

    def finish(self) -> tuple[IntList, IntList]:
        return (self.starts, self.ends)

class _discarded(ExampleProperty):

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.result: set[int] = set()

    def finish(self) -> frozenset[int]:
        return frozenset(self.result)

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if discarded:
            self.result.add(i)

class _parentage(ExampleProperty):

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def stop_example(self, i: int, *, discarded: bool) -> None:
        if i > 0:
            self.result[i] = self.example_stack[-1]

    def finish(self) -> IntList:
        return self.result

class _depths(ExampleProperty):

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = len(self.example_stack)

    def finish(self) -> IntList:
        return self.result

class _label_indices(ExampleProperty):

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        self.result[i] = label_index

    def finish(self) -> IntList:
        return self.result

class _mutator_groups(ExampleProperty):

    def __init__(self, examples: 'Examples') -> None:
        super().__init__(examples)
        self.groups: dict[int, set[tuple[int, int]]] = defaultdict(set)

    def start_example(self, i: int, label_index: int) -> None:
        key = (self.examples[i].start, self.examples[i].end)
        self.groups[label_index].add(key)

    def finish(self) -> Iterable[set[tuple[int, int]]]:
        return [g for g in self.groups.values() if len(g) >= 2]

class Examples:
    """A lazy collection of ``Example`` objects, derived from
    the record of recorded behaviour in ``ExampleRecord``.

    Behaves logically as if it were a list of ``Example`` objects,
    but actually mostly exists as a compact store of information
    for them to reference into. All properties on here are best
    understood as the backing storage for ``Example`` and are
    described there.
    """

    def __init__(self, record: ExampleRecord) -> None:
        self.trail = record.trail
        self.labels = record.labels
        self.__length = self.trail.count(STOP_EXAMPLE_DISCARD_RECORD) + record.trail.count(STOP_EXAMPLE_NO_DISCARD_RECORD)
        self.__children: Optional[list[Sequence[int]]] = None

    @cached_property
    def starts_and_ends(self) -> tuple[IntList, IntList]:
        return _starts_and_ends(self).run()

    @property
    def starts(self) -> IntList:
        return self.starts_and_ends[0]

    @property
    def ends(self) -> IntList:
        return self.starts_and_ends[1]

    @cached_property
    def discarded(self) -> frozenset[int]:
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
    def mutator_groups(self) -> list[set[tuple[int, int]]]:
        return _mutator_groups(self).run()

    @property
    def children(self) -> list[Sequence[int]]:
        if self.__children is None:
            children = [IntList() for _ in range(len(self))]
            for (i, p) in enumerate(self.parentage):
                if i > 0:
                    children[p].append(i)
            for (i, c) in enumerate(children):
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
    status: Status = Status.OVERRUN

    def __repr__(self) -> str:
        return 'Overrun'
Overrun: _Overrun = _Overrun()
global_test_counter: int = 0
MAX_DEPTH: int = 100

class DataObserver:
    """Observer class for recording the behaviour of a
    ConjectureData object, primarily used for tracking
    the behaviour in the tree cache."""

    def conclude_test(self, status: Status, interesting_origin: Optional[InterestingOrigin]) -> None:
        """Called when ``conclude_test`` is called on the
        observed ``ConjectureData``, with the same arguments.

        Note that this is called after ``freeze`` has completed.
        """

    def kill_branch(self) -> None:
        """Mark this part of the tree as not worth re-exploring."""

    def draw_integer(self, value: int, *, kwargs: IntegerKWargs, was_forced: bool) -> None:
        pass

    def draw_float(self, value: float, *, kwargs: FloatKWargs, was_forced: bool) -> None:
        pass

    def draw_string(self, value: str, *, kwargs: StringKWargs, was_forced: bool) -> None:
        pass

    def draw_bytes(self, value: bytes, *, kwargs: