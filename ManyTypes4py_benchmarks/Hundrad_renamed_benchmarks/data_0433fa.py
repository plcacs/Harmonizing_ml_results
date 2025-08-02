import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum
from functools import cached_property
from random import Random
from typing import TYPE_CHECKING, Any, NoReturn, Optional, TypeVar, Union
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

def __getattr__(name):
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

    def __repr__(self):
        return 'ExtraInformation({})'.format(', '.join((f'{k}={v!r}' for k, v in self.__dict__.items())))

    def has_information(self):
        return bool(self.__dict__)

class Status(IntEnum):
    OVERRUN = 0
    INVALID = 1
    VALID = 2
    INTERESTING = 3

    def __repr__(self):
        return f'Status.{self.name}'

@attr.s(slots=True, frozen=True)
class StructuralCoverageTag:
    label = attr.ib()
STRUCTURAL_COVERAGE_CACHE = {}

def structural_coverage(label):
    try:
        return STRUCTURAL_COVERAGE_CACHE[label]
    except KeyError:
        return STRUCTURAL_COVERAGE_CACHE.setdefault(label, StructuralCoverageTag(label))
POOLED_KWARGS_CACHE = LRUCache(4096)

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

    def __init__(self, owner, index):
        self.owner = owner
        self.index = index

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is other.owner and self.index == other.index

    def __ne__(self, other):
        if self is other:
            return False
        if not isinstance(other, Example):
            return NotImplemented
        return self.owner is not other.owner or self.index != other.index

    def __repr__(self):
        return f'examples[{self.index}]'

    @property
    def label(self):
        """A label is an opaque value that associates each example with its
        approximate origin, such as a particular strategy class or a particular
        kind of draw."""
        return self.owner.labels[self.owner.label_indices[self.index]]

    @property
    def parent(self):
        """The index of the example that this one is nested directly within."""
        if self.index == 0:
            return None
        return self.owner.parentage[self.index]

    @property
    def start(self):
        return self.owner.starts[self.index]

    @property
    def end(self):
        return self.owner.ends[self.index]

    @property
    def depth(self):
        """Depth of this example in the example tree. The top-level example has a
        depth of 0."""
        return self.owner.depths[self.index]

    @property
    def discarded(self):
        """True if this is example's ``stop_example`` call had ``discard`` set to
        ``True``. This means we believe that the shrinker should be able to delete
        this example completely, without affecting the value produced by its enclosing
        strategy. Typically set when a rejection sampler decides to reject a
        generated value and try again."""
        return self.index in self.owner.discarded

    @property
    def choice_count(self):
        """The number of choices in this example."""
        return self.end - self.start

    @property
    def children(self):
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

    def __init__(self, examples):
        self.example_stack = []
        self.examples = examples
        self.example_count = 0
        self.choice_count = 0

    def run(self):
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

    def __push(self, label_index):
        i = self.example_count
        assert i < len(self.examples)
        self.start_example(i, label_index=label_index)
        self.example_count += 1
        self.example_stack.append(i)

    def __pop(self, *, discarded):
        i = self.example_stack.pop()
        self.stop_example(i, discarded=discarded)

    def start_example(self, i, label_index):
        """Called at the start of each example, with ``i`` the
        index of the example and ``label_index`` the index of
        its label in ``self.examples.labels``."""

    def stop_example(self, i, *, discarded):
        """Called at the end of each example, with ``i`` the
        index of the example and ``discarded`` being ``True`` if ``stop_example``
        was called with ``discard=True``."""

    def finish(self):
        raise NotImplementedError
STOP_EXAMPLE_DISCARD_RECORD = 1
STOP_EXAMPLE_NO_DISCARD_RECORD = 2
START_EXAMPLE_RECORD = 3
CHOICE_RECORD = calc_label_from_name('ir draw record')

class ExampleRecord:
    """Records the series of ``start_example``, ``stop_example``, and
    ``draw_bits`` calls so that these may be stored in ``Examples`` and
    replayed when we need to know about the structure of individual
    ``Example`` objects.

    Note that there is significant similarity between this class and
    ``DataObserver``, and the plan is to eventually unify them, but
    they currently have slightly different functions and implementations.
    """

    def __init__(self):
        self.labels = []
        self.__index_of_labels = {}
        self.trail = IntList()
        self.nodes = []

    def freeze(self):
        self.__index_of_labels = None

    def record_choice(self):
        self.trail.append(CHOICE_RECORD)

    def start_example(self, label):
        assert self.__index_of_labels is not None
        try:
            i = self.__index_of_labels[label]
        except KeyError:
            i = self.__index_of_labels.setdefault(label, len(self.labels))
            self.labels.append(label)
        self.trail.append(START_EXAMPLE_RECORD + i)

    def stop_example(self, *, discard):
        if discard:
            self.trail.append(STOP_EXAMPLE_DISCARD_RECORD)
        else:
            self.trail.append(STOP_EXAMPLE_NO_DISCARD_RECORD)

class _starts_and_ends(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.starts = IntList.of_length(len(self.examples))
        self.ends = IntList.of_length(len(self.examples))

    def start_example(self, i, label_index):
        self.starts[i] = self.choice_count

    def stop_example(self, i, *, discarded):
        self.ends[i] = self.choice_count

    def finish(self):
        return (self.starts, self.ends)

class _discarded(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.result = set()

    def finish(self):
        return frozenset(self.result)

    def stop_example(self, i, *, discarded):
        if discarded:
            self.result.add(i)

class _parentage(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def stop_example(self, i, *, discarded):
        if i > 0:
            self.result[i] = self.example_stack[-1]

    def finish(self):
        return self.result

class _depths(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i, label_index):
        self.result[i] = len(self.example_stack)

    def finish(self):
        return self.result

class _label_indices(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i, label_index):
        self.result[i] = label_index

    def finish(self):
        return self.result

class _mutator_groups(ExampleProperty):

    def __init__(self, examples):
        super().__init__(examples)
        self.groups = defaultdict(set)

    def start_example(self, i, label_index):
        key = (self.examples[i].start, self.examples[i].end)
        self.groups[label_index].add(key)

    def finish(self):
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

    def __init__(self, record):
        self.trail = record.trail
        self.labels = record.labels
        self.__length = self.trail.count(STOP_EXAMPLE_DISCARD_RECORD) + record.trail.count(STOP_EXAMPLE_NO_DISCARD_RECORD)
        self.__children = None

    @cached_property
    def starts_and_ends(self):
        return _starts_and_ends(self).run()

    @property
    def starts(self):
        return self.starts_and_ends[0]

    @property
    def ends(self):
        return self.starts_and_ends[1]

    @cached_property
    def discarded(self):
        return _discarded(self).run()

    @cached_property
    def parentage(self):
        return _parentage(self).run()

    @cached_property
    def depths(self):
        return _depths(self).run()

    @cached_property
    def label_indices(self):
        return _label_indices(self).run()

    @cached_property
    def mutator_groups(self):
        return _mutator_groups(self).run()

    @property
    def children(self):
        if self.__children is None:
            children = [IntList() for _ in range(len(self))]
            for i, p in enumerate(self.parentage):
                if i > 0:
                    children[p].append(i)
            for i, c in enumerate(children):
                if not c:
                    children[i] = ()
            self.__children = children
        return self.__children

    def __len__(self):
        return self.__length

    def __getitem__(self, i):
        n = self.__length
        if i < -n or i >= n:
            raise IndexError(f'Index {i} out of range [-{n}, {n})')
        if i < 0:
            i += n
        return Example(self, i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class _Overrun:
    status = Status.OVERRUN

    def __repr__(self):
        return 'Overrun'
Overrun = _Overrun()
global_test_counter = 0
MAX_DEPTH = 100

class DataObserver:
    """Observer class for recording the behaviour of a
    ConjectureData object, primarily used for tracking
    the behaviour in the tree cache."""

    def conclude_test(self, status, interesting_origin):
        """Called when ``conclude_test`` is called on the
        observed ``ConjectureData``, with the same arguments.

        Note that this is called after ``freeze`` has completed.
        """

    def kill_branch(self):
        """Mark this part of the tree as not worth re-exploring."""

    def draw_integer(self, value, *, kwargs, was_forced):
        pass

    def draw_float(self, value, *, kwargs, was_forced):
        pass

    def draw_string(self, value, *, kwargs, was_forced):
        pass

    def draw_bytes(self, value, *, kwargs, was_forced):
        pass

    def draw_boolean(self, value, *, kwargs, was_forced):
        pass

@attr.s(slots=True)
class ConjectureResult:
    """Result class storing the parts of ConjectureData that we
    will care about after the original ConjectureData has outlived its
    usefulness."""
    status = attr.ib()
    interesting_origin = attr.ib()
    nodes = attr.ib(eq=False, repr=False)
    length = attr.ib()
    output = attr.ib()
    extra_information = attr.ib()
    expected_exception = attr.ib()
    expected_traceback = attr.ib()
    has_discards = attr.ib()
    target_observations = attr.ib()
    tags = attr.ib()
    examples = attr.ib(repr=False, eq=False)
    arg_slices = attr.ib(repr=False)
    slice_comments = attr.ib(repr=False)
    misaligned_at = attr.ib(repr=False)

    def as_result(self):
        return self

    @property
    def choices(self):
        return tuple((node.value for node in self.nodes))

class ConjectureData:

    @classmethod
    def for_choices(cls, choices, *, observer=None, provider=HypothesisProvider, random=None):
        from hypothesis.internal.conjecture.engine import choice_count
        return cls(max_choices=choice_count(choices), random=random, prefix=choices, observer=observer, provider=provider)

    def __init__(self, *, random, observer=None, provider=HypothesisProvider, prefix=None, max_choices=None, provider_kw=None):
        from hypothesis.internal.conjecture.engine import BUFFER_SIZE
        if observer is None:
            observer = DataObserver()
        if provider_kw is None:
            provider_kw = {}
        elif not isinstance(provider, type):
            raise InvalidArgument(f'Expected provider={provider!r} to be a class since provider_kw={provider_kw!r} was passed, but got an instance instead.')
        assert isinstance(observer, DataObserver)
        self.observer = observer
        self.max_choices = max_choices
        self.max_length = BUFFER_SIZE
        self.is_find = False
        self.overdraw = 0
        self._random = random
        self.length = 0
        self.index = 0
        self.output = ''
        self.status = Status.VALID
        self.frozen = False
        global global_test_counter
        self.testcounter = global_test_counter
        global_test_counter += 1
        self.start_time = time.perf_counter()
        self.gc_start_time = gc_cumulative_time()
        self.events = {}
        self.interesting_origin = None
        self.draw_times = {}
        self._stateful_run_times = defaultdict(float)
        self.max_depth = 0
        self.has_discards = False
        self.provider = provider(self, **provider_kw) if isinstance(provider, type) else provider
        assert isinstance(self.provider, PrimitiveProvider)
        self.__result = None
        self.target_observations = {}
        self.tags = set()
        self.labels_for_structure_stack = []
        self.__examples = None
        self.depth = -1
        self.__example_record = ExampleRecord()
        self.arg_slices = set()
        self.slice_comments = {}
        self._observability_args = {}
        self._observability_predicates = defaultdict(lambda: {'satisfied': 0, 'unsatisfied': 0})
        self._sampled_from_all_strategies_elements_message = None
        self.expected_exception = None
        self.expected_traceback = None
        self.extra_information = ExtraInformation()
        self.prefix = prefix
        self.nodes = ()
        self.misaligned_at = None
        self.start_example(TOP_LABEL)

    def __repr__(self):
        return 'ConjectureData(%s, %d choices%s)' % (self.status.name, len(self.nodes), ', frozen' if self.frozen else '')

    @property
    def choices(self):
        return tuple((node.value for node in self.nodes))

    def _draw(self, choice_type, kwargs, *, observe, forced):
        if self.length == self.max_length:
            debug_report(f'overrun because hit self.max_length={self.max_length!r}')
            self.mark_overrun()
        if len(self.nodes) == self.max_choices:
            debug_report(f'overrun because hit self.max_choices={self.max_choices!r}')
            self.mark_overrun()
        if observe and self.prefix is not None and (self.index < len(self.prefix)):
            value = self._pop_choice(choice_type, kwargs, forced=forced)
        elif forced is None:
            value = getattr(self.provider, f'draw_{choice_type}')(**kwargs)
        if forced is not None:
            value = forced
        if choice_type == 'float' and math.isnan(value):
            value = int_to_float(float_to_int(value))
        if observe:
            was_forced = forced is not None
            getattr(self.observer, f'draw_{choice_type}')(value, kwargs=kwargs, was_forced=was_forced)
            size = 0 if self.provider.avoid_realization else choices_size([value])
            if self.length + size > self.max_length:
                debug_report(f'overrun because self.length={self.length!r} + size={size!r} > self.max_length={self.max_length!r}')
                self.mark_overrun()
            node = ChoiceNode(type=choice_type, value=value, kwargs=kwargs, was_forced=was_forced, index=len(self.nodes))
            self.__example_record.record_choice()
            self.nodes += (node,)
            self.length += size
        return value

    def draw_integer(self, min_value=None, max_value=None, *, weights=None, shrink_towards=0, forced=None, observe=True):
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
        kwargs = self._pooled_kwargs('integer', {'min_value': min_value, 'max_value': max_value, 'weights': weights, 'shrink_towards': shrink_towards})
        return self._draw('integer', kwargs, observe=observe, forced=forced)

    def draw_float(self, min_value=-math.inf, max_value=math.inf, *, allow_nan=True, smallest_nonzero_magnitude=SMALLEST_SUBNORMAL, forced=None, observe=True):
        assert smallest_nonzero_magnitude > 0
        assert not math.isnan(min_value)
        assert not math.isnan(max_value)
        if forced is not None:
            assert allow_nan or not math.isnan(forced)
            assert math.isnan(forced) or (sign_aware_lte(min_value, forced) and sign_aware_lte(forced, max_value))
        kwargs = self._pooled_kwargs('float', {'min_value': min_value, 'max_value': max_value, 'allow_nan': allow_nan, 'smallest_nonzero_magnitude': smallest_nonzero_magnitude})
        return self._draw('float', kwargs, observe=observe, forced=forced)

    def draw_string(self, intervals, *, min_size=0, max_size=COLLECTION_DEFAULT_MAX_SIZE, forced=None, observe=True):
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0
        if len(intervals) == 0:
            assert min_size == 0
        kwargs = self._pooled_kwargs('string', {'intervals': intervals, 'min_size': min_size, 'max_size': max_size})
        return self._draw('string', kwargs, observe=observe, forced=forced)

    def draw_bytes(self, min_size=0, max_size=COLLECTION_DEFAULT_MAX_SIZE, *, forced=None, observe=True):
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0
        kwargs = self._pooled_kwargs('bytes', {'min_size': min_size, 'max_size': max_size})
        return self._draw('bytes', kwargs, observe=observe, forced=forced)

    def draw_boolean(self, p=0.5, *, forced=None, observe=True):
        assert forced is not True or p > 0
        assert forced is not False or p < 1
        kwargs = self._pooled_kwargs('boolean', {'p': p})
        return self._draw('boolean', kwargs, observe=observe, forced=forced)

    def _pooled_kwargs(self, choice_type, kwargs):
        """Memoize common dictionary objects to reduce memory pressure."""
        if self.provider.avoid_realization:
            return kwargs
        key = (choice_type, *choice_kwargs_key(choice_type, kwargs))
        try:
            return POOLED_KWARGS_CACHE[key]
        except KeyError:
            POOLED_KWARGS_CACHE[key] = kwargs
            return kwargs

    def _pop_choice(self, choice_type, kwargs, *, forced):
        assert self.prefix is not None
        assert self.index < len(self.prefix)
        value = self.prefix[self.index]
        if isinstance(value, ChoiceTemplate):
            node = value
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
        node_choice_type = {str: 'string', float: 'float', int: 'integer', bool: 'boolean', bytes: 'bytes'}[type(choice)]
        if node_choice_type != choice_type or not choice_permitted(choice, kwargs):
            if self.misaligned_at is None:
                self.misaligned_at = (self.index, choice_type, kwargs, forced)
            try:
                choice = choice_from_index(0, choice_type, kwargs)
            except ChoiceTooLarge:
                self.mark_overrun()
        self.index += 1
        return choice

    def as_result(self):
        """Convert the result of running this test into
        either an Overrun object or a ConjectureResult."""
        assert self.frozen
        if self.status == Status.OVERRUN:
            return Overrun
        if self.__result is None:
            self.__result = ConjectureResult(status=self.status, interesting_origin=self.interesting_origin, examples=self.examples, nodes=self.nodes, length=self.length, output=self.output, expected_traceback=self.expected_traceback, expected_exception=self.expected_exception, extra_information=self.extra_information if self.extra_information.has_information() else None, has_discards=self.has_discards, target_observations=self.target_observations, tags=frozenset(self.tags), arg_slices=self.arg_slices, slice_comments=self.slice_comments, misaligned_at=self.misaligned_at)
            assert self.__result is not None
        return self.__result

    def __assert_not_frozen(self, name):
        if self.frozen:
            raise Frozen(f'Cannot call {name} on frozen ConjectureData')

    def note(self, value):
        self.__assert_not_frozen('note')
        if not isinstance(value, str):
            value = repr(value)
        self.output += value

    def draw(self, strategy, label=None, observe_as=None):
        from hypothesis.internal.observability import TESTCASE_CALLBACKS
        from hypothesis.strategies._internal.utils import to_jsonable
        if self.is_find and (not strategy.supports_find):
            raise InvalidArgument(f'Cannot use strategy {strategy!r} within a call to find (presumably because it would be invalid after the call had ended).')
        at_top_level = self.depth == 0
        start_time = None
        if at_top_level:
            start_time = time.perf_counter()
            gc_start_time = gc_cumulative_time()
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
            key = observe_as or f'generate:unlabeled_{len(self.draw_times)}'
            try:
                strategy.validate()
                try:
                    v = strategy.do_draw(self)
                finally:
                    in_gctime = gc_cumulative_time() - gc_start_time
                    self.draw_times[key] = time.perf_counter() - start_time - in_gctime
            except Exception as err:
                add_note(err, f'while generating {key.removeprefix('generate:')!r} from {strategy!r}')
                raise
            if TESTCASE_CALLBACKS:
                self._observability_args[key] = to_jsonable(v)
            return v
        finally:
            self.stop_example()

    def start_example(self, label):
        self.provider.span_start(label)
        self.__assert_not_frozen('start_example')
        self.depth += 1
        if self.depth > self.max_depth:
            self.max_depth = self.depth
        self.__example_record.start_example(label)
        self.labels_for_structure_stack.append({label})

    def stop_example(self, *, discard=False):
        self.provider.span_end(discard)
        if self.frozen:
            return
        if discard:
            self.has_discards = True
        self.depth -= 1
        assert self.depth >= -1
        self.__example_record.stop_example(discard=discard)
        labels_for_structure = self.labels_for_structure_stack.pop()
        if not discard:
            if self.labels_for_structure_stack:
                self.labels_for_structure_stack[-1].update(labels_for_structure)
            else:
                self.tags.update([structural_coverage(l) for l in labels_for_structure])
        if discard:
            self.observer.kill_branch()

    @property
    def examples(self):
        assert self.frozen
        if self.__examples is None:
            self.__examples = Examples(record=self.__example_record)
        return self.__examples

    def freeze(self):
        if self.frozen:
            return
        self.finish_time = time.perf_counter()
        self.gc_finish_time = gc_cumulative_time()
        while self.depth >= 0:
            self.stop_example()
        self.__example_record.freeze()
        self.frozen = True
        self.observer.conclude_test(self.status, self.interesting_origin)

    def choice(self, values, *, forced=None, observe=True):
        forced_i = None if forced is None else values.index(forced)
        i = self.draw_integer(0, len(values) - 1, forced=forced_i, observe=observe)
        return values[i]

    def conclude_test(self, status, interesting_origin=None):
        assert interesting_origin is None or status == Status.INTERESTING
        self.__assert_not_frozen('conclude_test')
        self.interesting_origin = interesting_origin
        self.status = status
        self.freeze()
        raise StopTest(self.testcounter)

    def mark_interesting(self, interesting_origin=None):
        self.conclude_test(Status.INTERESTING, interesting_origin)

    def mark_invalid(self, why=None):
        if why is not None:
            self.events['invalid because'] = why
        self.conclude_test(Status.INVALID)

    def mark_overrun(self):
        self.conclude_test(Status.OVERRUN)

def draw_choice(choice_type, kwargs, *, random):
    cd = ConjectureData(random=random)
    return getattr(cd.provider, f'draw_{choice_type}')(**kwargs)