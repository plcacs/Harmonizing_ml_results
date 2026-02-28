import inspect
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    NoReturn,
    Optional,
    Union,
    Type,
    TypeVar,
    Generic,
    Dict,
    List,
    Tuple,
    DefaultDict,
    Sequence,
    ContextManager,
    Set,
    FrozenSet,
    Iterator,
    Iterable,
    TypeAlias,
)

from hypothesis import Verbosity, settings
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.observability import TESTCASE_CALLBACKS
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.reporting import report, verbose_report
from hypothesis.utils.dynamicvariables import DynamicVariable
from hypothesis.vendor.pretty import IDKey, PrettyPrintFunction, pretty

T = TypeVar('T')

class RandomSeeder(Generic[T]):
    def __init__(self, seed: T) -> None:
        self.seed = seed

    def __repr__(self) -> str:
        return f'RandomSeeder({self.seed!r})'

class _Checker:
    def __init__(self) -> None:
        self.saw_global_random = False

    def __call__(self, x: Any) -> Any:
        self.saw_global_random |= isinstance(x, RandomSeeder)
        return x

@contextmanager
def deprecate_random_in_strategy(fmt: str, *args: Any) -> Iterator[_Checker]:
    _global_rand_state = random.getstate()
    yield (checker := _Checker())
    if _global_rand_state != random.getstate() and (not checker.saw_global_random):
        note_deprecation('Do not use the `random` module inside strategies; instead consider  `st.randoms()`, `st.sampled_from()`, etc.  ' + fmt.format(*args), since='2024-02-05', has_codemod=False, stacklevel=1)

class BuildContext:
    def __init__(self, data: ConjectureData, *, is_final: bool = False, close_on_capture: bool = True) -> None:
        self.data = data
        self.tasks: List[Callable[[], None]] = []
        self.is_final = is_final
        self.close_on_capture = close_on_capture
        self.close_on_del = False
        self.known_object_printers: DefaultDict[IDKey, List[Callable[[Any, PrettyPrintFunction, int, *, _func: Callable[..., Any]], str]]] = defaultdict(list)

    def record_call(self, obj: Any, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.known_object_printers[IDKey(obj)].append(lambda obj, p, cycle, *, _func=func: p.maybe_repr_known_object_as_call(obj, cycle, get_pretty_function_description(_func), args, kwargs))

    def prep_args_kwargs_from_strategies(self, kwarg_strategies: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]:
        arg_labels = {}
        kwargs = {}
        for k, s in kwarg_strategies.items():
            start_idx = len(self.data.nodes)
            with deprecate_random_in_strategy('from {}={!r}', k, s) as check:
                obj = check(self.data.draw(s, observe_as=f'generate:{k}'))
            end_idx = len(self.data.nodes)
            kwargs[k] = obj
            if start_idx != end_idx:
                arg_labels[k] = (start_idx, end_idx)
                self.data.arg_slices.add((start_idx, end_idx))
        return (kwargs, arg_labels)

    def __enter__(self) -> 'BuildContext':
        self.assign_variable = _current_build_context.with_value(self)
        self.assign_variable.__enter__()
        return self

    def __exit__(self, exc_type: Type[BaseException], exc_value: BaseException, tb: Any) -> None:
        self.assign_variable.__exit__(exc_type, exc_value, tb)
        errors = []
        for task in self.tasks:
            try:
                task()
            except BaseException as err:
                errors.append(err)
        if errors:
            if len(errors) == 1:
                raise errors[0] from exc_value
            raise BaseExceptionGroup('Cleanup failed', errors) from exc_value

def cleanup(teardown: Callable[[], None]) -> None:
    """Register a function to be called when the current test has finished
    executing. Any exceptions thrown in teardown will be printed but not
    rethrown.

    Inside a test this isn't very interesting, because you can just use
    a finally block, but note that you can use this inside map, flatmap,
    etc. in order to e.g. insist that a value is closed at the end.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('Cannot register cleanup outside of build context')
    context.tasks.append(teardown)

def should_note() -> bool:
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('Cannot make notes outside of a test')
    return context.is_final or settings.default.verbosity >= Verbosity.verbose

def note(value: Any) -> None:
    """Report this value for the minimal failing example."""
    if should_note():
        if not isinstance(value, str):
            value = pretty(value)
        report(value)

def event(value: Any, payload: str = '') -> None:
    """Record an event that occurred during this test. Statistics on the number of test
    runs with each event will be reported at the end if you run Hypothesis in
    statistics reporting mode.

    Event values should be strings or convertible to them.  If an optional
    payload is given, it will be included in the string for :ref:`statistics`.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('Cannot make record events outside of a test')
    payload = _event_to_string(payload, (str, int, float))
    context.data.events[_event_to_string(value)] = payload
_events_to_strings: Dict[Any, str] = {}

def _event_to_string(event: Any, allowed_types: Type[Union[str, int, float]]) -> str:
    if isinstance(event, allowed_types):
        return event
    try:
        return _events_to_strings[event]
    except (KeyError, TypeError):
        pass
    result = str(event)
    try:
        _events_to_strings[event] = result
    except TypeError:
        pass
    return result

def target(observation: Union[int, float], *, label: str = '') -> Union[int, float]:
    """Calling this function with an ``int`` or ``float`` observation gives it feedback
    with which to guide our search for inputs that will cause an error, in
    addition to all the usual heuristics.  Observations must always be finite.

    Hypothesis will try to maximize the observed value over several examples;
    almost any metric will work so long as it makes sense to increase it.
    For example, ``-abs(error)`` is a metric that increases as ``error``
    approaches zero.

    Example metrics:

    - Number of elements in a collection, or tasks in a queue
    - Mean or maximum runtime of a task (or both, if you use ``label``)
    - Compression ratio for data (perhaps per-algorithm or per-level)
    - Number of steps taken by a state machine

    The optional ``label`` argument can be used to distinguish between
    and therefore separately optimise distinct observations, such as the
    mean and standard deviation of a dataset.  It is an error to call
    ``target()`` with any label more than once per test case.

    .. note::
        **The more examples you run, the better this technique works.**

        As a rule of thumb, the targeting effect is noticeable above
        :obj:`max_examples=1000 <hypothesis.settings.max_examples>`,
        and immediately obvious by around ten thousand examples
        *per label* used by your test.

    :ref:`statistics` include the best score seen for each label,
    which can help avoid `the threshold problem
    <https://hypothesis.works/articles/threshold-problem/>`__ when the minimal
    example shrinks right down to the threshold of failure (:issue:`2180`).
    """
    check_type((int, float), observation, 'observation')
    if not math.isfinite(observation):
        raise InvalidArgument(f'observation={observation!r} must be a finite float.')
    check_type(str, label, 'label')
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('Calling target() outside of a test is invalid.  Consider guarding this call with `if currently_in_test_context(): ...`')
    elif context.data.provider.avoid_realization:
        return observation
    verbose_report(f'Saw target({observation!r}, label={label!r})')
    if label in context.data.target_observations:
        raise InvalidArgument(f'Calling target({observation!r}, label={label!r}) would overwrite target({context.data.target_observations[label]!r}, label={label!r})')
    else:
        context.data.target_observations[label] = observation
    return observation
