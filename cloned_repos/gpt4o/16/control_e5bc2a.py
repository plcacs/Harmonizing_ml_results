import inspect
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, NoReturn, Optional, Union, Dict, List, Tuple
from weakref import WeakKeyDictionary
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

def _calling_function_location(what: str, frame: inspect.FrameInfo) -> str:
    where = frame.f_back
    return f'{what}() in {where.f_code.co_name} (line {where.f_lineno})'

def reject() -> NoReturn:
    if _current_build_context.value is None:
        note_deprecation('Using `reject` outside a property-based test is deprecated', since='2023-09-25', has_codemod=False)
    where = _calling_function_location('reject', inspect.currentframe())
    if currently_in_test_context():
        count = current_build_context().data._observability_predicates[where]
        count['unsatisfied'] += 1
    raise UnsatisfiedAssumption(where)

def assume(condition: bool) -> bool:
    if _current_build_context.value is None:
        note_deprecation('Using `assume` outside a property-based test is deprecated', since='2023-09-25', has_codemod=False)
    if TESTCASE_CALLBACKS or not condition:
        where = _calling_function_location('assume', inspect.currentframe())
        if TESTCASE_CALLBACKS and currently_in_test_context():
            predicates = current_build_context().data._observability_predicates
            predicates[where]['satisfied' if condition else 'unsatisfied'] += 1
        if not condition:
            raise UnsatisfiedAssumption(f'failed to satisfy {where}')
    return True

_current_build_context: DynamicVariable[Optional['BuildContext']] = DynamicVariable(None)

def currently_in_test_context() -> bool:
    return _current_build_context.value is not None

def current_build_context() -> 'BuildContext':
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('No build context registered')
    return context

class RandomSeeder:
    def __init__(self, seed: Any) -> None:
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
def deprecate_random_in_strategy(fmt: str, *args: Any) -> Callable[[], _Checker]:
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
        self.known_object_printers: Dict[IDKey, List[Callable[[Any, PrettyPrintFunction, bool], None]]] = defaultdict(list)

    def record_call(self, obj: Any, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.known_object_printers[IDKey(obj)].append(lambda obj, p, cycle, *, _func=func: p.maybe_repr_known_object_as_call(obj, cycle, get_pretty_function_description(_func), args, kwargs))

    def prep_args_kwargs_from_strategies(self, kwarg_strategies: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]:
        arg_labels: Dict[str, Tuple[int, int]] = {}
        kwargs: Dict[str, Any] = {}
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

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], tb: Optional[inspect.TracebackType]) -> None:
        self.assign_variable.__exit__(exc_type, exc_value, tb)
        errors: List[BaseException] = []
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
    if should_note():
        if not isinstance(value, str):
            value = pretty(value)
        report(value)

def event(value: Any, payload: Union[str, int, float] = '') -> None:
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument('Cannot make record events outside of a test')
    payload = _event_to_string(payload, (str, int, float))
    context.data.events[_event_to_string(value)] = payload

_events_to_strings: WeakKeyDictionary = WeakKeyDictionary()

def _event_to_string(event: Any, allowed_types: Union[type, Tuple[type, ...]] = str) -> str:
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
