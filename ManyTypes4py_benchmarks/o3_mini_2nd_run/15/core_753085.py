#!/usr/bin/env python3
"""
This module provides the core primitives of Hypothesis, such as given.
"""

import base64
import contextlib
import datetime
import inspect
import io
import math
import sys
import time
import traceback
import types
import unittest
import warnings
import zlib
from collections import defaultdict
from collections.abc import Coroutine, Generator, Hashable, Iterable, Sequence
from functools import partial
from inspect import Parameter
from random import Random
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    overload,
    Sequence as TypingSequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import attr
from hypothesis import strategies as st
from hypothesis._settings import HealthCheck, Phase, Verbosity, all_settings, local_settings, settings as Settings
from hypothesis.control import BuildContext
from hypothesis.database import choices_from_bytes, choices_to_bytes
from hypothesis.errors import (
    BackendCannotProceed,
    DeadlineExceeded,
    DidNotReproduce,
    FailedHealthCheck,
    FlakyFailure,
    FlakyReplay,
    Found,
    Frozen,
    HypothesisException,
    HypothesisWarning,
    InvalidArgument,
    NoSuchExample,
    StopTest,
    Unsatisfiable,
    UnsatisfiedAssumption,
)
from hypothesis.internal.compat import PYPY, BaseExceptionGroup, add_note, bad_django_TestCase, get_type_hints, int_from_bytes
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.internal.conjecture.data import ConjectureData, Status
from hypothesis.internal.conjecture.engine import BUFFER_SIZE, ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import ensure_free_stackframes, gc_cumulative_time
from hypothesis.internal.conjecture.providers import BytestringProvider, PrimitiveProvider
from hypothesis.internal.conjecture.shrinker import sort_key
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.internal.escalation import InterestingOrigin, current_pytest_item, format_exception, get_trimmed_traceback, is_hypothesis_file
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import OBSERVABILITY_COLLECT_COVERAGE, TESTCASE_CALLBACKS, _system_metadata, deliver_json_blob, make_testcase
from hypothesis.internal.reflection import convert_positional_arguments, define_function_signature, function_digest, get_pretty_function_description, get_signature, impersonate, is_mock, nicerepr, proxies, repr_call
from hypothesis.internal.scrutineer import MONITORING_TOOL_ID, Trace, Tracer, explanatory_lines, tractable_coverage_report
from hypothesis.internal.validation import check_type
from hypothesis.reporting import current_verbosity, report, verbose_report, with_reporter
from hypothesis.statistics import describe_statistics, describe_targets, note_statistics
from hypothesis.strategies._internal.misc import NOTHING
from hypothesis.strategies._internal.strategies import Ex, SearchStrategy, check_strategy
from hypothesis.strategies._internal.utils import to_jsonable
from hypothesis.vendor.pretty import RepresentationPrinter
from hypothesis.version import __version__

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)

TestFunc = TypeVar('TestFunc', bound=Callable[..., Any])
running_under_pytest: bool = False
pytest_shows_exceptiongroups: bool = True
global_force_seed: Optional[Any] = None
_hypothesis_global_random: Optional[Random] = None


@attr.s(auto_attribs=True)
class Example:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    raises: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]] = None
    reason: Optional[str] = None


class example:
    """A decorator which ensures a specific example is always tested."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise InvalidArgument('Cannot mix positional and keyword arguments for examples')
        if not (args or kwargs):
            raise InvalidArgument('An example must provide at least one argument')
        self.hypothesis_explicit_examples: List[Example] = []
        self._this_example: Example = Example(tuple(args), kwargs)

    def __call__(self, test: Callable[..., Any]) -> Callable[..., Any]:
        if not hasattr(test, 'hypothesis_explicit_examples'):
            test.hypothesis_explicit_examples = self.hypothesis_explicit_examples
        test.hypothesis_explicit_examples.append(self._this_example)
        return test

    def xfail(
        self,
        condition: bool = True,
        *,
        reason: str = '',
        raises: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = BaseException,
    ) -> "example":
        check_type(bool, condition, 'condition')
        check_type(str, reason, 'reason')
        if not ((isinstance(raises, type) and issubclass(raises, BaseException)) or (
            isinstance(raises, tuple) and raises and all((isinstance(r, type) and issubclass(r, BaseException) for r in raises))
        )):
            raise InvalidArgument(f'raises={raises!r} must be an exception type or tuple of exception types')
        if condition:
            self._this_example = attr.evolve(self._this_example, raises=raises, reason=reason)
        return self

    def via(self, whence: str) -> "example":
        if not isinstance(whence, str):
            raise InvalidArgument('.via() must be passed a string')
        return self


def seed(seed: Hashable) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """seed: Start the test execution from a specific seed.

    May be any hashable object. No exact meaning for seed is provided
    other than that for a fixed seed value Hypothesis will try the same
    actions (insofar as it can given external sources of non-
    determinism. e.g. timing and hash randomization).

    Overrides the derandomize setting, which is designed to enable
    deterministic builds rather than reproducing observed failures.
    """
    def accept(test: Callable[..., Any]) -> Callable[..., Any]:
        test._hypothesis_internal_use_seed = seed
        current_settings = getattr(test, '_hypothesis_internal_use_settings', None)
        test._hypothesis_internal_use_settings = Settings(current_settings, database=None)
        return test
    return accept


def reproduce_failure(version: str, blob: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run the example that corresponds to this data blob in order to reproduce
    a failure.

    A test with this decorator *always* runs only one example and always fails.
    If the provided example does not cause a failure, or is in some way invalid
    for this test, then this will fail with a DidNotReproduce error.
    """
    def accept(test: Callable[..., Any]) -> Callable[..., Any]:
        test._hypothesis_internal_use_reproduce_failure = (version, blob)
        return test
    return accept


def encode_failure(choices: Any) -> bytes:
    blob: bytes = choices_to_bytes(choices)
    compressed: bytes = zlib.compress(blob)
    if len(compressed) < len(blob):
        blob = b'\x01' + compressed
    else:
        blob = b'\x00' + blob
    return base64.b64encode(blob)


def decode_failure(blob: str) -> Any:
    try:
        decoded: bytes = base64.b64decode(blob)
    except Exception:
        raise InvalidArgument(f'Invalid base64 encoded string: {blob!r}') from None
    prefix: bytes = decoded[:1]
    if prefix == b'\x00':
        decoded = decoded[1:]
    elif prefix == b'\x01':
        try:
            decoded = zlib.decompress(decoded[1:])
        except zlib.error as err:
            raise InvalidArgument(f'Invalid zlib compression for blob {blob!r}') from err
    else:
        raise InvalidArgument(f'Could not decode blob {blob!r}: Invalid start byte {prefix!r}')
    choices = choices_from_bytes(decoded)
    if choices is None:
        raise InvalidArgument(f'Invalid serialized choice sequence for blob {blob!r}')
    return choices


def _invalid(
    message: str, *, exc: Type[BaseException] = InvalidArgument, test: Callable[..., Any], given_kwargs: Dict[str, Any]
) -> Callable[..., Any]:
    @impersonate(test)
    def wrapped_test(*arguments: Any, **kwargs: Any) -> Any:
        raise exc(message)
    wrapped_test.is_hypothesis_test = True
    wrapped_test.hypothesis = HypothesisHandle(inner_test=test, _get_fuzz_target=wrapped_test, _given_kwargs=given_kwargs)
    return wrapped_test


def is_invalid_test(
    test: Callable[..., Any],
    original_sig: inspect.Signature,
    given_arguments: Tuple[Any, ...],
    given_kwargs: Dict[str, Any],
) -> Optional[Callable[..., Any]]:
    invalid = partial(_invalid, test=test, given_kwargs=given_kwargs)
    if not (given_arguments or given_kwargs):
        return invalid('given must be called with at least one argument')
    params = list(original_sig.parameters.values())
    pos_params = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    kwonly_params = [p for p in params if p.kind is p.KEYWORD_ONLY]
    if given_arguments and params != pos_params:
        return invalid('positional arguments to @given are not supported with varargs, varkeywords, positional-only, or keyword-only arguments')
    if len(given_arguments) > len(pos_params):
        return invalid(
            f'Too many positional arguments for {test.__name__}() were passed to @given - expected at most {len(pos_params)} arguments, but got {len(given_arguments)} {given_arguments!r}'
        )
    if ... in given_arguments:
        return invalid('... was passed as a positional argument to @given, but may only be passed as a keyword argument or as the sole argument of @given')
    if given_arguments and given_kwargs:
        return invalid('cannot mix positional and keyword arguments to @given')
    extra_kwargs = [k for k in given_kwargs if k not in {p.name for p in pos_params + kwonly_params}]
    if extra_kwargs and (params == [] or params[-1].kind is not params[-1].VAR_KEYWORD):
        arg = extra_kwargs[0]
        extra = ''
        if arg in all_settings:
            extra = f'. Did you mean @settings({arg}={given_kwargs[arg]!r})?'
        return invalid(f'{test.__name__}() got an unexpected keyword argument {arg!r}, from `{arg}={given_kwargs[arg]!r}` in @given{extra}')
    if any((p.default is not p.empty for p in params)):
        return invalid('Cannot apply @given to a function with defaults.')
    empty = [f'{s!r} (arg {idx})' for idx, s in enumerate(given_arguments) if s is NOTHING] + [
        f'{name}={s!r}' for name, s in given_kwargs.items() if s is NOTHING
    ]
    if empty:
        strats = 'strategies' if len(empty) > 1 else 'strategy'
        return invalid('Cannot generate examples from empty ' + strats + ': ' + ', '.join(empty), exc=Unsatisfiable)
    return None


def execute_explicit_examples(
    state: "StateForActualGivenExecution",
    wrapped_test: Callable[..., Any],
    arguments: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    original_sig: inspect.Signature,
) -> Iterator[Tuple[List[str], BaseException]]:
    posargs: List[str] = [p.name for p in original_sig.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    for example_obj in reversed(getattr(wrapped_test, 'hypothesis_explicit_examples', ())):
        assert isinstance(example_obj, Example)
        if example_obj.args:
            assert not example_obj.kwargs
            if any((p.kind is p.POSITIONAL_ONLY for p in original_sig.parameters.values())):
                raise InvalidArgument('Cannot pass positional arguments to @example() when decorating a test function which has positional-only parameters.')
            if len(example_obj.args) > len(posargs):
                raise InvalidArgument(f'example has too many arguments for test. Expected at most {len(posargs)} but got {len(example_obj.args)}')
            example_kwargs: Dict[str, Any] = dict(zip(posargs[-len(example_obj.args):], example_obj.args))
        else:
            example_kwargs = dict(example_obj.kwargs)
        given_kws = ', '.join((repr(k) for k in sorted(wrapped_test.hypothesis._given_kwargs)))
        example_kws = ', '.join((repr(k) for k in sorted(example_kwargs)))
        if given_kws != example_kws:
            raise InvalidArgument(
                f'Inconsistent args: @given() got strategies for {given_kws}, but @example() got arguments for {example_kws}'
            ) from None
        assert set(example_kwargs).isdisjoint(kwargs)
        example_kwargs.update(kwargs)
        if Phase.explicit not in state.settings.phases:
            continue
        with local_settings(state.settings):
            fragments_reported: List[str] = []
            empty_data: ConjectureData = ConjectureData.for_choices([])
            try:
                execute_example = partial(state.execute_once, empty_data, is_final=True, print_example=True, example_kwargs=example_kwargs)
                with with_reporter(fragments_reported.append):
                    if example_obj.raises is None:
                        execute_example()
                    else:
                        bits = ', '.join((nicerepr(x) for x in arguments)) + ', '.join((f'{k}={nicerepr(v)}' for k, v in example_kwargs.items()))
                        try:
                            execute_example()
                        except failure_exceptions_to_catch() as err:
                            if not isinstance(err, example_obj.raises):
                                raise
                            state.xfail_example_reprs.add(repr_call(state.test, arguments, example_kwargs))
                        except example_obj.raises as err:
                            raise InvalidArgument(f'@example({bits}) raised an expected {err!r}, but Hypothesis does not treat this as a test failure') from err
                        else:
                            reason = f' because {example_obj.reason}' * bool(example_obj.reason)
                            if example_obj.raises is BaseException:
                                name = 'exception'
                            elif not isinstance(example_obj.raises, tuple):
                                name = example_obj.raises.__name__
                            elif len(example_obj.raises) == 1:
                                name = example_obj.raises[0].__name__
                            else:
                                name = ', '.join((ex.__name__ for ex in example_obj.raises[:-1])) + f', or {example_obj.raises[-1].__name__}'
                            vowel = name.upper()[0] in 'AEIOU'
                            raise AssertionError(f'Expected a{"n" if vowel else ""} {name} from @example({bits}){reason}, but no exception was raised.')
            except UnsatisfiedAssumption:
                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
            except BaseException as err:
                err = err.with_traceback(get_trimmed_traceback())
                if isinstance(err, failure_exceptions_to_catch()) and any((isinstance(arg, SearchStrategy) for arg in example_obj.args + tuple(example_obj.kwargs.values()))):
                    new = HypothesisWarning('The @example() decorator expects to be passed values, but you passed strategies instead.  See https://hypothesis.readthedocs.io/en/latest/reproducing.html for details.')
                    new.__cause__ = err
                    err = new
                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
                yield (fragments_reported, err)
                if state.settings.report_multiple_bugs and pytest_shows_exceptiongroups and isinstance(err, failure_exceptions_to_catch()) and (not isinstance(err, skip_exceptions_to_reraise())):
                    continue
                break
            finally:
                if fragments_reported:
                    assert fragments_reported[0].startswith('Falsifying example')
                    fragments_reported[0] = fragments_reported[0].replace('Falsifying example', 'Falsifying explicit example', 1)
                tc = make_testcase(
                    start_timestamp=state._start_timestamp,
                    test_name_or_nodeid=state.test_identifier,
                    data=empty_data,
                    how_generated='explicit example',
                    string_repr=state._string_repr,
                    timing=state._timing_features,
                )
                deliver_json_blob(tc)
            if fragments_reported:
                verbose_report(fragments_reported[0].replace('Falsifying', 'Trying', 1))
                for f in fragments_reported[1:]:
                    verbose_report(f)


def get_random_for_wrapped_test(test: Callable[..., Any], wrapped_test: Any) -> Random:
    settings = wrapped_test._hypothesis_internal_use_settings
    wrapped_test._hypothesis_internal_use_generated_seed = None
    if wrapped_test._hypothesis_internal_use_seed is not None:
        return Random(wrapped_test._hypothesis_internal_use_seed)
    elif settings.derandomize:
        return Random(int_from_bytes(function_digest(test)))
    elif global_force_seed is not None:
        return Random(global_force_seed)
    else:
        global _hypothesis_global_random
        if _hypothesis_global_random is None:
            _hypothesis_global_random = Random()
        seed_value = _hypothesis_global_random.getrandbits(128)
        wrapped_test._hypothesis_internal_use_generated_seed = seed_value
        return Random(seed_value)


@attr.s(auto_attribs=True)
class Stuff:
    selfy: Any = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = attr.Factory(dict)
    given_kwargs: Dict[str, Any] = attr.Factory(dict)


def process_arguments_to_given(
    wrapped_test: Callable[..., Any],
    arguments: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    given_kwargs: Dict[str, Any],
    params: Dict[str, inspect.Parameter],
) -> Tuple[Tuple[Any, ...], Dict[str, Any], Stuff]:
    selfy: Optional[Any] = None
    arguments, kwargs = convert_positional_arguments(wrapped_test, arguments, kwargs)
    posargs: List[str] = [p.name for p in params.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    if posargs:
        selfy = kwargs.get(posargs[0])
    elif arguments:
        selfy = arguments[0]
    if is_mock(selfy):
        selfy = None
    arguments = tuple(arguments)
    with ensure_free_stackframes():
        for k, s in given_kwargs.items():
            check_strategy(s, name=k)
            s.validate()
    stuff = Stuff(selfy=selfy, args=arguments, kwargs=kwargs, given_kwargs=given_kwargs)
    return (arguments, kwargs, stuff)


def skip_exceptions_to_reraise() -> Tuple[Type[BaseException], ...]:
    exceptions = set()
    if 'unittest' in sys.modules:
        exceptions.add(sys.modules['unittest'].SkipTest)
    if 'unittest2' in sys.modules:
        exceptions.add(sys.modules['unittest2'].SkipTest)
    if 'nose' in sys.modules:
        exceptions.add(sys.modules['nose'].SkipTest)
    if '_pytest' in sys.modules:
        exceptions.add(sys.modules['_pytest'].outcomes.Skipped)
    return tuple(sorted(exceptions, key=str))


def failure_exceptions_to_catch() -> Tuple[Type[BaseException], ...]:
    exceptions: List[Type[BaseException]] = [Exception, SystemExit, GeneratorExit]
    if '_pytest' in sys.modules:
        exceptions.append(sys.modules['_pytest'].outcomes.Failed)
    return tuple(exceptions)


def new_given_signature(original_sig: inspect.Signature, given_kwargs: Dict[str, Any]) -> inspect.Signature:
    return original_sig.replace(
        parameters=[p for p in original_sig.parameters.values() if not (p.name in given_kwargs and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))],
        return_annotation=None,
    )


def default_executor(data: Any, function: Callable[[Any], Any]) -> Any:
    return function(data)


def get_executor(runner: Any) -> Callable[[Any, Callable[[Any], Any]], Any]:
    try:
        execute_example = runner.execute_example
    except AttributeError:
        pass
    else:
        return lambda data, function: execute_example(partial(function, data))
    if hasattr(runner, 'setup_example') or hasattr(runner, 'teardown_example'):
        setup = getattr(runner, 'setup_example', None) or (lambda: None)
        teardown = getattr(runner, 'teardown_example', None) or (lambda ex: None)

        def execute(data: Any, function: Callable[[Any], Any]) -> Any:
            token = None
            try:
                token = setup()
                return function(data)
            finally:
                teardown(token)
        return execute
    return default_executor


@contextlib.contextmanager
def unwrap_markers_from_group() -> Iterator[None]:
    T = TypeVar('T', bound=BaseException)

    def _flatten_group(excgroup: BaseExceptionGroup) -> List[BaseException]:
        found_exceptions: List[BaseException] = []
        for exc in excgroup.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                found_exceptions.extend(_flatten_group(exc))
            else:
                found_exceptions.append(exc)
        return found_exceptions
    try:
        yield
    except BaseExceptionGroup as excgroup:
        frozen_exceptions, non_frozen_exceptions = excgroup.split(Frozen)
        if non_frozen_exceptions is None:
            raise
        _, user_exceptions = non_frozen_exceptions.split(lambda e: isinstance(e, (StopTest, HypothesisException)))
        if user_exceptions is not None:
            raise
        flattened_non_frozen_exceptions = _flatten_group(non_frozen_exceptions)
        if len(flattened_non_frozen_exceptions) == 1:
            e = flattened_non_frozen_exceptions[0]
            raise e from e.__cause__
        stoptests, non_stoptests = non_frozen_exceptions.split(StopTest)
        if non_stoptests:
            e = _flatten_group(non_stoptests)[0]
            raise e from e.__cause__
        assert stoptests is not None
        raise min(_flatten_group(stoptests), key=lambda s_e: s_e.testcounter)


class StateForActualGivenExecution:
    def __init__(
        self,
        stuff: Stuff,
        test: Callable[..., Any],
        settings: Settings,
        random: Random,
        wrapped_test: Callable[..., Any],
    ) -> None:
        self.test_runner: Callable[[Any, Callable[[Any], Any]], Any] = get_executor(stuff.selfy)
        self.stuff: Stuff = stuff
        self.settings: Settings = settings
        self.last_exception: Optional[BaseException] = None
        self.falsifying_examples: Tuple[Any, ...] = ()
        self.random: Random = random
        self.ever_executed: bool = False
        self.is_find: bool = getattr(wrapped_test, '_hypothesis_internal_is_find', False)
        self.wrapped_test: Callable[..., Any] = wrapped_test
        self.xfail_example_reprs: set = set()
        self.test: Callable[..., Any] = test
        self.print_given_args: bool = getattr(wrapped_test, '_hypothesis_internal_print_given_args', True)
        self.files_to_propagate: set = set()
        self.failed_normally: bool = False
        self.failed_due_to_deadline: bool = False
        self.explain_traces: Dict[Any, set] = defaultdict(set)
        self._start_timestamp: float = time.time()
        self._string_repr: str = ''
        self._timing_features: Dict[str, Any] = {}

    @property
    def test_identifier(self) -> Optional[str]:
        return getattr(current_pytest_item.value, 'nodeid', None) or get_pretty_function_description(self.wrapped_test)

    def _should_trace(self) -> bool:
        _trace_obs: bool = bool(TESTCASE_CALLBACKS and OBSERVABILITY_COLLECT_COVERAGE)
        _trace_failure: bool = self.failed_normally and (not self.failed_due_to_deadline) and {Phase.shrink, Phase.explain}.issubset(self.settings.phases)
        return _trace_obs or _trace_failure

    def execute_once(
        self,
        data: ConjectureData,
        *,
        print_example: bool = False,
        is_final: bool = False,
        expected_failure: Optional[Any] = None,
        example_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self.ever_executed = True
        data.is_find = self.is_find
        self._string_repr = ''
        text_repr: Optional[str] = None
        if self.settings.deadline is None and (not TESTCASE_CALLBACKS):

            @proxies(self.test)
            def test(*args: Any, **kwargs: Any) -> Any:
                with unwrap_markers_from_group(), ensure_free_stackframes():
                    return self.test(*args, **kwargs)
        else:

            @proxies(self.test)
            def test(*args: Any, **kwargs: Any) -> Any:
                arg_drawtime = math.fsum(data.draw_times.values())
                arg_stateful = math.fsum(data._stateful_run_times.values())
                arg_gctime = gc_cumulative_time()
                start = time.perf_counter()
                try:
                    with unwrap_markers_from_group(), ensure_free_stackframes():
                        result = self.test(*args, **kwargs)
                finally:
                    finish = time.perf_counter()
                    in_drawtime = math.fsum(data.draw_times.values()) - arg_drawtime
                    in_stateful = math.fsum(data._stateful_run_times.values()) - arg_stateful
                    in_gctime = gc_cumulative_time() - arg_gctime
                    runtime = finish - start - in_drawtime - in_stateful - in_gctime
                    self._timing_features = {'execute:test': runtime, 'overall:gc': in_gctime, **data.draw_times, **data._stateful_run_times}
                if (current_deadline := self.settings.deadline) is not None:
                    if not is_final:
                        current_deadline = current_deadline // 4 * 5
                    if runtime >= current_deadline.total_seconds():
                        raise DeadlineExceeded(datetime.timedelta(seconds=runtime), self.settings.deadline)
                return result

        def run(data: ConjectureData) -> Any:
            if self.stuff.selfy is not None:
                data.hypothesis_runner = self.stuff.selfy
            args = self.stuff.args
            kwargs = dict(self.stuff.kwargs)
            if example_kwargs is None:
                kw, argslices = context.prep_args_kwargs_from_strategies(self.stuff.given_kwargs)
            else:
                kw = example_kwargs
                argslices = {}
            kwargs.update(kw)
            if expected_failure is not None:
                nonlocal text_repr
                text_repr = repr_call(test, args, kwargs)
            if print_example or current_verbosity() >= Verbosity.verbose:
                printer = RepresentationPrinter(context=context)  # type: ignore
                if print_example:
                    printer.text('Falsifying example:')
                else:
                    printer.text('Trying example:')
                if self.print_given_args:
                    if data.provider.avoid_realization and (not print_example):
                        printer.text(' <symbolics>')
                    else:
                        printer.text(' ')
                        printer.repr_call(test.__name__, args, kwargs, force_split=True, arg_slices=argslices, leading_comment='# ' + (context.data.slice_comments[0, 0] if (0, 0) in context.data.slice_comments else None))
                report(printer.getvalue())
            if TESTCASE_CALLBACKS:
                printer = RepresentationPrinter(context=context)  # type: ignore
                printer.repr_call(test.__name__, args, kwargs, force_split=True, arg_slices=argslices, leading_comment='# ' + (context.data.slice_comments[0, 0] if (0, 0) in context.data.slice_comments else None))
                self._string_repr = printer.getvalue()
                data._observability_arguments = {**dict(enumerate(map(to_jsonable, args))), **{k: to_jsonable(v) for k, v in kwargs.items()}}
            try:
                return test(*args, **kwargs)
            except TypeError as e:
                if 'SearchStrategy' in str(e) and data._sampled_from_all_strategies_elements_message is not None:
                    msg, format_arg = data._sampled_from_all_strategies_elements_message
                    add_note(e, msg.format(format_arg))
                raise
            finally:
                if (parts := getattr(data, '_stateful_repr_parts', None)):
                    self._string_repr = '\n'.join(parts)
        with local_settings(self.settings):
            with deterministic_PRNG():
                with BuildContext(data, is_final=is_final) as context:
                    result = None
                    with data.provider.per_test_case_context_manager():
                        result = self.test_runner(data, run)
        if expected_failure is not None:
            exception, traceback_str = expected_failure
            if isinstance(exception, DeadlineExceeded) and (runtime_secs := math.fsum((v for k, v in self._timing_features.items() if k.startswith('execute:')))):
                report('Unreliable test timings! On an initial run, this test took %.2fms, which exceeded the deadline of %.2fms, but on a subsequent run it took %.2f ms, which did not. If you expect this sort of variability in your test timings, consider turning deadlines off for this test by setting deadline=None.' % (exception.runtime.total_seconds() * 1000, self.settings.deadline.total_seconds() * 1000, runtime_secs * 1000))
            else:
                report('Failed to reproduce exception. Expected: \n' + traceback_str)
            raise FlakyFailure(f'Hypothesis {text_repr} produces unreliable results: Falsified on the first call but did not on a subsequent one', [exception])
        return result

    def _flaky_replay_to_failure(self, err: BaseException, context: BaseException) -> FlakyFailure:
        interesting_examples = [self._runner.interesting_examples[io] for io in err._interesting_origins if io in self._runner.interesting_examples]
        exceptions = [ie.expected_exception for ie in interesting_examples]
        exceptions.append(context)
        return FlakyFailure(err.reason, exceptions)

    def _execute_once_for_engine(self, data: ConjectureData) -> None:
        trace = set()
        try:
            with Tracer(should_trace=self._should_trace()) as tracer:
                try:
                    result = self.execute_once(data)
                    if data.status == Status.VALID and tracer.branches:
                        self.explain_traces[None].add(frozenset(tracer.branches))
                finally:
                    trace = tracer.branches
            if result is not None:
                fail_health_check(self.settings, f'Tests run under @given should return None, but {self.test.__name__} returned {result!r} instead.', HealthCheck.return_value)
        except UnsatisfiedAssumption as e:
            try:
                data.mark_invalid(e.reason)
            except FlakyReplay as err:
                raise self._flaky_replay_to_failure(err, e) from None
        except (StopTest, BackendCannotProceed):
            raise
        except (FailedHealthCheck, *skip_exceptions_to_reraise()):
            raise
        except failure_exceptions_to_catch() as e:
            tb = e.__traceback__
            filepath = traceback.extract_tb(tb)[-1][0]
            if is_hypothesis_file(filepath) and (not isinstance(e, HypothesisException)):
                raise
            if data.frozen:
                raise StopTest(data.testcounter) from e
            else:
                tb_trim = get_trimmed_traceback()
                data.expected_traceback = format_exception(e, tb_trim)
                data.expected_exception = e
                assert data.expected_traceback is not None
                verbose_report(data.expected_traceback)
                self.failed_normally = True
                interesting_origin = InterestingOrigin.from_exception(e)
                if trace:
                    self.explain_traces[interesting_origin].add(frozenset(trace))
                if interesting_origin[0] == DeadlineExceeded:
                    self.failed_due_to_deadline = True
                    self.explain_traces.clear()
                try:
                    data.mark_interesting(interesting_origin)
                except FlakyReplay as err:
                    raise self._flaky_replay_to_failure(err, e) from None
        finally:
            if TESTCASE_CALLBACKS:
                if (runner := getattr(self, '_runner', None)):
                    phase = runner._current_phase
                elif self.failed_normally or self.failed_due_to_deadline:
                    phase = 'shrink'
                else:
                    phase = 'unknown'
                backend_desc = f', using backend={self.settings.backend!r}' * (self.settings.backend != 'hypothesis' and (not getattr(runner, '_switch_to_hypothesis_provider', False)))
                try:
                    data._observability_args = data.provider.realize(data._observability_args)
                    self._string_repr = data.provider.realize(self._string_repr)
                except BackendCannotProceed:
                    data._observability_args = {}
                    self._string_repr = '<backend failed to realize symbolic arguments>'
                tc = make_testcase(
                    start_timestamp=self._start_timestamp,
                    test_name_or_nodeid=self.test_identifier,
                    data=data,
                    how_generated=f'during {phase} phase{backend_desc}',
                    string_repr=self._string_repr,
                    timing=self._timing_features,
                    coverage=tractable_coverage_report(trace) or None,
                    phase=phase,
                    backend_metadata=data.provider.observe_test_case(),
                )
                deliver_json_blob(tc)
                for msg in data.provider.observe_information_messages(lifetime='test_case'):
                    self._deliver_information_message(**msg)
            self._timing_features = {}

    def _deliver_information_message(self, *, type: str, title: str, content: str) -> None:
        deliver_json_blob({'type': type, 'run_start': self._start_timestamp, 'property': self.test_identifier, 'title': title, 'content': content})

    def run_engine(self) -> None:
        __tracebackhide__ = True
        try:
            database_key: Optional[bytes]
            try:
                database_key = self.wrapped_test._hypothesis_internal_database_key
            except AttributeError:
                if global_force_seed is None:
                    database_key = function_digest(self.test)
                else:
                    database_key = None
            runner = self._runner = ConjectureRunner(self._execute_once_for_engine, settings=self.settings, random=self.random, database_key=database_key)
            runner.run()
            note_statistics(runner.statistics)
            if TESTCASE_CALLBACKS:
                self._deliver_information_message(type='info', title='Hypothesis Statistics', content=describe_statistics(runner.statistics))
                for msg in (p if isinstance((p := runner.provider), PrimitiveProvider) else p(None)).observe_information_messages(lifetime='test_function'):
                    self._deliver_information_message(**msg)
            if runner.call_count == 0:
                return
            if runner.interesting_examples:
                self.falsifying_examples = sorted(runner.interesting_examples.values(), key=lambda d: sort_key(d.nodes), reverse=True)
            elif runner.valid_examples == 0:
                explanations: List[str] = []
                if runner.invalid_examples > min(20, runner.call_count // 5):
                    explanations.append(f'{runner.invalid_examples} of {runner.call_count} examples failed a .filter() or assume() condition. Try making your filters or assumes less strict, or rewrite using strategy parameters: st.integers().filter(lambda x: x > 0) fails less often (that is, never) when rewritten as st.integers(min_value=1).')
                if runner.overrun_examples > min(20, runner.call_count // 5):
                    explanations.append(f'{runner.overrun_examples} of {runner.call_count} examples were too large to finish generating; try reducing the typical size of your inputs?')
                rep = get_pretty_function_description(self.test)
                raise Unsatisfiable(f'Unable to satisfy assumptions of {rep}. {" Also, ".join(explanations)}')
            if sys.version_info[:2] >= (3, 12) and (not PYPY) and self._should_trace() and (not Tracer.can_trace()):
                warnings.warn(f'avoiding tracing test function because tool id {MONITORING_TOOL_ID} is already taken by tool {sys.monitoring.get_tool(MONITORING_TOOL_ID)}.', HypothesisWarning, stacklevel=3)
            if not self.falsifying_examples:
                return
            elif not (self.settings.report_multiple_bugs and pytest_shows_exceptiongroups):
                del self.falsifying_examples[:-1]
            errors_to_report: List[Tuple[List[str], BaseException]] = []
            report_lines: List[str] = describe_targets(runner.best_observed_targets)
            if report_lines:
                report_lines.append('')
            explanations = explanatory_lines(self.explain_traces, self.settings)
            for falsifying_example in self.falsifying_examples:
                fragments: List[str] = []
                ran_example = runner.new_conjecture_data_ir(falsifying_example.choices, max_choices=len(falsifying_example.choices))
                ran_example.slice_comments = falsifying_example.slice_comments
                tb = None
                origin = None
                assert falsifying_example.expected_exception is not None
                assert falsifying_example.expected_traceback is not None
                try:
                    with with_reporter(fragments.append):
                        self.execute_once(ran_example, print_example=not self.is_find, is_final=True, expected_failure=(falsifying_example.expected_exception, falsifying_example.expected_traceback))
                except StopTest as e:
                    err = FlakyFailure('Inconsistent results: An example failed on the first run but now succeeds (or fails with another error, or is for some reason not runnable).', [falsifying_example.expected_exception or e])
                    errors_to_report.append((fragments, err))
                except UnsatisfiedAssumption as e:
                    err = FlakyFailure('Unreliable assumption: An example which satisfied assumptions on the first run now fails it.', [e])
                    errors_to_report.append((fragments, err))
                except BaseException as e:
                    fragments.extend(explanations[falsifying_example.interesting_origin])
                    errors_to_report.append((fragments, e.with_traceback(get_trimmed_traceback())))
                    tb = format_exception(e, get_trimmed_traceback(e))
                    origin = InterestingOrigin.from_exception(e)
                else:
                    raise NotImplementedError('This should be unreachable')
                finally:
                    tc = {
                        'type': 'test_case',
                        'run_start': self._start_timestamp,
                        'property': self.test_identifier,
                        'status': 'passed' if sys.exc_info()[0] is None else 'failed',
                        'status_reason': str(origin or 'unexpected/flaky pass'),
                        'representation': self._string_repr,
                        'arguments': ran_example._observability_args,
                        'how_generated': 'minimal failing example',
                        'features': {**{f'target:{k}'.strip(':'): v for k, v in ran_example.target_observations.items()}, **ran_example.events},
                        'timing': self._timing_features,
                        'coverage': None,
                        'metadata': {'traceback': tb, 'predicates': dict(ran_example._observability_predicates), **_system_metadata()},
                    }
                    deliver_json_blob(tc)
                    if self.settings.print_blob:
                        fragments.append('\nYou can reproduce this example by temporarily adding @reproduce_failure(%r, %r) as a decorator on your test case' % (__version__, encode_failure(falsifying_example.choices)))
                    ran_example.freeze()
            _raise_to_user(errors_to_report, self.settings, report_lines, verified_by=runner._verified_by)

def _raise_to_user(
    errors_to_report: List[Tuple[List[str], BaseException]],
    settings: Settings,
    target_lines: List[str],
    trailer: str = '',
    verified_by: Optional[Any] = None,
) -> None:
    failing_prefix = 'Falsifying example: '
    ls: List[str] = []
    for fragments, err in errors_to_report:
        for note in fragments:
            add_note(err, note)
            if note.startswith(failing_prefix):
                ls.append(note.removeprefix(failing_prefix))
    if current_pytest_item.value:
        current_pytest_item.value._hypothesis_failing_examples = ls
    if len(errors_to_report) == 1:
        _, the_error_hypothesis_found = errors_to_report[0]
    else:
        assert errors_to_report
        the_error_hypothesis_found = BaseExceptionGroup(f'Hypothesis found {len(errors_to_report)} distinct failures{trailer}.', [e for _, e in errors_to_report])
    if settings.verbosity >= Verbosity.normal:
        for line in target_lines:
            add_note(the_error_hypothesis_found, line)
    if verified_by:
        msg = f'backend={verified_by!r} claimed to verify this test passes - please send them a bug report!'
        add_note(err, msg)
    raise the_error_hypothesis_found


@contextlib.contextmanager
def fake_subTest(self: Any, msg: Optional[str] = None, **__: Any) -> Iterator[None]:
    warnings.warn('subTest per-example reporting interacts badly with Hypothesis trying hundreds of examples, so we disable it for the duration of any test that uses `@given`.', HypothesisWarning, stacklevel=2)
    yield


@attr.s(auto_attribs=True)
class HypothesisHandle:
    inner_test: Callable[..., Any]
    _get_fuzz_target: Callable[[], Any]
    _given_kwargs: Dict[str, Any]

    @property
    def fuzz_one_input(self) -> Any:
        try:
            return self.__cached_target
        except AttributeError:
            self.__cached_target = self._get_fuzz_target()
            return self.__cached_target


@overload
def given(_: Any, /) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


@overload
def given(*_given_arguments: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


@overload
def given(**_given_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


def given(*_given_arguments: Any, **_given_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator for turning a test function that accepts arguments into a randomized test."""
    def run_test_as_given(test: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.isclass(test):
            raise InvalidArgument('@given cannot be applied to a class.')
        given_arguments: Tuple[Any, ...] = tuple(_given_arguments)
        given_kwargs: Dict[str, Any] = dict(_given_kwargs)
        original_sig: inspect.Signature = get_signature(test)
        if given_arguments == (Ellipsis,) and (not given_kwargs):
            given_kwargs = {p.name: Ellipsis for p in original_sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            given_arguments = ()
        check_invalid = is_invalid_test(test, original_sig, given_arguments, given_kwargs)
        if check_invalid is not None:
            return check_invalid
        if given_arguments:
            assert not given_kwargs
            posargs: List[str] = [p.name for p in original_sig.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
            given_kwargs = dict(list(zip(posargs[::-1], given_arguments[::-1]))[::-1])
        del given_arguments
        new_signature: inspect.Signature = new_given_signature(original_sig, given_kwargs)
        if ... in given_kwargs.values():
            hints = get_type_hints(test)
        for name in [name for name, value in given_kwargs.items() if value is ...]:
            if name not in hints:
                return _invalid(f'passed {name}=... for {test.__name__}, but {name} has no type annotation', test=test, given_kwargs=given_kwargs)
            given_kwargs[name] = st.from_type(hints[name])
        prev_self = Unset = object()

        @impersonate(test)
        @define_function_signature(test.__name__, test.__doc__, new_signature)
        def wrapped_test(*arguments: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            test_func = wrapped_test.hypothesis.inner_test
            if getattr(test_func, 'is_hypothesis_test', False):
                raise InvalidArgument(f'You have applied @given to the test {test_func.__name__} more than once, which wraps the test several times and is extremely slow. A similar effect can be gained by combining the arguments of the two calls to given. For example, instead of @given(booleans()) @given(integers()), you could write @given(booleans(), integers())')
            settings = wrapped_test._hypothesis_internal_use_settings
            random_inst = get_random_for_wrapped_test(test_func, wrapped_test)
            arguments, kwargs, stuff = process_arguments_to_given(wrapped_test, arguments, kwargs, given_kwargs, new_signature.parameters)
            if inspect.iscoroutinefunction(test_func) and get_executor(stuff.selfy) is default_executor:
                raise InvalidArgument(f"Hypothesis doesn't know how to run async test functions like {test_func.__name__}.  You'll need to write a custom executor, or use a library like pytest-asyncio or pytest-trio which can handle the translation for you.\n    See https://hypothesis.readthedocs.io/en/latest/details.html#custom-function-execution")
            runner = stuff.selfy
            if isinstance(stuff.selfy, TestCase) and test_func.__name__ in dir(TestCase):
                msg = f'You have applied @given to the method {test_func.__name__}, which is used by the unittest runner but is not itself a test.  This is not useful in any way.'
                fail_health_check(settings, msg, HealthCheck.not_a_test_method)
            if bad_django_TestCase(runner):
                raise InvalidArgument(f'You have applied @given to a method on {type(runner).__qualname__}, but this class does not inherit from the supported versions in `hypothesis.extra.django`.  Use the Hypothesis variants to ensure that each example is run in a separate database transaction.')
            nonlocal prev_self
            cur_self = stuff.selfy if getattr(type(stuff.selfy), test_func.__name__, None) is wrapped_test else None
            if prev_self is Unset:
                prev_self = cur_self
            elif cur_self is not prev_self:
                msg = f'The method {test_func.__qualname__} was called from multiple different executors. This may lead to flaky tests and nonreproducible errors when replaying from database.'
                fail_health_check(settings, msg, HealthCheck.differing_executors)
            state = StateForActualGivenExecution(stuff, test_func, settings, random_inst, wrapped_test)
            reproduce_failure_info = wrapped_test._hypothesis_internal_use_reproduce_failure
            if reproduce_failure_info is not None:
                expected_version, failure = reproduce_failure_info
                if expected_version != __version__:
                    raise InvalidArgument('Attempting to reproduce a failure from a different version of Hypothesis. This failure is from %s, but you are currently running %r. Please change your Hypothesis version to a matching one.' % (expected_version, __version__))
                try:
                    state.execute_once(ConjectureData.for_choices(decode_failure(failure)), print_example=True, is_final=True)
                    raise DidNotReproduce('Expected the test to raise an error, but it completed successfully.')
                except StopTest:
                    raise DidNotReproduce("The shape of the test data has changed in some way from where this blob was defined. Are you sure you're running the same test?") from None
                except UnsatisfiedAssumption:
                    raise DidNotReproduce('The test data failed to satisfy an assumption in the test. Have you added it since this blob was generated?') from None
            errors = list(execute_explicit_examples(state, wrapped_test, arguments, kwargs, original_sig))
            if errors:
                assert len(errors) == 1 or state.settings.report_multiple_bugs
                if isinstance(errors[-1][1], skip_exceptions_to_reraise()):
                    del errors[:-1]
                _raise_to_user(errors, state.settings, [], ' in explicit examples')
            ran_explicit_examples = Phase.explicit in state.settings.phases and getattr(wrapped_test, 'hypothesis_explicit_examples', ())
            SKIP_BECAUSE_NO_EXAMPLES = unittest.SkipTest('Hypothesis has been told to run no examples for this test.')
            if not (Phase.reuse in settings.phases or Phase.generate in settings.phases):
                if not ran_explicit_examples:
                    raise SKIP_BECAUSE_NO_EXAMPLES
                return
            try:
                if isinstance(runner, TestCase) and hasattr(runner, 'subTest'):
                    subTest = runner.subTest
                    try:
                        runner.subTest = types.MethodType(fake_subTest, runner)
                        state.run_engine()
                    finally:
                        runner.subTest = subTest
                else:
                    state.run_engine()
            except BaseException as e:
                generated_seed = wrapped_test._hypothesis_internal_use_generated_seed
                with local_settings(settings):
                    if not (state.failed_normally or generated_seed is None):
                        if running_under_pytest:
                            report(f'You can add @seed({generated_seed}) to this test or run pytest with --hypothesis-seed={generated_seed} to reproduce this failure.')
                        else:
                            report(f'You can add @seed({generated_seed}) to this test to reproduce this failure.')
                    the_error_hypothesis_found = e.with_traceback(None if isinstance(e, BaseExceptionGroup) else get_trimmed_traceback())
                    raise the_error_hypothesis_found
            if not (ran_explicit_examples or state.ever_executed):
                raise SKIP_BECAUSE_NO_EXAMPLES
        for attrib in dir(test):
            if not (attrib.startswith('_') or hasattr(wrapped_test, attrib)):
                setattr(wrapped_test, attrib, getattr(test, attrib))
        wrapped_test.is_hypothesis_test = True
        if hasattr(test, '_hypothesis_internal_settings_applied'):
            wrapped_test._hypothesis_internal_settings_applied = True
        wrapped_test._hypothesis_internal_use_seed = getattr(test, '_hypothesis_internal_use_seed', None)
        wrapped_test._hypothesis_internal_use_settings = getattr(test, '_hypothesis_internal_use_settings', None) or Settings.default
        wrapped_test._hypothesis_internal_use_reproduce_failure = getattr(test, '_hypothesis_internal_use_reproduce_failure', None)
        wrapped_test.hypothesis = HypothesisHandle(test, _get_fuzz_target, given_kwargs)
        return wrapped_test

    def _get_fuzz_target() -> Any:
        test_func = wrapped_test.hypothesis.inner_test
        new_settings = Settings(parent=wrapped_test._hypothesis_internal_use_settings, deadline=None)
        random_inst = get_random_for_wrapped_test(test_func, wrapped_test)
        _args, _kwargs, stuff = process_arguments_to_given(wrapped_test, (), {}, given_kwargs, new_signature.parameters)
        assert not _args
        assert not _kwargs
        state = StateForActualGivenExecution(stuff, test_func, new_settings, random_inst, wrapped_test)
        database_key = function_digest(test_func) + b'.secondary'
        minimal_failures: Dict[Any, Any] = {}

        def fuzz_one_input(buffer: Union[bytes, bytearray, memoryview, io.IOBase]) -> Optional[bytes]:
            if isinstance(buffer, io.IOBase):
                buffer = buffer.read(BUFFER_SIZE)
            assert isinstance(buffer, (bytes, bytearray, memoryview))
            data = ConjectureData(random=None, provider=BytestringProvider, provider_kw={'bytestring': buffer})
            try:
                state.execute_once(data)
            except (StopTest, UnsatisfiedAssumption):
                return None
            except BaseException:
                known = minimal_failures.get(data.interesting_origin)
                if new_settings.database is not None and (known is None or sort_key(data.nodes) <= sort_key(known)):
                    new_settings.database.save(database_key, choices_to_bytes(data.choices))
                    minimal_failures[data.interesting_origin] = data.nodes
                raise
            assert isinstance(data.provider, BytestringProvider)
            return bytes(data.provider.drawn)
        fuzz_one_input.__doc__ = HypothesisHandle.fuzz_one_input.__doc__
        return fuzz_one_input

    return run_test_as_given


def find(
    specifier: SearchStrategy,
    condition: Callable[[Any], bool],
    *,
    settings: Optional[Settings] = None,
    random: Optional[Random] = None,
    database_key: Optional[bytes] = None,
) -> Any:
    """Returns the minimal example from the given strategy ``specifier`` that matches the predicate function ``condition``."""
    if settings is None:
        settings = Settings(max_examples=2000)
    settings = Settings(settings, suppress_health_check=list(HealthCheck), report_multiple_bugs=False)
    if database_key is None and settings.database is not None:
        database_key = function_digest(condition)
    if not isinstance(specifier, SearchStrategy):
        raise InvalidArgument(f'Expected SearchStrategy but got {specifier!r} of type {type(specifier).__name__}')
    specifier.validate()
    last: List[Any] = []

    @settings
    @given(specifier)
    def test(v: Any) -> None:
        if condition(v):
            last[:] = [v]
            raise Found

    if random is not None:
        test = seed(random.getrandbits(64))(test)
    _test = test
    _test._hypothesis_internal_is_find = True
    _test._hypothesis_internal_database_key = database_key
    try:
        test()
    except Found:
        return last[0]
    raise NoSuchExample(get_pretty_function_description(condition))
