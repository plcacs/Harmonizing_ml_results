"""This module provides the core primitives of Hypothesis, such as given."""
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
    TYPE_CHECKING, Any, BinaryIO, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, 
    overload, FrozenSet, cast
)
from unittest import TestCase
import attr
from hypothesis import strategies as st
from hypothesis._settings import HealthCheck, Phase, Verbosity, all_settings, local_settings, settings as Settings
from hypothesis.control import BuildContext
from hypothesis.database import choices_from_bytes, choices_to_bytes
from hypothesis.errors import (
    BackendCannotProceed, DeadlineExceeded, DidNotReproduce, FailedHealthCheck, FlakyFailure, 
    FlakyReplay, Found, Frozen, HypothesisException, HypothesisWarning, InvalidArgument, 
    NoSuchExample, StopTest, Unsatisfiable, UnsatisfiedAssumption
)
from hypothesis.internal.compat import (
    PYPY, BaseExceptionGroup, add_note, bad_django_TestCase, get_type_hints, int_from_bytes
)
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.internal.conjecture.data import ConjectureData, Status
from hypothesis.internal.conjecture.engine import BUFFER_SIZE, ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import ensure_free_stackframes, gc_cumulative_time
from hypothesis.internal.conjecture.providers import BytestringProvider, PrimitiveProvider
from hypothesis.internal.conjecture.shrinker import sort_key
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.internal.escalation import (
    InterestingOrigin, current_pytest_item, format_exception, get_trimmed_traceback, is_hypothesis_file
)
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import (
    OBSERVABILITY_COLLECT_COVERAGE, TESTCASE_CALLBACKS, _system_metadata, deliver_json_blob, make_testcase
)
from hypothesis.internal.reflection import (
    convert_positional_arguments, define_function_signature, function_digest, 
    get_pretty_function_description, get_signature, impersonate, is_mock, nicerepr, proxies, repr_call
)
from hypothesis.internal.scrutineer import (
    MONITORING_TOOL_ID, Trace, Tracer, explanatory_lines, tractable_coverage_report
)
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
    raises: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]] = None
    reason: Optional[str] = None

class example:
    """A decorator which ensures a specific example is always tested."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise InvalidArgument('Cannot mix positional and keyword arguments for examples')
        if not (args or kwargs):
            raise InvalidArgument('An example must provide at least one argument')
        self.hypothesis_explicit_examples: List[Example] = []
        self._this_example = Example(tuple(args), kwargs)

    def __call__(self, test: TestFunc) -> TestFunc:
        if not hasattr(test, 'hypothesis_explicit_examples'):
            test.hypothesis_explicit_examples = self.hypothesis_explicit_examples
        test.hypothesis_explicit_examples.append(self._this_example)
        return test

    def xfail(self, condition: bool = True, *, reason: str = '', raises: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = BaseException) -> 'example':
        """Mark this example as an expected failure."""
        check_type(bool, condition, 'condition')
        check_type(str, reason, 'reason')
        if not (isinstance(raises, type) and issubclass(raises, BaseException)) and (
            not (isinstance(raises, tuple) and raises and 
            all(isinstance(r, type) and issubclass(r, BaseException) for r in raises)
        ):
            raise InvalidArgument(f'raises={raises!r} must be an exception type or tuple of exception types')
        if condition:
            self._this_example = attr.evolve(self._this_example, raises=raises, reason=reason)
        return self

    def via(self, whence: str, /) -> 'example':
        """Attach a machine-readable label noting whence this example came."""
        if not isinstance(whence, str):
            raise InvalidArgument('.via() must be passed a string')
        return self

def seed(seed: Hashable) -> Callable[[TestFunc], TestFunc]:
    """Set a specific random seed for test execution."""

    def accept(test: TestFunc) -> TestFunc:
        test._hypothesis_internal_use_seed = seed
        current_settings = getattr(test, '_hypothesis_internal_use_settings', None)
        test._hypothesis_internal_use_settings = Settings(current_settings, database=None)
        return test
    return accept

def reproduce_failure(version: str, blob: bytes) -> Callable[[TestFunc], TestFunc]:
    """Run the example that corresponds to this data blob to reproduce a failure."""

    def accept(test: TestFunc) -> TestFunc:
        test._hypothesis_internal_use_reproduce_failure = (version, blob)
        return test
    return accept

def encode_failure(choices: bytes) -> bytes:
    blob = choices_to_bytes(choices)
    compressed = zlib.compress(blob)
    if len(compressed) < len(blob):
        blob = b'\x01' + compressed
    else:
        blob = b'\x00' + blob
    return base64.b64encode(blob)

def decode_failure(blob: bytes) -> bytes:
    try:
        decoded = base64.b64decode(blob)
    except Exception:
        raise InvalidArgument(f'Invalid base64 encoded string: {blob!r}') from None
    prefix = decoded[:1]
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

def _invalid(message: str, *, exc: Type[Exception] = InvalidArgument, test: Callable[..., Any], given_kwargs: Dict[str, Any]) -> Callable[..., Any]:
    @impersonate(test)
    def wrapped_test(*arguments: Any, **kwargs: Any) -> None:
        raise exc(message)
    wrapped_test.is_hypothesis_test = True
    wrapped_test.hypothesis = HypothesisHandle(inner_test=test, get_fuzz_target=wrapped_test, given_kwargs=given_kwargs)
    return wrapped_test

def is_invalid_test(test: Callable[..., Any], original_sig: inspect.Signature, given_arguments: Tuple[Any, ...], given_kwargs: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    """Check the arguments to @given for basic usage constraints."""
    invalid = partial(_invalid, test=test, given_kwargs=given_kwargs)
    if not (given_arguments or given_kwargs):
        return invalid('given must be called with at least one argument')
    params = list(original_sig.parameters.values())
    pos_params = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    kwonly_params = [p for p in params if p.kind is p.KEYWORD_ONLY]
    if given_arguments and params != pos_params:
        return invalid('positional arguments to @given are not supported with varargs, varkeywords, positional-only, or keyword-only arguments')
    if len(given_arguments) > len(pos_params):
        return invalid(f'Too many positional arguments for {test.__name__}() were passed to @given - expected at most {len(pos_params)} arguments, but got {len(given_arguments)} {given_arguments!r}')
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
    if any(p.default is not p.empty for p in params):
        return invalid('Cannot apply @given to a function with defaults.')
    empty = [f'{s!r} (arg {idx})' for idx, s in enumerate(given_arguments) if s is NOTHING] + [
        f'{name}={s!r}' for name, s in given_kwargs.items() if s is NOTHING
    ]
    if empty:
        strats = 'strategies' if len(empty) > 1 else 'strategy'
        return invalid(f'Cannot generate examples from empty {strats}: ' + ', '.join(empty), exc=Unsatisfiable)
    return None

def execute_explicit_examples(
    state: 'StateForActualGivenExecution',
    wrapped_test: Callable[..., Any],
    arguments: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    original_sig: inspect.Signature
) -> Generator[Tuple[List[str], BaseException], None, None]:
    assert isinstance(state, StateForActualGivenExecution)
    posargs = [p.name for p in original_sig.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    for example in reversed(getattr(wrapped_test, 'hypothesis_explicit_examples', ())):
        assert isinstance(example, Example)
        if example.args:
            assert not example.kwargs
            if any(p.kind is p.POSITIONAL_ONLY for p in original_sig.parameters.values()):
                raise InvalidArgument('Cannot pass positional arguments to @example() when decorating a test function which has positional-only parameters.')
            if len(example.args) > len(posargs):
                raise InvalidArgument(f'example has too many arguments for test. Expected at most {len(posargs)} but got {len(example.args)}')
            example_kwargs = dict(zip(posargs[-len(example.args):], example.args))
        else:
            example_kwargs = dict(example.kwargs)
        given_kws = ', '.join(repr(k) for k in sorted(wrapped_test.hypothesis._given_kwargs))
        example_kws = ', '.join(repr(k) for k in sorted(example_kwargs))
        if given_kws != example_kws:
            raise InvalidArgument(f'Inconsistent args: @given() got strategies for {given_kws}, but @example() got arguments for {example_kws}') from None
        assert set(example_kwargs).isdisjoint(kwargs)
        example_kwargs.update(kwargs)
        if Phase.explicit not in state.settings.phases:
            continue
        with local_settings(state.settings):
            fragments_reported: List[str] = []
            empty_data = ConjectureData.for_choices([])
            try:
                execute_example = partial(state.execute_once, empty_data, is_final=True, print_example=True, example_kwargs=example_kwargs)
                with with_reporter(fragments_reported.append):
                    if example.raises is None:
                        execute_example()
                    else:
                        bits = ', '.join(nicerepr(x) for x in arguments) + ', '.join(f'{k}={nicerepr(v)}' for k, v in example_kwargs.items())
                        try:
                            execute_example()
                        except failure_exceptions_to_catch() as err:
                            if not isinstance(err, example.raises):
                                raise
                            state.xfail_example_reprs.add(repr_call(state.test, arguments, example_kwargs))
                        except example.raises as err:
                            raise InvalidArgument(f'@example({bits}) raised an expected {err!r}, but Hypothesis does not treat this as a test failure') from err
                        else:
                            reason = f' because {example.reason}' * bool(example.reason)
                            if example.raises is BaseException:
                                name = 'exception'
                            elif not isinstance(example.raises, tuple):
                                name = example.raises.__name__
                            elif len(example.raises) == 1:
                                name = example.raises[0].__name__
                            else:
                                name = ', '.join(ex.__name__ for ex in example.raises[:-1]) + f', or {example.raises[-1].__name__}'
                            vowel = name.upper()[0] in 'AEIOU'
                            raise AssertionError(f'Expected a{"n" * vowel} {name} from @example({bits}){reason}, but no exception was raised.')
            except UnsatisfiedAssumption:
                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
            except BaseException as err:
                err = err.with_traceback(get_trimmed_traceback())
                if isinstance(err, failure_exceptions_to_catch()) and any(
                    isinstance(arg, SearchStrategy) for arg in example.args + tuple(example.kwargs.values())
                ):
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
                    timing=state._timing_features
                )
                deliver_json_blob(tc)
            if fragments_reported:
                verbose_report(fragments_reported[0].replace('Falsifying', 'Trying', 1))
                for f in fragments_reported[1:]:
                    verbose_report(f)

def get_random_for_wrapped_test(test: Callable[..., Any], wrapped_test: Callable[..., Any]) -> Random:
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
        seed = _hypothesis_global_random.getrandbits(128)
        wrapped_test._hypothesis_internal_use_generated_seed = seed
        return Random(seed)

@attr.s(auto_attribs=True)
class Stuff:
    selfy: Optional[Any] = None
    args: Tuple[Any, ...] = attr.Factory(tuple)
    kwargs: Dict[str, Any] = attr.Factory(dict)
    given_kwargs: Dict[str, Any] = attr.Factory(dict)

def process_arguments_to_given(
    wrapped_test: Callable