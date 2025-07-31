#!/usr/bin/env python3
"""
The settings module configures runtime options for Hypothesis.

Either an explicit settings object can be used or the default object on
this module can be modified.
"""
import contextlib
import datetime
import inspect
import os
import warnings
from collections.abc import Collection, Generator, Sequence
from enum import Enum, EnumMeta, IntEnum, unique
from typing import TYPE_CHECKING, Any, Callable, ClassVar, NoReturn, Optional, TypeVar, Union, List, Tuple
import attr
from hypothesis.errors import HypothesisDeprecationWarning, InvalidArgument, InvalidState
from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type, try_convert
from hypothesis.utils.conventions import not_set
from hypothesis.utils.dynamicvariables import DynamicVariable

if TYPE_CHECKING:
    from typing import TypeAlias
    from hypothesis.database import ExampleDatabase

__all__ = ['settings']
ValidatorT = Callable[[Any], object]
all_settings: dict[str, "Setting"] = {}
T = TypeVar('T')


class settingsProperty:
    def __init__(self, name: str, *, show_default: bool) -> None:
        self.name: str = name
        self.show_default: bool = show_default

    def __get__(self, obj: Optional[Any], type: Optional[Any] = None) -> Any:
        if obj is None:
            return self
        else:
            try:
                result = obj.__dict__[self.name]
                if self.name == 'database' and result is not_set:
                    from hypothesis.database import ExampleDatabase
                    result = ExampleDatabase(not_set)
                assert result is not not_set
                return result
            except KeyError:
                raise AttributeError(self.name) from None

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.name] = value

    def __delete__(self, obj: Any) -> None:
        raise AttributeError(f'Cannot delete attribute {self.name}')

    @property
    def __doc__(self) -> str:
        description: str = all_settings[self.name].description
        default: str = repr(getattr(settings.default, self.name)) if self.show_default else '(dynamically calculated)'
        return f'{description}\n\ndefault value: ``{default}``'


default_variable: DynamicVariable[Optional["settings"]] = DynamicVariable(None)


class settingsMeta(type):
    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def default(cls) -> "settings":
        v: Optional["settings"] = default_variable.value
        if v is not None:
            return v
        if getattr(settings, '_current_profile', None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value  # type: ignore

    def _assign_default_internal(cls, value: "settings") -> None:
        default_variable.value = value

    def __setattr__(cls, name: str, value: Any) -> None:
        if name == 'default':
            raise AttributeError('Cannot assign to the property settings.default - consider using settings.load_profile instead.')
        elif not (isinstance(value, settingsProperty) or name.startswith('_')):
            raise AttributeError(f'Cannot assign hypothesis.settings.{name}={value!r} - the settings class is immutable.  You can change the global default settings with settings.load_profile, or use @settings(...) to decorate your test instead.')
        super().__setattr__(name, value)


class settings(metaclass=settingsMeta):
    """A settings object configures options including verbosity, runtime controls,
    persistence, determinism, and more.

    Default values are picked up from the settings.default object and
    changes made there will be picked up in newly created settings.
    """
    __definitions_are_locked: bool = False
    _profiles: dict[str, "settings"] = {}
    __module__ = 'hypothesis'
    _current_profile: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f'settings has no attribute {name}')

    def __init__(
        self,
        parent: Optional["settings"] = None,
        *,
        max_examples: Any = not_set,
        derandomize: Any = not_set,
        database: Any = not_set,
        verbosity: Any = not_set,
        phases: Any = not_set,
        stateful_step_count: Any = not_set,
        report_multiple_bugs: Any = not_set,
        suppress_health_check: Any = not_set,
        deadline: Any = not_set,
        print_blob: Any = not_set,
        backend: Any = not_set
    ) -> None:
        if parent is not None:
            check_type(settings, parent, 'parent')
        if derandomize not in (not_set, False):
            if database not in (not_set, None):
                raise InvalidArgument(f'derandomize=True implies database=None, so passing database={database!r} too is invalid.')
            database = None
        defaults: Optional["settings"] = parent or settings.default
        if defaults is not None:
            for setting in all_settings.values():
                value = locals()[setting.name]
                if value is not_set:
                    object.__setattr__(self, setting.name, getattr(defaults, setting.name))
                else:
                    object.__setattr__(self, setting.name, setting.validator(value))

    def __call__(self, test: Callable[..., Any]) -> Callable[..., Any]:
        """Make the settings object (self) an attribute of the test.

        The settings are later discovered by looking them up on the test itself.
        """
        _test: Any = test
        if not callable(_test):
            raise InvalidArgument(f'settings objects can be called as a decorator with @given, but decorated test={test!r} is not callable.')
        if inspect.isclass(test):
            from hypothesis.stateful import RuleBasedStateMachine
            if issubclass(_test, RuleBasedStateMachine):
                attr_name: str = '_hypothesis_internal_settings_applied'
                if getattr(test, attr_name, False):
                    raise InvalidArgument('Applying the @settings decorator twice would overwrite the first version; merge their arguments instead.')
                setattr(test, attr_name, True)
                _test.TestCase.settings = self
                return test
            else:
                raise InvalidArgument('@settings(...) can only be used as a decorator on functions, or on subclasses of RuleBasedStateMachine.')
        if hasattr(_test, '_hypothesis_internal_settings_applied'):
            descr: str = get_pretty_function_description(test)
            raise InvalidArgument(f'{descr} has already been decorated with a settings object.\n    Previous:  {_test._hypothesis_internal_use_settings!r}\n    This:  {self!r}')
        _test._hypothesis_internal_use_settings = self
        _test._hypothesis_internal_settings_applied = True
        return test

    @classmethod
    def _define_setting(
        cls,
        name: str,
        description: str,
        *,
        default: Any,
        options: Optional[Collection[Any]] = None,
        validator: Optional[ValidatorT] = None,
        show_default: bool = True
    ) -> None:
        """Add a new setting.

        - name is the name of the property that will be used to access the
          setting. This must be a valid python identifier.
        - description will appear in the property's docstring
        - default is the default value. This may be a zero argument
          function in which case it is evaluated and its result is stored
          the first time it is accessed on any given settings object.
        """
        if settings.__definitions_are_locked:
            raise InvalidState('settings have been locked and may no longer be defined.')
        if options is not None:
            options = tuple(options)
            assert default in options

            def validator_inner(value: Any) -> Any:
                if value not in options:
                    msg = f'Invalid {name}, {value!r}. Valid options: {options!r}'
                    raise InvalidArgument(msg)
                return value
            validator = validator_inner  # type: ignore
        else:
            assert validator is not None
        all_settings[name] = Setting(name=name, description=description.strip(), default=default, validator=validator)  # type: ignore
        setattr(settings, name, settingsProperty(name, show_default=show_default))

    @classmethod
    def lock_further_definitions(cls) -> None:
        settings.__definitions_are_locked = True

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('settings objects are immutable')

    def __repr__(self) -> str:
        bits = sorted(
            (f'{name}={getattr(self, name)!r}'
             for name in all_settings if name != 'backend' or len(AVAILABLE_PROVIDERS) > 1)
        )
        return 'settings({})'.format(', '.join(bits))

    def show_changed(self) -> str:
        bits: List[str] = []
        for name, setting in all_settings.items():
            value = getattr(self, name)
            if value != setting.default:
                bits.append(f'{name}={value!r}')
        return ', '.join(sorted(bits, key=len))

    @staticmethod
    def register_profile(name: str, parent: Optional["settings"] = None, **kwargs: Any) -> None:
        """Registers a collection of values to be used as a settings profile.

        Settings profiles can be loaded by name - for example, you might
        create a 'fast' profile which runs fewer examples, keep the 'default'
        profile, and create a 'ci' profile that increases the number of
        examples and uses a different database to store failures.

        The arguments to this method are exactly as for
        :class:`~hypothesis.settings`: optional ``parent`` settings, and
        keyword arguments for each setting that will be set differently to
        parent (or settings.default, if parent is None).

        If you register a profile that has already been defined and that profile
        is the currently loaded profile, the new changes will take effect immediately,
        and do not require reloading the profile.
        """
        check_type(str, name, 'name')
        settings._profiles[name] = settings(parent=parent, **kwargs)
        if settings._current_profile == name:
            settings.load_profile(name)

    @staticmethod
    def get_profile(name: str) -> "settings":
        """Return the profile with the given name."""
        check_type(str, name, 'name')
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f'Profile {name!r} is not registered') from None

    @staticmethod
    def load_profile(name: str) -> None:
        """Loads in the settings defined in the profile provided.

        If the profile does not exist, InvalidArgument will be raised.
        Any setting not defined in the profile will be the library
        defined default for that setting.
        """
        check_type(str, name, 'name')
        settings._current_profile = name
        settings._assign_default_internal(settings.get_profile(name))


@contextlib.contextmanager
def local_settings(s: "settings") -> Generator["settings", None, None]:
    with default_variable.with_value(s):
        yield s


@attr.s()
class Setting:
    name: str = attr.ib()
    description: str = attr.ib()
    default: Any = attr.ib()
    validator: Callable[[Any], Any] = attr.ib()


def _max_examples_validator(x: Any) -> int:
    check_type(int, x, name='max_examples')
    if x < 1:
        raise InvalidArgument(f'max_examples={x!r} should be at least one. You can disable example generation with the `phases` setting instead.')
    return x


def _validate_database(db: Any) -> Optional["ExampleDatabase"]:
    from hypothesis.database import ExampleDatabase
    if db is None or isinstance(db, ExampleDatabase):
        return db
    raise InvalidArgument(f'Arguments to the database setting must be None or an instance of ExampleDatabase.  Try passing database=ExampleDatabase({db!r}), or construct and use one of the specific subclasses in hypothesis.database')


@unique
class Phase(IntEnum):
    explicit = 0
    reuse = 1
    generate = 2
    target = 3
    shrink = 4
    explain = 5

    def __repr__(self) -> str:
        return f'Phase.{self.name}'


class HealthCheckMeta(EnumMeta):
    def __iter__(self) -> Generator["HealthCheck", None, None]:
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return (x for x in super().__iter__() if x not in deprecated)


@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    """Arguments for :attr:`~hypothesis.settings.suppress_health_check`.

    Each member of this enum is a type of health check to suppress.
    """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'

    @classmethod
    def all(cls) -> List["HealthCheck"]:
        note_deprecation('`HealthCheck.all()` is deprecated; use `list(HealthCheck)` instead.', since='2023-04-16', has_codemod=True, stacklevel=1)
        return list(HealthCheck)

    data_too_large = 1
    'Checks if too many examples are aborted for being too large.\n\n    This is measured by the number of random choices that Hypothesis makes\n    in order to generate something, not the size of the generated object.\n    For example, choosing a 100MB object from a predefined list would take\n    only a few bits, while generating 10KB of JSON from scratch might trigger\n    this health check.\n    '
    filter_too_much = 2
    'Check for when the test is filtering out too many examples, either\n    through use of :func:`~hypothesis.assume()` or :ref:`filter() <filtering>`,\n    or occasionally for Hypothesis internal reasons.'
    too_slow = 3
    'Check for when your data generation is extremely slow and likely to hurt\n    testing.'
    return_value = 5
    'Deprecated; we always error if a test returns a non-None value.'
    large_base_example = 7
    'Checks if the natural example to shrink towards is very large.'
    not_a_test_method = 8
    'Deprecated; we always error if :func:`@given <hypothesis.given>` is applied\n    to a method defined by :class:`python:unittest.TestCase` (i.e. not a test).'
    function_scoped_fixture = 9
    'Checks if :func:`@given <hypothesis.given>` has been applied to a test\n    with a pytest function-scoped fixture. Function-scoped fixtures run once\n    for the whole function, not once per example, and this is usually not what\n    you want.\n\n    Because of this limitation, tests that need to set up or reset\n    state for every example need to do so manually within the test itself,\n    typically using an appropriate context manager.\n\n    Suppress this health check only in the rare case that you are using a\n    function-scoped fixture that does not need to be reset between individual\n    examples, but for some reason you cannot use a wider fixture scope\n    (e.g. session scope, module scope, class scope).'
    differing_executors = 10
    'Checks if :func:`@given <hypothesis.given>` has been applied to a test\n    which is executed by different :ref:`executors<custom-function-execution>`.\n    If your test function is defined as a method on a class, that class will be\n    your executor, and subclasses executing an inherited test is a common way\n    for things to go wrong.\n\n    The correct fix is often to bring the executor instance under the control\n    of hypothesis by explicit parametrization over, or sampling from,\n    subclasses, or to refactor so that :func:`@given <hypothesis.given>` is\n    specified on leaf subclasses.'


@unique
class Verbosity(IntEnum):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3

    def __repr__(self) -> str:
        return f'Verbosity.{self.name}'


settings._define_setting(
    'verbosity',
    options=tuple(Verbosity),
    default=Verbosity.normal,
    description='Control the verbosity level of Hypothesis messages'
)

def _validate_phases(phases: Any) -> Tuple[Phase, ...]:
    phases_tuple: Tuple[Phase, ...] = tuple(phases)
    for a in phases_tuple:
        if not isinstance(a, Phase):
            raise InvalidArgument(f'{a!r} is not a valid phase')
    return tuple((p for p in list(Phase) if p in phases_tuple))

settings._define_setting(
    'phases',
    default=tuple(Phase),
    description='Control which phases should be run. See :ref:`the full documentation for more details <phases>`',
    validator=_validate_phases
)

def _validate_stateful_step_count(x: Any) -> int:
    check_type(int, x, name='stateful_step_count')
    if x < 1:
        raise InvalidArgument(f'stateful_step_count={x!r} must be at least one.')
    return x

settings._define_setting(
    name='stateful_step_count',
    default=50,
    validator=_validate_stateful_step_count,
    description='\nNumber of steps to run a stateful program for before giving up on it breaking.\n'
)

settings._define_setting(
    name='report_multiple_bugs',
    default=True,
    options=(True, False),
    description='\nBecause Hypothesis runs the test many times, it can sometimes find multiple\nbugs in a single run.  Reporting all of them at once is usually very useful,\nbut replacing the exceptions can occasionally clash with debuggers.\nIf disabled, only the exception with the smallest minimal example is raised.\n'
)

def validate_health_check_suppressions(suppressions: Any) -> List[HealthCheck]:
    suppressions_list: List[Any] = try_convert(list, suppressions, 'suppress_health_check')
    for s in suppressions_list:
        if not isinstance(s, HealthCheck):
            raise InvalidArgument(f'Non-HealthCheck value {s!r} of type {type(s).__name__} is invalid in suppress_health_check.')
        if s in (HealthCheck.return_value, HealthCheck.not_a_test_method):
            note_deprecation(f'The {s.name} health check is deprecated, because this is always an error.', since='2023-03-15', has_codemod=False, stacklevel=2)
    return suppressions_list  # type: ignore

settings._define_setting(
    'suppress_health_check',
    default=(),
    description='A list of :class:`~hypothesis.HealthCheck` items to disable.',
    validator=validate_health_check_suppressions
)

class duration(datetime.timedelta):
    """A timedelta specifically measured in milliseconds."""
    def __repr__(self) -> str:
        ms: float = self.total_seconds() * 1000
        return f'timedelta(milliseconds={(int(ms) if ms == int(ms) else ms)!r})'

def _validate_deadline(x: Any) -> Optional[duration]:
    if x is None:
        return x
    invalid_deadline_error = InvalidArgument(f'deadline={x!r} (type {type(x).__name__}) must be a timedelta object, an integer or float number of milliseconds, or None to disable the per-test-case deadline.')
    if isinstance(x, (int, float)):
        if isinstance(x, bool):
            raise invalid_deadline_error
        try:
            x = duration(milliseconds=x)
        except OverflowError:
            raise InvalidArgument(f'deadline={x!r} is invalid, because it is too large to represent as a timedelta. Use deadline=None to disable deadlines.') from None
    if isinstance(x, datetime.timedelta):
        if x <= datetime.timedelta(0):
            raise InvalidArgument(f'deadline={x!r} is invalid, because it is impossible to meet a deadline <= 0. Use deadline=None to disable deadlines.')
        return duration(seconds=x.total_seconds())
    raise invalid_deadline_error

settings._define_setting(
    'deadline',
    default=duration(milliseconds=200),
    validator=_validate_deadline,
    description='\nIf set, a duration (as timedelta, or integer or float number of milliseconds)\nthat each individual example (i.e. each time your test\nfunction is called, not the whole decorated test) within a test is not\nallowed to exceed. Tests which take longer than that may be converted into\nerrors (but will not necessarily be if close to the deadline, to allow some\nvariability in test run time).\n\nSet this to ``None`` to disable this behaviour entirely.\n\nBy default when running on CI, this will be set to None.\n'
)

def is_in_ci() -> bool:
    return 'CI' in os.environ or 'TF_BUILD' in os.environ

settings._define_setting(
    'print_blob',
    default=False,
    options=(True, False),
    description='\nIf set to ``True``, Hypothesis will print code for failing examples that can be used with\n:func:`@reproduce_failure <hypothesis.reproduce_failure>` to reproduce the failing example.\n'
)

def _backend_validator(value: Any) -> str:
    if value not in AVAILABLE_PROVIDERS:
        if value == 'crosshair':
            install: str = '`pip install "hypothesis[crosshair]"` and try again.'
            raise InvalidArgument(f'backend={value!r} is not available.  {install}')
        raise InvalidArgument(f'backend={value!r} is not available - maybe you need to install a plugin?\n    Installed backends: {sorted(AVAILABLE_PROVIDERS)!r}')
    return value

settings._define_setting(
    'backend',
    default='hypothesis',
    show_default=False,
    validator=_backend_validator,
    description='\nEXPERIMENTAL AND UNSTABLE - see :ref:`alternative-backends`.\nThe importable name of a backend which Hypothesis should use to generate primitive\ntypes.  We aim to support heuristic-random, solver-based, and fuzzing-based backends.\n'
)

settings.lock_further_definitions()

def note_deprecation(message: str, *, since: str, has_codemod: bool, stacklevel: int = 0) -> None:
    if since != 'RELEASEDAY':
        date: datetime.date = datetime.date.fromisoformat(since)
        assert datetime.date(2021, 1, 1) <= date
    if has_codemod:
        message += '\n    The `hypothesis codemod` command-line tool can automatically refactor your code to fix this warning.'
    warnings.warn(HypothesisDeprecationWarning(message), stacklevel=2 + stacklevel)

settings.register_profile('default', settings())
settings.load_profile('default')
assert settings.default is not None
CI = settings(derandomize=True, deadline=None, database=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile('ci', CI)
if is_in_ci():
    settings.load_profile('ci')
assert settings.default is not None
assert set(all_settings) == {p.name for p in inspect.signature(settings.__init__).parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY}