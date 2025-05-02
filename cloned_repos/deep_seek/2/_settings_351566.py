"""The settings module configures runtime options for Hypothesis.

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
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, Iterator, List, NoReturn, 
    Optional, Set, Tuple, Type, TypeVar, Union, cast
)
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
all_settings: Dict[str, 'Setting'] = {}
T = TypeVar('T')

class settingsProperty:
    def __init__(self, name: str, *, show_default: bool) -> None:
        self.name = name
        self.show_default = show_default

    def __get__(self, obj: Optional['settings'], type: Optional[Type['settings']] = None) -> Any:
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

    def __set__(self, obj: 'settings', value: Any) -> None:
        obj.__dict__[self.name] = value

    def __delete__(self, obj: 'settings') -> NoReturn:
        raise AttributeError(f'Cannot delete attribute {self.name}')

    @property
    def __doc__(self) -> str:
        description = all_settings[self.name].description
        default = repr(getattr(settings.default, self.name)) if self.show_default else '(dynamically calculated)'
        return f'{description}\n\ndefault value: ``{default}``'

default_variable: DynamicVariable[Optional['settings']] = DynamicVariable[Optional['settings']](None)

class settingsMeta(type):
    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def default(cls) -> 'settings':
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, '_current_profile', None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def _assign_default_internal(cls, value: 'settings') -> None:
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
    __definitions_are_locked: ClassVar[bool] = False
    _profiles: ClassVar[Dict[str, 'settings']] = {}
    __module__: ClassVar[str] = 'hypothesis'
    _current_profile: ClassVar[Optional[str]] = None

    def __getattr__(self, name: str) -> Any:
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f'settings has no attribute {name}')

    def __init__(
        self,
        parent: Optional['settings'] = None,
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
        defaults = parent or settings.default
        if defaults is not None:
            for setting in all_settings.values():
                value = locals()[setting.name]
                if value is not_set:
                    object.__setattr__(self, setting.name, getattr(defaults, setting.name))
                else:
                    object.__setattr__(self, setting.name, setting.validator(value))

    def __call__(self, test: Any) -> Any:
        """Make the settings object (self) an attribute of the test.

        The settings are later discovered by looking them up on the test itself.
        """
        _test = test
        if not callable(_test):
            raise InvalidArgument(f'settings objects can be called as a decorator with @given, but decorated test={test!r} is not callable.')
        if inspect.isclass(test):
            from hypothesis.stateful import RuleBasedStateMachine
            if issubclass(_test, RuleBasedStateMachine):
                attr_name = '_hypothesis_internal_settings_applied'
                if getattr(test, attr_name, False):
                    raise InvalidArgument('Applying the @settings decorator twice would overwrite the first version; merge their arguments instead.')
                setattr(test, attr_name, True)
                _test.TestCase.settings = self
                return test
            else:
                raise InvalidArgument('@settings(...) can only be used as a decorator on functions, or on subclasses of RuleBasedStateMachine.')
        if hasattr(_test, '_hypothesis_internal_settings_applied'):
            descr = get_pretty_function_description(test)
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

            def validator(value: Any) -> Any:
                if value not in options:
                    msg = f'Invalid {name}, {value!r}. Valid options: {options!r}'
                    raise InvalidArgument(msg)
                return value
        else:
            assert validator is not None
        all_settings[name] = Setting(name=name, description=description.strip(), default=default, validator=validator)
        setattr(settings, name, settingsProperty(name, show_default=show_default))

    @classmethod
    def lock_further_definitions(cls) -> None:
        settings.__definitions_are_locked = True

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError('settings objects are immutable')

    def __repr__(self) -> str:
        bits = sorted((f'{name}={getattr(self, name)!r}' for name in all_settings if name != 'backend' or len(AVAILABLE_PROVIDERS) > 1))
        return 'settings({})'.format(', '.join(bits))

    def show_changed(self) -> str:
        bits = []
        for name, setting in all_settings.items():
            value = getattr(self, name)
            if value != setting.default:
                bits.append(f'{name}={value!r}')
        return ', '.join(sorted(bits, key=len))

    @staticmethod
    def register_profile(name: str, parent: Optional['settings'] = None, **kwargs: Any) -> None:
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
    def get_profile(name: str) -> 'settings':
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
def local_settings(s: settings) -> Iterator[settings]:
    with default_variable.with_value(s):
        yield s

@attr.s()
class Setting:
    name: str = attr.ib()
    description: str = attr.ib()
    default: Any = attr.ib()
    validator: ValidatorT = attr.ib()

def _max_examples_validator(x: Any) -> int:
    check_type(int, x, name='max_examples')
    if x < 1:
        raise InvalidArgument(f'max_examples={x!r} should be at least one. You can disable example generation with the `phases` setting instead.')
    return x

settings._define_setting(
    'max_examples',
    default=100,
    validator=_max_examples_validator,
    description="\nOnce this many satisfying examples have been considered without finding any\ncounter-example, Hypothesis will stop looking.\n\nNote that we might call your test function fewer times if we find a bug early\nor can tell that we've exhausted the search space; or more if we discard some\nexamples due to use of .filter(), assume(), or a few other things that can\nprevent the test case from completing successfully.\n\nThe default value is chosen to suit a workflow where the test will be part of\na suite that is regularly executed locally or on a CI server, balancing total\nrunning time against the chance of missing a bug.\n\nIf you are writing one-off tests, running tens of thousands of examples is\nquite reasonable as Hypothesis may miss uncommon bugs with default settings.\nFor very complex code, we have observed Hypothesis finding novel bugs after\n*several million* examples while testing :pypi:`SymPy <sympy>`.\nIf you are running more than 100k examples for a test, consider using our\n:ref:`integration for coverage-guided fuzzing <fuzz_one_input>` - it really\nshines when given minutes or hours to run.\n"
)

settings._define_setting(
    'derandomize',
    default=False,
    options=(True, False),
    description="\nIf True, seed Hypothesis' random number generator using a hash of the test\nfunction, so that every run will test the same set of examples until you\nupdate Hypothesis, Python, or the test function.\n\nThis allows you to `check for regressions and look for bugs\n<https://blog.nelhage.com/post/two-kinds-of-testing/>`__ using\n:ref:`separate settings profiles <settings_profiles>` - for example running\nquick deterministic tests on every commit, and a longer non-deterministic\nnightly testing run.\n\nBy default when running on CI, this will be set to True.\n"
)

def _validate_database(db: Any) -> Optional['ExampleDatabase']:
    from hypothesis.database import ExampleDatabase
    if db is None or isinstance(db, ExampleDatabase):
        return db
    raise InvalidArgument(f'Arguments to the database setting must be None or an instance of ExampleDatabase.  Try passing database=ExampleDatabase({db!r}), or construct and use one of the specific subclasses in hypothesis.database')

settings._define_setting(
    'database',
    default=not_set,
    show_default=False,
    description='\nAn instance of :class:`~hypothesis.database.ExampleDatabase` that will be\nused to save examples to and load previous examples from. May be ``None``\nin which case no storage will be used.\n\nSee the :doc:`example database documentation <database>` for a list of built-in\nexample database implementations, and how to define custom implementations.\n',
    validator=_validate_database
)

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
    def __iter__(self) -> Iterator['HealthCheck']:
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return iter((x for x in super().__iter__() if x not in deprecated))

@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    """Arguments for :attr:`~hypothesis.settings.suppress_health_check`.

    Each member of this enum is a type of health check to suppress.
    """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'

    @classmethod
    def all(cls) -> List['HealthCheck']:
        note_deprecation('`HealthCheck.all()` is deprecated; use `list(HealthCheck)` instead.', since='2023-04-16', has_codemod=True, stacklevel=1)
        return list(HealthCheck)
    
    data_too_large = 1
    filter_too_much = 2
    too_slow = 3
    return_value = 5
    large_base_example = 7
    not_a_test_method = 8
    function_scoped_fixture = 9
    differing_executors = 10

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
    phases = tuple(phases)
    for a in phases:
        if not isinstance(a, Phase):
            raise InvalidArgument(f'{a!r} is not a valid phase')
    return tuple((p for p in list(Phase) if p in phases))

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

def validate_health_check_s