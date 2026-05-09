from typing import Callable, ClassVar, Optional, TypeVar, Union
from enum import Enum, EnumMeta, IntEnum, unique
from attr import s
from collections import abc
import datetime
import os
import warnings
import contextlib
import inspect
import hypothesis
from hypothesis.errors import HypothesisDeprecationWarning, InvalidArgument, InvalidState
from hypothesis.database import ExampleDatabase
from hypothesis.utils.conventions import not_set
from hypothesis.utils.dynamicvariables import DynamicVariable

class settingsProperty:
    def __init__(self, name: str, *, show_default: bool):
        self.name = name
        self.show_default = show_default

    def __get__(self, obj, type=None):
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

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

    def __delete__(self, obj):
        raise AttributeError(f'Cannot delete attribute {self.name}')

    @property
    def __doc__(self):
        description = all_settings[self.name].description
        default = repr(getattr(settings.default, self.name)) if self.show_default else '(dynamically calculated)'
        return f'{description}\n\ndefault value: ``{default}``'

class settingsMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default(cls):
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, '_current_profile', None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def _assign_default_internal(cls, value):
        default_variable.value = value

    def __setattr__(cls, name, value):
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
    __definitions_are_locked = False
    _profiles = {}
    __module__ = 'hypothesis'
    _current_profile = None

    def __getattr__(self, name):
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f'settings has no attribute {name}')

    def __init__(self, parent=None, *, max_examples: int, derandomize: bool, database: Optional[ExampleDatabase], verbosity: int, phases: tuple, stateful_step_count: int, report_multiple_bugs: bool, suppress_health_check: Union[tuple, list], deadline: Union[datetime.timedelta, int, float], print_blob: bool, backend: str):
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

    def __call__(self, test: Callable):
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
    def _define_setting(cls, name: str, description: str, *, default, options: tuple = None, validator: Callable = None, show_default: bool = True):
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

            def validator(value):
                if value not in options:
                    msg = f'Invalid {name}, {value!r}. Valid options: {options!r}'
                    raise InvalidArgument(msg)
                return value
        else:
            assert validator is not None
        all_settings[name] = Setting(name=name, description=description.strip(), default=default, validator=validator)
        setattr(settings, name, settingsProperty(name, show_default=show_default))

    @classmethod
    def lock_further_definitions(cls):
        settings.__definitions_are_locked = True

    def __setattr__(self, name, value):
        raise AttributeError('settings objects are immutable')

    def __repr__(self):
        bits = sorted((f'{name}={getattr(self, name)!r}' for name in all_settings if name != 'backend' or len(AVAILABLE_PROVIDERS) > 1))
        return 'settings({})'.format(', '.join(bits))

    def show_changed(self):
        bits = []
        for name, setting in all_settings.items():
            value = getattr(self, name)
            if value != setting.default:
                bits.append(f'{name}={value!r}')
        return ', '.join(sorted(bits, key=len))

    @staticmethod
    def register_profile(name: str, parent=None, **kwargs):
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
    def get_profile(name: str):
        """Return the profile with the given name."""
        check_type(str, name, 'name')
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f'Profile {name!r} is not registered') from None

    @staticmethod
    def load_profile(name: str):
        """Loads in the settings defined in the profile provided.

        If the profile does not exist, InvalidArgument will be raised.
        Any setting not defined in the profile will be the library
        defined default for that setting.
        """
        check_type(str, name, 'name')
        settings._current_profile = name
        settings._assign_default_internal(settings.get_profile(name))

@contextlib.contextmanager
def local_settings(s):
    with default_variable.with_value(s):
        yield s

@attr.s()
class Setting:
    name = attr.ib()
    description = attr.ib()
    default = attr.ib()
    validator = attr.ib()

def _max_examples_validator(x):
    check_type(int, x, name='max_examples')
    if x < 1:
        raise InvalidArgument(f'max_examples={x!r} should be at least one. You can disable example generation with the `phases` setting instead.')
    return x
settings._define_setting('max_examples', default=100, validator=_max_examples_validator, description="\nOnce this many satisfying examples have been considered without finding any\ncounter-example, Hypothesis will stop looking.\n\nNote that we might call your test function fewer times if we find a bug early\nor can tell that we've exhausted the search space; or more if we discard some\nexamples due to use of .filter(), assume(), or a few other things that can\nprevent the test case from completing successfully.\n\nThe default value is chosen to suit a workflow where the test will be part of\na suite that is regularly executed locally or on a CI server, balancing total\nrunning time against the chance of missing a bug.\n\nIf you are writing one-off tests, running tens of thousands of examples is\nquite reasonable as Hypothesis may miss uncommon bugs with default settings.\nFor very complex code, we have observed Hypothesis finding novel bugs after\n*several million* examples while testing :pypi:`SymPy <sympy>`.\nIf you are running more than 100k examples for a test, consider using our\n:ref:`integration for coverage-guided fuzzing <fuzz_one_input>` - it really\nshines when given minutes or hours to run.\n")

def _validate_database(db):
    from hypothesis.database import ExampleDatabase
    if db is None or isinstance(db, ExampleDatabase):
        return db
    raise InvalidArgument(f'Arguments to the database setting must be None or an instance of ExampleDatabase.  Try passing database=ExampleDatabase({db!r}), or construct and use one of the specific subclasses in hypothesis.database')
settings._define_setting('database', default=not_set, show_default=False, description='\nAn instance of :class:`~hypothesis.database.ExampleDatabase` that will be\nused to save examples to and load previous examples from. May be ``None``\nin which case no storage will be used.\n\nSee the :doc:`example database documentation <database>` for a list of built-in\nexample database implementations, and how to define custom implementations.\n', validator=_validate_database)

@unique
class Phase(IntEnum):
    """Phases of the test.

    - explicit: The test is run with explicit examples.
    - reuse: The test is run with reused examples.
    - generate: The test is run with generated examples.
    - target: The test is run with targeted examples.
    - shrink: The test is run with shrunk examples.
    - explain: The test is run with explained examples.
    """
    explicit = 0
    reuse = 1
    generate = 2
    target = 3
    shrink = 4
    explain = 5

    def __repr__(self):
        return f'Phase.{self.name}'

class HealthCheckMeta(EnumMeta):

    def __iter__(self):
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return iter((x for x in super().__iter__() if x not in deprecated))

@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    """Arguments for :attr:`~hypothesis.settings.suppress_health_check`.

    Each member of this enum is a type of health check to suppress.
    """

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'

    @classmethod
    def all(cls):
        note_deprecation('`HealthCheck.all()` is deprecated; use `list(HealthCheck)` instead.', since='2023-04-16', has_codemod=True, stacklevel=1)
        return list(HealthCheck)
    data_too_large = 1
    'Checks if too many examples are aborted for being too large.\n\n    This is