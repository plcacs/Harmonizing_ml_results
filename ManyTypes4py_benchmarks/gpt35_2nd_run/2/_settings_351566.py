from typing import TYPE_CHECKING, Any, Callable, ClassVar, NoReturn, Optional, TypeVar, Union

ValidatorT = Callable[[Any], object]
T = TypeVar('T')

class settingsProperty:
    def __init__(self, name: str, *, show_default: bool) -> None:
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
    def __doc__(self) -> str:
        description = all_settings[self.name].description
        default = repr(getattr(settings.default, self.name)) if self.show_default else '(dynamically calculated)'
        return f'{description}\n\ndefault value: ``{default}``'

default_variable = DynamicVariable[Optional['settings']](None)

class settingsMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default(cls) -> Optional['settings']:
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, '_current_profile', None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def _assign_default_internal(cls, value: Optional['settings']) -> None:
        default_variable.value = value

    def __setattr__(cls, name: str, value: Any) -> None:
        if name == 'default':
            raise AttributeError('Cannot assign to the property settings.default - consider using settings.load_profile instead.')
        elif not (isinstance(value, settingsProperty) or name.startswith('_')):
            raise AttributeError(f'Cannot assign hypothesis.settings.{name}={value!r} - the settings class is immutable.  You can change the global default settings with settings.load_profile, or use @settings(...) to decorate your test instead.')
        super().__setattr__(name, value)

class settings(metaclass=settingsMeta):
    __definitions_are_locked: bool = False
    _profiles: dict = {}
    __module__: str = 'hypothesis'
    _current_profile: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f'settings has no attribute {name}')

    def __init__(self, parent: Optional['settings'] = None, *, max_examples: int = not_set, derandomize: bool = not_set, database: Any = not_set, verbosity: Any = not_set, phases: Any = not_set, stateful_step_count: Any = not_set, report_multiple_bugs: Any = not_set, suppress_health_check: Any = not_set, deadline: Any = not_set, print_blob: Any = not_set, backend: Any = not_set) -> None:
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

    def __call__(self, test: Callable) -> Callable:
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
    def _define_setting(cls, name: str, description: str, *, default: Any, options: Optional[tuple] = None, validator: Optional[Callable[[Any], Any]] = None, show_default: bool = True) -> None:
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
    def lock_further_definitions(cls) -> None:
        settings.__definitions_are_locked = True

    def __setattr__(self, name: str, value: Any) -> None:
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
        check_type(str, name, 'name')
        settings._profiles[name] = settings(parent=parent, **kwargs)
        if settings._current_profile == name:
            settings.load_profile(name)

    @staticmethod
    def get_profile(name: str) -> 'settings':
        check_type(str, name, 'name')
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f'Profile {name!r} is not registered') from None

    @staticmethod
    def load_profile(name: str) -> None:
        check_type(str, name, 'name')
        settings._current_profile = name
        settings._assign_default_internal(settings.get_profile(name))

@contextlib.contextmanager
def local_settings(s: 'settings') -> None:
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

def _validate_database(db: Any) -> Any:
    from hypothesis.database import ExampleDatabase
    if db is None or isinstance(db, ExampleDatabase):
        return db
    raise InvalidArgument(f'Arguments to the database setting must be None or an instance of ExampleDatabase.  Try passing database=ExampleDatabase({db!r}), or construct and use one of the specific subclasses in hypothesis.database')

def _validate_phases(phases: Any) -> tuple:
    phases = tuple(phases)
    for a in phases:
        if not isinstance(a, Phase):
            raise InvalidArgument(f'{a!r} is not a valid phase')
    return tuple((p for p in list(Phase) if p in phases))

def _validate_stateful_step_count(x: Any) -> int:
    check_type(int, x, name='stateful_step_count')
    if x < 1:
        raise InvalidArgument(f'stateful_step_count={x!r} must be at least one.')
    return x

def validate_health_check_suppressions(suppressions: Any) -> list:
    suppressions = try_convert(list, suppressions, 'suppress_health_check')
    for s in suppressions:
        if not isinstance(s, HealthCheck):
            raise InvalidArgument(f'Non-HealthCheck value {s!r} of type {type(s).__name__} is invalid in suppress_health_check.')
        if s in (HealthCheck.return_value, HealthCheck.not_a_test_method):
            note_deprecation(f'The {s.name} health check is deprecated, because this is always an error.', since='2023-03-15', has_codemod=False, stacklevel=2)
    return suppressions

def _validate_deadline(x: Any) -> Any:
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

def _backend_validator(value: Any) -> str:
    if value not in AVAILABLE_PROVIDERS:
        if value == 'crosshair':
            install = '`pip install "hypothesis[crosshair]"` and try again.'
            raise InvalidArgument(f'backend={value!r} is not available.  {install}')
        raise InvalidArgument(f'backend={value!r} is not available - maybe you need to install a plugin?\n    Installed backends: {sorted(AVAILABLE_PROVIDERS)!r}')
    return value
