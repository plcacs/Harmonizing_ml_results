```python
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The settings module configures runtime options for Hypothesis.

Either an explicit settings object can be used or the default object on
this module can be modified.
"""

from __future__ import annotations

import contextlib
import datetime
import inspect
import os
import warnings
from collections.abc import Collection, Generator, Sequence
from enum import Enum, EnumMeta, IntEnum, unique
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    cast,
)

import attr

from hypothesis.errors import (
    HypothesisDeprecationWarning,
    InvalidArgument,
    InvalidState,
)
from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type, try_convert
from hypothesis.utils.conventions import not_set
from hypothesis.utils.dynamicvariables import DynamicVariable

if TYPE_CHECKING:
    from typing import TypeAlias

    from hypothesis.database import ExampleDatabase

__all__ = ["settings"]

ValidatorT: TypeAlias = Callable[[Any], object]
all_settings: dict[str, Setting] = {}

T = TypeVar("T")


class settingsProperty:
    def __init__(self, name: str, *, show_default: bool) -> None:
        self.name = name
        self.show_default = show_default

    def __get__(self, obj: object | None, type: type | None = None) -> Any:
        if obj is None:
            return self
        else:
            try:
                result = obj.__dict__[self.name]
                if self.name == "database" and result is not_set:
                    from hypothesis.database import ExampleDatabase

                    result = ExampleDatabase(not_set)
                assert result is not not_set
                return result
            except KeyError:
                raise AttributeError(self.name) from None

    def __set__(self, obj: object, value: Any) -> None:
        obj.__dict__[self.name] = value

    def __delete__(self, obj: object) -> NoReturn:
        raise AttributeError(f"Cannot delete attribute {self.name}")

    @property
    def __doc__(self) -> str:
        description = all_settings[self.name].description
        default = (
            repr(getattr(settings.default, self.name))
            if self.show_default
            else "(dynamically calculated)"
        )
        return f"{description}\n\ndefault value: ``{default}``"


default_variable = DynamicVariable[Optional["settings"]](None)


class settingsMeta(type):
    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def default(cls) -> Optional["settings"]:
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, "_current_profile", None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def _assign_default_internal(cls, value: settings) -> None:
        default_variable.value = value

    def __setattr__(cls, name: str, value: object) -> None:
        if name == "default":
            raise AttributeError(
                "Cannot assign to the property settings.default - "
                "consider using settings.load_profile instead."
            )
        elif not (isinstance(value, settingsProperty) or name.startswith("_")):
            raise AttributeError(
                f"Cannot assign hypothesis.settings.{name}={value!r} - the settings "
                "class is immutable.  You can change the global default "
                "settings with settings.load_profile, or use @settings(...) "
                "to decorate your test instead."
            )
        super().__setattr__(name, value)


class settings(metaclass=settingsMeta):
    __definitions_are_locked = False
    _profiles: ClassVar[dict[str, settings]] = {}
    __module__ = "hypothesis"
    _current_profile: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f"settings has no attribute {name}")

    def __init__(
        self,
        parent: Optional[settings] = None,
        *,
        max_examples: int = not_set,  # type: ignore[assignment]
        derandomize: bool = not_set,  # type: ignore[assignment]
        database: Optional[ExampleDatabase] = not_set,  # type: ignore[assignment]
        verbosity: Verbosity = not_set,  # type: ignore[assignment]
        phases: Collection[Phase] = not_set,  # type: ignore[assignment]
        stateful_step_count: int = not_set,  # type: ignore[assignment]
        report_multiple_bugs: bool = not_set,  # type: ignore[assignment]
        suppress_health_check: Collection[HealthCheck] = not_set,  # type: ignore[assignment]
        deadline: Union[int, float, datetime.timedelta, None] = not_set,  # type: ignore[assignment]
        print_blob: bool = not_set,  # type: ignore[assignment]
        backend: str = not_set,  # type: ignore[assignment]
    ) -> None:
        if derandomize not in (not_set, False):
            if database not in (not_set, None):  # type: ignore[comparison-overlap]
                raise InvalidArgument(
                    "derandomize=True implies database=None, so passing "
                    f"{database=} too is invalid."
                )
            database = None

        defaults = parent or settings.default
        if defaults is not None:
            for setting in all_settings.values():
                value = locals()[setting.name]
                if value is not_set:
                    object.__setattr__(
                        self, setting.name, getattr(defaults, setting.name)
                    )
                else:
                    object.__setattr__(self, setting.name, setting.validator(value))

    def __call__(self, test: T) -> T:
        _test: Any = test
        if not callable(_test):
            raise InvalidArgument(
                "settings objects can be called as a decorator with @given, "
                f"but decorated {test=} is not callable."
            )
        if inspect.isclass(test):
            from hypothesis.stateful import RuleBasedStateMachine

            if issubclass(_test, RuleBasedStateMachine):
                attr_name = "_hypothesis_internal_settings_applied"
                if getattr(test, attr_name, False):
                    raise InvalidArgument(
                        "Applying the @settings decorator twice would "
                        "overwrite the first version; merge their arguments "
                        "instead."
                    )
                setattr(test, attr_name, True)
                _test.TestCase.settings = self
                return cast(T, test)
            else:
                raise InvalidArgument(
                    "@settings(...) can only be used as a decorator on "
                    "functions, or on subclasses of RuleBasedStateMachine."
                )
        if hasattr(_test, "_hypothesis_internal_settings_applied"):
            descr = get_pretty_function_description(test)
            raise InvalidArgument(
                f"{descr} has already been decorated with a settings object.\n"
                f"    Previous:  {_test._hypothesis_internal_use_settings!r}\n"
                f"    This:  {self!r}"
            )

        _test._hypothesis_internal_use_settings = self
        _test._hypothesis_internal_settings_applied = True
        return test

    @classmethod
    def _define_setting(
        cls,
        name: str,
        description: str,
        *,
        default: object,
        options: Optional[Sequence[object]] = None,
        validator: Optional[ValidatorT] = None,
        show_default: bool = True,
    ) -> None:
        if cls.__definitions_are_locked:
            raise InvalidState(
                "settings have been locked and may no longer be defined."
            )
        if options is not None:
            options = tuple(options)
            assert default in options

            def validator(value: object) -> object:
                if value not in options:
                    msg = f"Invalid {name}, {value!r}. Valid options: {options!r}"
                    raise InvalidArgument(msg)
                return value

        else:
            assert validator is not None

        all_settings[name] = Setting(
            name=name,
            description=description.strip(),
            default=default,
            validator=validator,
        )
        setattr(settings, name, settingsProperty(name, show_default=show_default))

    @classmethod
    def lock_further_definitions(cls) -> None:
        cls.__definitions_are_locked = True

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("settings objects are immutable")

    def __repr__(self) -> str:
        bits = sorted(
            f"{name}={getattr(self, name)!r}"
            for name in all_settings
            if (name != "backend" or len(AVAILABLE_PROVIDERS) > 1)
        )
        return "settings({})".format(", ".join(bits))

    def show_changed(self) -> str:
        bits = []
        for name, setting in all_settings.items():
            value = getattr(self, name)
            if value != setting.default:
                bits.append(f"{name}={value!r}")
        return ", ".join(sorted(bits, key=len))

    @staticmethod
    def register_profile(
        name: str,
        parent: Optional[settings] = None,
        **kwargs: Any,
    ) -> None:
        check_type(str, name, "name")
        settings._profiles[name] = settings(parent=parent, **kwargs)
        if settings._current_profile == name:
            settings.load_profile(name)

    @staticmethod
    def get_profile(name: str) -> settings:
        check_type(str, name, "name")
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f"Profile {name!r} is not registered") from None

    @staticmethod
    def load_profile(name: str) -> None:
        check_type(str, name, "name")
        settings._current_profile = name
        settings._assign_default_internal(settings.get_profile(name))


@contextlib.contextmanager
def local_settings(s: settings) -> Generator[settings, None, None]:
    with default_variable.with_value(s):
        yield s


@attr.s()
class Setting:
    name: str = attr.ib()
    description: str = attr.ib()
    default: object = attr.ib()
    validator: ValidatorT = attr.ib()


def _max_examples_validator(x: int) -> int:
    check_type(int, x, name="max_examples")
    if x < 1:
        raise InvalidArgument(
            f"max_examples={x!r} should be at least one. You can disable "
            "example generation with the `phases` setting instead."
        )
    return x


settings._define_setting(
    "max_examples",
    default=100,
    validator=_max_examples_validator,
    description="...",
)


settings._define_setting(
    "derandomize",
    default=False,
    options=(True, False),
    description="...",
)


def _validate_database(db: ExampleDatabase | None) -> ExampleDatabase | None:
    from hypothesis.database import ExampleDatabase

    if db is None or isinstance(db, ExampleDatabase):
        return db
    raise InvalidArgument(
        "Arguments to the database setting must be None or an instance of "
        f"ExampleDatabase.  Try passing database=ExampleDatabase({db!r}), or "
        "construct and use one of the specific subclasses in "
        "hypothesis.database"
    )


settings._define_setting(
    "database",
    default=not_set,
    show_default=False,
    description="...",
    validator=_validate_database,
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
        return f"Phase.{self.name}"


class HealthCheckMeta(EnumMeta):
    def __iter__(self) -> Generator[HealthCheck, None, None]:
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return (x for x in super().__iter__() if x not in deprecated)


@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    data_too_large = 1
    filter_too_much = 2
    too_slow = 3
    return_value = 5
    large_base_example = 7
    not_a_test_method = 8
    function_scoped_fixture = 9
    differing_executors = 10

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@unique
class Verbosity(IntEnum):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3

    def __repr__(self) -> str:
        return f"Verbosity.{self.name}"


settings._define_setting(
    "verbosity",
    options=tuple(Verbosity),
    default=Verbosity.normal,
    description="...",
)


def _validate_phases(phases: Collection[Phase]) -> Sequence[Phase]:
    phases = tuple(phases)
    for a in phases:
        if not isinstance(a, Phase):
            raise InvalidArgument(f"{a!r} is not a valid phase")
    return tuple(p for p in list(Phase) if p in phases)


settings._define_setting(
    "phases",
    default=tuple(Phase),
    description="...",
    validator=_validate_phases,
)


def _validate_stateful_step_count(x: int) -> int:
    check_type(int, x, name="stateful_step_count")
    if x < 1:
        raise InvalidArgument(f"stateful_step_count={x!r} must be at least one.")
    return x


settings._define_setting(
    name="stateful_step_count",
    default=50,
    validator=_validate_stateful_step_count,
    description="...",
)

settings._define_setting(
    name="report_multiple_bugs",
    default=True,
    options=(True, False),
    description="...",
)


def validate_health_check_suppressions(suppressions: Any) -> list[HealthCheck]:
    suppressions = try_convert(list, suppressions, "suppress_health_check")
    for s in suppressions:
        if not isinstance(s, HealthCheck):
            raise InvalidArgument(
                f"Non-HealthCheck value {s!r} of type {type(s).__name__} "
                "is invalid in suppress_health_check."
            )
        if s in (HealthCheck.return_value, HealthCheck.not_a_test_method):
            warnings.warn(
                f"The {s.name} health check is deprecated, because this is always an error.",
                HypothesisDeprecationWarning,
                stacklevel=2,
            )
    return suppressions


settings._define_setting(
    "suppress_health_check",
    default=(),
    description="...",
    validator=validate_health_check_suppressions,
)


class duration(datetime.timedelta):
    def __repr__(self) -> str:
        ms = self.total_seconds() * 1000
        return f"timedelta(milliseconds={int(ms) if ms == int(ms) else ms!r})"


def _validate_deadline(
    x: Union[int, float, datetime.timedelta, None],
) -> Optional[duration]:
    if x is None:
        return x
    invalid_deadline_error = InvalidArgument(
        f"deadline={x!r} (type {type(x).__name__}) must be a timedelta object, "
        "an integer or float number of milliseconds, or None to disable the "
        "per-test-case deadline."
    )
    if isinstance(x, (int, float)):
        if isinstance(x, bool):
            raise invalid_deadline_error
        try:
            x = duration(milliseconds=x)
        except OverflowError:
            raise InvalidArgument(
                f"deadline={x!r} is invalid, because it is too large to represent "
                "as a timedelta. Use deadline=None to disable deadlines."
            ) from None
    if isinstance(x, datetime.timedelta):
        if x <= datetime.timedelta(0):
            raise InvalidArgument(
                f"deadline={x!r} is invalid, because it is impossible to meet a "
                "deadline <= 0. Use deadline=None to disable deadlines."
            )
        return duration(seconds=x.total_seconds())
    raise invalid_deadline_error


settings._define_setting(
    "deadline",
    default=duration(milliseconds=200),
    validator=_validate_deadline,
    description="...",
)


def is_in_ci() -> bool:
    return "CI" in os.environ or "TF_BUILD" in os.environ


settings._define_setting(
    "print_blob",
    default=False,
    options=(True, False),
    description="...",
)


def _backend_validator(value: str) -> str:
    if value not in AVAILABLE_PROVIDERS:
        if value == "crosshair":
            install = '`pip install "hypothesis[crosshair]"` and try again.'
            raise InvalidArgument(f"backend={value!r} is not available.  {install}")
        raise InvalidArgument(
            f"backend={value!r} is not available - maybe you need to install a plugin?"
            f"\n    Installed backends: {sorted(AVAILABLE_PROVIDERS)!r}"
        )
    return value


settings._define_setting(
    "backend",
    default="hypothesis",
    show_default=False,
    validator=_backend_validator,
    description="...",
)

settings.l