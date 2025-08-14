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
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Type,
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

ValidatorT: "TypeAlias" = Callable[[Any], object]
all_settings: Dict[str, "Setting"] = {}

T = TypeVar("T")


class settingsProperty:
    def __init__(self, name: str, *, show_default: bool) -> None:
        self.name = name
        self.show_default = show_default

    def __get__(self, obj: Optional[object], type: Optional[Type[object]] = None) -> Any:
        if obj is None:
            return self
        else:
            try:
                result = obj.__dict__[self.name]  # type: ignore
                # This is a gross hack, but it preserves the old behaviour that
                # you can change the storage directory and it will be reflected
                # in the default database.
                if self.name == "database" and result is not_set:
                    from hypothesis.database import ExampleDatabase

                    result = ExampleDatabase(not_set)
                assert result is not not_set
                return result
            except KeyError:
                raise AttributeError(self.name) from None

    def __set__(self, obj: object, value: Any) -> None:
        obj.__dict__[self.name] = value  # type: ignore

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
    def __init__(cls: Type["settings"], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def default(cls: Type["settings"]) -> Optional["settings"]:
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, "_current_profile", None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def _assign_default_internal(cls: Type["settings"], value: "settings") -> None:
        default_variable.value = value

    def __setattr__(cls: Type["settings"], name: str, value: object) -> None:
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
    """A settings object configures options including verbosity, runtime controls,
    persistence, determinism, and more.

    Default values are picked up from the settings.default object and
    changes made there will be picked up in newly created settings.
    """

    __definitions_are_locked: ClassVar[bool] = False
    _profiles: ClassVar[Dict[str, "settings"]] = {}
    __module__: ClassVar[str] = "hypothesis"
    _current_profile: ClassVar[Optional[str]] = None

    def __getattr__(self, name: str) -> Any:
        if name in all_settings:
            return all_settings[name].default
        else:
            raise AttributeError(f"settings has no attribute {name}")

    def __init__(
        self,
        parent: Optional["settings"] = None,
        *,
        max_examples: int = not_set,  # type: ignore
        derandomize: bool = not_set,  # type: ignore
        database: Optional["ExampleDatabase"] = not_set,  # type: ignore
        verbosity: "Verbosity" = not_set,  # type: ignore
        phases: Collection["Phase"] = not_set,  # type: ignore
        stateful_step_count: int = not_set,  # type: ignore
        report_multiple_bugs: bool = not_set,  # type: ignore
        suppress_health_check: Collection["HealthCheck"] = not_set,  # type: ignore
        deadline: Union[int, float, datetime.timedelta, None] = not_set,  # type: ignore
        print_blob: bool = not_set,  # type: ignore
        backend: str = not_set,  # type: ignore
    ) -> None:
        if parent is not None:
            check_type(settings, parent, "parent")
        if derandomize not in (not_set, False):
            if database not in (not_set, None):  # type: ignore
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
        """Make the settings object (self) an attribute of the test.

        The settings are later discovered by looking them up on the test itself.
        """
        # Aliasing as Any avoids mypy errors (attr-defined) when accessing and
        # setting custom attributes on the decorated function or class.
        _test: Any = test

        # Using the alias here avoids a mypy error (return-value) later when
        # ``test`` is returned, because this check results in type refinement.
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
                return test  # type: ignore
            else:
                raise InvalidArgument(
                    "@settings(...) can only be used as a decorator on "
                    "functions, or on subclasses of RuleBasedStateMachine."
                )
        if hasattr(_test, "_hypothesis_internal_settings_applied"):
            # Can't use _hypothesis_internal_use_settings as an indicator that
            # @settings was applied, because @given also assigns that attribute.
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
        """Add a new setting.

        - name is the name of the property that will be used to access the
          setting. This must be a valid python identifier.
        - description will appear in the property's docstring
        - default is the default value. This may be a zero argument
          function in which case it is evaluated and its result is stored
          the first time it is accessed on any given settings object.
        """
        if settings.__definitions_are_locked:
            raise InvalidState(
                "settings have been locked and may no longer be defined."
            )
        if options is not None:
            options = tuple(options)
            assert default in options

            def validator(value: Any) -> Any:
                if value not in options:  # type: ignore
                    msg = f"Invalid {name}, {value!r}. Valid options: {options!r}"
                    raise InvalidArgument(msg)
                return value

        else:
            assert validator is not None

        all_settings[name] = Setting(
            name=name,
            description=description.strip(),
            default=default,
            validator=validator,  # type: ignore
        )
        setattr(settings, name, settingsProperty(name, show_default=show_default))

    @classmethod
    def lock_further_definitions(cls) -> None:
        settings.__definitions_are_locked = True

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("settings objects are immutable")

    def __repr__(self) -> str:
        bits = sorted(
            f"{name}={getattr(self, name)!r}"
            for name in all_settings
            if (name != "backend" or len(AVAILABLE_PROVIDERS) > 1)  # experimental
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
        parent: Optional["settings"] = None,
        **kwargs: Any,
    ) -> None:
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
        check_type(str, name, "name")
        settings._profiles[name] = settings(parent=parent, **kwargs)
        if settings._current_profile == name:
            settings.load_profile(name)

    @staticmethod
    def get_profile(name: str) -> "settings":
        """Return the profile with the given name."""
        check_type(str, name, "name")
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f"Profile {name!r} is not registered") from None

    @staticmethod
    def load_profile(name: str) -> None:
        """Loads in the settings defined in the profile provided.

        If the profile does not exist, InvalidArgument will be raised.
        Any setting not defined in the profile will be the library
        defined default for that setting.
        """
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
    description="""
Once this many satisfying examples have been considered without finding any
counter-example, Hypothesis will stop looking.

Note that we might call your test function fewer times if we find a bug early
or can tell that we've exhausted the search space; or more if we discard some
examples due to use of .filter(), assume(), or a few other things that can
prevent the test case from completing successfully.

The default value is chosen to suit a workflow where the test will be part of
a suite that is regularly executed locally or on a CI server, balancing total
running time against the chance of missing a bug.

If you are writing one-off tests, running tens of thousands of examples is
quite reasonable as Hypothesis may miss uncommon bugs with default settings.
For very complex code, we have observed Hypothesis finding novel bugs after
*several million* examples while testing :pypi:`SymPy <sympy>`.
If you are running more than 100k examples for a test, consider using our
:ref:`integration for coverage-guided fuzzing <fuzz_one_input>` - it really
shines when given minutes or hours to run.
""",
)


settings._define_setting(
    "derandomize",
    default=False,
    options=(True, False),
    description="""
If True, seed Hypothesis' random number generator using a hash of the test
function, so that every run will test the same set of examples until you
update Hypothesis, Python, or the test function.

This allows you to `check for regressions and look for bugs
<https://blog.nelhage.com/post/two-kinds-of-testing/>`__ using
:ref:`separate settings profiles <settings_profiles>` - for example running
quick deterministic tests on every commit, and a longer non-deterministic
nightly testing run.

By default when running on CI, this will be set to True.
""",
)


def _validate_database(db: "ExampleDatabase") -> "ExampleDatabase":
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
    description="""
An instance of :class:`~hypothesis.database.ExampleDatabase` that will be
used to save examples to and load previous examples from. May be ``None``
in which case no storage will be used.

See the :doc:`example database documentation <database>` for a list of built-in
example database implementations, and how to define custom implementations.
""",
    validator=_validate_database,
)


@unique
class Phase(IntEnum):
    explicit = 0  #: controls whether explicit examples are run.
    reuse = 1  #: controls whether previous examples will be reused.
    generate = 2  #: controls whether new examples will be generated.
    target = 3  #: controls whether examples will be mutated for targeting.
    shrink = 4  #: controls whether examples will be shrunk.
    explain = 5  #: controls whether Hypothesis attempts to explain test failures.

    def __repr__(self) -> str:
        return f"Phase.{self.name}"


class HealthCheckMeta(EnumMeta):
    def __iter__(self) -> Any:
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return iter(x for x in super().__iter__() if x not in deprecated)


@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    """Arguments for :attr:`~hypothesis.settings.suppress_health_check`.

    Each member of this enum is a type of health check to suppress.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def all(cls) -> List["HealthCheck"]:
        # Skipping of