#!/usr/bin/env python3
"""
Utilities for deprecated items.

When a deprecated item is used, a warning will be displayed. Warnings may not be
disabled with Prefect settings. Instead, the standard Python warnings filters can be
used.

Deprecated items require a start or end date. If a start date is given, the end date
will be calculated 6 months later. Start and end dates are always in the format MMM YYYY
e.g. Jan 2023.
"""

import functools
import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from pydantic import BaseModel
from typing_extensions import ParamSpec, TypeAlias, TypeVar

from prefect.types._datetime import DateTime, from_format
from prefect.utilities.callables import get_call_parameters
from prefect.utilities.importtools import (
    AliasedModuleDefinition,
    AliasedModuleFinder,
    to_qualified_name,
)

P = ParamSpec("P")
R = TypeVar("R", bound=Any)
M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")

DEPRECATED_WARNING: str = (
    "{name} has been deprecated{when}. It will not be available in new releases after {end_date}."
    " {help}"
)
DEPRECATED_MOVED_WARNING: str = (
    "{name} has moved to {new_location}. It will not be available at the old import "
    "path after {end_date}. {help}"
)
DEPRECATED_DATEFMT: str = "MMM YYYY"  # e.g. Feb 2023
DEPRECATED_MODULE_ALIASES: list[AliasedModuleDefinition] = []


class PrefectDeprecationWarning(DeprecationWarning):
    """
    A deprecation warning.
    """
    pass


def generate_deprecation_message(
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    help: str = "",
    when: str = "",
) -> str:
    if not start_date and not end_date:
        raise ValueError(
            "A start date is required if an end date is not provided. Suggested start"
            f" date is {DateTime.now('UTC').format(DEPRECATED_DATEFMT)!r}"
        )

    if not end_date:
        if TYPE_CHECKING:
            assert start_date is not None
        parsed_start_date = from_format(start_date, DEPRECATED_DATEFMT)
        parsed_end_date = parsed_start_date.add(months=6)
        end_date = parsed_end_date.format(DEPRECATED_DATEFMT)
    else:
        # Validate format
        from_format(end_date, DEPRECATED_DATEFMT)

    if when:
        when = " when " + when

    message: str = DEPRECATED_WARNING.format(
        name=name, when=when, end_date=end_date, help=help
    )
    return message.rstrip()


def deprecated_callable(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stacklevel: int = 2,
    help: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        message: str = generate_deprecation_message(
            name=to_qualified_name(fn),
            start_date=start_date,
            end_date=end_date,
            help=help,
        )

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_class(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stacklevel: int = 2,
    help: str = "",
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        message: str = generate_deprecation_message(
            name=to_qualified_name(cls),
            start_date=start_date,
            end_date=end_date,
            help=help,
        )

        original_init: Callable[..., None] = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: T, *args: Any, **kwargs: Any) -> None:
            warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def deprecated_parameter(
    name: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stacklevel: int = 2,
    help: str = "",
    when: Optional[Callable[[Any], bool]] = None,
    when_message: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Mark a parameter in a callable as deprecated.

    Example:

        @deprecated_parameter("y", when=lambda y: y is not None)
        def foo(x, y = None):
            return x + 1 + (y or 0)
    """
    when_callable: Callable[[Any], bool] = when or (lambda _: True)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        message: str = generate_deprecation_message(
            name=f"The parameter {name!r} for {fn.__name__!r}",
            start_date=start_date,
            end_date=end_date,
            help=help,
            when=when_message,
        )

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                parameters: dict[str, Any] = get_call_parameters(fn, args, kwargs, apply_defaults=False)
            except Exception:
                # Avoid raising any parsing exceptions here
                parameters = kwargs  # type: ignore

            if name in parameters and when_callable(parameters[name]):
                warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)

            return fn(*args, **kwargs)

        return wrapper

    return decorator


JsonValue: TypeAlias = Union[int, float, str, bool, None, list["JsonValue"], "JsonDict"]
JsonDict: TypeAlias = dict[str, JsonValue]


def deprecated_field(
    name: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    when_message: str = "",
    help: str = "",
    when: Optional[Callable[[Any], bool]] = None,
    stacklevel: int = 2,
) -> Callable[[type[M]], type[M]]:
    """
    Mark a field in a Pydantic model as deprecated.

    Raises warning only if the field is specified during init.

    Example:

        @deprecated_field("x", when=lambda x: x is not None)
        class Model(BaseModel):
            x: Optional[int] = None
            y: str
    """
    when_callable: Callable[[Any], bool] = when or (lambda _: True)

    def decorator(model_cls: type[M]) -> type[M]:
        message: str = generate_deprecation_message(
            name=f"The field {name!r} in {model_cls.__name__!r}",
            start_date=start_date,
            end_date=end_date,
            help=help,
            when=when_message,
        )

        cls_init: Callable[..., None] = model_cls.__init__

        @functools.wraps(model_cls.__init__)
        def __init__(__pydantic_self__: M, **data: Any) -> None:
            if name in data and when_callable(data[name]):
                warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            cls_init(__pydantic_self__, **data)
            field = __pydantic_self__.model_fields.get(name)
            if field is not None:
                json_schema_extra: Any = field.json_schema_extra or {}
                if not isinstance(json_schema_extra, dict):
                    # json_schema_extra is a hook function; wrap it to add the deprecated flag.
                    extra_func: Callable[[JsonDict], None] = json_schema_extra

                    @functools.wraps(extra_func)
                    def wrapped(__json_schema: JsonDict) -> None:
                        extra_func(__json_schema)
                        __json_schema["deprecated"] = True

                    json_schema_extra = wrapped
                else:
                    json_schema_extra["deprecated"] = True
                field.json_schema_extra = json_schema_extra

        model_cls.__init__ = __init__
        return model_cls

    return decorator


def inject_renamed_module_alias_finder() -> None:
    """
    Insert an aliased module finder into Python's import machinery.

    Required for `register_renamed_module` to work.
    """
    sys.meta_path.insert(0, AliasedModuleFinder(DEPRECATED_MODULE_ALIASES))


def register_renamed_module(old_name: str, new_name: str, start_date: str) -> None:
    """
    Register a renamed module.

    Adds backwards compatibility imports for the old module name and displays a
    deprecation warning on import of the module.
    """
    message: str = generate_deprecation_message(
        name=f"The {old_name!r} module",
        start_date=start_date,
        help=f"Use {new_name!r} instead.",
    )

    def callback(_: Any) -> None:
        warnings.warn(message, DeprecationWarning, stacklevel=3)

    DEPRECATED_MODULE_ALIASES.append(
        AliasedModuleDefinition(old_name, new_name, callback)
    )