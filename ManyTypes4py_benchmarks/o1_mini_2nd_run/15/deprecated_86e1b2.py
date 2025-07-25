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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Dict
from pydantic import BaseModel
from typing_extensions import ParamSpec, TypeAlias, TypeVar
from prefect.types._datetime import DateTime, from_format
from prefect.utilities.callables import get_call_parameters
from prefect.utilities.importtools import AliasedModuleDefinition, AliasedModuleFinder, to_qualified_name

P = ParamSpec('P')
R = TypeVar('R', infer_variance=True)
M = TypeVar('M', bound=BaseModel)
T = TypeVar('T')

DEPRECATED_WARNING: str = '{name} has been deprecated{when}. It will not be available in new releases after {end_date}. {help}'
DEPRECATED_MOVED_WARNING: str = '{name} has moved to {new_location}. It will not be available at the old import path after {end_date}. {help}'
DEPRECATED_DATEFMT: str = 'MMM YYYY'
DEPRECATED_MODULE_ALIASES: list[AliasedModuleDefinition] = []

class PrefectDeprecationWarning(DeprecationWarning):
    """
    A deprecation warning.
    """

def generate_deprecation_message(
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    help: str = '',
    when: str = ''
) -> str:
    if not start_date and not end_date:
        raise ValueError(
            f'A start date is required if an end date is not provided. Suggested start date is {DateTime.now("UTC").format(DEPRECATED_DATEFMT)!r}'
        )
    if not end_date:
        if TYPE_CHECKING:
            assert start_date is not None
        parsed_start_date: DateTime = from_format(start_date, DEPRECATED_DATEFMT)
        parsed_end_date: DateTime = parsed_start_date.add(months=6)
        end_date = parsed_end_date.format(DEPRECATED_DATEFMT)
    else:
        from_format(end_date, DEPRECATED_DATEFMT)
    if when:
        when = ' when ' + when
    message: str = DEPRECATED_WARNING.format(
        name=name, when=when, end_date=end_date, help=help
    )
    return message.rstrip()

def deprecated_callable(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stacklevel: int = 2,
    help: str = ''
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        message: str = generate_deprecation_message(
            name=to_qualified_name(fn),
            start_date=start_date,
            end_date=end_date,
            help=help
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
    help: str = ''
) -> Callable[[type], type]:
    
    def decorator(cls: type) -> type:
        message: str = generate_deprecation_message(
            name=to_qualified_name(cls),
            start_date=start_date,
            end_date=end_date,
            help=help
        )
        original_init: Callable[..., None] = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: M, *args: Any, **kwargs: Any) -> None:
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
    help: str = '',
    when: Optional[Callable[[Any], bool]] = None,
    when_message: str = ''
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Mark a parameter in a callable as deprecated.

    Example:

        