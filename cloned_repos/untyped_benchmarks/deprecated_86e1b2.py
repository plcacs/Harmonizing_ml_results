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
from prefect.utilities.importtools import AliasedModuleDefinition, AliasedModuleFinder, to_qualified_name
P = ParamSpec('P')
R = TypeVar('R', infer_variance=True)
M = TypeVar('M', bound=BaseModel)
T = TypeVar('T')
DEPRECATED_WARNING = '{name} has been deprecated{when}. It will not be available in new releases after {end_date}. {help}'
DEPRECATED_MOVED_WARNING = '{name} has moved to {new_location}. It will not be available at the old import path after {end_date}. {help}'
DEPRECATED_DATEFMT = 'MMM YYYY'
DEPRECATED_MODULE_ALIASES = []

class PrefectDeprecationWarning(DeprecationWarning):
    """
    A deprecation warning.
    """

def generate_deprecation_message(name, start_date=None, end_date=None, help='', when=''):
    if not start_date and (not end_date):
        raise ValueError(f'A start date is required if an end date is not provided. Suggested start date is {DateTime.now('UTC').format(DEPRECATED_DATEFMT)!r}')
    if not end_date:
        if TYPE_CHECKING:
            assert start_date is not None
        parsed_start_date = from_format(start_date, DEPRECATED_DATEFMT)
        parsed_end_date = parsed_start_date.add(months=6)
        end_date = parsed_end_date.format(DEPRECATED_DATEFMT)
    else:
        from_format(end_date, DEPRECATED_DATEFMT)
    if when:
        when = ' when ' + when
    message = DEPRECATED_WARNING.format(name=name, when=when, end_date=end_date, help=help)
    return message.rstrip()

def deprecated_callable(*, start_date=None, end_date=None, stacklevel=2, help=''):

    def decorator(fn):
        message = generate_deprecation_message(name=to_qualified_name(fn), start_date=start_date, end_date=end_date, help=help)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def deprecated_class(*, start_date=None, end_date=None, stacklevel=2, help=''):

    def decorator(cls):
        message = generate_deprecation_message(name=to_qualified_name(cls), start_date=start_date, end_date=end_date, help=help)
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            original_init(self, *args, **kwargs)
        cls.__init__ = new_init
        return cls
    return decorator

def deprecated_parameter(name, *, start_date=None, end_date=None, stacklevel=2, help='', when=None, when_message=''):
    """
    Mark a parameter in a callable as deprecated.

    Example:

        ```python

        @deprecated_parameter("y", when=lambda y: y is not None)
        def foo(x, y = None):
            return x + 1 + (y or 0)
        ```
    """
    when = when or (lambda _: True)

    def decorator(fn):
        message = generate_deprecation_message(name=f'The parameter {name!r} for {fn.__name__!r}', start_date=start_date, end_date=end_date, help=help, when=when_message)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                parameters = get_call_parameters(fn, args, kwargs, apply_defaults=False)
            except Exception:
                parameters = kwargs
            if name in parameters and when(parameters[name]):
                warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            return fn(*args, **kwargs)
        return wrapper
    return decorator
JsonValue = Union[int, float, str, bool, None, list['JsonValue'], 'JsonDict']
JsonDict = dict[str, JsonValue]

def deprecated_field(name, *, start_date=None, end_date=None, when_message='', help='', when=None, stacklevel=2):
    """
    Mark a field in a Pydantic model as deprecated.

    Raises warning only if the field is specified during init.

    Example:

        ```python

        @deprecated_field("x", when=lambda x: x is not None)
        class Model(BaseModel)
            x: Optional[int] = None
            y: str
        ```
    """
    when = when or (lambda _: True)

    def decorator(model_cls):
        message = generate_deprecation_message(name=f'The field {name!r} in {model_cls.__name__!r}', start_date=start_date, end_date=end_date, help=help, when=when_message)
        cls_init = model_cls.__init__

        @functools.wraps(model_cls.__init__)
        def __init__(__pydantic_self__, **data):
            if name in data.keys() and when(data[name]):
                warnings.warn(message, PrefectDeprecationWarning, stacklevel=stacklevel)
            cls_init(__pydantic_self__, **data)
            field = __pydantic_self__.model_fields.get(name)
            if field is not None:
                json_schema_extra = field.json_schema_extra or {}
                if not isinstance(json_schema_extra, dict):
                    extra_func = json_schema_extra

                    @functools.wraps(extra_func)
                    def wrapped(__json_schema):
                        extra_func(__json_schema)
                        __json_schema['deprecated'] = True
                    json_schema_extra = wrapped
                else:
                    json_schema_extra['deprecated'] = True
                field.json_schema_extra = json_schema_extra
        model_cls.__init__ = __init__
        return model_cls
    return decorator

def inject_renamed_module_alias_finder():
    """
    Insert an aliased module finder into Python's import machinery.

    Required for `register_renamed_module` to work.
    """
    sys.meta_path.insert(0, AliasedModuleFinder(DEPRECATED_MODULE_ALIASES))

def register_renamed_module(old_name, new_name, start_date):
    """
    Register a renamed module.

    Adds backwwards compatibility imports for the old module name and displays a
    deprecation warnings on import of the module.
    """
    message = generate_deprecation_message(name=f'The {old_name!r} module', start_date=start_date, help=f'Use {new_name!r} instead.')

    def callback(_):
        return warnings.warn(message, DeprecationWarning, stacklevel=3)
    DEPRECATED_MODULE_ALIASES.append(AliasedModuleDefinition(old_name, new_name, callback))