from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from pydantic import BaseModel
from typing_extensions import ParamSpec, TypeAlias, TypeVar
from prefect.types._datetime import DateTime
from prefect.utilities.callables import get_call_parameters
from prefect.utilities.importtools import AliasedModuleDefinition, AliasedModuleFinder, to_qualified_name

P = ParamSpec('P')
R = TypeVar('R', covariant=True)
M = TypeVar('M', bound=BaseModel)
T = TypeVar('T')

DEPRECATED_WARNING: str = '{name} has been deprecated{when}. It will not be available in new releases after {end_date}. {help}'
DEPRECATED_MOVED_WARNING: str = '{name} has moved to {new_location}. It will not be available at the old import path after {end_date}. {help}'
DEPRECATED_DATEFMT: str = 'MMM YYYY'
DEPRECATED_MODULE_ALIASES: list = []

class PrefectDeprecationWarning(DeprecationWarning):
    """
    A deprecation warning.
    """

def generate_deprecation_message(name: str, start_date: Optional[str] = None, end_date: Optional[str] = None, help: str = '', when: str = '') -> str:

def deprecated_callable(*, start_date: Optional[str] = None, end_date: Optional[str] = None, stacklevel: int = 2, help: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:

def deprecated_class(*, start_date: Optional[str] = None, end_date: Optional[str] = None, stacklevel: int = 2, help: str = '') -> Callable[[Type[M]], Type[M]]:

def deprecated_parameter(name: str, *, start_date: Optional[str] = None, end_date: Optional[str] = None, stacklevel: int = 2, help: str = '', when: Optional[Callable[[Any], bool]] = None, when_message: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:

JsonValue: TypeAlias = Union[int, float, str, bool, None, list['JsonValue'], 'JsonDict']
JsonDict: TypeAlias = dict[str, JsonValue]

def deprecated_field(name: str, *, start_date: Optional[str] = None, end_date: Optional[str] = None, when_message: str = '', help: str = '', when: Optional[Callable[[Any], bool]] = None, stacklevel: int = 2) -> Callable[[Type[M]], Type[M]]:

def inject_renamed_module_alias_finder() -> None:

def register_renamed_module(old_name: str, new_name: str, start_date: str) -> None:
