import collections.abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast, Dict, Iterable, List, Mapping, Set, Tuple, Type, TypeVar, Union, Optional
import inspect
import logging
from allennlp.common.checks import ConfigurationError
from allennlp.common.lazy import Lazy
from allennlp.common.params import Params
logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar('T', bound='FromParams')
_NO_DEFAULT: Any = inspect.Parameter.empty

def takes_arg(obj: Any, arg: str) -> bool:
    ...

def takes_kwargs(obj: Any) -> bool:
    ...

def can_construct_from_params(type_: Type) -> bool:
    ...

def is_base_registrable(cls: Type) -> bool:
    ...

def remove_optional(annotation: Type) -> Type:
    ...

def infer_constructor_params(cls: Type, constructor: Optional[Callable]) -> Dict[str, inspect.Parameter]:
    ...

def create_kwargs(constructor: Callable, cls: Type, params: Params, **extras: Any) -> Dict[str, Any]:
    ...

def create_extras(cls: Type, extras: Dict[str, Any]) -> Dict[str, Any]:
    ...

def pop_and_construct_arg(class_name: str, argument_name: str, annotation: Type, default: Any, params: Params, **extras: Any) -> Any:
    ...

def construct_arg(class_name: str, argument_name: str, popped_params: Any, annotation: Type, default: Any, **extras: Any) -> Any:
    ...

class FromParams:
    @classmethod
    def from_params(cls: Type, params: Union[str, Params], constructor_to_call: Optional[Callable] = None, constructor_to_inspect: Optional[Callable] = None, **extras: Any) -> Any:
        ...

    def to_params(self) -> Params:
        ...

    def _to_params(self) -> Dict[str, Any]:
        ...
