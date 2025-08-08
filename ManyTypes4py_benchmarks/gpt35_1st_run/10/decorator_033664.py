from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from pydantic.v1 import validator
from pydantic.v1.config import Extra
from pydantic.v1.errors import ConfigError
from pydantic.v1.main import BaseModel, create_model
from pydantic.v1.typing import get_all_type_hints
from pydantic.v1.utils import to_camel

__all__: Tuple[str] = ('validate_arguments',)

if TYPE_CHECKING:
    from pydantic.v1.typing import AnyCallable
    AnyCallableT = TypeVar('AnyCallableT', bound=AnyCallable)
    ConfigType = Union[None, Type[Any], Dict[str, Any]]

@overload
def validate_arguments(func: Callable) -> Callable:
    ...

@overload
def validate_arguments(func: Callable = None, *, config: Optional[ConfigType] = None) -> Callable:
    ...

def validate_arguments(func: Callable = None, *, config: Optional[ConfigType] = None) -> Callable:
    ...

ALT_V_ARGS: str = 'v__args'
ALT_V_KWARGS: str = 'v__kwargs'
V_POSITIONAL_ONLY_NAME: str = 'v__positional_only'
V_DUPLICATE_KWARGS: str = 'v__duplicate_kwargs'

class ValidatedFunction:

    def __init__(self, function: Callable, config: Optional[ConfigType]) -> None:
        ...

    def init_model_instance(self, *args: Any, **kwargs: Any) -> BaseModel:
        ...

    def call(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def build_values(self, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def execute(self, m: BaseModel) -> Any:
        ...

    def create_model(self, fields: Dict[str, Tuple[Type, Any]], takes_args: bool, takes_kwargs: bool, config: Optional[ConfigType]) -> None:
        ...

