import warnings
from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import deprecated

if not TYPE_CHECKING:
    DeprecationWarning = PydanticDeprecatedSince20

__all__: tuple[str] = ('validate_arguments',)

if TYPE_CHECKING:
    AnyCallable = Callable[..., Any]
    AnyCallableT = TypeVar('AnyCallableT', bound=AnyCallable)
    ConfigType = Union[None, type[Any], dict[str, Any]]

@overload
def validate_arguments(func: Optional[Callable] = None, *, config: Optional[ConfigType] = None) -> Callable:
    ...

@deprecated('The `validate_arguments` method is deprecated; use `validate_call` instead.', category=None)
def validate_arguments(func: Optional[Callable] = None, *, config: Optional[ConfigType] = None) -> Callable:
    ...

ALT_V_ARGS: str = 'v__args'
ALT_V_KWARGS: str = 'v__kwargs'
V_POSITIONAL_ONLY_NAME: str = 'v__positional_only'
V_DUPLICATE_KWARGS: str = 'v__duplicate_kwargs'

class ValidatedFunction:
    def __init__(self, function: Callable, config: ConfigType) -> None:
        ...

    def init_model_instance(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def call(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def build_values(self, args: tuple[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        ...

    def execute(self, m: Any) -> Any:
        ...

    def create_model(self, fields: dict[str, tuple[type, Any]], takes_args: bool, takes_kwargs: bool, config: ConfigType) -> None:
        ...

