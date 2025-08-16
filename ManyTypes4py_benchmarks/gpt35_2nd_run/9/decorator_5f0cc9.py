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
def validate_arguments(func: AnyCallableT = None, *, config: ConfigType = None) -> AnyCallableT:
    ...

@overload
def validate_arguments(func: AnyCallableT) -> AnyCallableT:
    ...

@deprecated('The `validate_arguments` method is deprecated; use `validate_call` instead.', category=None)
def validate_arguments(func: AnyCallableT = None, *, config: ConfigType = None) -> AnyCallableT:
    """Decorator to validate the arguments passed to a function."""
    warnings.warn('The `validate_arguments` method is deprecated; use `validate_call` instead.', PydanticDeprecatedSince20, stacklevel=2)

    def validate(_func: AnyCallable) -> AnyCallable:
        vd = ValidatedFunction(_func, config)

        @wraps(_func)
        def wrapper_function(*args: Any, **kwargs: Any) -> Any:
            return vd.call(*args, **kwargs)
        wrapper_function.vd = vd
        wrapper_function.validate = vd.init_model_instance
        wrapper_function.raw_function = vd.raw_function
        wrapper_function.model = vd.model
        return wrapper_function
    if func:
        return validate(func)
    else:
        return validate
