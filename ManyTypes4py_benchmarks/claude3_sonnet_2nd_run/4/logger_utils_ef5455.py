from __future__ import annotations
import logging
from functools import wraps
from inspect import getcallargs, getmembers, getmodule, isclass, isfunction, ismethod, Signature, signature
from logging import Logger
from typing import Any, Callable, cast, Union, Optional, TypeVar, Dict, List, Tuple, overload

_DEFAULT_ENTER_MSG_PREFIX: str = 'enter to '
_DEFAULT_ENTER_MSG_SUFFIX: str = ''
_DEFAULT_WITH_ARGUMENTS_MSG_PART: str = ' with: '
_DEFAULT_EXIT_MSG_PREFIX: str = 'exit from '
_DEFAULT_EXIT_MSG_SUFFIX: str = ''
_DEFAULT_RETURN_VALUE_MSG_PART: str = ' with return value: '
_CLS_PARAM: str = 'cls'
_SELF_PARAM: str = 'self'
_PRIVATE_PREFIX_SYMBOL: str = '_'
_FIXTURE_ATTRIBUTE: str = '_pytestfixturefunction'
_LOGGER_VAR_NAME: str = 'logger'
empty_and_none: set = {Signature.empty, 'None'}
Function = Callable[..., Any]
Decorated = Union[type[Any], Function]
T = TypeVar('T', bound=Decorated)

@overload
def log(decorated: None = None, *, prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX, suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX, with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART, prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX, suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX, return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART) -> Callable[[T], T]: ...

@overload
def log(decorated: T, *, prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX, suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX, with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART, prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX, suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX, return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART) -> T: ...

def log(decorated: Optional[T] = None, *, prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX, suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX, with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART, prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX, suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX, return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART) -> Union[Callable[[T], T], T]:
    decorator = _make_decorator(prefix_enter_msg, suffix_enter_msg, with_arguments_msg_part, prefix_exit_msg, suffix_exit_msg, return_value_msg_part)
    if decorated is None:
        return decorator
    return decorator(decorated)

def _make_decorator(prefix_enter_msg: str, suffix_enter_msg: str, with_arguments_msg_part: str, prefix_out_msg: str, suffix_out_msg: str, return_value_msg_part: str) -> Callable[[T], T]:

    def decorator(decorated: T) -> T:
        decorated_logger: Logger = _get_logger(decorated)

        def decorator_class(clazz: type[Any]) -> type[Any]:
            _decorate_class_members_with_logs(clazz)
            return clazz

        def _decorate_class_members_with_logs(clazz: type[Any]) -> None:
            members: List[Tuple[str, Any]] = getmembers(clazz, predicate=lambda val: ismethod(val) or isfunction(val))
            for member_name, member in members:
                setattr(clazz, member_name, decorator_func(member, f'{clazz.__name__}'))

        def decorator_func(func: Function, prefix_name: str = '') -> Function:
            func_name: str = func.__name__
            func_signature: Signature = signature(func)
            is_fixture: bool = hasattr(func, _FIXTURE_ATTRIBUTE)
            has_return_value: bool = func_signature.return_annotation not in empty_and_none
            is_private: bool = func_name.startswith(_PRIVATE_PREFIX_SYMBOL)
            full_func_name: str = f'{prefix_name}.{func_name}'
            under_info: Optional[bool] = None
            debug_enable: Optional[bool] = None

            @wraps(func)
            def _wrapper_func(*args: Any, **kwargs: Any) -> Any:
                _log_enter_to_function(*args, **kwargs)
                val = func(*args, **kwargs)
                _log_exit_of_function(val)
                return val

            def _log_enter_to_function(*args: Any, **kwargs: Any) -> None:
                if _is_log_info():
                    decorated_logger.info(f"{prefix_enter_msg}'{full_func_name}'{suffix_enter_msg}")
                elif _is_debug_enable():
                    _log_debug(*args, **kwargs)

            def _is_log_info() -> bool:
                return not (_is_under_info() or is_private or is_fixture)

            def _is_under_info() -> bool:
                nonlocal under_info
                if under_info is None:
                    under_info = decorated_logger.getEffectiveLevel() < logging.INFO
                return under_info

            def _is_debug_enable() -> bool:
                nonlocal debug_enable
                if debug_enable is None:
                    debug_enable = decorated_logger.isEnabledFor(logging.DEBUG)
                return debug_enable

            def _log_debug(*args: Any, **kwargs: Any) -> None:
                used_parameters: Dict[str, Any] = getcallargs(func, *args, **kwargs)
                _SELF_PARAM in used_parameters and used_parameters.pop(_SELF_PARAM)
                _CLS_PARAM in used_parameters and used_parameters.pop(_CLS_PARAM)
                if used_parameters:
                    decorated_logger.debug(f"{prefix_enter_msg}'{full_func_name}'{with_arguments_msg_part}{used_parameters}{suffix_enter_msg}")
                else:
                    decorated_logger.debug(f"{prefix_enter_msg}'{full_func_name}'{suffix_enter_msg}")

            def _log_exit_of_function(return_value: Any) -> None:
                if _is_debug_enable() and has_return_value:
                    decorated_logger.debug(f"{prefix_out_msg}'{full_func_name}'{return_value_msg_part}'{return_value}'{suffix_out_msg}")
            return _wrapper_func
        if isclass(decorated):
            return decorator_class(cast(type[Any], decorated))
        return decorator_func(cast(Function, decorated))
    return decorator

def _get_logger(decorated: Decorated) -> Logger:
    module = getmodule(decorated)
    return module.__dict__.get(_LOGGER_VAR_NAME, logging.getLogger(module.__name__))
