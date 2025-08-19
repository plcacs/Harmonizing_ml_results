from __future__ import annotations
import logging
from functools import wraps
from inspect import getcallargs, getmembers, getmodule, isclass, isfunction, ismethod, Signature, signature
from logging import Logger
from types import ModuleType
from typing import Any, Callable, Optional, Union, TypeVar, ParamSpec, overload, cast

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
empty_and_none: set[object] = {Signature.empty, 'None'}

P = ParamSpec('P')
R = TypeVar('R')
C = TypeVar('C', bound=type[Any])

Function = Callable[..., Any]
Decorated = Union[type[Any], Function]


@overload
def log(
    decorated: None = ...,
    *,
    prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX,
    suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX,
    with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART,
    prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX,
    suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX,
    return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART,
) -> Callable[[Decorated], Decorated]: ...
@overload
def log(
    decorated: Callable[P, R],
    *,
    prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX,
    suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX,
    with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART,
    prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX,
    suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX,
    return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART,
) -> Callable[P, R]: ...
@overload
def log(
    decorated: C,
    *,
    prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX,
    suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX,
    with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART,
    prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX,
    suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX,
    return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART,
) -> C: ...
def log(
    decorated: Optional[Decorated] = None,
    *,
    prefix_enter_msg: str = _DEFAULT_ENTER_MSG_PREFIX,
    suffix_enter_msg: str = _DEFAULT_ENTER_MSG_SUFFIX,
    with_arguments_msg_part: str = _DEFAULT_WITH_ARGUMENTS_MSG_PART,
    prefix_exit_msg: str = _DEFAULT_EXIT_MSG_PREFIX,
    suffix_exit_msg: str = _DEFAULT_EXIT_MSG_SUFFIX,
    return_value_msg_part: str = _DEFAULT_RETURN_VALUE_MSG_PART,
) -> Union[Callable[[Decorated], Decorated], Decorated]:
    decorator = _make_decorator(
        prefix_enter_msg,
        suffix_enter_msg,
        with_arguments_msg_part,
        prefix_exit_msg,
        suffix_exit_msg,
        return_value_msg_part,
    )
    if decorated is None:
        return decorator
    return decorator(decorated)


def _make_decorator(
    prefix_enter_msg: str,
    suffix_enter_msg: str,
    with_arguments_msg_part: str,
    prefix_out_msg: str,
    suffix_out_msg: str,
    return_value_msg_part: str,
) -> Callable[[Decorated], Decorated]:

    def decorator(decorated: Decorated) -> Decorated:
        decorated_logger: Logger = _get_logger(decorated)

        def decorator_class(clazz: C) -> C:
            _decorate_class_members_with_logs(clazz)
            return clazz

        def _decorate_class_members_with_logs(clazz: type[Any]) -> None:
            members = getmembers(clazz, predicate=lambda val: ismethod(val) or isfunction(val))
            for member_name, member in members:
                setattr(clazz, member_name, decorator_func(cast(Callable[..., Any], member), f'{clazz.__name__}'))

        def decorator_func(func: Callable[P, R], prefix_name: str = '') -> Callable[P, R]:
            func_name: str = func.__name__
            func_signature: Signature = signature(func)
            is_fixture: bool = hasattr(func, _FIXTURE_ATTRIBUTE)
            has_return_value: bool = func_signature.return_annotation not in empty_and_none
            is_private: bool = func_name.startswith(_PRIVATE_PREFIX_SYMBOL)
            full_func_name: str = f'{prefix_name}.{func_name}'
            under_info: Optional[bool] = None
            debug_enable: Optional[bool] = None

            @wraps(func)
            def _wrapper_func(*args: P.args, **kwargs: P.kwargs) -> R:
                _log_enter_to_function(*args, **kwargs)
                val: R = func(*args, **kwargs)
                _log_exit_of_function(val)
                return val

            def _log_enter_to_function(*args: P.args, **kwargs: P.kwargs) -> None:
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
                return cast(bool, debug_enable)

            def _log_debug(*args: P.args, **kwargs: P.kwargs) -> None:
                used_parameters: dict[str, Any] = getcallargs(func, *args, **kwargs)
                _SELF_PARAM in used_parameters and used_parameters.pop(_SELF_PARAM)
                _CLS_PARAM in used_parameters and used_parameters.pop(_CLS_PARAM)
                if used_parameters:
                    decorated_logger.debug(
                        f"{prefix_enter_msg}'{full_func_name}'{with_arguments_msg_part}{used_parameters}{suffix_enter_msg}"
                    )
                else:
                    decorated_logger.debug(f"{prefix_enter_msg}'{full_func_name}'{suffix_enter_msg}")

            def _log_exit_of_function(return_value: R) -> None:
                if _is_debug_enable() and has_return_value:
                    decorated_logger.debug(
                        f"{prefix_out_msg}'{full_func_name}'{return_value_msg_part}'{return_value}'{suffix_out_msg}"
                    )

            return _wrapper_func

        if isclass(decorated):
            return decorator_class(cast(type[Any], decorated))
        return decorator_func(cast(Callable[P, R], decorated))

    return decorator


def _get_logger(decorated: Any) -> Logger:
    module = cast(ModuleType, getmodule(decorated))
    logger_obj: Any = module.__dict__.get(_LOGGER_VAR_NAME)
    if isinstance(logger_obj, Logger):
        return logger_obj
    return logging.getLogger(module.__name__)