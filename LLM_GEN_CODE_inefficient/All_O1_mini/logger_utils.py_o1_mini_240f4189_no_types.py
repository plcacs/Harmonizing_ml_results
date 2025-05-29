from __future__ import annotations
import logging
from functools import wraps
from inspect import getcallargs, getmembers, getmodule, isclass, isfunction, ismethod, Signature, signature
from logging import Logger
from typing import Any, Callable, cast, Dict, Optional, Union
_DEFAULT_ENTER_MSG_PREFIX = 'enter to '
_DEFAULT_ENTER_MSG_SUFFIX = ''
_DEFAULT_WITH_ARGUMENTS_MSG_PART = ' with: '
_DEFAULT_EXIT_MSG_PREFIX = 'exit from '
_DEFAULT_EXIT_MSG_SUFFIX = ''
_DEFAULT_RETURN_VALUE_MSG_PART = ' with return value: '
_CLS_PARAM = 'cls'
_SELF_PARAM = 'self'
_PRIVATE_PREFIX_SYMBOL = '_'
_FIXTURE_ATTRIBUTE = '_pytestfixturefunction'
_LOGGER_VAR_NAME = 'logger'
empty_and_none: set[Union[Any, str]] = {Signature.empty, 'None'}
Function = Callable[..., Any]
Decorated = Union[type[Any], Function]


def log(decorated=None, *, prefix_enter_msg: str=_DEFAULT_ENTER_MSG_PREFIX,
    suffix_enter_msg: str=_DEFAULT_ENTER_MSG_SUFFIX,
    with_arguments_msg_part: str=_DEFAULT_WITH_ARGUMENTS_MSG_PART,
    prefix_exit_msg: str=_DEFAULT_EXIT_MSG_PREFIX, suffix_exit_msg: str=
    _DEFAULT_EXIT_MSG_SUFFIX, return_value_msg_part: str=
    _DEFAULT_RETURN_VALUE_MSG_PART):
    decorator: Decorated = _make_decorator(prefix_enter_msg,
        suffix_enter_msg, with_arguments_msg_part, prefix_exit_msg,
        suffix_exit_msg, return_value_msg_part)
    if decorated is None:
        return decorator
    return decorator(decorated)


def _make_decorator(prefix_enter_msg, suffix_enter_msg,
    with_arguments_msg_part, prefix_out_msg, suffix_out_msg,
    return_value_msg_part):

    def decorator(decorated):
        decorated_logger: Logger = _get_logger(decorated)

        def decorator_class(clazz):
            _decorate_class_members_with_logs(clazz)
            return clazz

        def _decorate_class_members_with_logs(clazz):
            members = getmembers(clazz, predicate=lambda val: ismethod(val) or
                isfunction(val))
            for member_name, member in members:
                setattr(clazz, member_name, decorator_func(member,
                    f'{clazz.__name__}'))

        def decorator_func(func, prefix_name=''):
            func_name: str = func.__name__
            func_signature: Signature = signature(func)
            is_fixture: bool = hasattr(func, _FIXTURE_ATTRIBUTE)
            has_return_value: bool = (func_signature.return_annotation not in
                empty_and_none)
            is_private: bool = func_name.startswith(_PRIVATE_PREFIX_SYMBOL)
            full_func_name: str = (f'{prefix_name}.{func_name}' if
                prefix_name else func_name)
            under_info: Optional[bool] = None
            debug_enable: Optional[bool] = None

            @wraps(func)
            def _wrapper_func(*args: Any, **kwargs: Any):
                _log_enter_to_function(*args, **kwargs)
                val: Any = func(*args, **kwargs)
                _log_exit_of_function(val)
                return val

            def _log_enter_to_function(*args: Any, **kwargs: Any):
                if _is_log_info():
                    decorated_logger.info(
                        f"{prefix_enter_msg}'{full_func_name}'{suffix_enter_msg}"
                        )
                elif _is_debug_enable():
                    _log_debug(*args, **kwargs)

            def _is_log_info():
                return not (_is_under_info() or is_private or is_fixture)

            def _is_under_info():
                nonlocal under_info
                if under_info is None:
                    under_info = decorated_logger.getEffectiveLevel(
                        ) < logging.INFO
                return under_info

            def _is_debug_enable():
                nonlocal debug_enable
                if debug_enable is None:
                    debug_enable = decorated_logger.isEnabledFor(logging.DEBUG)
                return debug_enable

            def _log_debug(*args: Any, **kwargs: Any):
                used_parameters: Dict[str, Any] = getcallargs(func, *args,
                    **kwargs)
                if _SELF_PARAM in used_parameters:
                    used_parameters.pop(_SELF_PARAM)
                if _CLS_PARAM in used_parameters:
                    used_parameters.pop(_CLS_PARAM)
                if used_parameters:
                    decorated_logger.debug(
                        f"{prefix_enter_msg}'{full_func_name}'{with_arguments_msg_part}{used_parameters}{suffix_enter_msg}"
                        )
                else:
                    decorated_logger.debug(
                        f"{prefix_enter_msg}'{full_func_name}'{suffix_enter_msg}"
                        )

            def _log_exit_of_function(return_value):
                if _is_debug_enable() and has_return_value:
                    decorated_logger.debug(
                        f"{prefix_out_msg}'{full_func_name}'{return_value_msg_part}'{return_value}'{suffix_out_msg}"
                        )
            return _wrapper_func
        if isclass(decorated):
            return decorator_class(cast(type[Any], decorated))
        return decorator_func(cast(Function, decorated))
    return decorator


def _get_logger(decorated):
    module = getmodule(decorated)
    return module.__dict__.get(_LOGGER_VAR_NAME, logging.getLogger(module.
        __name__))
