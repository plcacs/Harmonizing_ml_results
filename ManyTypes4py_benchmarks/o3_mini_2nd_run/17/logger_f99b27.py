import sys
import logging
import functools
import sublime
from typing import Any, Callable, Type, Tuple, Dict, cast
from .helpers import get_settings, active_view

T = TypeVar("T", bound=Callable[..., Any])


def prevent_spam(func: T) -> T:
    """Prevent spamming in the logger"""
    _last_messages: Dict[Tuple[Any, ...], int] = {}

    def _remove_from_cache(args: Tuple[Any, ...]) -> None:
        m = _last_messages.pop(args)
        if m == 1:
            return
        method = getattr(Log._logger, args[0])
        message: str = '{}\n ...last message repeated {} in the last 10s'.format(
            args[1],
            'one more time' if m == 2 else '{} times'.format(m)
        )
        method(message)

    @functools.wraps(func)
    def wrapper(cls: Any, *args: Any, **kwargs: Any) -> None:
        key: Tuple[Any, ...] = args
        if key in _last_messages:
            _last_messages[key] += 1
            return
        _last_messages[key] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(key), 10000)
        func(cls, args[0], *args[1:], **kwargs)

    return cast(T, wrapper)


class MetaLog(type):

    def __new__(cls: Type['MetaLog'], name: str, bases: Tuple[type, ...], attrs: Dict[str, Any], **kwargs: Any) -> Any:
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'
        cls._logger = logging.getLogger('anacondaST3')
        cls._logger.setLevel(logging.__getattribute__(log_level.upper()))
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        cls._logger.addHandler(log_handler)
        cls._logger.propagate = False
        obj = super().__new__(cls, name, bases, attrs)
        for method in ['debug', 'info', 'warning', 'error', 'fatal']:
            setattr(obj, method, functools.partial(obj.write, method))
        return obj


class Log(metaclass=MetaLog):
    """The class is responsible to log errors"""

    @classmethod
    @prevent_spam
    def write(cls, method: str, *args: Any, **kwargs: Any) -> None:
        """Info wrapper"""
        f: Callable[..., Any] = getattr(cls._logger, method)
        f(*args, **kwargs)