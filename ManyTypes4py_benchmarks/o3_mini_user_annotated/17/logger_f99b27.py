import sys
import logging
import functools
from typing import Any, Callable, Dict, Tuple, Type

import sublime

from .helpers import get_settings, active_view


def prevent_spam(func: Callable[..., None]) -> Callable[..., None]:
    """Prevent spamming in the logger."""
    _last_messages: Dict[Tuple[Any, ...], int] = {}

    def _remove_from_cache(args: Tuple[Any, ...]) -> None:
        m: int = _last_messages.pop(args)
        if m == 1:
            return

        method: Callable[..., None] = getattr(Log._logger, args[0])
        method(
            '{}\n ...last message repeated {} in the last 10s'.format(
                args[1], 'one more time' if m == 2 else '{} times'.format(m)
            )
        )

    @functools.wraps(func)
    def wrapper(cls: Any, *args: Any, **kwargs: Any) -> None:
        if args in _last_messages:
            _last_messages[args] += 1
            return

        _last_messages[args] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(args), 10000)
        func(cls, args[0], *args[1:], **kwargs)

    return wrapper


class MetaLog(type):

    def __new__(
        cls: Type["MetaLog"],
        name: str,
        bases: Tuple[Type[Any], ...],
        attrs: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'

        cls._logger: logging.Logger = logging.getLogger('anacondaST3')
        cls._logger.setLevel(logging.__getattribute__(log_level.upper()))
        log_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter(
            '%(name)s: %(levelname)s - %(message)s'
        ))

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
        f: Callable[..., None] = getattr(cls._logger, method)
        f(*args, **kwargs)