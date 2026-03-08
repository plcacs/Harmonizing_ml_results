import sys
import logging
import functools
import sublime
from typing import Any, Callable, Dict, Tuple, Type, ClassVar, cast
from .helpers import get_settings, active_view


def prevent_spam(func: Callable[..., Any]) -> Callable[..., None]:
    """Prevent spamming in the logger
    """
    _last_messages: Dict[Tuple[Any, ...], int] = {}

    def _remove_from_cache(args: Tuple[Any, ...]) -> None:
        m = _last_messages.pop(args)
        if m == 1:
            return
        method = cast(Callable[..., None], getattr(Log._logger, args[0]))
        method('{}\n ...last message repeated {} in the last 10s'.format(args[1], 'one more time' if m == 2 else '{} times'.format(m)))

    @functools.wraps(func)
    def wrapper(cls: Type[Any], *args: Any, **kwargs: Any) -> None:
        if args in _last_messages:
            _last_messages[args] += 1
            return
        _last_messages[args] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(args), 10000)
        func(cls, args[0], *args[1:], **kwargs)
    return wrapper


class MetaLog(type):
    _logger: ClassVar[logging.Logger]

    def __new__(mcls: Type['MetaLog'], name: str, bases: Tuple[type, ...], attrs: Dict[str, Any], **kwargs: Any) -> 'MetaLog':
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'
        mcls._logger = logging.getLogger('anacondaST3')
        mcls._logger.setLevel(logging.__getattribute__(log_level.upper()))
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        mcls._logger.addHandler(log_handler)
        mcls._logger.propagate = False
        obj = super().__new__(mcls, name, bases, attrs)
        for method in ['debug', 'info', 'warning', 'error', 'fatal']:
            write = cast(Callable[..., None], getattr(obj, 'write'))
            setattr(obj, method, functools.partial(write, method))
        return obj


class Log(metaclass=MetaLog):
    """The class is responsible to log errors
    """
    _logger: ClassVar[logging.Logger]

    @classmethod
    @prevent_spam
    def write(cls: Type['Log'], method: str, *args: Any, **kwargs: Any) -> None:
        """Info wrapper
        """
        f = cast(Callable[..., None], getattr(cls._logger, method))
        f(*args, **kwargs)