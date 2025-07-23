import sys
import logging
import functools
import sublime
from typing import Any, Callable, Dict, Tuple, Type, TypeVar
from .helpers import get_settings, active_view

F = TypeVar('F', bound=Callable[..., Any])

def prevent_spam(func: Callable[..., Any]) -> Callable[..., Any]:
    """Prevent spamming in the logger
    """
    _last_messages: Dict[Tuple[Any, ...], int] = {}

    def _remove_from_cache(args: Tuple[Any, ...]) -> None:
        m: int = _last_messages.pop(args)
        if m == 1:
            return
        method = getattr(Log._logger, args[0])
        message_suffix = 'one more time' if m == 2 else f'{m} times'
        method(f'{args[1]}\n ...last message repeated {message_suffix} in the last 10s')

    @functools.wraps(func)
    def wrapper(cls: Type['Log'], *args: Any, **kwargs: Any) -> None:
        if args in _last_messages:
            _last_messages[args] += 1
            return
        _last_messages[args] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(args), 10000)
        func(cls, args[0], *args[1:], **kwargs)
    return wrapper

class MetaLog(type):
    _logger: logging.Logger

    def __new__(cls: Type[type], name: str, bases: Tuple[type, ...], attrs: Dict[str, Any], **kwargs: Any) -> Type[Any]:
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'
        cls._logger = logging.getLogger('anacondaST3')
        cls._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        cls._logger.addHandler(log_handler)
        cls._logger.propagate = False
        obj = super().__new__(cls, name, bases, attrs)
        for method in ['debug', 'info', 'warning', 'error', 'fatal']:
            setattr(obj, method, functools.partial(obj.write, method))
        return obj

class Log(metaclass=MetaLog):
    """The class is responsible to log errors
    """

    @classmethod
    @prevent_spam
    def write(cls: Type['Log'], method: str, *args: Any, **kwargs: Any) -> None:
        """Info wrapper
        """
        f: Callable[..., None] = getattr(cls._logger, method)
        f(*args, **kwargs)
