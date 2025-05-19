import sys
import logging
import functools
from typing import Callable, Any, TypeVar, Dict, Tuple, Optional
import sublime
from .helpers import get_settings, active_view

T = TypeVar('T', bound=Callable[..., Any])

def prevent_spam(func: Callable[..., Any]) -> Callable[..., Any]:
    """Prevent spamming in the logger
    """
    _last_messages: Dict[Tuple[Any, ...], int] = {}

    def _remove_from_cache(args: Tuple[Any, ...]) -> None:
        m: int = _last_messages.pop(args)
        if m == 1:
            return
        method = getattr(Log._logger, args[0])  # type: Callable[..., Any]
        if m == 2:
            repeat = 'one more time'
        else:
            repeat = f'{m} times'
        message = f'{args[1]}\n ...last message repeated {repeat} in the last 10s'
        method(message)

    @functools.wraps(func)
    def wrapper(cls: Any, *args: Any, **kwargs: Any) -> Optional[Any]:
        if args in _last_messages:
            _last_messages[args] += 1
            return
        _last_messages[args] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(args), 10000)
        return func(cls, args[0], *args[1:], **kwargs)
    return wrapper

class MetaLog(type):
    def __new__(
        cls: type,
        name: str,
        bases: Tuple[type, ...],
        attrs: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'
        cls._logger: logging.Logger = logging.getLogger('anacondaST3')
        cls._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
        log_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        cls._logger.addHandler(log_handler)
        cls._logger.propagate = False
        obj: Any = super().__new__(cls, name, bases, attrs)
        for method in ['debug', 'info', 'warning', 'error', 'fatal']:
            setattr(obj, method, functools.partial(obj.write, method))
        return obj

class Log(metaclass=MetaLog):
    """The class is responsible to log errors
    """

    _logger: logging.Logger

    @classmethod
    @prevent_spam
    def write(cls, method: str, *args: Any, **kwargs: Any) -> None:
        """Info wrapper
        """
        f: Callable[..., Any] = getattr(cls._logger, method)
        f(*args, **kwargs)
