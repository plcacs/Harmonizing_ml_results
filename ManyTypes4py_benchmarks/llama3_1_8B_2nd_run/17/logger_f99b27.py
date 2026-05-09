import sys
import logging
import functools
import sublime
from .helpers import get_settings, active_view

def prevent_spam(func: callable) -> callable:
    """Prevent spamming in the logger
    """
    _last_messages: dict[tuple, int] = {}

    def _remove_from_cache(args: tuple) -> None:
        m = _last_messages.pop(args)
        if m == 1:
            return
        method = getattr(Log._logger, args[0])
        method('{}\n ...last message repeated {} in the last 10s'.format(args[1], 'one more time' if m == 2 else '{} times'.format(m)))

    @functools.wraps(func)
    def wrapper(cls: type, *args: object, **kwargs: object) -> None:
        if args in _last_messages:
            _last_messages[args] += 1
            return
        _last_messages[args] = 1
        sublime.set_timeout_async(lambda: _remove_from_cache(args), 10000)
        func(cls, args[0], *args[1:], **kwargs)
    return wrapper

class MetaLog(type):
    """Meta class for logging
    """

    def __new__(cls: type, name: str, bases: tuple[type], attrs: dict[str, object], **kwargs: object) -> type:
        log_level: str = get_settings(active_view(), 'log_level', 'info')
        if log_level not in ['debug', 'info', 'warning', 'error', 'fatal']:
            log_level = 'warning'
        cls._logger: logging.Logger = logging.getLogger('anacondaST3')
        cls._logger.setLevel(getattr(logging, log_level.upper()))
        log_handler: logging.Handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(logging.Formatter('%(name)s: %(levelname)s - %(message)s'))
        cls._logger.addHandler(log_handler)
        cls._logger.propagate = False
        obj: type = super().__new__(cls, name, bases, attrs)
        for method in ['debug', 'info', 'warning', 'error', 'fatal']:
            setattr(obj, method, functools.partial(obj.write, method))
        return obj

class Log(metaclass=MetaLog):
    """The class is responsible to log errors
    """

    @classmethod
    @prevent_spam
    def write(cls: type, method: str, *args: object, **kwargs: object) -> None:
        """Info wrapper
        """
        f = getattr(cls._logger, method)
        f(*args, **kwargs)
