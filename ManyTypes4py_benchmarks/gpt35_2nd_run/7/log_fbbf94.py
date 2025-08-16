import logging
import logging.handlers
import sys
from tornado.escape import _unicode
from tornado.util import unicode_type, basestring_type
try:
    import colorama
except ImportError:
    colorama = None
try:
    import curses
except ImportError:
    curses = None
from typing import Dict, Any, Optional

access_log: logging.Logger = logging.getLogger('tornado.access')
app_log: logging.Logger = logging.getLogger('tornado.application')
gen_log: logging.Logger = logging.getLogger('tornado.general')

def _stderr_supports_color() -> bool:
    try:
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            if curses:
                curses.setupterm()
                if curses.tigetnum('colors') > 0:
                    return True
            elif colorama:
                if sys.stderr is getattr(colorama.initialise, 'wrapped_stderr', object()):
                    return True
    except Exception:
        pass
    return False

def _safe_unicode(s: Any) -> str:
    try:
        return _unicode(s)
    except UnicodeDecodeError:
        return repr(s)

class LogFormatter(logging.Formatter):
    DEFAULT_FORMAT: str = '%(color)s[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s'
    DEFAULT_DATE_FORMAT: str = '%y%m%d %H:%M:%S'
    DEFAULT_COLORS: Dict[int, int] = {logging.DEBUG: 4, logging.INFO: 2, logging.WARNING: 3, logging.ERROR: 1, logging.CRITICAL: 5}

    def __init__(self, fmt: str = DEFAULT_FORMAT, datefmt: str = DEFAULT_DATE_FORMAT, style: str = '%', color: bool = True, colors: Dict[int, int] = DEFAULT_COLORS) -> None:
        ...

    def format(self, record: logging.LogRecord) -> str:
        ...

def enable_pretty_logging(options: Optional[Any] = None, logger: Optional[logging.Logger] = None) -> None:
    ...

def define_logging_options(options: Optional[Any] = None) -> None:
    ...
