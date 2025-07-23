"""Logging support for Tornado.

Tornado uses three logger streams:

* ``tornado.access``: Per-request logging for Tornado's HTTP servers (and
  potentially other servers in the future)
* ``tornado.application``: Logging of errors from application code (i.e.
  uncaught exceptions from callbacks)
* ``tornado.general``: General-purpose logging, including any errors
  or warnings from Tornado itself.

These streams may be configured independently using the standard library's
`logging` module.  For example, you may wish to send ``tornado.access`` logs
to a separate file for analysis.
"""
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
from typing import Dict, Any, cast, Optional, Union, Type

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
    """Log formatter used in Tornado.

    Key features of this formatter are:

    * Color support when logging to a terminal that supports it.
    * Timestamps on every log line.
    * Robust against str/bytes encoding problems.
    """
    DEFAULT_FORMAT: str = '%(color)s[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s'
    DEFAULT_DATE_FORMAT: str = '%y%m%d %H:%M:%S'
    DEFAULT_COLORS: Dict[int, int] = {logging.DEBUG: 4, logging.INFO: 2, logging.WARNING: 3, logging.ERROR: 1, logging.CRITICAL: 5}

    def __init__(
        self,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        style: str = '%',
        color: bool = True,
        colors: Dict[int, int] = DEFAULT_COLORS
    ) -> None:
        logging.Formatter.__init__(self, datefmt=datefmt)
        self._fmt = fmt
        self._colors: Dict[int, str] = {}
        if color and _stderr_supports_color():
            if curses is not None:
                fg_color = curses.tigetstr('setaf') or curses.tigetstr('setf') or b''
                for levelno, code in colors.items():
                    self._colors[levelno] = unicode_type(curses.tparm(fg_color, code), 'ascii')
                normal = curses.tigetstr('sgr0')
                if normal is not None:
                    self._normal = unicode_type(normal, 'ascii')
                else:
                    self._normal = ''
            else:
                for levelno, code in colors.items():
                    self._colors[levelno] = '\x1b[2;3%dm' % code
                self._normal = '\x1b[0m'
        else:
            self._normal = ''

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
            assert isinstance(message, basestring_type)
            record.message = _safe_unicode(message)
        except Exception as e:
            record.message = f'Bad message ({e!r}): {record.__dict__!r}'
        record.asctime = self.formatTime(record, cast(str, self.datefmt))
        if record.levelno in self._colors:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        else:
            record.color = record.end_color = ''
        formatted = self._fmt % record.__dict__
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = [formatted.rstrip()]
            lines.extend((_safe_unicode(ln) for ln in record.exc_text.split('\n')))
            formatted = '\n'.join(lines)
        return formatted.replace('\n', '\n    ')

def enable_pretty_logging(options: Optional[Any] = None, logger: Optional[logging.Logger] = None) -> None:
    """Turns on formatted logging output as configured."""
    if options is None:
        import tornado.options
        options = tornado.options.options
    if options.logging is None or options.logging.lower() == 'none':
        return
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(getattr(logging, options.logging.upper()))
    if options.log_file_prefix:
        rotate_mode = options.log_rotate_mode
        if rotate_mode == 'size':
            channel = logging.handlers.RotatingFileHandler(
                filename=options.log_file_prefix,
                maxBytes=options.log_file_max_size,
                backupCount=options.log_file_num_backups,
                encoding='utf-8'
            )
        elif rotate_mode == 'time':
            channel = logging.handlers.TimedRotatingFileHandler(
                filename=options.log_file_prefix,
                when=options.log_rotate_when,
                interval=options.log_rotate_interval,
                backupCount=options.log_file_num_backups,
                encoding='utf-8'
            )
        else:
            error_message = 'The value of log_rotate_mode option should be ' + '"size" or "time", not "%s".' % rotate_mode
            raise ValueError(error_message)
        channel.setFormatter(LogFormatter(color=False))
        logger.addHandler(channel)
    if options.log_to_stderr or (options.log_to_stderr is None and (not logger.handlers)):
        channel = logging.StreamHandler()
        channel.setFormatter(LogFormatter())
        logger.addHandler(channel)

def define_logging_options(options: Optional[Any] = None) -> None:
    """Add logging-related flags to ``options``."""
    if options is None:
        import tornado.options
        options = tornado.options.options
    options.define('logging', default='info', help="Set the Python log level. If 'none', tornado won't touch the logging configuration.", metavar='debug|info|warning|error|none')
    options.define('log_to_stderr', type=bool, default=None, help='Send log output to stderr (colorized if possible). By default use stderr if --log_file_prefix is not set and no other logging is configured.')
    options.define('log_file_prefix', type=str, default=None, metavar='PATH', help='Path prefix for log files. Note that if you are running multiple tornado processes, log_file_prefix must be different for each of them (e.g. include the port number)')
    options.define('log_file_max_size', type=int, default=100 * 1000 * 1000, help='max size of log files before rollover')
    options.define('log_file_num_backups', type=int, default=10, help='number of log files to keep')
    options.define('log_rotate_when', type=str, default='midnight', help="specify the type of TimedRotatingFileHandler interval other options:('S', 'M', 'H', 'D', 'W0'-'W6')")
    options.define('log_rotate_interval', type=int, default=1, help='The interval value of timed rotating')
    options.define('log_rotate_mode', type=str, default='size', help='The mode of rotating files(time or size)')
    options.add_parse_callback(lambda: enable_pretty_logging(options))
