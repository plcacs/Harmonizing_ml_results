from __future__ import annotations
from typing import Any, Optional, Union, Tuple, Dict, List, Callable, Type
from string import Template
from org.transcrypt.stubs.browser import __pragma__
import time
import warnings
import collections

__author__ = 'Vinay Sajip <vinay_sajip@red-dove.com>, Carl Allendorph'
__status__ = 'experimental'
__version__ = '0.5.1.2'
__date__ = '15 November 2016'
_startTime: float = time.time()
raiseExceptions: bool = True
logThreads: bool = True
logMultiprocessing: bool = True
logProcesses: bool = True

CRITICAL: int = 50
FATAL: int = CRITICAL
ERROR: int = 40
WARNING: int = 30
WARN: int = WARNING
INFO: int = 20
DEBUG: int = 10
NOTSET: int = 0

_levelToName: Dict[int, str] = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET'
}
_nameToLevel: Dict[str, int] = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET
}


def getLevelName(level: Union[int, str]) -> str:
    return _levelToName.get(level) or _nameToLevel.get(level) or 'Level {}'.format(level)


def addLevelName(level: Union[int, str], levelName: str) -> None:
    _acquireLock()
    try:
        _levelToName[level if isinstance(level, int) else int(level)] = levelName  # type: ignore
        _nameToLevel[levelName] = level if isinstance(level, int) else int(level)  # type: ignore
    except Exception as exc:
        raise exc
    finally:
        _releaseLock()


def currentframe() -> Any:
    return None


_srcfile: Optional[str] = None


def _checkLevel(level: Union[int, str]) -> int:
    if isinstance(level, int):
        rv: int = level
    elif isinstance(level, str):
        if level not in _nameToLevel:
            raise ValueError('Unknown level: {}'.format(level))
        rv = _nameToLevel[level]
    else:
        raise TypeError('Level not an integer or a valid string: {}'.format(level))
    return rv


_lock: Optional[Any] = None


def _acquireLock() -> None:
    if _lock:
        _lock.acquire()


def _releaseLock() -> None:
    if _lock:
        _lock.release()


class LogRecord(object):
    def __init__(self, name: Optional[str], level: int, pathname: str, lineno: int, msg: Any, args: Tuple[Any, ...],
                 exc_info: Any, func: Optional[str] = None, sinfo: Optional[Any] = None, **kwargs: Any) -> None:
        ct: float = time.time()
        self.name: Optional[str] = name
        self.msg: Any = msg
        if args and len(args) == 1 and isinstance(args[0], collections.Mapping) and args[0]:
            if raiseExceptions:
                raise NotImplementedError('No Dict Args to Log Record')
        self.args: Tuple[Any, ...] = args
        self.levelname: str = getLevelName(level)
        self.levelno: int = level
        self.pathname: str = pathname
        self.filename: str = pathname
        self.module: str = 'Unknown module'
        self.exc_info: Any = exc_info
        self.exc_text: Optional[str] = None
        self.stack_info: Optional[Any] = sinfo
        self.lineno: int = lineno
        self.funcName: Optional[str] = func
        self.created: float = ct
        self.msecs: float = (ct - int(ct)) * 1000
        self.relativeCreated: float = (self.created - _startTime) * 1000
        self.thread: Optional[Any] = None
        self.threadName: Optional[str] = None
        self.processName: Optional[Any] = None
        self.process: Optional[Any] = None

    def getMessage(self) -> str:
        msg: str = str(self.msg)
        if self.args:
            msg = msg.format(*self.args)
        return msg

    def toDict(self) -> Dict[str, Any]:
        keysToPick: List[str] = ['name', 'msg', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno',
                                   'funcName', 'created', 'asctime', 'msecs', 'relativeCreated', 'thread', 'threadName',
                                   'process']
        ret: Dict[str, Any] = {}
        for k in keysToPick:
            if k == 'name':
                ret[k] = getattr(self, 'py_name', None)
            else:
                ret[k] = getattr(self, k, None)
        ret['message'] = self.getMessage()
        return ret

    def __str__(self) -> str:
        return '<LogRecord: {}, {}, {}, {}, "{}">'.format(self.name, self.levelno, self.pathname, self.lineno, self.msg)

    def __repr__(self) -> str:
        return str(self)


_logRecordFactory: Callable[..., LogRecord] = LogRecord


def setLogRecordFactory(factory: Callable[..., LogRecord]) -> None:
    global _logRecordFactory
    _logRecordFactory = factory


def getLogRecordFactory() -> Callable[..., LogRecord]:
    return _logRecordFactory


def makeLogRecord(dict: Dict[str, Any]) -> LogRecord:
    rv = _logRecordFactory(None, 0, '', 0, '', (), None, None)
    rv.__dict__.update(dict)
    return rv


class PercentStyle(object):
    default_format: str = '%(message)s'
    asctime_format: str = '%(asctime)s'
    asctime_search: str = '%(asctime)'

    def __init__(self, fmt: Optional[str]) -> None:
        self._fmt: str = fmt or self.default_format

    def usesTime(self) -> bool:
        return self._fmt.find(self.asctime_search) >= 0

    def format(self, record: LogRecord) -> str:
        return self._fmt % record.__dict__


class StrFormatStyle(PercentStyle):
    default_format: str = '{message}'
    asctime_format: str = '{asctime}'
    asctime_search: str = '{asctime'

    __pragma__('kwargs')

    def format(self, record: LogRecord) -> str:
        return self._fmt.format(**record.toDict())
    __pragma__('nokwargs')


class StringTemplateStyle(PercentStyle):
    default_format: str = '${message}'
    asctime_format: str = '${asctime}'
    asctime_search: str = '${asctime}'

    def __init__(self, fmt: Optional[str]) -> None:
        self._fmt = fmt or self.default_format
        self._tpl: Template = Template(self._fmt)

    def usesTime(self) -> bool:
        fmt: str = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_format) >= 0

    __pragma__('kwargs')
    def format(self, record: LogRecord) -> str:
        return self._tpl.substitute(**record.__dict__)
    __pragma__('nokwargs')


BASIC_FORMAT: str = '{levelname}:{name}:{message}'
_STYLES: Dict[str, Tuple[Callable[[Optional[str]], PercentStyle], str]] = {
    '{': (StrFormatStyle, BASIC_FORMAT)
}


class Formatter(object):
    converter: Callable[[float], Any] = time.localtime

    __pragma__('kwargs')
    def __init__(self, format: Optional[str] = None, datefmt: Optional[str] = None, style: str = '{') -> None:
        if style != '{':
            raise NotImplementedError('{} format only')
        self._style: PercentStyle = _STYLES[style][0](format)
        self._fmt: str = self._style._fmt
        self.datefmt: Optional[str] = datefmt
    __pragma__('nokwargs')
    default_time_format: str = '%Y-%m-%d %H:%M:%S'
    default_msec_format: str = '{},{:03d}'

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        ct: Any = self.converter(record.created)
        if datefmt:
            s: str = time.strftime(datefmt, ct)
        else:
            t: str = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format.format(t, int(record.msecs))
        return s

    def formatException(self, ei: Any) -> str:
        return str(ei)

    def usesTime(self) -> bool:
        return self._style.usesTime()

    def formatMessage(self, record: LogRecord) -> str:
        return self._style.format(record)

    def formatStack(self, stack_info: Any) -> str:
        return stack_info

    def format(self, record: LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s: str = self.formatMessage(record)
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s and s[-1] != '\n':
                s = s + '\n'
            s = s + record.exc_text
        if record.stack_info:
            if s and s[-1] != '\n':
                s = s + '\n'
            s = s + self.formatStack(record.stack_info)
        return s


_defaultFormatter: Formatter = Formatter()


class BufferingFormatter(object):
    def __init__(self, linefmt: Optional[Formatter] = None) -> None:
        if linefmt:
            self.linefmt: Formatter = linefmt
        else:
            self.linefmt = _defaultFormatter

    def formatHeader(self, records: List[LogRecord]) -> str:
        return ''

    def formatFooter(self, records: List[LogRecord]) -> str:
        return ''

    def format(self, records: List[LogRecord]) -> str:
        rv: str = ''
        if len(records) > 0:
            rv = rv + self.formatHeader(records)
            for record in records:
                rv = rv + self.linefmt.format(record)
                rv = rv + self.formatFooter(records)
        return rv


class Filter(object):
    def __init__(self, name: str = '') -> None:
        self.name: str = name
        self.nlen: int = len(name)

    def filter(self, record: LogRecord) -> bool:
        if self.nlen == 0:
            return True
        elif self.name == record.name:
            return True
        elif record.name is None or record.name.find(self.name, 0, self.nlen) != 0:
            return False
        return record.name[self.nlen:self.nlen+1] == '.'


class Filterer(object):
    def __init__(self) -> None:
        self.filters: List[Any] = []

    def addFilter(self, filt: Any) -> None:
        if filt not in self.filters:
            self.filters.append(filt)

    def removeFilter(self, filt: Any) -> None:
        if filt in self.filters:
            self.filters.remove(filt)

    def filter(self, record: LogRecord) -> bool:
        rv: bool = True
        for f in self.filters:
            result: Any = f.filter(record) if hasattr(f, 'filter') else f(record)
            if not result:
                rv = False
                break
        return rv


class ConsoleLogStream(object):
    def __init__(self) -> None:
        self.name: str = 'console'

    def write(self, msg: str) -> None:
        msg = msg.rstrip('\n\r')
        if len(msg) > 0:
            console.log(msg)


_consoleStream: ConsoleLogStream = ConsoleLogStream()
_handlers: Dict[str, Any] = {}
_handlerList: List[Any] = []


def _removeHandlerRef(wr: Any) -> None:
    acquire, release, handlers = (_acquireLock, _releaseLock, _handlerList)
    if acquire and release and handlers:
        acquire()
        try:
            if wr in handlers:
                handlers.remove(wr)
        finally:
            release()


def _addHandlerRef(handler: Handler) -> None:
    _acquireLock()
    try:
        _handlerList.append(handler)
    finally:
        _releaseLock()


class Handler(Filterer):
    def __init__(self, level: Union[int, str] = NOTSET) -> None:
        super().__init__()
        self._name: Optional[str] = None
        self.level: int = _checkLevel(level)
        self.formatter: Optional[Formatter] = None
        _addHandlerRef(self)
        self.createLock()

    def get_name(self) -> Optional[str]:
        return self._name

    def set_name(self, name: Optional[str]) -> None:
        _acquireLock()
        try:
            if self._name in _handlers:
                del _handlers[self._name]
            self._name = name
            if name:
                _handlers[name] = self
        finally:
            _releaseLock()

    name = property(get_name, set_name)

    def createLock(self) -> None:
        self.lock: Optional[Any] = None

    def acquire(self) -> None:
        if self.lock:
            self.lock.acquire()

    def release(self) -> None:
        if self.lock:
            self.lock.release()

    def setLevel(self, level: Union[int, str]) -> None:
        self.level = _checkLevel(level)

    def format(self, record: LogRecord) -> str:
        fmt: Formatter = self.formatter if self.formatter else _defaultFormatter
        return fmt.format(record)

    def emit(self, record: LogRecord) -> None:
        raise NotImplementedError('Must be implemented by handler')

    def handle(self, record: LogRecord) -> bool:
        rv: bool = self.filter(record)
        if rv:
            self.acquire()
            try:
                self.emit(record)
            finally:
                self.release()
        return rv

    def setFormatter(self, fmt: Formatter) -> None:
        self.formatter = fmt

    def flush(self) -> None:
        pass

    def close(self) -> None:
        _acquireLock()
        try:
            if self._name and self._name in _handlers:
                del _handlers[self._name]
        finally:
            _releaseLock()

    def handleError(self, record: LogRecord) -> None:
        if raiseExceptions:
            raise Exception('Failed to log: {}'.format(record))
        else:
            _consoleStream.write('--- Logging Error ---\n')

    def __repr__(self) -> str:
        level_str: str = getLevelName(self.level)
        return '<{} ({})>'.format(self.__class__.__name__, level_str)


class StreamHandler(Handler):
    terminator: str = '\n'

    def __init__(self, stream: Optional[Any] = None, level: Union[int, str] = NOTSET) -> None:
        super().__init__(level)
        if stream is None:
            stream = _consoleStream
        self.stream: Any = stream

    def flush(self) -> None:
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: LogRecord) -> None:
        try:
            msg: str = self.format(record)
            stream: Any = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def __repr__(self) -> str:
        level_str: str = getLevelName(self.level)
        name: str = getattr(self.stream, 'name', '')
        if name:
            name += ' '
        return '<{} {}({})>'.format(self.__class__.__name__, name, level_str)


class FileHandler(StreamHandler):
    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False) -> None:
        raise NotImplementedError('No Filesystem for FileHandler')


class _StderrHandler(StreamHandler):
    def __init__(self, level: Union[int, str] = NOTSET) -> None:
        super().__init__(None, level)

    def _getStream(self) -> Any:
        return _consoleStream
    stream = property(_getStream)


_defaultLastResort: _StderrHandler = _StderrHandler(WARNING)
lastResort: _StderrHandler = _defaultLastResort


class PlaceHolder(object):
    def __init__(self, alogger: Logger) -> None:
        n: Optional[str] = alogger.name
        self.loggerMap: Dict[str, Logger] = {n: alogger} if n else {}

    def append(self, alogger: Logger) -> None:
        n: Optional[str] = alogger.name
        if n and n not in self.loggerMap:
            self.loggerMap[n] = alogger


def setLoggerClass(klass: Type[Logger]) -> None:
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
    global _loggerClass
    _loggerClass = klass


def getLoggerClass() -> Type[Logger]:
    return _loggerClass


class Manager(object):
    def __init__(self, rootnode: Logger) -> None:
        self.root: Logger = rootnode
        self.disable: int = 0
        self.emittedNoHandlerWarning: bool = False
        self.loggerDict: Dict[str, Union[Logger, PlaceHolder]] = {}
        self.loggerClass: Optional[Type[Logger]] = None
        self.logRecordFactory: Optional[Callable[..., LogRecord]] = None

    def getLogger(self, name: str) -> Logger:
        rv: Optional[Union[Logger, PlaceHolder]] = None
        if not isinstance(name, str):
            raise TypeError('A logger name must be a string')
        _acquireLock()
        try:
            if name in self.loggerDict:
                rv = self.loggerDict[name]
                if isinstance(rv, PlaceHolder):
                    ph: PlaceHolder = rv
                    rv = (self.loggerClass or _loggerClass)(name)
                    rv.manager = self
                    self.loggerDict[name] = rv
                    self._fixupChildren(ph, rv)
                    self._fixupParents(rv)
            else:
                rv = (self.loggerClass or _loggerClass)(name)
                rv.manager = self
                self.loggerDict[name] = rv
                self._fixupParents(rv)
            assert isinstance(rv, Logger)
        finally:
            _releaseLock()
        return rv

    def setLoggerClass(self, klass: Type[Logger]) -> None:
        if klass != Logger:
            if not issubclass(klass, Logger):
                raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
        self.loggerClass = klass

    def setLogRecordFactory(self, factory: Callable[..., LogRecord]) -> None:
        self.logRecordFactory = factory

    def _fixupParents(self, alogger: Logger) -> None:
        name: str = alogger.name or ""
        i: int = name.rfind('.')
        rv: Optional[Logger] = None
        while i > 0 and not rv:
            substr: str = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = PlaceHolder(alogger)
            else:
                obj: Union[Logger, PlaceHolder] = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    assert isinstance(obj, PlaceHolder)
                    obj.append(alogger)
            i = name.rfind('.', 0, i - 1)
        if not rv:
            rv = self.root
        alogger.parent = rv

    def _fixupChildren(self, ph: PlaceHolder, alogger: Logger) -> None:
        name: str = alogger.name or ""
        for c in list(ph.loggerMap.keys()):
            log: Logger = ph.loggerMap[c]
            if not log.parent.name.startswith(name):
                alogger.parent = log.parent
                log.parent = alogger


class Logger(Filterer):
    def __init__(self, name: str, level: Union[int, str] = NOTSET) -> None:
        super().__init__()
        self.name: str = name
        self.level: int = _checkLevel(level)
        self.parent: Optional[Logger] = None
        self.propagate: bool = True
        self.handlers: List[Handler] = []
        self.disabled: bool = False

    def setLevel(self, level: Union[int, str]) -> None:
        self.level = _checkLevel(level)

    __pragma__('kwargs')
    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1388, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg: Any, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)
    critical.fatal = critical

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
        if not isinstance(level, int):
            if raiseExceptions:
                raise TypeError('level must be an integer')
            else:
                return
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def findCaller(self, stack_info: bool = False) -> Tuple[str, int, str, Optional[Any]]:
        f: Any = currentframe()
        rv: Tuple[str, int, str, Optional[Any]] = ('(unknown file)', 0, '(unknown function)', None)
        return rv

    def makeRecord(self, name: str, level: int, fn: str, lno: int, msg: Any, args: Tuple[Any, ...],
                   exc_info: Any, func: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
                   sinfo: Optional[Any] = None) -> LogRecord:
        rv: LogRecord = _logRecordFactory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in ['message', 'asctime'] or key in rv.__dict__:
                    raise KeyError('Attempt to overwrite %r in LogRecord' % key)
                rv.__dict__[key] = extra[key]
        return rv

    def _log(self, level: int, msg: Any, args: Tuple[Any, ...], exc_info: Any = None,
             extra: Optional[Dict[str, Any]] = None, stack_info: bool = False) -> None:
        sinfo: Optional[Any] = None
        if _srcfile:
            try:
                fn, lno, func, sinfo = self.findCaller(stack_info)
            except ValueError:
                fn, lno, func = ('(unknown file)', 0, '(unknown function)')
        else:
            fn, lno, func = ('(unknown file)', 0, '(unknown function)')
        record: LogRecord = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
        self.handle(record)

    def handle(self, record: LogRecord) -> None:
        if not self.disabled and self.filter(record):
            self.callHandlers(record)

    def addHandler(self, hdlr: Handler) -> None:
        _acquireLock()
        try:
            if hdlr not in self.handlers:
                self.handlers.append(hdlr)
        finally:
            _releaseLock()

    def removeHandler(self, hdlr: Handler) -> None:
        _acquireLock()
        try:
            if hdlr in self.handlers:
                self.handlers.remove(hdlr)
        finally:
            _releaseLock()

    def hasHandlers(self) -> bool:
        c: Optional[Logger] = self
        rv: bool = False
        while c:
            if len(c.handlers) > 0:
                rv = True
                break
            if not c.propagate:
                break
            else:
                c = c.parent
        return rv

    def callHandlers(self, record: LogRecord) -> None:
        c: Optional[Logger] = self
        found: int = 0
        while c:
            for hdlr in c.handlers:
                found += 1
                if record.levelno >= hdlr.level:
                    hdlr.handle(record)
            if not c.propagate:
                c = None
            else:
                c = c.parent
        if found == 0:
            if lastResort:
                if record.levelno >= lastResort.level:
                    lastResort.handle(record)
            elif raiseExceptions and (not self.manager.emittedNoHandlerWarning):
                _consoleStream.write('No handlers could be found for logger "{}"'.format(self.name))
                self.manager.emittedNoHandlerWarning = True

    def getEffectiveLevel(self) -> int:
        logger: Optional[Logger] = self
        while logger:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level: int) -> bool:
        if self.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getChild(self, suffix: str) -> Logger:
        if self.parent is not self:
            suffix = '.'.join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def __repr__(self) -> str:
        level_str: str = getLevelName(self.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, self.name, level_str)


class RootLogger(Logger):
    def __init__(self, level: Union[int, str]) -> None:
        super().__init__('root', level)


_loggerClass: Type[Logger] = Logger


class LoggerAdapter(object):
    def __init__(self, logger: Logger, extra: Dict[str, Any]) -> None:
        self.logger: Logger = logger
        self.extra: Dict[str, Any] = extra

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        kwargs['extra'] = self.extra
        return msg, kwargs

    __pragma__('kwargs')
    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.log(WARNING, msg, *args, **kwargs)

    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1719, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.log(ERROR, msg, *args, **kwargs)

    def exception(self, msg: Any, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        self.log(ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.log(CRITICAL, msg, *args, **kwargs)

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def isEnabledFor(self, level: int) -> bool:
        if self.logger.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def setLevel(self, level: Union[int, str]) -> None:
        self.logger.setLevel(level)

    def getEffectiveLevel(self) -> int:
        return self.logger.getEffectiveLevel()

    def hasHandlers(self) -> bool:
        return self.logger.hasHandlers()

    def __repr__(self) -> str:
        level_str: str = getLevelName(self.logger.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, self.logger.name, level_str)


root: RootLogger = RootLogger(WARNING)
Logger.root = root
Logger.manager = Manager(Logger.root)
root.manager = Logger.manager


def _resetLogging() -> None:
    global _handlerList, _handlers, root
    _handlerList = []
    _handlers = {}
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)
    root.manager = Logger.manager
__pragma__('kwargs')


def basicConfig(**kwargs: Any) -> None:
    _acquireLock()
    try:
        if len(root.handlers) == 0:
            handlers: Optional[List[Handler]] = kwargs.pop('handlers', None)
            if handlers is not None:
                if 'stream' in kwargs:
                    raise ValueError("'stream' should not be specified together with 'handlers'")
            if handlers is None:
                stream: Optional[Any] = kwargs.pop('stream', None)
                h: StreamHandler = StreamHandler(stream)
                handlers = [h]
            dfs: Optional[str] = kwargs.pop('datefmt', None)
            style: str = kwargs.pop('style', '{')
            if style not in _STYLES:
                raise ValueError('Style must be one of: {}'.format(','.join(_STYLES.keys())))
            fs: str = kwargs.pop('format', _STYLES[style][1])
            fmt: Formatter = Formatter(fs, dfs, style)
            for h in handlers:
                if h.formatter is None:
                    h.setFormatter(fmt)
                root.addHandler(h)
            level: Optional[Union[int, str]] = kwargs.pop('level', None)
            if level is not None:
                root.setLevel(level)
            if len(kwargs) > 0:
                keys: str = ', '.join(kwargs.keys())
                raise ValueError('Unrecognised argument(s): {}'.format(keys))
    finally:
        _releaseLock()
__pragma__('nokwargs')


def getLogger(name: Optional[str] = None) -> Logger:
    if name:
        return Logger.manager.getLogger(name)
    else:
        return root
__pragma__('kwargs')


def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.critical(msg, *args, **kwargs)
fatal = critical


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.error(msg, *args, **kwargs)


def exception(msg: Any, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
    error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.warning(msg, *args, **kwargs)


def warn(msg: Any, *args: Any, **kwargs: Any) -> None:
    warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1944, 'logging')
    warning(msg, *args, **kwargs)


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.info(msg, *args, **kwargs)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.debug(msg, *args, **kwargs)


def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    if len(root.handlers) == 0:
        basicConfig()
    root.log(level, msg, *args, **kwargs)
__pragma__('nokwargs')


def disable(level: int) -> None:
    root.manager.disable = level


def shutdown(handlerList: List[Any] = _handlerList) -> None:
    for wr in reversed(handlerList[:]):
        try:
            h: Optional[Handler] = wr if isinstance(wr, Handler) else None
            if h:
                try:
                    h.acquire()
                    h.flush()
                    h.close()
                except (OSError, ValueError):
                    pass
                finally:
                    h.release()
        except Exception as exc:
            if raiseExceptions:
                raise exc


class NullHandler(Handler):
    def handle(self, record: LogRecord) -> None:
        pass

    def emit(self, record: LogRecord) -> None:
        pass

    def createLock(self) -> None:
        self.lock = None


_warnings_showwarning: Optional[Callable[..., Any]] = None


def _showwarning(message: Any, category: Any, filename: str, lineno: int, file: Optional[Any] = None,
                 line: Optional[Any] = None) -> None:
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(message, category, filename, lineno, file, line)
    else:
        s: str = warnings.formatwarning(message, category, filename, lineno, line)
        logger: Logger = getLogger('py.warnings')
        if not logger.handlers:
            logger.addHandler(NullHandler())
        logger.warning(s)


def captureWarnings(capture: bool) -> None:
    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.setShowWarning(_showwarning)
    elif _warnings_showwarning is not None:
        warnings.setShowWarnings(_warnings_showwarning)
        _warnings_showwarning = None