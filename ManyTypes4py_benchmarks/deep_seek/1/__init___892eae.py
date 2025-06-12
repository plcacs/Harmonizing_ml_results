from org.transcrypt.stubs.browser import __pragma__
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Mapping, Iterable, Type, TypeVar, cast

__author__: str = 'Vinay Sajip <vinay_sajip@red-dove.com>, Carl Allendorph'
__status__: str = 'experimental'
__version__: str = '0.5.1.2'
__date__: str = '15 November 2016'
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
_levelToName: Dict[int, str] = {CRITICAL: 'CRITICAL', ERROR: 'ERROR', WARNING: 'WARNING', INFO: 'INFO', DEBUG: 'DEBUG', NOTSET: 'NOTSET'}
_nameToLevel: Dict[str, int] = {'CRITICAL': CRITICAL, 'FATAL': FATAL, 'ERROR': ERROR, 'WARN': WARNING, 'WARNING': WARNING, 'INFO': INFO, 'DEBUG': DEBUG, 'NOTSET': NOTSET}

def getLevelName(level: Union[int, str]) -> str:
    return _levelToName.get(level) or _nameToLevel.get(level) or 'Level {}'.format(level)

def addLevelName(level: int, levelName: str) -> None:
    _acquireLock()
    try:
        _levelToName[level] = levelName
        _nameToLevel[levelName] = level
    except Exception as exc:
        raise exc
    finally:
        _releaseLock()

def currentframe() -> None:
    return None
_srcfile: Optional[str] = None

def _checkLevel(level: Union[int, str]) -> int:
    if isinstance(level, int):
        rv = level
    elif str(level) == level:
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

class LogRecord:
    def __init__(self, name: str, level: int, pathname: str, lineno: int, msg: Any, args: Tuple[Any, ...], exc_info: Optional[Any], func: Optional[str] = None, sinfo: Optional[str] = None, **kwargs: Any) -> None:
        ct = time.time()
        self.name = name
        self.msg = msg
        if args and len(args) == 1 and isinstance(args[0], collections.Mapping) and args[0]:
            if raiseExceptions:
                raise NotImplementedError('No Dict Args to Log Record')
        self.args = args
        self.levelname = getLevelName(level)
        self.levelno = level
        self.pathname = pathname
        self.filename = pathname
        self.module = 'Unknown module'
        self.exc_info = exc_info
        self.exc_text = None
        self.stack_info = sinfo
        self.lineno = lineno
        self.funcName = func
        self.created = ct
        self.msecs = (ct - int(ct)) * 1000
        self.relativeCreated = (self.created - _startTime) * 1000
        self.thread = None
        self.threadName = None
        self.processName = None
        self.process = None

    def getMessage(self) -> str:
        msg = str(self.msg)
        if self.args:
            msg = msg.format(*self.args)
        return msg

    def toDict(self) -> Dict[str, Any]:
        keysToPick = ['name', 'msg', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno', 'funcName', 'created', 'asctime', 'msecs', 'relativeCreated', 'thread', 'threadName', 'process']
        ret = {}
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
_logRecordFactory: Type[LogRecord] = LogRecord

def setLogRecordFactory(factory: Type[LogRecord]) -> None:
    global _logRecordFactory
    _logRecordFactory = factory

def getLogRecordFactory() -> Type[LogRecord]:
    return _logRecordFactory

def makeLogRecord(dict: Dict[str, Any]) -> LogRecord:
    rv = _logRecordFactory(None, None, '', 0, '', (), None, None)
    rv.__dict__.update(dict)
    return rv

class PercentStyle:
    default_format = '%(message)s'
    asctime_format = '%(asctime)s'
    asctime_search = '%(asctime)'

    def __init__(self, fmt: Optional[str]) -> None:
        self._fmt = fmt or self.default_format

    def usesTime(self) -> bool:
        return self._fmt.find(self.asctime_search) >= 0

    def format(self, record: LogRecord) -> str:
        return self._fmt % record.__dict__

class StrFormatStyle(PercentStyle):
    default_format = '{message}'
    asctime_format = '{asctime}'
    asctime_search = '{asctime'
    __pragma__('kwargs')

    def format(self, record: LogRecord) -> str:
        return self._fmt.format(**record.toDict())
    __pragma__('nokwargs')

class StringTemplateStyle(PercentStyle):
    default_format = '${message}'
    asctime_format = '${asctime}'
    asctime_search = '${asctime}'

    def __init__(self, fmt: Optional[str]) -> None:
        self._fmt = fmt or self.default_format
        self._tpl = Template(self._fmt)

    def usesTime(self) -> bool:
        fmt = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_format) >= 0
    __pragma__('kwargs')

    def format(self, record: LogRecord) -> str:
        return self._tpl.substitute(**record.__dict__)
    __pragma__('nokwargs')
BASIC_FORMAT: str = '{levelname}:{name}:{message}'
_STYLES: Dict[str, Tuple[Type[PercentStyle], str]] = {'{': (StrFormatStyle, BASIC_FORMAT)}

class Formatter:
    converter: Callable[[float], time.struct_time] = time.localtime
    __pragma__('kwargs')

    def __init__(self, format: Optional[str] = None, datefmt: Optional[str] = None, style: str = '{') -> None:
        if style != '{':
            raise NotImplementedError('{} format only')
        self._style = _STYLES[style][0](format)
        self._fmt = self._style._fmt
        self.datefmt = datefmt
    __pragma__('nokwargs')
    default_time_format: str = '%Y-%m-%d %H:%M:%S'
    default_msec_format: str = '{},{:03d}'

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s

    def formatException(self, ei: Any) -> str:
        return str(ei)

    def usesTime(self) -> bool:
        return self._style.usesTime()

    def formatMessage(self, record: LogRecord) -> str:
        return self._style.format(record)

    def formatStack(self, stack_info: str) -> str:
        return stack_info

    def format(self, record: LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[len(s) - 1] != '\n':
                s = s + '\n'
                s = s + record.exc_text
        if record.stack_info:
            if s[len(s) - 1] != '\n':
                s = s + '\n'
                s = s + self.formatStack(record.stack_info)
        return s
_defaultFormatter = Formatter()

class BufferingFormatter:
    def __init__(self, linefmt: Optional[Formatter] = None) -> None:
        if linefmt:
            self.linefmt = linefmt
        else:
            self.linefmt = _defaultFormatter

    def formatHeader(self, records: List[LogRecord]) -> str:
        return ''

    def formatFooter(self, records: List[LogRecord]) -> str:
        return ''

    def format(self, records: List[LogRecord]) -> str:
        rv = ''
        if len(records) > 0:
            rv = rv + self.formatHeader(records)
            for record in records:
                rv = rv + self.linefmt.format(record)
                rv = rv + self.formatFooter(records)
        return rv

class Filter:
    def __init__(self, name: str = '') -> None:
        self.name = name
        self.nlen = len(name)

    def filter(self, record: LogRecord) -> bool:
        if self.nlen == 0:
            return True
        elif self.name == record.name:
            return True
        elif record.name.find(self.name, 0, self.nlen) != 0:
            return False
        return record.name[self.nlen] == '.'

class Filterer:
    def __init__(self) -> None:
        self.filters: List[Filter] = []

    def addFilter(self, filt: Filter) -> None:
        if not filt in self.filters:
            self.filters.append(filt)

    def removeFilter(self, filt: Filter) -> None:
        if filt in self.filters:
            self.filters.remove(filt)

    def filter(self, record: LogRecord) -> bool:
        rv = True
        for f in self.filters:
            if hasattr(f, 'filter'):
                result = f.filter(record)
            else:
                result = f(record)
            if not result:
                rv = False
                break
        return rv

class ConsoleLogStream:
    def __init__(self) -> None:
        self.name = 'console'

    def write(self, msg: str) -> None:
        msg = msg.rstrip('\n\r')
        if len(msg) > 0:
            console.log(msg)
_consoleStream = ConsoleLogStream()
_handlers: Dict[str, 'Handler'] = {}
_handlerList: List['Handler'] = []

def _removeHandlerRef(wr: 'Handler') -> None:
    acquire, release, handlers = (_acquireLock, _releaseLock, _handlerList)
    if acquire and release and handlers:
        acquire()
        try:
            if wr in handlers:
                handlers.remove(wr)
        finally:
            release()

def _addHandlerRef(handler: 'Handler') -> None:
    _acquireLock()
    try:
        _handlerList.append(handler)
    finally:
        _releaseLock()

class Handler(Filterer):
    def __init__(self, level: int = NOTSET) -> None:
        Filterer.__init__(self)
        self._name: Optional[str] = None
        self.level = _checkLevel(level)
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
        if self.formatter:
            fmt = self.formatter
        else:
            fmt = _defaultFormatter
        return fmt.format(record)

    def emit(self, record: LogRecord) -> None:
        raise NotImplementedError('Must be implemented by handler')

    def handle(self, record: LogRecord) -> bool:
        rv = self.filter(record)
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
        level = getLevelName(self.level)
        return '<{} ({})>'.format(self.__class__.__name__, level)

class StreamHandler(Handler):
    terminator = '\n'

    def __init__(self, stream: Optional[ConsoleLogStream] = None, level: int = NOTSET) -> None:
        Handler.__init__(self, level)
        if stream is None:
            stream = _consoleStream
        self.stream = stream

    def flush(self) -> None:
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def __repr__(self) -> str:
        level = getLevelName(self.level)
        name = getattr(self.stream, 'name', '')
        if name:
            name += ' '
        return '<{} {}({})>'.format(self.__class__.__name__, name, level)

class FileHandler(StreamHandler):
    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False) -> None:
        raise NotImplementedError('No Filesystem for FileHandler')

class _StderrHandler(StreamHandler):
    def __init__(self, level: int = NOTSET) -> None:
        StreamHandler.__init__(self, None, level)

    def _getStream(self) -> ConsoleLogStream:
        return _consoleStream
    stream = property(_getStream)
_defaultLastResort = _StderrHandler(WARNING)
lastResort = _defaultLastResort

class PlaceHolder:
    def __init__(self, alogger: 'Logger') -> None:
        n = alogger.name
        self.loggerMap: Dict[str, 'Logger'] = {n: alogger}

    def append(self, alogger: 'Logger') -> None:
        n = alogger.name
        if n not in self.loggerMap.keys():
            self.loggerMap[n] = alogger

def setLoggerClass(klass: Type['Logger']) -> None:
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
    global _loggerClass
    _loggerClass = klass

def getLoggerClass() -> Type['Logger']:
    return _loggerClass

class Manager:
    def __init__(self, rootnode: 'Logger') -> None:
        self.root = rootnode
        self.disable = 0
        self.emittedNoHandlerWarning = False
        self.loggerDict: Dict[str, Union['Logger', PlaceHolder]] = {}
        self.loggerClass: Optional[Type['Logger']] = None
        self.logRecordFactory: Optional[Callable[..., LogRecord]] = None

    def getLogger(self, name: str) -> 'Logger':
        rv = None
        if not isinstance(name, str):
            raise TypeError('A logger name must be a string')
        _acquireLock()
        try:
            if name in self.loggerDict:
                rv = self.loggerDict[name]
                if isinstance(rv, PlaceHolder):
                    ph = rv
                    rv = (self.loggerClass or _loggerClass)(name)
                    rv.manager