"""
Logging package for Python. Based on PEP 282 and comments thereto in
comp.lang.python.

Copyright (C) 2001-2016 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!

Edited By Carl Allendorph 2016 for the Transcrypt Project

This code base was originally pulled from the Python project to
maintain as much compatibility in the interfaces as possible.
"""
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Type, Mapping, 
    Iterable, Set, cast, TypeVar, Generic, Sequence, TextIO, IO
)
from org.transcrypt.stubs.browser import __pragma__
import time
import warnings
import collections

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
    """
    Return the textual representation of logging level 'level'.
    """
    if isinstance(level, int):
        return _levelToName.get(level, f'Level {level}')
    return str(_nameToLevel.get(str(level), f'Level {level}')

def addLevelName(level: int, levelName: str) -> None:
    """
    Associate 'levelName' with 'level'.
    """
    _acquireLock()
    try:
        _levelToName[level] = levelName
        _nameToLevel[levelName] = level
    finally:
        _releaseLock()

def currentframe() -> Optional[Any]:
    return None

_srcfile: Optional[str] = None

def _checkLevel(level: Union[int, str]) -> int:
    if isinstance(level, int):
        rv = level
    elif isinstance(level, str):
        if level not in _nameToLevel:
            raise ValueError(f'Unknown level: {level}')
        rv = _nameToLevel[level]
    else:
        raise TypeError(f'Level not an integer or a valid string: {level}')
    return rv

_lock: Optional[Any] = None

def _acquireLock() -> None:
    """
    Acquire the module-level lock for serializing access to shared data.
    """
    if _lock:
        _lock.acquire()

def _releaseLock() -> None:
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()

class LogRecord:
    """
    A LogRecord instance represents an event being logged.
    """

    def __init__(
        self,
        name: str,
        level: int,
        pathname: str,
        lineno: int,
        msg: str,
        args: Tuple[Any, ...],
        exc_info: Optional[Any],
        func: Optional[str] = None,
        sinfo: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        ct: float = time.time()
        self.name: str = name
        self.msg: str = msg
        if args and len(args) == 1 and isinstance(args[0], collections.Mapping) and args[0]:
            if raiseExceptions:
                raise NotImplementedError('No Dict Args to Log Record')
        self.args: Tuple[Any, ...] = args
        self.levelname: str = getLevelName(level)
        self.levelno: int = level
        self.pathname: str = pathname
        self.filename: str = pathname
        self.module: str = 'Unknown module'
        self.exc_info: Optional[Any] = exc_info
        self.exc_text: Optional[str] = None
        self.stack_info: Optional[str] = sinfo
        self.lineno: int = lineno
        self.funcName: Optional[str] = func
        self.created: float = ct
        self.msecs: float = (ct - int(ct)) * 1000
        self.relativeCreated: float = (self.created - _startTime) * 1000
        self.thread: Optional[Any] = None
        self.threadName: Optional[str] = None
        self.processName: Optional[str] = None
        self.process: Optional[Any] = None

    def getMessage(self) -> str:
        """
        Return the message for this LogRecord.
        """
        msg = str(self.msg)
        if self.args:
            msg = msg.format(*self.args)
        return msg

    def toDict(self) -> Dict[str, Any]:
        """ Convert the LogRecord object into a dict. """
        keysToPick: List[str] = [
            'name', 'msg', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'asctime', 'msecs',
            'relativeCreated', 'thread', 'threadName', 'process'
        ]
        ret: Dict[str, Any] = {}
        for k in keysToPick:
            if k == 'name':
                ret[k] = getattr(self, 'py_name', None)
            else:
                ret[k] = getattr(self, k, None)
        ret['message'] = self.getMessage()
        return ret

    def __str__(self) -> str:
        return f'<LogRecord: {self.name}, {self.levelno}, {self.pathname}, {self.lineno}, "{self.msg}">'

    def __repr__(self) -> str:
        return str(self)

_logRecordFactory: Type[LogRecord] = LogRecord

def setLogRecordFactory(factory: Type[LogRecord]) -> None:
    """
    Set the factory to be used when instantiating a log record.
    """
    global _logRecordFactory
    _logRecordFactory = factory

def getLogRecordFactory() -> Type[LogRecord]:
    """
    Return the factory to be used when instantiating a log record.
    """
    return _logRecordFactory

def makeLogRecord(dict: Dict[str, Any]) -> LogRecord:
    """
    Make a LogRecord whose attributes are defined by the specified dictionary.
    """
    rv = _logRecordFactory(None, None, '', 0, '', (), None, None)
    rv.__dict__.update(dict)
    return rv

class PercentStyle:
    default_format: str = '%(message)s'
    asctime_format: str = '%(asctime)s'
    asctime_search: str = '%(asctime)'

    def __init__(self, fmt: Optional[str] = None) -> None:
        self._fmt: str = fmt or self.default_format

    def usesTime(self) -> bool:
        return self._fmt.find(self.asctime_search) >= 0

    def format(self, record: LogRecord) -> str:
        return self._fmt % record.__dict__

class StrFormatStyle(PercentStyle):
    default_format: str = '{message}'
    asctime_format: str = '{asctime}'
    asctime_search: str = '{asctime'

    def format(self, record: LogRecord) -> str:
        return self._fmt.format(**record.toDict())

class StringTemplateStyle(PercentStyle):
    default_format: str = '${message}'
    asctime_format: str = '${asctime}'
    asctime_search: str = '${asctime}'

    def __init__(self, fmt: Optional[str] = None) -> None:
        self._fmt: str = fmt or self.default_format
        self._tpl: Any = Template(self._fmt)

    def usesTime(self) -> bool:
        fmt = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_format) >= 0

    def format(self, record: LogRecord) -> str:
        return self._tpl.substitute(**record.__dict__)

BASIC_FORMAT: str = '{levelname}:{name}:{message}'
_STYLES: Dict[str, Tuple[Type[PercentStyle], str]] = {
    '{': (StrFormatStyle, BASIC_FORMAT)
}

class Formatter:
    """
    Formatter instances are used to convert a LogRecord to text.
    """
    converter: Callable[[float], time.struct_time] = time.localtime
    default_time_format: str = '%Y-%m-%d %H:%M:%S'
    default_msec_format: str = '{},{:03d}'

    def __init__(
        self,
        format: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '{'
    ) -> None:
        if style != '{':
            raise NotImplementedError('{} format only')
        self._style = _STYLES[style][0](format)
        self._fmt = self._style._fmt
        self.datefmt = datefmt

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Return the creation time of the specified LogRecord as formatted text.
        """
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s

    def formatException(self, ei: Any) -> str:
        """
        Format and return the specified exception information as a string.
        """
        return str(ei)

    def usesTime(self) -> bool:
        """
        Check if the format uses the creation time of the record.
        """
        return self._style.usesTime()

    def formatMessage(self, record: LogRecord) -> str:
        return self._style.format(record)

    def formatStack(self, stack_info: str) -> str:
        """
        This method is provided as an extension point for specialized
        formatting of stack information.
        """
        return stack_info

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record as text.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1] != '\n':
                s = s + '\n'
                s = s + record.exc_text
        if record.stack_info:
            if s[-1] != '\n':
                s = s + '\n'
                s = s + self.formatStack(record.stack_info)
        return s

_defaultFormatter: Formatter = Formatter()

class BufferingFormatter:
    """
    A formatter suitable for formatting a number of records.
    """

    def __init__(self, linefmt: Optional[Formatter] = None) -> None:
        """
        Optionally specify a formatter which will be used to format each
        individual record.
        """
        if linefmt:
            self.linefmt = linefmt
        else:
            self.linefmt = _defaultFormatter

    def formatHeader(self, records: List[LogRecord]) -> str:
        """
        Return the header string for the specified records.
        """
        return ''

    def formatFooter(self, records: List[LogRecord]) -> str:
        """
        Return the footer string for the specified records.
        """
        return ''

    def format(self, records: List[LogRecord]) -> str:
        """
        Format the specified records and return the result as a string.
        """
        rv = ''
        if len(records) > 0:
            rv = rv + self.formatHeader(records)
            for record in records:
                rv = rv + self.linefmt.format(record)
                rv = rv + self.formatFooter(records)
        return rv

class Filter:
    """
    Filter instances are used to perform arbitrary filtering of LogRecords.
    """

    def __init__(self, name: str = '') -> None:
        """
        Initialize a filter.
        """
        self.name = name
        self.nlen = len(name)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if the specified record is to be logged.
        """
        if self.nlen == 0:
            return True
        elif self.name == record.name:
            return True
        elif record.name.find(self.name, 0, self.nlen) != 0:
            return False
        return record.name[self.nlen] == '.'

class Filterer:
    """
    A base class for loggers and handlers which allows them to share
    common code.
    """

    def __init__(self) -> None:
        """
        Initialize the list of filters to be an empty list.
        """
        self.filters: List[Filter] = []

    def addFilter(self, filt: Filter) -> None:
        """
        Add the specified filter to this handler.
        """
        if filt not in self.filters:
            self.filters.append(filt)

    def removeFilter(self, filt: Filter) -> None:
        """
        Remove the specified filter from this handler.
        """
        if filt in self.filters:
            self.filters.remove(filt)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if a record is loggable by consulting all the filters.
        """
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
    """ This implements a quasi "Stream" like object for mimicing
    the sys.stderr object.
    """

    def __init__(self) -> None:
        self.name = 'console'

    def write(self, msg: str) -> None:
        msg = msg.rstrip('\n\r')
        if len(msg) > 0:
            console.log(msg)

_consoleStream: ConsoleLogStream = ConsoleLogStream()
_handlers: Dict[str, 'Handler'] = {}
_handlerList: List['Handler'] = []

def _removeHandlerRef(wr: 'Handler') -> None:
    """
    Remove a handler reference from the internal cleanup list.
    """
    acquire, release, handlers = (_acquireLock, _releaseLock, _handlerList)
    if acquire and release and handlers:
        acquire()
        try:
            if wr in handlers:
                handlers.remove(wr)
        finally:
            release()

def _addHandlerRef(handler: 'Handler') -> None:
    """
    Add a handler to the internal cleanup list using a weak reference.
    """
    _acquireLock()
    try:
        _handlerList.append(handler)
    finally:
        _releaseLock()

class Handler(Filterer):
    """
    Handler instances dispatch logging events to specific destinations.
    """

    def __init__(self, level: int = NOTSET) -> None:
        """
        Initializes the instance - basically setting the formatter to None
        and the filter list to empty.
        """
        Filterer.__init__(self)
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
        """
        Acquire a thread lock for serializing access to the underlying I/O.
        """
        self.lock: Optional[Any] = None

    def acquire(self) -> None:
        """
        Acquire the I/O thread lock.
        """
        if self.lock:
            self.lock.acquire()

    def release(self) -> None:
        """
        Release the I/O thread lock.
        """
        if self.lock:
            self.lock.release()

    def setLevel(self, level: Union[int, str]) -> None:
        """
        Set the logging level of this handler.  level must be an int or a str.
        """
        self.level = _checkLevel(level)

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record.
        """
        if self.formatter:
            fmt = self.formatter
        else:
            fmt = _defaultFormatter
        return fmt.format(record)

    def emit(self, record: LogRecord) -> None:
        """
        Do whatever it takes to actually log the specified logging record.
        """
        raise NotImplementedError('Must be implemented by handler')

    def handle(self, record: LogRecord) -> bool:
        """
        Conditionally emit the specified logging record.
        """
        rv = self.filter(record)
        if rv:
            self.acquire()
            try:
                self.emit(record)
            finally:
                self.release()
        return rv

    def setFormatter(self, fmt: Formatter) -> None:
        """
        Set the formatter