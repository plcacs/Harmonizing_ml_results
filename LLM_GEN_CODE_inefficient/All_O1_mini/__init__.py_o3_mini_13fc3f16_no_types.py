from org.transcrypt.stubs.browser import __pragma__
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping
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
_levelToName: Dict[int, str] = {CRITICAL: 'CRITICAL', ERROR: 'ERROR',
    WARNING: 'WARNING', INFO: 'INFO', DEBUG: 'DEBUG', NOTSET: 'NOTSET'}
_nameToLevel: Dict[str, int] = {'CRITICAL': CRITICAL, 'FATAL': FATAL,
    'ERROR': ERROR, 'WARN': WARNING, 'WARNING': WARNING, 'INFO': INFO,
    'DEBUG': DEBUG, 'NOTSET': NOTSET}


def getLevelName(level):
    """
    Return the textual representation of logging level 'level'.
    """
    return _levelToName.get(level) or _nameToLevel.get(level
        ) or 'Level {}'.format(level)


def addLevelName(level, levelName):
    """
    Associate 'levelName' with 'level'.
    """
    _acquireLock()
    try:
        _levelToName[level] = levelName
        _nameToLevel[levelName] = level
    except Exception as exc:
        raise exc
    finally:
        _releaseLock()


def currentframe():
    return None


_srcfile: Optional[Any] = None


def _checkLevel(level):
    if isinstance(level, int):
        rv = level
    elif isinstance(level, str):
        if level not in _nameToLevel:
            raise ValueError('Unknown level: {}'.format(level))
        rv = _nameToLevel[level]
    else:
        raise TypeError('Level not an integer or a valid string: {}'.format
            (level))
    return rv


_lock: Optional[Any] = None


def _acquireLock():
    """
    Acquire the module-level lock for serializing access to shared data.
    """
    if _lock:
        _lock.acquire()


def _releaseLock():
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()


class LogRecord(object):
    """
    A LogRecord instance represents an event being logged.
    """

    def __init__(self, name, level, pathname, lineno, msg, args, exc_info,
        func=None, sinfo=None, **kwargs: Any):
        """
        Initialize a logging record with interesting information.
        """
        ct: float = time.time()
        self.name: str = name
        self.msg: str = msg
        if args and len(args) == 1 and isinstance(args[0], Mapping) and args[0
            ]:
            if raiseExceptions:
                raise NotImplementedError('No Dict Args to Log Record')
        self.args: Tuple[Any, ...] = args
        self.levelname: str = getLevelName(level)
        self.levelno: int = level
        self.pathname: str = pathname
        self.filename: str = pathname
        self.module: str = 'Unknown module'
        self.exc_info: Optional[Any] = exc_info
        self.exc_text: Optional[Any] = None
        self.stack_info: Optional[str] = sinfo
        self.lineno: int = lineno
        self.funcName: Optional[str] = func
        self.created: float = ct
        self.msecs: float = (ct - int(ct)) * 1000
        self.relativeCreated: float = (self.created - _startTime) * 1000
        self.thread: Optional[Any] = None
        self.threadName: Optional[Any] = None
        self.processName: Optional[Any] = None
        self.process: Optional[Any] = None

    def getMessage(self):
        """
        Return the message for this LogRecord.
        """
        msg: str = str(self.msg)
        if self.args:
            msg = msg.format(*self.args)
        return msg

    def toDict(self):
        """
        Utility method to convert the LogRecord object into a dict.
        """
        keysToPick: List[str] = ['name', 'msg', 'levelname', 'levelno',
            'pathname', 'filename', 'module', 'lineno', 'funcName',
            'created', 'asctime', 'msecs', 'relativeCreated', 'thread',
            'threadName', 'process']
        ret: Dict[str, Any] = {}
        for k in keysToPick:
            if k == 'name':
                ret[k] = getattr(self, 'py_name', None)
            else:
                ret[k] = getattr(self, k, None)
        ret['message'] = self.getMessage()
        return ret

    def __str__(self):
        return '<LogRecord: {}, {}, {}, {}, "{}">'.format(self.name, self.
            levelno, self.pathname, self.lineno, self.msg)

    def __repr__(self):
        return str(self)


_logRecordFactory: Callable[..., LogRecord] = LogRecord


def setLogRecordFactory(factory):
    """
    Set the factory to be used when instantiating a log record.
    """
    global _logRecordFactory
    _logRecordFactory = factory


def getLogRecordFactory():
    """
    Return the factory to be used when instantiating a log record.
    """
    return _logRecordFactory


def makeLogRecord(d):
    """
    Make a LogRecord from a dictionary.
    """
    rv: LogRecord = _logRecordFactory(None, None, '', 0, '', (), None, None)
    rv.__dict__.update(d)
    return rv


class PercentStyle(object):
    default_format: str = '%(message)s'
    asctime_format: str = '%(asctime)s'
    asctime_search: str = '%(asctime)'

    def __init__(self, fmt):
        self._fmt: str = fmt or self.default_format

    def usesTime(self):
        return self._fmt.find(self.asctime_search) >= 0

    def format(self, record):
        return self._fmt % record.__dict__


class StrFormatStyle(PercentStyle):
    default_format: str = '{message}'
    asctime_format: str = '{asctime}'
    asctime_search: str = '{asctime'
    __pragma__('kwargs')

    def format(self, record):
        return self._fmt.format(**record.toDict())
    __pragma__('nokwargs')


class StringTemplateStyle(PercentStyle):
    default_format: str = '${message}'
    asctime_format: str = '${asctime}'
    asctime_search: str = '${asctime}'

    def __init__(self, fmt):
        self._fmt: str = fmt or self.default_format
        self._tpl: Any = Template(self._fmt)

    def usesTime(self):
        fmt: str = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_format) >= 0
    __pragma__('kwargs')

    def format(self, record):
        return self._tpl.substitute(**record.__dict__)
    __pragma__('nokwargs')


BASIC_FORMAT: str = '{levelname}:{name}:{message}'
_STYLES: Dict[str, Tuple[Any, str]] = {'{': (StrFormatStyle, BASIC_FORMAT)}


class Formatter(object):
    """
    Formatter instances are used to convert a LogRecord to text.
    """
    converter: Callable[[float], Any] = time.localtime
    __pragma__('kwargs')

    def __init__(self, format=None, datefmt=None, style='{'):
        if style != '{':
            raise NotImplementedError('{} format only')
        self._style: Any = _STYLES[style][0](format)
        self._fmt: str = self._style._fmt
        self.datefmt: Optional[str] = datefmt
    __pragma__('nokwargs')
    default_time_format: str = '%Y-%m-%d %H:%M:%S'
    default_msec_format: str = '{}, {:03d}'

    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as formatted text.
        """
        ct: Any = self.converter(record.created)
        if datefmt:
            s: str = time.strftime(datefmt, ct)
        else:
            t: str = time.strftime(self.default_time_format, ct)
            s: str = self.default_msec_format.format(t, int(record.msecs))
        return s

    def formatException(self, ei):
        """
        Format and return the specified exception information as a string.
        """
        return str(ei)

    def usesTime(self):
        """
        Check if the format uses the creation time of the record.
        """
        return self._style.usesTime()

    def formatMessage(self, record):
        return self._style.format(record)

    def formatStack(self, stack_info):
        """
        Format stack information.
        """
        return stack_info if stack_info else ''

    def format(self, record):
        """
        Format the specified record as text.
        """
        record.message: str = record.getMessage()
        if self.usesTime():
            record.asctime: str = self.formatTime(record, self.datefmt)
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


class BufferingFormatter(object):
    """
    A formatter suitable for formatting a number of records.
    """

    def __init__(self, linefmt=None):
        """
        Optionally specify a formatter which will be used to format each
        individual record.
        """
        if linefmt:
            self.linefmt: Union[Formatter, str] = linefmt
        else:
            self.linefmt = _defaultFormatter

    def formatHeader(self, records):
        """
        Return the header string for the specified records.
        """
        return ''

    def formatFooter(self, records):
        """
        Return the footer string for the specified records.
        """
        return ''

    def format(self, records):
        """
        Format the specified records and return the result as a string.
        """
        rv: str = ''
        if len(records) > 0:
            rv = rv + self.formatHeader(records)
            for record in records:
                rv = rv + self.linefmt.format(record)
                rv = rv + self.formatFooter(records)
        return rv


class Filter(object):
    """
    Filter instances are used to perform arbitrary filtering of LogRecords.
    """

    def __init__(self, name=''):
        """
        Initialize a filter.
        """
        self.name: str = name
        self.nlen: int = len(name)

    def filter(self, record):
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


class Filterer(object):
    """
    A base class for loggers and handlers which allows them to share
    common code.
    """

    def __init__(self):
        """
        Initialize the list of filters to be an empty list.
        """
        self.filters: List[Union[Filter, Callable[[LogRecord], bool]]] = []

    def addFilter(self, filt):
        """
        Add the specified filter to this handler.
        """
        if filt not in self.filters:
            self.filters.append(filt)

    def removeFilter(self, filt):
        """
        Remove the specified filter from this handler.
        """
        if filt in self.filters:
            self.filters.remove(filt)

    def filter(self, record):
        """
        Determine if a record is loggable by consulting all the filters.
        """
        rv: bool = True
        for f in self.filters:
            if hasattr(f, 'filter'):
                result: bool = f.filter(record)
            else:
                result = f(record)
            if not result:
                rv = False
                break
        return rv


class ConsoleLogStream(object):
    """ This implements a quasi "Stream" like object for mimicing
    the sys.stderr object.
    """

    def __init__(self):
        self.name: str = 'console'

    def write(self, msg):
        """
        Write message to console.
        """
        msg = msg.rstrip('\n\r')
        if len(msg) > 0:
            console.log(msg)


_consoleStream: ConsoleLogStream = ConsoleLogStream()
_handlers: Dict[str, Any] = {}
_handlerList: List[Callable[[], Any]] = []


def _removeHandlerRef(wr):
    """
    Remove a handler reference from the internal cleanup list.
    """
    acquire, release, handlers = _acquireLock, _releaseLock, _handlerList
    if acquire and release and handlers:
        acquire()
        try:
            if wr in handlers:
                handlers.remove(wr)
        finally:
            release()


def _addHandlerRef(handler):
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

    def __init__(self, level=NOTSET):
        """
        Initializes the instance.
        """
        super().__init__()
        self._name: Optional[str] = None
        self.level: int = _checkLevel(level)
        self.formatter: Optional[Formatter] = None
        _addHandlerRef(self)
        self.createLock()

    def get_name(self):
        return self._name

    def set_name(self, name):
        _acquireLock()
        try:
            if self._name in _handlers:
                del _handlers[self._name]
            self._name = name
            if name:
                _handlers[name] = self
        finally:
            _releaseLock()
    name: property = property(get_name, set_name)

    def createLock(self):
        """
        Acquire a thread lock for serializing access to the underlying I/O.
        """
        self.lock: Optional[Any] = None

    def acquire(self):
        """
        Acquire the I/O thread lock.
        """
        if self.lock:
            self.lock.acquire()

    def release(self):
        """
        Release the I/O thread lock.
        """
        if self.lock:
            self.lock.release()

    def setLevel(self, level):
        """
        Set the logging level of this handler.
        """
        self.level = _checkLevel(level)

    def format(self, record):
        """
        Format the specified record.
        """
        if self.formatter:
            fmt: Formatter = self.formatter
        else:
            fmt = _defaultFormatter
        return fmt.format(record)

    def emit(self, record):
        """
        Emit the record.
        """
        raise NotImplementedError('Must be implemented by handler')

    def handle(self, record):
        """
        Conditionally emit the specified logging record.
        """
        rv: bool = self.filter(record)
        if rv:
            self.acquire()
            try:
                self.emit(record)
            finally:
                self.release()
        return rv

    def setFormatter(self, fmt):
        """
        Set the formatter for this handler.
        """
        self.formatter = fmt

    def flush(self):
        """
        Ensure all logging output has been flushed.
        """
        pass

    def close(self):
        """
        Tidy up any resources used by the handler.
        """
        _acquireLock()
        try:
            if self._name and self._name in _handlers:
                del _handlers[self._name]
        finally:
            _releaseLock()

    def handleError(self, record):
        """
        Handle errors which occur during an emit() call.
        """
        if raiseExceptions:
            raise Exception('Failed to log: {}'.format(record))
        else:
            _consoleStream.write('--- Logging Error ---\n')

    def __repr__(self):
        level: str = getLevelName(self.level)
        return '<{} ({})>'.format(self.__class__.__name__, level)


class StreamHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream.
    """
    terminator: str = '\n'

    def __init__(self, stream=None, level=NOTSET):
        """
        Initialize the handler.
        """
        super().__init__(level)
        if stream is None:
            stream = _consoleStream
        self.stream: Any = stream

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            msg: str = self.format(record)
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def __repr__(self):
        level: str = getLevelName(self.level)
        name: str = getattr(self.stream, 'name', '')
        if name:
            name += ' '
        return '<{} {}({})>'.format(self.__class__.__name__, name, level)


class FileHandler(StreamHandler):
    """ Handler Class that is suppose to write to disk - we haven't
    implemented this in transcrypt.
    """

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        """
        """
        raise NotImplementedError('No Filesystem for FileHandler')


class _StderrHandler(StreamHandler):
    """
    This class is like a StreamHandler using sys.stderr
    """

    def __init__(self, level=NOTSET):
        """
        Initialize the handler.
        """
        super().__init__(None, level)

    def _getStream(self):
        return _consoleStream
    stream: Callable[[], Any] = property(_getStream)


_defaultLastResort: _StderrHandler = _StderrHandler(WARNING)
lastResort: _StderrHandler = _defaultLastResort


class PlaceHolder(object):
    """
    PlaceHolder instances are used in the Manager logger hierarchy.
    """

    def __init__(self, alogger):
        """
        Initialize with the specified logger.
        """
        n: str = alogger.name
        self.loggerMap: Dict[str, 'Logger'] = {n: alogger}

    def append(self, alogger):
        """
        Add the specified logger as a child of this placeholder.
        """
        n: str = alogger.name
        if n not in self.loggerMap.keys():
            self.loggerMap[n] = alogger


def setLoggerClass(klass):
    """
    Set the class to be used when instantiating a logger.
    """
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError('logger not derived from logging.Logger: ' +
                klass.__name__)
    global _loggerClass
    _loggerClass = klass


def getLoggerClass():
    """
    Return the class to be used when instantiating a logger.
    """
    return _loggerClass


class Manager(object):
    """
    Manager holds the hierarchy of loggers.
    """

    def __init__(self, rootnode):
        """
        Initialize the manager with the root node.
        """
        self.root: 'RootLogger' = rootnode
        self.disable: int = 0
        self.emittedNoHandlerWarning: bool = False
        self.loggerDict: Dict[str, Union['Logger', PlaceHolder]] = {}
        self.loggerClass: Optional[type] = None
        self.logRecordFactory: Optional[Callable[..., LogRecord]] = None

    def getLogger(self, name):
        """
        Get a logger with the specified name, creating it if necessary.
        """
        rv: Optional['Logger'] = None
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
        finally:
            _releaseLock()
        assert rv is not None
        return rv

    def setLoggerClass(self, klass):
        """
        Set the class to be used when instantiating a logger with this Manager.
        """
        if klass != Logger:
            if not issubclass(klass, Logger):
                raise TypeError('logger not derived from logging.Logger: ' +
                    klass.__name__)
        self.loggerClass = klass

    def setLogRecordFactory(self, factory):
        """
        Set the factory to be used when instantiating a log record with this
        Manager.
        """
        self.logRecordFactory = factory

    def _fixupParents(self, alogger):
        """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
        name: str = alogger.name
        i: int = name.rfind('.')
        rv: Optional['Logger'] = None
        while i > 0 and not rv:
            substr: str = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = PlaceHolder(alogger)
            else:
                obj = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    assert isinstance(obj, PlaceHolder)
                    obj.append(alogger)
            i = name.rfind('.', 0, i - 1)
        if not rv:
            rv = self.root
        alogger.parent = rv

    def _fixupChildren(self, ph, alogger):
        """
        Ensure that children of the placeholder ph are connected to the
        specified logger.
        """
        name: str = alogger.name
        namelen: int = len(name)
        for c in ph.loggerMap.keys():
            log: 'Logger' = ph.loggerMap[c]
            if not log.parent.name.startswith(name):
                alogger.parent = log.parent
                log.parent = alogger


class Logger(Filterer):
    """
    Instances of the Logger class represent a single logging channel.
    """

    def __init__(self, name, level=NOTSET):
        """
        Initialize the logger with a name and an optional level.
        """
        super().__init__()
        self.name: str = name
        self.level: int = _checkLevel(level)
        self.parent: Optional['Logger'] = None
        self.propagate: bool = True
        self.handlers: List[Handler] = []
        self.disabled: bool = False

    def setLevel(self, level):
        """
        Set the logging level of this logger.
        """
        self.level = _checkLevel(level)
    __pragma__('kwargs')

    def debug(self, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with severity 'DEBUG'.
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with severity 'INFO'.
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with severity 'WARNING'.
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def warn(self, msg, *args: Any, **kwargs: Any):
        warnings.warn_explicit(
            'The `warn` method is deprecated - use `warning`',
            DeprecationWarning, 'logging/__init__.py', 1388, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with severity 'ERROR'.
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg, *args: Any, exc_info: bool=True, **kwargs: Any):
        """
        Convenience method for logging an ERROR with exception information.
        """
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with severity 'CRITICAL'.
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)
        fatal = critical

    def log(self, level, msg, *args: Any, **kwargs: Any):
        """
        Log 'msg.format(args)' with the integer severity 'level'.
        """
        if not isinstance(level, int):
            if raiseExceptions:
                raise TypeError('level must be an integer')
            else:
                return
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def findCaller(self, stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        return '(unknown file)', 0, '(unknown function)', None

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=
        None, extra=None, sinfo=None):
        """
        A factory method to create a LogRecord.
        """
        rv: LogRecord = _logRecordFactory(name, level, fn, lno, msg, args,
            exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in ['message', 'asctime'] or key in rv.__dict__:
                    raise KeyError('Attempt to overwrite %r in LogRecord' % key
                        )
                rv.__dict__[key] = extra[key]
        return rv

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=
        False):
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        sinfo: Optional[str] = None
        if _srcfile:
            try:
                fn, lno, func, sinfo = self.findCaller(stack_info)
            except ValueError:
                fn, lno, func = '(unknown file)', 0, '(unknown function)'
        else:
            fn, lno, func = '(unknown file)', 0, '(unknown function)'
        record: LogRecord = self.makeRecord(self.name, level, fn, lno, msg,
            args, exc_info, func, extra, sinfo)
        self.handle(record)

    def handle(self, record):
        """
        Call the handlers for the specified record.
        """
        if not self.disabled and self.filter(record):
            self.callHandlers(record)

    def addHandler(self, hdlr):
        """
        Add the specified handler to this logger.
        """
        _acquireLock()
        try:
            if hdlr not in self.handlers:
                self.handlers.append(hdlr)
        finally:
            _releaseLock()

    def removeHandler(self, hdlr):
        """
        Remove the specified handler from this logger.
        """
        _acquireLock()
        try:
            if hdlr in self.handlers:
                self.handlers.remove(hdlr)
        finally:
            _releaseLock()

    def hasHandlers(self):
        """
        See if this logger has any handlers configured.
        """
        c: Optional['Logger'] = self
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

    def callHandlers(self, record):
        """
        Pass a record to all relevant handlers.
        """
        c: Optional['Logger'] = self
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
            elif raiseExceptions and not self.manager.emittedNoHandlerWarning:
                _consoleStream.write(
                    'No handlers could be found for logger "{}"'.format(
                    self.name))
                self.manager.emittedNoHandlerWarning = True

    def getEffectiveLevel(self):
        """
        Get the effective level for this logger.
        """
        logger: Optional['Logger'] = self
        while logger:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        if self.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getChild(self, suffix):
        """
        Get a logger which is a descendant to this one.
        """
        if self.root is not self:
            suffix = '.'.join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def __repr__(self):
        level: str = getLevelName(self.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, self.name, level)


class RootLogger(Logger):
    """
    A root logger.
    """

    def __init__(self, level):
        """
        Initialize the logger with the name "root".
        """
        super().__init__('root', level)


_loggerClass: type = Logger


class LoggerAdapter(object):
    """
    An adapter for loggers which makes it easier to specify contextual
    information in logging output.
    """

    def __init__(self, logger, extra):
        """
        Initialize the adapter with a logger and a dict-like object.
        """
        self.logger: Logger = logger
        self.extra: Dict[str, Any] = extra

    def process(self, msg, kwargs):
        """
        Process the logging message and keyword arguments.
        """
        kwargs['extra'] = self.extra
        return msg, kwargs
    __pragma__('kwargs')

    def debug(self, msg, *args: Any, **kwargs: Any):
        """
        Delegate a debug call to the underlying logger.
        """
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args: Any, **kwargs: Any):
        """
        Delegate an info call to the underlying logger.
        """
        self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg, *args: Any, **kwargs: Any):
        """
        Delegate a warning call to the underlying logger.
        """
        self.log(WARNING, msg, *args, **kwargs)

    def warn(self, msg, *args: Any, **kwargs: Any):
        warnings.warn_explicit(
            'The `warn` method is deprecated - use `warning`',
            DeprecationWarning, 'logging/__init__.py', 1719, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args: Any, **kwargs: Any):
        """
        Delegate an error call to the underlying logger.
        """
        self.log(ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args: Any, exc_info: bool=True, **kwargs: Any):
        """
        Delegate an exception call to the underlying logger.
        """
        self.log(ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args: Any, **kwargs: Any):
        """
        Delegate a critical call to the underlying logger.
        """
        self.log(CRITICAL, msg, *args, **kwargs)

    def log(self, level, msg, *args: Any, **kwargs: Any):
        """
        Delegate a log call to the underlying logger, with contextual information.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        if self.logger.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def setLevel(self, level):
        """
        Set the specified level on the underlying logger.
        """
        self.logger.setLevel(level)

    def getEffectiveLevel(self):
        """
        Get the effective level for the underlying logger.
        """
        return self.logger.getEffectiveLevel()

    def hasHandlers(self):
        """
        See if the underlying logger has any handlers.
        """
        return self.logger.hasHandlers()

    def __repr__(self):
        logger: Logger = self.logger
        level: str = getLevelName(logger.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, logger.name,
            level)


root: RootLogger = RootLogger(WARNING)
Logger.root = root
Logger.manager = Manager(Logger.root)
root.manager = Logger.manager


def _resetLogging():
    """ This is a utility method to help with testing so that
    we can start from a clean slate.
    """
    global _handlerList, _handlers, root, _loggerClass
    _handlerList.clear()
    _handlers.clear()
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)
    root.manager = Logger.manager


__pragma__('kwargs')


def basicConfig(**kwargs: Any):
    """
    Do basic configuration for the logging system.
    """
    _acquireLock()
    try:
        if len(root.handlers) == 0:
            handlers_arg: Optional[List[Handler]] = kwargs.pop('handlers', None
                )
            if handlers_arg is not None:
                if 'stream' in kwargs:
                    raise ValueError(
                        "'stream' should not be specified together with 'handlers'"
                        )
            if handlers_arg is None:
                stream_arg: Optional[Any] = kwargs.pop('stream', None)
                h: Handler = StreamHandler(stream_arg)
                handlers_arg = [h]
            dfs: Optional[str] = kwargs.pop('datefmt', None)
            style: str = kwargs.pop('style', '{')
            if style not in _STYLES:
                raise ValueError('Style must be one of: {}'.format(','.join
                    (_STYLES.keys())))
            fs: str = kwargs.pop('format', _STYLES[style][1])
            fmt: Formatter = Formatter(fs, dfs, style)
            for h in handlers_arg:
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


def getLogger(name=None):
    """
    Return a logger with the specified name, creating it if necessary.
    """
    if name:
        return Logger.manager.getLogger(name)
    else:
        return root


__pragma__('kwargs')


def critical(msg, *args: Any, **kwargs: Any):
    """
    Log a message with severity 'CRITICAL' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.critical(msg, *args, **kwargs)


fatal: Callable[..., None] = critical


def error(msg, *args: Any, **kwargs: Any):
    """
    Log a message with severity 'ERROR' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.error(msg, *args, **kwargs)


def exception(msg, *args: Any, exc_info: bool=True, **kwargs: Any):
    """
    Log a message with severity 'ERROR' on the root logger, with exception information.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg, *args: Any, **kwargs: Any):
    """
    Log a message with severity 'WARNING' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.warning(msg, *args, **kwargs)


def warn(msg, *args: Any, **kwargs: Any):
    warnings.warn_explicit('The `warn` method is deprecated - use `warning`',
        DeprecationWarning, 'logging/__init__.py', 1944, 'logging')
    warning(msg, *args, **kwargs)


def info(msg, *args: Any, **kwargs: Any):
    """
    Log a message with severity 'INFO' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.info(msg, *args, **kwargs)


def debug(msg, *args: Any, **kwargs: Any):
    """
    Log a message with severity 'DEBUG' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.debug(msg, *args, **kwargs)


def log(level, msg, *args: Any, **kwargs: Any):
    """
    Log 'msg.format(args)' with the integer severity 'level' on the root logger.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.log(level, msg, *args, **kwargs)


__pragma__('nokwargs')


def disable(level):
    """
    Disable all logging calls of severity 'level' and below.
    """
    root.manager.disable = level


def shutdown(handlerList=_handlerList):
    """
    Perform any cleanup actions in the logging system.
    """
    for wr in reversed(handlerList[:]):
        try:
            h = wr
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
    """
    This handler does nothing.
    """

    def handle(self, record):
        """Stub."""

    def emit(self, record):
        """Stub."""

    def createLock(self):
        self.lock = None


_warnings_showwarning: Optional[Callable[..., None]] = None


def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to logging.
    """
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(message, category, filename, lineno, file,
                line)
    else:
        s: str = warnings.formatwarning(message, category, filename, lineno,
            line)
        logger: Logger = getLogger('py.warnings')
        if not logger.handlers:
            logger.addHandler(NullHandler())
        logger.warning(s)


def captureWarnings(capture):
    """
    Redirect warnings to the logging package.
    """
    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.setShowWarning(_showwarning)
    elif _warnings_showwarning is not None:
        warnings.setShowWarnings(_warnings_showwarning)
        _warnings_showwarning = None
