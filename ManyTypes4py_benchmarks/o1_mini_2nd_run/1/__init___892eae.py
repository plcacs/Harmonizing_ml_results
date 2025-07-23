"""
Logging package for Python. Based on PEP 282 and comments thereto in
comp.lang.python.

Copyright (C) 2001-2016 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!

Edited By Carl Allendorph 2016 for the Transcrypt Project

This code base was originally pulled from the Python project to
maintain as much compatibility in the interfaces as possible.

Limitations:
- The output formatting is currently limited to string.format() style
    formatting because of Transcrypt. This means that PercentStyle
    formatting and string template formatting is disabled and not
    available.
- I've had to makes some hacks to work around shortcomings in Transcrypt
    at this point but most of these are superficial.
- The components of the logging dealing with exceptions, stack traces,
    and other content are not implemented at this point
- Anything related to threads and processes is set to default values
    of None.
- StreamHandler publishes content to console.log as sys.stderr is not
    available.
- Integration with the `warnings` python module is not implemented because
    `warnings` does not exist yet.

Automated tests are available in the logging test module.

"""
from org.transcrypt.stubs.browser import __pragma__
import time
import warnings
import collections.abc
from typing import Any, Callable, Dict, List, Optional, Union
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
_levelToName: Dict[int, str] = {CRITICAL: 'CRITICAL', ERROR: 'ERROR', WARNING: 'WARNING', INFO: 'INFO', DEBUG: 'DEBUG', NOTSET: 'NOTSET'}
_nameToLevel: Dict[str, int] = {'CRITICAL': CRITICAL, 'FATAL': FATAL, 'ERROR': ERROR, 'WARN': WARNING, 'WARNING': WARNING, 'INFO': INFO, 'DEBUG': DEBUG, 'NOTSET': NOTSET}

def getLevelName(level: Union[int, str]) -> str:
    """
    Return the textual representation of logging level 'level'.

    If the level is one of the predefined levels (CRITICAL, ERROR, WARNING,
    INFO, DEBUG) then you get the corresponding string. If you have
    associated levels with names using addLevelName then the name you have
    associated with 'level' is returned.

    If a numeric value corresponding to one of the defined levels is passed
    in, the corresponding string representation is returned.

    Otherwise, the string "Level %s" % level is returned.
    """
    return _levelToName.get(level) or _nameToLevel.get(level) or 'Level {}'.format(level)

def addLevelName(level: int, levelName: str) -> None:
    """
    Associate 'levelName' with 'level'.

    This is used when converting levels to text during message formatting.
    """
    _acquireLock()
    try:
        _levelToName[level] = levelName
        _nameToLevel[levelName] = level
    except Exception as exc:
        raise exc
    finally:
        _releaseLock()

def currentframe() -> Optional[Any]:
    return None
_srcfile: Optional[Any] = None

def _checkLevel(level: Union[int, str]) -> int:
    if isinstance(level, int):
        rv = level
    elif isinstance(level, str):
        if level not in _nameToLevel:
            raise ValueError('Unknown level: {}'.format(level))
        rv = _nameToLevel[level]
    else:
        raise TypeError('Level not an integer or a valid string: {}'.format(level))
    return rv
_lock: Optional[Any] = None

def _acquireLock() -> None:
    """
    Acquire the module-level lock for serializing access to shared data.

    This should be released with _releaseLock().
    """
    if _lock:
        _lock.acquire()

def _releaseLock() -> None:
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()

class LogRecord(object):
    """
    A LogRecord instance represents an event being logged.

    LogRecord instances are created every time something is logged. They
    contain all the information pertinent to the event being logged. The
    main information passed in is in msg and args, which are combined
    using str(msg) % args to create the message field of the record. The
    record also includes information such as when the record was created,
    the source line where the logging call was made, and any exception
    information to be logged.
    """

    def __init__(self, name: str, level: int, pathname: str, lineno: int, msg: str, args: Union[tuple, list], exc_info: Optional[Any], func: Optional[str] = None, sinfo: Optional[Any] = None, **kwargs: Any) -> None:
        """
        Initialize a logging record with interesting information.
        """
        ct: float = time.time()
        self.name: str = name
        self.msg: str = msg
        if args and len(args) == 1 and isinstance(args[0], collections.abc.Mapping) and args[0]:
            if raiseExceptions:
                raise NotImplementedError('No Dict Args to Log Record')
        self.args: Union[tuple, list] = args
        self.levelname: str = getLevelName(level)
        self.levelno: int = level
        self.pathname: str = pathname
        self.filename: str = pathname
        self.module: str = 'Unknown module'
        self.exc_info: Optional[Any] = exc_info
        self.exc_text: Optional[str] = None
        self.stack_info: Optional[Any] = sinfo
        self.lineno: int = lineno
        self.funcName: Optional[str] = func
        self.created: float = ct
        self.msecs: float = (ct - int(ct)) * 1000
        self.relativeCreated: float = (self.created - _startTime) * 1000
        self.thread: Optional[Any] = None
        self.threadName: Optional[Any] = None
        self.processName: Optional[Any] = None
        self.process: Optional[Any] = None

    def getMessage(self) -> str:
        """
        Return the message for this LogRecord.

        Return the message for this LogRecord after merging any
        user-supplied arguments with the message.
        """
        msg: str = str(self.msg)
        if self.args:
            msg = msg.format(*self.args)
        return msg

    def toDict(self) -> Dict[str, Any]:
        """ Utility method to convert the LogRecord object into a
        an object that can be passed to the str.format function.
        This is needed to allow for named string format arguments.
        @note - if you create a new LogRecord object type, then
           you will likely want to override this method to add
           more keys to the returned dict
        """
        keysToPick: List[str] = ['name', 'msg', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno', 'funcName', 'created', 'asctime', 'msecs', 'relativeCreated', 'thread', 'threadName', 'process']
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
_logRecordFactory: Callable[..., 'LogRecord'] = LogRecord

def setLogRecordFactory(factory: Callable[..., 'LogRecord']) -> None:
    """
    Set the factory to be used when instantiating a log record.

    :param factory: A callable which will be called to instantiate
    a log record.
    """
    global _logRecordFactory
    _logRecordFactory = factory

def getLogRecordFactory() -> Callable[..., 'LogRecord']:
    """
    Return the factory to be used when instantiating a log record.
    """
    return _logRecordFactory

def makeLogRecord(dict_: Dict[str, Any]) -> LogRecord:
    """
    Make a LogRecord whose attributes are defined by the specified dictionary,
    This function is useful for converting a logging event received over
    a socket connection (which is sent as a dictionary) into a LogRecord
    instance.
    """
    rv: LogRecord = _logRecordFactory(None, None, '', 0, '', (), None, None)
    rv.__dict__.update(dict_)
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
        self._fmt: str = fmt or self.default_format
        self._tpl: Any = Template(self._fmt)

    def usesTime(self) -> bool:
        fmt: str = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_format) >= 0

    __pragma__('kwargs')

    def format(self, record: LogRecord) -> str:
        return self._tpl.substitute(**record.__dict__)
    __pragma__('nokwargs')
BASIC_FORMAT: str = '{levelname}:{name}:{message}'
_STYLES: Dict[str, tuple] = {'{': (StrFormatStyle, BASIC_FORMAT)}

class Formatter(object):
    """
    Formatter instances are used to convert a LogRecord to text.

    Formatters need to know how a LogRecord is constructed. They are
    responsible for converting a LogRecord to (usually) a string which can
    be interpreted by either a human or an external system. The base Formatter
    allows a formatting string to be specified. If none is supplied, the
    default value of "%s(message)" is used.

    The Formatter can be initialized with a format string which makes use of
    knowledge of the LogRecord attributes - e.g. the default value mentioned
    above makes use of the fact that the user's message and arguments are pre-
    formatted into a LogRecord's message attribute. Currently, the useful
    attributes in a LogRecord are described by:

    %(name)s                        Name of the logger (logging channel)
    %(levelno)s                 Numeric logging level for the message (DEBUG, INFO,
                                                WARNING, ERROR, CRITICAL)
    %(levelname)s               Text logging level for the message ("DEBUG", "INFO",
                                                "WARNING", "ERROR", "CRITICAL")
    %(pathname)s                Full pathname of the source file where the logging
                                                call was issued (if available)
    %(filename)s                Filename portion of pathname
    %(module)s                  Module (name portion of filename)
    %(lineno)d                  Source line number where the logging call was issued
                                                (if available)
    %(funcName)s                Function name
    %(created)f                 Time when the LogRecord was created (time.time()
                                                return value)
    %(asctime)s                 Textual time when the LogRecord was created
    %(msecs)d                       Millisecond portion of the creation time
    %(relativeCreated)d Time in milliseconds when the LogRecord was created,
                                                relative to the time the logging module was loaded
                                                (typically at application startup time)
    %(thread)d                  Thread ID (if available)
    %(threadName)s          Thread name (if available)
    %(process)d                 Process ID (if available)
    %(message)s                 The result of record.getMessage(), computed just as
                                                the record is emitted
    """
    converter: Callable[[float], Any] = time.localtime

    __pragma__('kwargs')

    def __init__(self, format: Optional[str] = None, datefmt: Optional[str] = None, style: str = '{') -> None:
        """
        Initialize the formatter with specified format strings.

        Initialize the formatter either with the specified format string, or a
        default as described above. Allow for specialized date formatting with
        the optional datefmt argument (if omitted, you get the ISO8601 format).

        Use a style parameter of '%', '{' or '$' to specify that you want to
        use one of %-formatting, :meth:`str.format` (``{}``) formatting or
        :class:`string.Template` formatting in your format string.

        .. versionchanged:: 3.2
        Added the ``style`` parameter.
        """
        if style != '{':
            raise NotImplementedError('{} format only')
        self._style: PercentStyle = _STYLES[style][0](format)
        self._fmt: str = self._style._fmt
        self.datefmt: Optional[str] = datefmt

    __pragma__('nokwargs')
    default_time_format: str = '%Y-%m-%d %H:%M:%S'
    default_msec_format: str = '{},{:03d}'

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        """
        Return the creation time of the specified LogRecord as formatted text.

        This method should be called from format() by a formatter which
        wants to make use of a formatted time. This method can be overridden
        in formatters to provide for any specific requirement, but the
        basic behaviour is as follows: if datefmt (a string) is specified,
        it is used with time.strftime() to format the creation time of the
        record. Otherwise, the ISO8601 format is used. The resulting
        string is returned. This function uses a user-configurable function
        to convert the creation time to a tuple. By default, time.localtime()
        is used; to change this for a particular formatter instance, set the
        'converter' attribute to a function with the same signature as
        time.localtime() or time.gmtime(). To change it for all formatters,
        for example if you want all logging times to be shown in GMT,
        set the 'converter' attribute in the Formatter class.
        """
        ct: Any = self.converter(record.created)
        if datefmt:
            s: str = time.strftime(datefmt, ct)
        else:
            t: str = time.strftime(self.default_time_format, ct)
            s: str = self.default_msec_format.format(t, int(record.msecs))
        return s

    def formatException(self, ei: Any) -> str:
        """
        Format and return the specified exception information as a string.

        This default implementation just uses
        traceback.print_exception()
        """
        return str(ei)

    def usesTime(self) -> bool:
        """
        Check if the format uses the creation time of the record.
        """
        return self._style.usesTime()

    def formatMessage(self, record: LogRecord) -> str:
        return self._style.format(record)

    def formatStack(self, stack_info: Any) -> str:
        """
        This method is provided as an extension point for specialized
        formatting of stack information.

        The input data is a string as returned from a call to
        :func:`traceback.print_stack`, but with the last trailing newline
        removed.

        The base implementation just returns the value passed in.
        """
        return stack_info

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record as text.

        The record's attribute dictionary is used as the operand to a
        string formatting operation which yields the returned string.
        Before formatting the dictionary, a couple of preparatory steps
        are carried out. The message attribute of the record is computed
        using LogRecord.getMessage(). If the formatting string uses the
        time (as determined by a call to usesTime(), formatTime() is
        called to format the event time. If there is exception information,
        it is formatted using formatException() and appended to the message.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s: str = self.formatMessage(record)
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
_defaultFormatter: Formatter = Formatter()

class BufferingFormatter(object):
    """
    A formatter suitable for formatting a number of records.
    """

    def __init__(self, linefmt: Optional[Formatter] = None) -> None:
        """
        Optionally specify a formatter which will be used to format each
        individual record.
        """
        if linefmt:
            self.linefmt: Formatter = linefmt
        else:
            self.linefmt: Formatter = _defaultFormatter

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

    Loggers and Handlers can optionally use Filter instances to filter
    records as desired. The base filter class only allows events which are
    below a certain point in the logger hierarchy. For example, a filter
    initialized with "A.B" will allow events logged by loggers "A.B",
    "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
    initialized with the empty string, all events are passed.
    """

    def __init__(self, name: str = '') -> None:
        """
        Initialize a filter.

        Initialize with the name of the logger which, together with its
        children, will have its events allowed through the filter. If no
        name is specified, allow every event.
        """
        self.name: str = name
        self.nlen: int = len(name)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if the specified record is to be logged.

        Is the specified record to be logged? Returns 0 for no, nonzero for
        yes. If deemed appropriate, the record may be modified in-place.
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

    def __init__(self) -> None:
        """
        Initialize the list of filters to be an empty list.
        """
        self.filters: List[Union[Filter, Callable[[LogRecord], bool]]] = []

    def addFilter(self, filt: Union[Filter, Callable[[LogRecord], bool]]) -> None:
        """
        Add the specified filter to this handler.
        """
        if filt not in self.filters:
            self.filters.append(filt)

    def removeFilter(self, filt: Union[Filter, Callable[[LogRecord], bool]]) -> None:
        """
        Remove the specified filter from this handler.
        """
        if filt in self.filters:
            self.filters.remove(filt)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if a record is loggable by consulting all the filters.

        The default is to allow the record to be logged; any filter can veto
        this and the record is then dropped. Returns a zero value if a record
        is to be dropped, else non-zero.

        .. versionchanged:: 3.2

        Allow filters to be just callables.
        """
        rv: bool = True
        for f in self.filters:
            if hasattr(f, 'filter'):
                result: bool = f.filter(record)
            else:
                result: bool = f(record)
            if not result:
                rv = False
                break
        return rv

class ConsoleLogStream(object):
    """ This implements a quasi "Stream" like object for mimicing
    the sys.stderr object.
    """

    def __init__(self) -> None:
        self.name: str = 'console'

    def write(self, msg: str) -> None:
        """
        """
        msg = msg.rstrip('\n\r')
        if len(msg) > 0:
            console.log(msg)
_consoleStream: ConsoleLogStream = ConsoleLogStream()
_handlers: Dict[str, 'Handler'] = {}
_handlerList: List[Callable[[], Optional['Handler']]] = []

def _removeHandlerRef(wr: Callable[[], Optional['Handler']]) -> None:
    """
    Remove a handler reference from the internal cleanup list.
    """
    acquire: Optional[Callable[[], None]] = _acquireLock
    release: Optional[Callable[[], None]] = _releaseLock
    handlers: List[Callable[[], Optional['Handler']]] = _handlerList
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

    The base handler class. Acts as a placeholder which defines the Handler
    interface. Handlers can optionally use Formatter instances to format
    records as desired. By default, no formatter is specified; in this case,
    the 'raw' message as determined by record.message is logged.
    """

    def __init__(self, level: int = NOTSET) -> None:
        """
        Initializes the instance - basically setting the formatter to None
        and the filter list to empty.
        """
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
    name: Optional[str] = property(get_name, set_name)

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

        If a formatter is set, use it. Otherwise, use the default formatter
        for the module.
        """
        if self.formatter:
            fmt: Formatter = self.formatter
        else:
            fmt: Formatter = _defaultFormatter
        return fmt.format(record)

    def emit(self, record: LogRecord) -> None:
        """
        Do whatever it takes to actually log the specified logging record.

        This version is intended to be implemented by subclasses and so
        raises a NotImplementedError.
        """
        raise NotImplementedError('Must be implemented by handler')

    def handle(self, record: LogRecord) -> bool:
        """
        Conditionally emit the specified logging record.

        Emission depends on filters which may have been added to the handler.
        Wrap the actual emission of the record with acquisition/release of
        the I/O thread lock. Returns whether the filter passed the record for
        emission.
        """
        rv: bool = self.filter(record)
        if rv:
            self.acquire()
            try:
                self.emit(record)
            finally:
                self.release()
        return rv

    def setFormatter(self, fmt: Formatter) -> None:
        """
        Set the formatter for this handler.
        """
        self.formatter = fmt

    def flush(self) -> None:
        """
        Ensure all logging output has been flushed.

        This version does nothing and is intended to be implemented by
        subclasses.
        """
        pass

    def close(self) -> None:
        """
        Tidy up any resources used by the handler.

        This version removes the handler from an internal map of handlers,
        _handlers, which is used for handler lookup by name. Subclasses
        should ensure that this gets called from overridden close()
        methods.
        """
        _acquireLock()
        try:
            if self._name and self._name in _handlers:
                del _handlers[self._name]
        finally:
            _releaseLock()

    def handleError(self, record: LogRecord) -> None:
        """
        Handle errors which occur during an emit() call.

        This method should be called from handlers when an exception is
        encountered during an emit() call. If raiseExceptions is false,
        exceptions get silently ignored. This is what is mostly wanted
        for a logging system - most users will not care about errors in
        the logging system, they are more interested in application errors.
        You could, however, replace this with a custom handler if you wish.
        The record which was being processed is passed in to this method.
        """
        if raiseExceptions:
            raise Exception('Failed to log: {}'.format(record))
        else:
            _consoleStream.write('--- Logging Error ---\n')

    def __repr__(self) -> str:
        level: str = getLevelName(self.level)
        return '<{} ({})>'.format(self.__class__.__name__, level)

class StreamHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """
    terminator: str = '\n'

    def __init__(self, stream: Optional[Any] = None, level: int = NOTSET) -> None:
        """
        Initialize the handler.

        If stream is not specified, sys.stderr is used.
        """
        super().__init__(level)
        if stream is None:
            stream = _consoleStream
        self.stream: Any = stream

    def flush(self) -> None:
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: LogRecord) -> None:
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.    If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.    If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg: str = self.format(record)
            stream: Any = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def __repr__(self) -> str:
        level: str = getLevelName(self.level)
        name: str = getattr(self.stream, 'name', '')
        if name:
            name += ' '
        return '<{} {}({})>'.format(self.__class__.__name__, name, level)

class FileHandler(StreamHandler):
    """ Handler Class that is suppose to write to disk - we haven't
    implemented this in transcrypt.
    """

    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False) -> None:
        """
        """
        raise NotImplementedError('No Filesystem for FileHandler')

class _StderrHandler(StreamHandler):
    """
    This class is like a StreamHandler using sys.stderr, but always uses
    whatever sys.stderr is currently set to rather than the value of
    sys.stderr at handler construction time.
    """

    def __init__(self, level: int = NOTSET) -> None:
        """
        Initialize the handler.
        """
        super().__init__(None, level)

    def _getStream(self) -> Any:
        return _consoleStream
    stream: Callable[[], Any] = property(_getStream)
_defaultLastResort: _StderrHandler = _StderrHandler(WARNING)
lastResort: _StderrHandler = _defaultLastResort

class PlaceHolder(object):
    """
    PlaceHolder instances are used in the Manager logger hierarchy to take
    the place of nodes for which no loggers have been defined. This class is
    intended for internal use only and not as part of the public API.
    """

    def __init__(self, alogger: 'Logger') -> None:
        """
        Initialize with the specified logger being a child of this placeholder.
        """
        n: str = alogger.name
        self.loggerMap: Dict[str, 'Logger'] = {n: alogger}

    def append(self, alogger: 'Logger') -> None:
        """
        Add the specified logger as a child of this placeholder.
        """
        n: str = alogger.name
        if n not in self.loggerMap.keys():
            self.loggerMap[n] = alogger

def setLoggerClass(klass: type) -> None:
    """
    Set the class to be used when instantiating a logger. The class should
    define __init__() such that only a name argument is required, and the
    __init__() should call Logger.__init__()
    """
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
    global _loggerClass
    _loggerClass = klass

def getLoggerClass() -> type:
    """
    Return the class to be used when instantiating a logger.
    """
    return _loggerClass

class Manager(object):
    """
    There is [under normal circumstances] just one Manager instance, which
    holds the hierarchy of loggers.
    """

    def __init__(self, rootnode: 'RootLogger') -> None:
        """
        Initialize the manager with the root node of the logger hierarchy.
        """
        self.root: 'RootLogger' = rootnode
        self.disable: int = 0
        self.emittedNoHandlerWarning: bool = False
        self.loggerDict: Dict[str, Union['Logger', PlaceHolder]] = {}
        self.loggerClass: Optional[type] = None
        self.logRecordFactory: Optional[Callable[..., LogRecord]] = None

    def getLogger(self, name: str) -> 'Logger':
        """
        Get a logger with the specified name (channel name), creating it
        if it doesn't yet exist. This name is a dot-separated hierarchical
        name, such as "a", "a.b", "a.b.c" or similar.

        If a PlaceHolder existed for the specified name [i.e. the logger
        didn't exist but a child of it did], replace it with the created
        logger and fix up the parent/child references which pointed to the
        placeholder to now point to the logger.
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

    def setLoggerClass(self, klass: type) -> None:
        """
        Set the class to be used when instantiating a logger with this Manager.
        """
        if klass != Logger:
            if not issubclass(klass, Logger):
                raise TypeError('logger not derived from logging.Logger: ' + klass.__name__)
        self.loggerClass = klass

    def setLogRecordFactory(self, factory: Callable[..., LogRecord]) -> None:
        """
        Set the factory to be used when instantiating a log record with this
        Manager.
        """
        self.logRecordFactory = factory

    def _fixupParents(self, alogger: 'Logger') -> None:
        """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
        name: str = alogger.name
        i: int = name.rfind('.')
        rv: Optional['Logger'] = None
        while i > 0 and (not rv):
            substr: str = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = PlaceHolder(alogger)
            else:
                obj: Union['Logger', PlaceHolder] = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    assert isinstance(obj, PlaceHolder)
                    obj.append(alogger)
            i = name.rfind('.', 0, i - 1)
        if not rv:
            rv = self.root
        alogger.parent = rv

    def _fixupChildren(self, ph: PlaceHolder, alogger: 'Logger') -> None:
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
    Instances of the Logger class represent a single logging channel. A
    "logging channel" indicates an area of an application. A
    "logging channel" indicates an area of an application.
    An "area" can be anything the user wishes, typically a module or a
    class.
    """

    def __init__(self, name: str, level: int = NOTSET) -> None:
        """
        Initialize the logger with a name and an optional level.
        """
        super().__init__()
        self.name: str = name
        self.level: int = _checkLevel(level)
        self.parent: Optional['Logger'] = None
        self.propagate: bool = True
        self.handlers: List['Handler'] = []
        self.disabled: bool = False

    def setLevel(self, level: Union[int, str]) -> None:
        """
        Set the logging level of this logger.    level must be an int or a str.
        """
        self.level = _checkLevel(level)

    __pragma__('kwargs')

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a {}", "thorny problem", exc_info=1)
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a {}", "interesting problem", exc_info=1)
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a {}", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1388, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a {}", "major problem", exc_info=1)
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        """
        Convenience method for logging an ERROR with exception information.
        """
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a {}", "major disaster", exc_info=1)
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)
        fatal: Callable[[str, Any, Any], None] = self.critical

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'msg.format(args)' with the integer severity 'level'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.log(level, "We have a {}", "mysterious problem", exc_info=1)
        """
        if not isinstance(level, int):
            if raiseExceptions:
                raise TypeError('level must be an integer')
            else:
                return
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def findCaller(self, stack_info: bool = False) -> tuple:
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = currentframe()
        rv: tuple = ('(unknown file)', 0, '(unknown function)', None)
        return rv

    def makeRecord(self, name: str, level: int, fn: str, lno: int, msg: str, args: Union[tuple, list], exc_info: Optional[Any], func: Optional[str] = None, extra: Optional[Dict[str, Any]] = None, sinfo: Optional[Any] = None) -> LogRecord:
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        rv: LogRecord = _logRecordFactory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in ['message', 'asctime'] or key in rv.__dict__:
                    raise KeyError('Attempt to overwrite %r in LogRecord' % key)
                rv.__dict__[key] = extra[key]
        return rv

    def _log(self, level: int, msg: str, args: Union[tuple, list], exc_info: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None, stack_info: bool = False) -> None:
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
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
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if not self.disabled and self.filter(record):
            self.callHandlers(record)

    def addHandler(self, hdlr: 'Handler') -> None:
        """
        Add the specified handler to this logger.
        """
        _acquireLock()
        try:
            if hdlr not in self.handlers:
                self.handlers.append(hdlr)
        finally:
            _releaseLock()

    def removeHandler(self, hdlr: 'Handler') -> None:
        """
        Remove the specified handler from this logger.
        """
        _acquireLock()
        try:
            if hdlr in self.handlers:
                self.handlers.remove(hdlr)
        finally:
            _releaseLock()

    def hasHandlers(self) -> bool:
        """
        See if this logger has any handlers configured.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. Return True if a handler was found, else False.
        Stop searching up the hierarchy whenever a logger with the "propagate"
        attribute set to zero is found - that will be the last logger which
        is checked for the existence of handlers.
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

    def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. If no handler was found, output a one-off error
        message to sys.stderr. Stop searching up the hierarchy whenever a
        logger with the "propagate" attribute set to zero is found - that
        will be the last logger whose handlers are called.
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
            elif raiseExceptions and (not self.manager.emittedNoHandlerWarning):
                _consoleStream.write('No handlers could be found for logger "{}"'.format(self.name))
                self.manager.emittedNoHandlerWarning = True

    def getEffectiveLevel(self) -> int:
        """
        Get the effective level for this logger.

        Loop through this logger and its parents in the logger hierarchy,
        looking for a non-zero logging level. Return the first one found.
        """
        logger: Optional['Logger'] = self
        while logger:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level: int) -> bool:
        """
        Is this logger enabled for level 'level'?
        """
        if self.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getChild(self, suffix: str) -> 'Logger':
        """
        Get a logger which is a descendant to this one.

        This is a convenience method, such that

        logging.getLogger('abc').getChild('def.ghi')

        is the same as

        logging.getLogger('abc.def.ghi')

        It's useful, for example, when the parent logger is named using
        __name__ rather than a literal string.
        """
        if self.root is not self:
            suffix = '.'.join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def __repr__(self) -> str:
        level: str = getLevelName(self.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, self.name, level)

class RootLogger(Logger):
    """
    A root logger is not that different to any other logger, except that
    it must have a logging level and there is only one instance of it in
    the hierarchy.
    """

    def __init__(self, level: int) -> None:
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

    def __init__(self, logger: 'Logger', extra: Dict[str, Any]) -> None:
        """
        Initialize the adapter with a logger and a dict-like object which
        provides contextual information. This constructor signature allows
        easy stacking of LoggerAdapters, if so desired.

        You can effectively pass keyword arguments as shown in the
        following example:

        adapter = LoggerAdapter(someLogger, dict(p1=v1, p2="v2"))
        """
        self.logger: 'Logger' = logger
        self.extra: Dict[str, Any] = extra

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the logging message and keyword arguments passed in to
        a logging call to insert contextual information. You can either
        manipulate the message itself, the keyword args or both. Return
        the message and kwargs modified (or not) to suit your needs.

        Normally, you'll only need to override this one method in a
        LoggerAdapter subclass for your specific needs.
        """
        kwargs['extra'] = self.extra
        return (msg, kwargs)
    __pragma__('kwargs')

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate a debug call to the underlying logger.
        """
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate an info call to the underlying logger.
        """
        self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate a warning call to the underlying logger.
        """
        self.log(WARNING, msg, *args, **kwargs)

    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1719, 'logging')
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate an error call to the underlying logger.
        """
        self.log(ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        """
        Delegate an exception call to the underlying logger.
        """
        self.log(ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate a critical call to the underlying logger.
        """
        self.log(CRITICAL, msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate a log call to the underlying logger, after adding
        contextual information from this adapter instance.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, msg, args, **kwargs)
    __pragma__('nokwargs')

    def isEnabledFor(self, level: int) -> bool:
        """
        Is this logger enabled for level 'level'?
        """
        if self.logger.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def setLevel(self, level: Union[int, str]) -> None:
        """
        Set the specified level on the underlying logger.
        """
        self.logger.setLevel(level)

    def getEffectiveLevel(self) -> int:
        """
        Get the effective level for the underlying logger.
        """
        return self.logger.getEffectiveLevel()

    def hasHandlers(self) -> bool:
        """
        See if the underlying logger has any handlers.
        """
        return self.logger.hasHandlers()

    def __repr__(self) -> str:
        logger: 'Logger' = self.logger
        level: str = getLevelName(logger.getEffectiveLevel())
        return '<{} {} ({})>'.format(self.__class__.__name__, logger.name, level)
root: 'RootLogger' = RootLogger(WARNING)
Logger.root = root
Logger.manager = Manager(Logger.root)
root.manager = Logger.manager

def _resetLogging() -> None:
    """ This is a utility method to help with testing so that
    we can start from a clean slate.
    """
    global _handlerList, _handlers, root, _loggerClass
    _handlerList = []
    _handlers = {}
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)
    root.manager = Logger.manager

__pragma__('kwargs')

def basicConfig(**kwargs: Any) -> None:
    """
    Do basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured. It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a StreamHandler which writes to
    console.log, set a formatter using the BASIC_FORMAT format string, and
    add the handler to the root logger.

    A number of optional keyword arguments may be specified, which can alter
    the default behaviour.

    format      Use the specified format string for the handler.
    datefmt     Use the specified date/time format.
    style           If a format string is specified, use this to specify the
                            type of format string (possible values '%', '{', '$', for
                            %-formatting, :meth:`str.format` and :class:`string.Template`
                            - defaults to '%').
    level           Set the root logger level to the specified level.
    stream      Use the specified stream to initialize the StreamHandler. Note
                            that this argument is incompatible with 'filename' - if both
                            are present, 'stream' is ignored.
    handlers    If specified, this should be an iterable of already created
                            handlers, which will be added to the root handler. Any handler
                            in the list which does not have a formatter assigned will be
                            assigned the formatter created in this function.

    .. versionchanged:: 3.2
    Added the ``style`` parameter.

    .. versionchanged:: 3.3
    Added the ``handlers`` parameter. A ``ValueError`` is now thrown for
    incompatible arguments (e.g. ``handlers`` specified together with
    ``filename``/``filemode``, or ``filename``/``filemode`` specified
    together with ``stream``, or ``handlers`` specified together with
    ``stream``.
    """
    _acquireLock()
    try:
        if len(root.handlers) == 0:
            handlers: Optional[List['Handler']] = kwargs.pop('handlers', None)
            if handlers is not None:
                if 'stream' in kwargs:
                    raise ValueError("'stream' should not be specified together with 'handlers'")
            if handlers is None:
                stream: Optional[Any] = kwargs.pop('stream', None)
                h: 'StreamHandler' = StreamHandler(stream)
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

def getLogger(name: Optional[str] = None) -> 'Logger':
    """
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
    """
    if name:
        return Logger.manager.getLogger(name)
    else:
        return root

__pragma__('kwargs')

def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with severity 'CRITICAL' on the root logger. If the logger
    has no handlers, call basicConfig() to add a console handler with a
    pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.critical(msg, *args, **kwargs)
fatal: Callable[[str, Any, Any], None] = critical

def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with severity 'ERROR' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.error(msg, *args, **kwargs)

def exception(msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
    """
    Log a message with severity 'ERROR' on the root logger, with exception
    information. If the logger has no handlers, basicConfig() is called to add
    a console handler with a pre-defined format.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)

def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with severity 'WARNING' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.warning(msg, *args, **kwargs)

def warn(msg: str, *args: Any, **kwargs: Any) -> None:
    warnings.warn_explicit('The `warn` method is deprecated - use `warning`', DeprecationWarning, 'logging/__init__.py', 1944, 'logging')
    warning(msg, *args, **kwargs)

def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.info(msg, *args, **kwargs)

def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log a message with severity 'DEBUG' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.debug(msg, *args, **kwargs)

def log(level: int, msg: str, *args: Any, **kwargs: Any) -> None:
    """
    Log 'msg.format(args)' with the integer severity 'level' on the root logger. If
    the logger has no handlers, call basicConfig() to add a console handler
    with a pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.log(level, msg, *args, **kwargs)
__pragma__('nokwargs')

def disable(level: int) -> None:
    """
    Disable all logging calls of severity 'level' and below.
    """
    root.manager.disable = level

def shutdown(handlerList: List[Callable[[], Optional['Handler']]] = _handlerList) -> None:
    """
    Perform any cleanup actions in the logging system (e.g. flushing
    buffers).

    Should be called at application exit.
    """
    for wr in reversed(handlerList[:]):
        try:
            h: Optional['Handler'] = wr()
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
    This handler does nothing. It's intended to be used to avoid the
    "No handlers could be found for logger XXX" one-off warning. This is
    important for library code, which may contain code to log events. If a user
    of the library does not configure logging, the one-off warning might be
    produced; to avoid this, the library developer simply needs to instantiate
    a NullHandler and add it to the top-level logger of the library module or
    package.
        """

    def handle(self, record: LogRecord) -> None:
        """Stub."""

    def emit(self, record: LogRecord) -> None:
        """Stub."""

    def createLock(self) -> None:
        self.lock = None
_warnings_showwarning: Optional[Callable[..., None]] = None

def _showwarning(message: warnings.WarningMessage, category: type, filename: str, lineno: int, file: Optional[Any] = None, line: Optional[str] = None) -> None:
    """
    Implementation of showwarnings which redirects to logging, which will first
    check to see if the file parameter is None. If a file is specified, it will
    delegate to the original warnings implementation of showwarning. Otherwise,
    it will call warnings.formatwarning and will log the resulting string to a
    warnings logger named "py.warnings" with level logging.WARNING.
    """
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(message, category, filename, lineno, file, line)
    else:
        s: str = warnings.formatwarning(message, category, filename, lineno, line)
        logger: 'Logger' = getLogger('py.warnings')
        if not logger.handlers:
            logger.addHandler(NullHandler())
        logger.warning(s)

def captureWarnings(capture: bool) -> None:
    """
    If capture is true, redirect all warnings to the logging package.
    If capture is False, ensure that warnings are not redirected to logging
    but to their original destinations.
    """
    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.setShowWarning(_showwarning)
    elif _warnings_showwarning is not None:
        warnings.setShowWarnings(_warnings_showwarning)
        _warnings_showwarning = None
