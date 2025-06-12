"""
Additional handlers for the logging package for Python. The core package is
based on PEP 282 and comments thereto in comp.lang.python.

Copyright (C) 2001-2016 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging.handlers' and log away!

Edited by Carl Allendorph 2016 for the Transcrypt project

I've kept some of the handlers but I've pruned anything related to
the file system at this point. There is a stub in preparation for
an AJAX based handler that will likely need to be coupled with
a QueueHandler and maybe a buffered handler.

!!!!!!!!!!!!!!!!!!!!!!!!!
This code is not well tested yet!
!!!!!!!!!!!!!!!!!!!!!!!!!
"""
from org.transcrypt.stubs.browser import __pragma__
import logging
import re
from typing import List, Tuple, Optional, Any, Dict, Callable

threading = None
DEFAULT_TCP_LOGGING_PORT: int = 9020
DEFAULT_UDP_LOGGING_PORT: int = 9021
DEFAULT_HTTP_LOGGING_PORT: int = 9022
DEFAULT_SOAP_LOGGING_PORT: int = 9023
SYSLOG_UDP_PORT: int = 514
SYSLOG_TCP_PORT: int = 514
_MIDNIGHT: int = 24 * 60 * 60

class AJAXHandler(logging.Handler):
    """
    This class provides a means of pushing log records to a webserver
    via AJAX requests to a host server. Likely will have cross-domain
    restrictions unless you do something special.
    """

    def __init__(self, url: str, method: str = 'GET', headers: List[Tuple[str, str]] = []) -> None:
        """
        Initialize the instance with url and the method type
        ("GET" or "POST")
        @param url the page to send the messages to. Generally
          this is going to be address relative to this host
          but could also be fullyqualified
        @param method "GET" or "POST"
        @param headers list of tuples that contains the
          headers that will be added to the HTTP request for the
          AJAX push. None are applied by default.
        """
        super().__init__()
        method = method.upper()
        if method not in ['GET', 'POST']:
            raise ValueError('method must be GET or POST')
        self.url: str = url
        self.method: str = method
        self.headers: List[Tuple[str, str]] = headers

    def mapLogRecord(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Default implementation of mapping the log record into a dict
        that is sent as the CGI data. Overwrite in your class.
        Contributed by Franz Glasner.
        """
        return record.__dict__

    def urlencode(self, msg: str) -> str:
        """ Encode the passed string with escapes for
        non-valid characters in a url.
        """

        def repl(m: re.Match) -> str:
            v = m.group(0)
            v = ord(v)
            hVal = format(v, 'x')
            if len(hVal) == 1:
                hVal = '0' + hVal
            hVal = '%' + hVal
            return hVal

        p = re.compile("[^-A-Za-z0-9\\-\\._~:/?#[\\]@!$&'()\\*+,;=`]")
        ret: str = p.sub(repl, msg)
        return ret

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.
        Send the record to the Web server as a percent-encoded dictionary
        """
        if isinstance(record, str):
            msg: str = record
        else:
            msg = self.format(record)
        try:
            url: str = self.url
            data: Optional[str] = None
            if self.method == 'GET':
                if url.find('?') >= 0:
                    sep = '&'
                else:
                    sep = '?'
                url = url + '{}msg={}'.format(sep, msg)
                url = self.urlencode(url)
            else:
                data = 'msg={}'.format(msg)
                data = self.urlencode(data)

            def ajaxCallback() -> int:
                return 0
            conn: Optional[Any] = None
            errObj: Optional[Exception] = None
            __pragma__('js', '{}', "\n                       try {\n                         conn = new(XMLHttpRequest || ActiveXObject)('MSXML2.XMLHTTP.3.0');\n                       } catch( err ) {\n                         errObj = err\n                       }\n                       ")
            if errObj is not None:
                raise Exception('Failed Create AJAX Request') from errObj
            if conn is None:
                raise Exception('Invalid Ajax Object')
            conn.open(self.method, url, 1)
            for key, val in self.headers:
                conn.setRequestHeader(key, val)
            conn.onreadystatechange = ajaxCallback
            conn.send(data)
        except Exception:
            self.handleError(record)

class BufferingHandler(logging.Handler):
    """
    A handler class which buffers logging records in memory. Whenever each
    record is added to the buffer, a check is made to see if the buffer should
    be flushed. If it should, then flush() is expected to do what's needed.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the handler with the buffer size.
        """
        super().__init__()
        self.capacity: int = capacity
        self.buffer: List[logging.LogRecord] = []

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Should the handler flush its buffer?

        Returns true if the buffer is up to capacity. This method can be
        overridden to implement custom flushing strategies.
        """
        return len(self.buffer) >= self.capacity

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.

        Append the record. If shouldFlush() tells us to, call flush() to process
        the buffer.
        """
        self.buffer.append(record)
        if self.shouldFlush(record):
            self.flush()

    def flush(self) -> None:
        """
        Override to implement custom flushing behaviour.

        This version just zaps the buffer to empty.
        """
        self.acquire()
        try:
            self.buffer = []
        finally:
            self.release()

    def close(self) -> None:
        """
        Close the handler.

        This version just flushes and chains to the parent class' close().
        """
        try:
            self.flush()
        finally:
            super().close()

class MemoryHandler(BufferingHandler):
    """
    A handler class which buffers logging records in memory, periodically
    flushing them to a target handler. Flushing occurs whenever the buffer
    is full, or when an event of a certain severity or greater is seen.
    """

    def __init__(
        self,
        capacity: int,
        flushLevel: int = logging.ERROR,
        target: Optional[logging.Handler] = None,
        flushOnClose: bool = True
    ) -> None:
        """
        Initialize the handler with the buffer size, the level at which
        flushing should occur and an optional target.

        Note that without a target being set either here or via setTarget(),
        a MemoryHandler is no use to anyone!

        The ``flushOnClose`` argument is ``True`` for backward compatibility
        reasons - the old behaviour is that when the handler is closed, the
        buffer is flushed, even if the flush level hasn't been exceeded nor the
        capacity exceeded. To prevent this, set ``flushOnClose`` to ``False``.
        """
        super().__init__(capacity)
        self.flushLevel: int = flushLevel
        self.target: Optional[logging.Handler] = target
        self.flushOnClose: bool = flushOnClose

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        return super().shouldFlush(record) or record.levelno >= self.flushLevel

    def setTarget(self, target: logging.Handler) -> None:
        """
        Set the target handler for this handler.
        """
        self.target = target

    def flush(self) -> None:
        """
        For a MemoryHandler, flushing means just sending the buffered
        records to the target, if there is one. Override if you want
        different behaviour.

        The record buffer is also cleared by this operation.
        """
        self.acquire()
        try:
            if self.target:
                for record in self.buffer:
                    self.target.handle(record)
                self.buffer = []
        finally:
            self.release()

    def close(self) -> None:
        """
        Flush, if appropriately configured, set the target to None and lose the
        buffer.
        """
        try:
            if self.flushOnClose:
                self.flush()
        finally:
            self.acquire()
            try:
                self.target = None
                super().close()
            finally:
                self.release()

class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.

    For transcrypt, this maybe useful for implementing a web worker in
    the background for processing logs and sending them to the server or
    elsewhere without blocking the main task.

    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue: Any) -> None:
        """
        Initialise an instance, using the passed queue.
        """
        super().__init__()
        raise NotImplementedError('No Working Implementation Yet')

    def enqueue(self, record: logging.LogRecord) -> None:
        """
        Enqueue a record.

        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.

        In transcrypt we will likely want to use a push to a webworker
        here instead of a normal queue.
        """
        self.queue.put_nowait(record)

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        self.format(record)
        record.msg = record.message
        record.args = None
        record.exc_info = None
        return record

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)
