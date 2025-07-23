"""
Additional handlers for the logging package for Python. The core package is
based on PEP 282 and comments thereto in comp.lang.python.

Copyright (C) 2001-2016 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging.handlers' and log away!

Edited by Carl Allendorph 2016 for the Transcrypt project
"""
from org.transcrypt.stubs.browser import __pragma__
import logging
import re
from typing import Optional, List, Tuple, Any, Dict, Union, Callable, TypeVar
import threading as threading_module

threading: Optional[Any] = None
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
    via AJAX requests to a host server.
    """

    def __init__(self, url: str, method: str = 'GET', headers: List[Tuple[str, str]] = []) -> None:
        """
        Initialize the instance with url and the method type
        ("GET" or "POST")
        """
        logging.Handler.__init__(self)
        method = method.upper()
        if method not in ['GET', 'POST']:
            raise ValueError('method must be GET or POST')
        self.url: str = url
        self.method: str = method
        self.headers: List[Tuple[str, str]] = headers

    def mapLogRecord(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Default implementation of mapping the log record into a dict
        that is sent as the CGI data.
        """
        return record.__dict__

    def urlencode(self, msg: str) -> str:
        """ Encode the passed string with escapes for
        non-valid characters in a url.
        """
        def repl(m: re.Match) -> str:
            v = m.group(0)
            v = ord(v)
            hVal = v.toString(16)
            if len(hVal) == 1:
                hVal = '0' + hVal
            hVal = '%' + hVal
            return hVal
        p = re.compile("[^-A-Za-z0-9\\-\\._~:/?#[\\]@!$&'()\\*+,;=`]")
        ret = p.sub(repl, msg)
        return ret

    def emit(self, record: Union[logging.LogRecord, str]) -> None:
        """
        Emit a record.
        Send the record to the Web server as a percent-encoded dictionary
        """
        if type(record) is str:
            msg = record
        else:
            msg = self.format(record)
        try:
            url = self.url
            data = None
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
            conn = None
            errObj = None
            __pragma__('js', '{}', "\n                       try {\n                         conn = new(XMLHttpRequest || ActiveXObject)('MSXML2.XMLHTTP.3.0');\n                       } catch( err ) {\n                         errObj = err\n                       }\n                       ")
            if errObj is not None:
                raise Exception('Failed Create AJAX Request', errObj)
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
    A handler class which buffers logging records in memory.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the handler with the buffer size.
        """
        logging.Handler.__init__(self)
        self.capacity: int = capacity
        self.buffer: List[logging.LogRecord] = []

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Should the handler flush its buffer?
        """
        return len(self.buffer) >= self.capacity

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.
        """
        self.buffer.append(record)
        if self.shouldFlush(record):
            self.flush()

    def flush(self) -> None:
        """
        Override to implement custom flushing behaviour.
        """
        self.acquire()
        try:
            self.buffer = []
        finally:
            self.release()

    def close(self) -> None:
        """
        Close the handler.
        """
        try:
            self.flush()
        finally:
            logging.Handler.close(self)

class MemoryHandler(BufferingHandler):
    """
    A handler class which buffers logging records in memory.
    """

    def __init__(self, capacity: int, flushLevel: int = logging.ERROR, target: Optional[logging.Handler] = None, flushOnClose: bool = True) -> None:
        """
        Initialize the handler.
        """
        BufferingHandler.__init__(self, capacity)
        self.flushLevel: int = flushLevel
        self.target: Optional[logging.Handler] = target
        self.flushOnClose: bool = flushOnClose

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        return len(self.buffer) >= self.capacity or record.levelno >= self.flushLevel

    def setTarget(self, target: logging.Handler) -> None:
        """
        Set the target handler for this handler.
        """
        self.target = target

    def flush(self) -> None:
        """
        Flush the buffered records to the target.
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
        Close the handler.
        """
        try:
            if self.flushOnClose:
                self.flush()
        finally:
            self.acquire()
            try:
                self.target = None
                BufferingHandler.close(self)
            finally:
                self.release()

class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue.
    """

    def __init__(self, queue: Any) -> None:
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        raise NotImplementedError('No Working Implementation Yet')

    def enqueue(self, record: logging.LogRecord) -> None:
        """
        Enqueue a record.
        """
        self.queue.put_nowait(record)

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Prepares a record for queuing.
        """
        self.format(record)
        record.msg = record.message
        record.args = None
        record.exc_info = None
        return record

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)
