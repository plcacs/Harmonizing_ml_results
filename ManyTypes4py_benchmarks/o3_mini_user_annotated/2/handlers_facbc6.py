from typing import Any, Dict, List, Optional, Union
from org.transcrypt.stubs.browser import __pragma__
import logging
import re

threading = None

DEFAULT_TCP_LOGGING_PORT = 9020
DEFAULT_UDP_LOGGING_PORT = 9021
DEFAULT_HTTP_LOGGING_PORT = 9022
DEFAULT_SOAP_LOGGING_PORT = 9023
SYSLOG_UDP_PORT = 514
SYSLOG_TCP_PORT = 514

_MIDNIGHT = 24 * 60 * 60  # number of seconds in a day

class AJAXHandler(logging.Handler):
    """
    This class provides a means of pushing log records to a webserver
    via AJAX requests to a host server.
    """
    def __init__(self, url: str, method: str = "GET", headers: List[tuple] = []) -> None:
        """
        Initialize the instance with url and the method type
        ("GET" or "POST")
        """
        super().__init__()
        method = method.upper()
        if method not in ["GET", "POST"]:
            raise ValueError("method must be GET or POST")
        self.url: str = url
        self.method: str = method
        self.headers: List[tuple] = headers

    def mapLogRecord(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Default implementation of mapping the log record into a dict
        that is sent as the CGI data. Overwrite in your class.
        """
        return record.__dict__

    def urlencode(self, msg: str) -> str:
        """Encode the passed string with escapes for non-valid characters in a url."""
        def repl(m: Any) -> str:
            v: Any = m.group(0)
            v = ord(v)
            hVal: str = format(v, 'x')
            if len(hVal) == 1:
                hVal = "0" + hVal
            hVal = "%" + hVal
            return hVal

        p = re.compile(r"[^-A-Za-z0-9\-\._~:/?#[\]@!$&'()\*+,;=`]")
        ret: str = p.sub(repl, msg)
        return ret

    def emit(self, record: Union[str, logging.LogRecord]) -> None:
        """
        Emit a record.
        Send the record to the Web server as a percent-encoded dictionary.
        """
        if isinstance(record, str):
            msg: str = record
        else:
            msg = self.format(record)

        try:
            url: str = self.url
            data: Optional[str] = None
            if self.method == "GET":
                sep: str = '&' if url.find('?') >= 0 else '?'
                url = url + "{}msg={}".format(sep, msg)
                url = self.urlencode(url)
            else:  # "POST"
                data = "msg={}".format(msg)
                data = self.urlencode(data)

            def ajaxCallback() -> int:
                return 0

            conn: Any = None
            errObj: Any = None
            __pragma__('js', '{}', '''
            try {
              conn = new(XMLHttpRequest || ActiveXObject)('MSXML2.XMLHTTP.3.0');
            } catch( err ) {
              errObj = err
            }
            ''')

            if errObj is not None:
                raise Exception("Failed Create AJAX Request", errObj)

            if conn is None:
                raise Exception("Invalid Ajax Object")

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
        super().__init__()
        self.capacity: int = capacity
        self.buffer: List[logging.LogRecord] = []

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Returns true if the buffer is up to capacity.
        """
        return len(self.buffer) >= self.capacity

    def emit(self, record: logging.LogRecord) -> None:
        """
        Append the record. If shouldFlush() tells us to, call flush() to process the buffer.
        """
        self.buffer.append(record)
        if self.shouldFlush(record):
            self.flush()

    def flush(self) -> None:
        """
        Override to implement custom flushing behaviour.
        This version just clears the buffer.
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
            super().close()

class MemoryHandler(BufferingHandler):
    """
    A handler class which buffers logging records in memory, periodically
    flushing them to a target handler.
    """
    def __init__(self, capacity: int, flushLevel: int = logging.ERROR, target: Optional[logging.Handler] = None, flushOnClose: bool = True) -> None:
        """
        Initialize the handler with the buffer size, the level at which flushing should occur and an optional target.
        """
        super().__init__(capacity)
        self.flushLevel: int = flushLevel
        self.target: Optional[logging.Handler] = target
        self.flushOnClose: bool = flushOnClose

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        return (len(self.buffer) >= self.capacity) or (record.levelno >= self.flushLevel)

    def setTarget(self, target: logging.Handler) -> None:
        """
        Set the target handler for this handler.
        """
        self.target = target

    def flush(self) -> None:
        """
        Flush buffered records to the target handler.
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
        Flush, if appropriately configured, set the target to None and clear the buffer.
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
    This handler sends events to a queue.
    """
    def __init__(self, queue: Any) -> None:
        """
        Initialise an instance using the passed queue.
        """
        super().__init__()
        raise NotImplementedError("No Working Implementation Yet")
        # self.queue = queue

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