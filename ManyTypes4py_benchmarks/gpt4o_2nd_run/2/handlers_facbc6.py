from org.transcrypt.stubs.browser import __pragma__
import logging
import re
from typing import List, Tuple, Optional, Any

threading = None
DEFAULT_TCP_LOGGING_PORT = 9020
DEFAULT_UDP_LOGGING_PORT = 9021
DEFAULT_HTTP_LOGGING_PORT = 9022
DEFAULT_SOAP_LOGGING_PORT = 9023
SYSLOG_UDP_PORT = 514
SYSLOG_TCP_PORT = 514
_MIDNIGHT = 24 * 60 * 60

class AJAXHandler(logging.Handler):
    def __init__(self, url: str, method: str = 'GET', headers: List[Tuple[str, str]] = []) -> None:
        logging.Handler.__init__(self)
        method = method.upper()
        if method not in ['GET', 'POST']:
            raise ValueError('method must be GET or POST')
        self.url = url
        self.method = method
        self.headers = headers

    def mapLogRecord(self, record: logging.LogRecord) -> dict:
        return record.__dict__

    def urlencode(self, msg: str) -> str:
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

    def emit(self, record: logging.LogRecord) -> None:
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
    def __init__(self, capacity: int) -> None:
        logging.Handler.__init__(self)
        self.capacity = capacity
        self.buffer: List[logging.LogRecord] = []

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        return len(self.buffer) >= self.capacity

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)
        if self.shouldFlush(record):
            self.flush()

    def flush(self) -> None:
        self.acquire()
        try:
            self.buffer = []
        finally:
            self.release()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            logging.Handler.close(self)

class MemoryHandler(BufferingHandler):
    def __init__(self, capacity: int, flushLevel: int = logging.ERROR, target: Optional[logging.Handler] = None, flushOnClose: bool = True) -> None:
        BufferingHandler.__init__(self, capacity)
        self.flushLevel = flushLevel
        self.target = target
        self.flushOnClose = flushOnClose

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        return len(self.buffer) >= self.capacity or record.levelno >= self.flushLevel

    def setTarget(self, target: logging.Handler) -> None:
        self.target = target

    def flush(self) -> None:
        self.acquire()
        try:
            if self.target:
                for record in self.buffer:
                    self.target.handle(record)
                self.buffer = []
        finally:
            self.release()

    def close(self) -> None:
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
    def __init__(self, queue: Any) -> None:
        logging.Handler.__init__(self)
        raise NotImplementedError('No Working Implementation Yet')

    def enqueue(self, record: logging.LogRecord) -> None:
        self.queue.put_nowait(record)

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        self.format(record)
        record.msg = record.message
        record.args = None
        record.exc_info = None
        return record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)
