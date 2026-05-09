from org.transcrypt.stubs.browser import __pragma__
import logging
import re
from typing import List, Tuple

threading = None
DEFAULT_TCP_LOGGING_PORT = 9020
DEFAULT_UDP_LOGGING_PORT = 9021
DEFAULT_HTTP_LOGGING_PORT = 9022
DEFAULT_SOAP_LOGGING_PORT = 9023
SYSLOG_UDP_PORT = 514
SYSLOG_TCP_PORT = 514
_MIDNIGHT = 24 * 60 * 60

class AJAXHandler(logging.Handler):
    """
    This class provides a means of pushing log records to a webserver
    via AJAX requests to a host server. Likely will have cross-domain
    restrictions unless you do something special.
    """

    def __init__(self, url: str, method: str = 'GET', headers: List[Tuple[str, str]] = []):
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
        logging.Handler.__init__(self)
        method = method.upper()
        if method not in ['GET', 'POST']:
            raise ValueError('method must be GET or POST')
        self.url: str = url
        self.method: str = method
        self.headers: List[Tuple[str, str]] = headers

    # ... rest of the code ...

class BufferingHandler(logging.Handler):
    """
    A handler class which buffers logging records in memory. Whenever each
    record is added to the buffer, a check is made to see if the buffer should
    be flushed. If it should, then flush() is expected to do what's needed.
    """

    def __init__(self, capacity: int):
        """
        Initialize the handler with the buffer size.
        """
        logging.Handler.__init__(self)
        self.capacity: int = capacity
        self.buffer: List[logging.LogRecord] = []

    # ... rest of the code ...

class MemoryHandler(BufferingHandler):
    """
    A handler class which buffers logging records in memory, periodically
    flushing them to a target handler. Flushing occurs whenever the buffer
    is full, or when an event of a certain severity or greater is seen.
    """

    def __init__(self, capacity: int, flushLevel: int = logging.ERROR, target: logging.Handler = None, flushOnClose: bool = True):
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
        BufferingHandler.__init__(self, capacity)
        self.flushLevel: int = flushLevel
        self.target: logging.Handler = target
        self.flushOnClose: bool = flushOnClose

    # ... rest of the code ...

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

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        raise NotImplementedError('No Working Implementation Yet')

    # ... rest of the code ...
