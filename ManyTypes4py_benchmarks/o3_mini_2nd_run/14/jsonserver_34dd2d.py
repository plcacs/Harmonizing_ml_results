#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import socket
import logging
import platform
import asyncore
import asynchat
import threading
import traceback
import subprocess
from logging import handlers
from optparse import OptionParser
from os import chmod
from os.path import dirname, join, abspath
from operator import xor
from typing import Any, Union, Optional, Callable, Type, Iterable

try:
    import ujson as json
except ImportError:
    import json

PROJECT_ROOT: str = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(PROJECT_ROOT, 'anaconda_lib'))
from lib.path import log_directory  # type: ignore
from jedi import set_debug_function  # type: ignore
from lib.contexts import json_decode  # type: ignore
from unix_socket import UnixSocketPath, get_current_umask  # type: ignore
from handlers import ANACONDA_HANDLERS  # type: ignore
from jedi import settings as jedi_settings  # type: ignore
from lib.anaconda_handler import AnacondaHandler  # type: ignore

DEBUG_MODE: bool = False
logger: logging.Logger = logging.getLogger('')
PY3: bool = True if sys.version_info >= (3,) else False

class JSONHandler(asynchat.async_chat):
    """Handles JSON messages from a client.
    """

    def __init__(self, sock: socket.socket, server: JSONServer) -> None:
        self.server: JSONServer = server
        self.rbuffer: list[bytes] = []
        super().__init__(sock)
        terminator: bytes = b'\r\n'
        self.set_terminator(terminator)

    def return_back(self, data: Any = None, **kwargs: Any) -> None:
        """Send data back to the client."""
        if kwargs:
            data = kwargs
        if data is not None:
            print(data)
            dumped: str = json.dumps(data)
            payload: str = f'{dumped}\r\n'
            payload_bytes: bytes = payload.encode('utf8') if PY3 else payload
            if DEBUG_MODE is True:
                msg: str = f'About push back to ST3: {payload_bytes}'
                print(msg)
                logging.info(msg)
            self.push(payload_bytes)

    def collect_incoming_data(self, data: bytes) -> None:
        """Called when data is ready to be read."""
        self.rbuffer.append(data)

    def found_terminator(self) -> None:
        """Called when the terminator is found in the buffer."""
        message: Union[bytes, str]
        if PY3:
            message = b''.join(self.rbuffer)
        else:
            message = ''.join(self.rbuffer)
        self.rbuffer = []
        with json_decode(message) as data:  # type: ignore
            if not data:
                logging.info('No data received in the handler')
                return
            if isinstance(data, dict) and data.get('method') == 'check':
                logging.info('Check received')
                self.return_back(message='Ok', uid=data['uid'])
                return
            self.server.last_call = time.time()
        if isinstance(data, dict):
            logging.info('client requests: {0}'.format(data.get('method')))
            method: str = data.pop('method')
            uid: Any = data.pop('uid')
            vid: Optional[Any] = data.pop('vid', None)
            settings: dict = data.pop('settings', {})
            handler_type: str = data.pop('handler')
            if DEBUG_MODE is True:
                print(f'Received method: {method}, handler: {handler_type}')
            try:
                self.handle_command(handler_type, method, uid, vid, settings, data)
            except Exception as error:
                logging.error(error)
                log_traceback()
                self.return_back({'success': False, 'uid': uid, 'vid': vid, 'error': str(error)})
        else:
            logging.error("client sent something that I don't understand: {0}".format(data))

    def handle_command(self, handler_type: str, method: str, uid: Any, vid: Optional[Any],
                       settings: dict, data: dict) -> None:
        """Call the right commands handler."""
        if not AnacondaHandler._registry.initialized:
            AnacondaHandler._registry.initialize()
        handler: Callable = ANACONDA_HANDLERS.get(handler_type, AnacondaHandler.get_handler(handler_type))
        if DEBUG_MODE is True:
            print(f'{handler} handler retrieved from registry')
        # Call the handler; assuming it returns an object with a run() method.
        handler(method, data, uid, vid, settings, self.return_back, DEBUG_MODE).run()

class JSONServer(asyncore.dispatcher):
    """Asynchronous standard library TCP JSON server."""
    allow_reuse_address = False
    request_queue_size: int = 5
    if platform.system().lower() != 'linux':
        address_family = socket.AF_INET
    else:
        address_family = socket.AF_UNIX
    socket_type = socket.SOCK_STREAM

    def __init__(self, address: Union[tuple, str], handler: Type[JSONHandler] = JSONHandler) -> None:
        self.address: Union[tuple, str] = address
        self.handler: Type[JSONHandler] = handler
        super().__init__()
        self.create_socket(self.address_family, self.socket_type)
        self.last_call: float = time.time()
        self.bind(self.address)
        if hasattr(socket, 'AF_UNIX') and self.address_family == socket.AF_UNIX:
            chmod(self.address, xor(511, get_current_umask()))
        logging.debug('bind: address=%s' % (address,))
        self.listen(self.request_queue_size)
        logging.debug('listen: backlog=%d' % (self.request_queue_size,))
        self.logger: Optional[logging.Logger] = None

    @property
    def fileno(self) -> int:
        return self.socket.fileno()

    def serve_forever(self) -> None:
        asyncore.loop()

    def shutdown(self) -> None:
        self.handle_close()

    def handle_accept(self) -> None:
        """Called when we accept an incoming connection."""
        result: Optional[tuple[socket.socket, Union[tuple, str]]] = self.accept()
        if result is None:
            return
        sock, addr = result
        if self.logger is not None:
            self.logger.info('Incoming connection from {0}'.format(repr(addr) or 'unix socket'))
        self.handler(sock, self)

    def handle_close(self) -> None:
        """Called when close."""
        logging.info('Closing the socket, server will be shutdown now...')
        self.close()

class Checker(threading.Thread):
    """Check that the ST3 PID already exists every delta seconds."""
    MAX_INACTIVITY: int = 1800

    def __init__(self, server: JSONServer, pid: Union[int, str], delta: int = 5) -> None:
        super().__init__()
        self.server: JSONServer = server
        self.delta: int = delta
        self.daemon = True
        self.die: bool = False
        self.pid: int = int(pid)

    def run(self) -> None:
        while not self.die:
            if time.time() - self.server.last_call > self.MAX_INACTIVITY:
                if self.server.logger is not None:
                    self.server.logger.info('detected inactivity for more than 30 minutes... shutting down...')
                break
            self._check()
            if not self.die:
                time.sleep(self.delta)
        self.server.shutdown()

    if os.name == 'nt':
        def _isprocessrunning(self, timeout: int = MAX_INACTIVITY * 1000) -> bool:
            """Blocking until process has exited or timeout is reached."""
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 1048576
            WAIT_TIMEOUT = 258
            hprocess = kernel32.OpenProcess(SYNCHRONIZE, False, self.pid)
            if hprocess == 0:
                return False
            ret = kernel32.WaitForSingleObject(hprocess, timeout)
            kernel32.CloseHandle(hprocess)
            return ret == WAIT_TIMEOUT
    else:
        def _isprocessrunning(self) -> bool:
            """Returning immediately whether process is running."""
            try:
                os.kill(self.pid, 0)
            except OSError:
                return False
            return True

    def _check(self) -> None:
        """Check for the ST3 pid."""
        if not self._isprocessrunning():
            if self.server.logger is not None:
                self.server.logger.info('process {0} does not exist, stopping server...'.format(self.pid))
            self.die = True

def get_logger(path: str) -> logging.Logger:
    """Build file logger."""
    if not os.path.exists(path):
        os.makedirs(path)
    log_obj: logging.Logger = logging.getLogger('')
    log_obj.setLevel(logging.DEBUG)
    hdlr = handlers.RotatingFileHandler(filename=join(path, 'anaconda_jsonserver.log'),
                                        maxBytes=10000000, backupCount=5, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s: %(message)s')
    hdlr.setFormatter(formatter)
    log_obj.addHandler(hdlr)
    return log_obj

def log_traceback() -> None:
    """Just log the traceback."""
    logging.error(traceback.format_exc())

if __name__ == '__main__':
    WINDOWS: bool = os.name == 'nt'
    LINUX: bool = platform.system().lower() == 'linux'
    usage: str
    if WINDOWS:
        usage = 'usage: %prog -p <project> -e <extra_paths> port'
    else:
        usage = 'usage: %prog -p <project> -e <extra_paths> ST3_PID'
    opt_parser: OptionParser = OptionParser(usage=usage)
    opt_parser.add_option('-p', '--project', action='store', dest='project', help='project name')
    opt_parser.add_option('-e', '--extra_paths', action='store', dest='extra_paths',
                          help='extra paths (separated by comma) that should be added to sys.paths')
    options, args = opt_parser.parse_args()
    port: Optional[int] = None
    PID: Optional[str] = None
    if not LINUX:
        if len(args) != 2:
            opt_parser.error('you have to pass a port number and PID')
        port = int(args[0])
        PID = args[1]
    else:
        if len(args) != 1:
            opt_parser.error('you have to pass a Sublime Text 3 PID')
        PID = args[0]
    if options.project is not None:
        jedi_settings.cache_directory = join(jedi_settings.cache_directory, options.project)
        log_directory = join(log_directory, options.project)
    if not os.path.exists(jedi_settings.cache_directory):
        os.makedirs(jedi_settings.cache_directory)
    if options.extra_paths is not None:
        for path in options.extra_paths.split(','):
            if path not in sys.path:
                sys.path.insert(0, path)
    logger = get_logger(log_directory)
    try:
        server: Optional[JSONServer] = None
        if not LINUX:
            server = JSONServer(('localhost', port))  # type: ignore
        else:
            unix_socket_path: UnixSocketPath = UnixSocketPath(options.project)
            if not os.path.exists(dirname(unix_socket_path.socket)):
                os.makedirs(dirname(unix_socket_path.socket))
            if os.path.exists(unix_socket_path.socket):
                os.unlink(unix_socket_path.socket)
            server = JSONServer(unix_socket_path.socket)
        logger.info('Anaconda Server started in {0} for PID {1} with cache dir {2}{3}'.format(
            port or getattr(unix_socket_path, 'socket', 'unknown'),
            PID,
            jedi_settings.cache_directory,
            ' and extra paths {0}'.format(options.extra_paths) if options.extra_paths is not None else ''
        ))
    except Exception as error:
        log_traceback()
        logger.error(str(error))
        if server is not None:
            server.shutdown()
        sys.exit(-1)
    server.logger = logger  # type: ignore
    if PID != 'DEBUG':
        checker = Checker(server, pid=PID, delta=1)
        checker.start()
    else:
        logger.info('Anaconda Server started in DEBUG mode...')
        print('DEBUG MODE')
        DEBUG_MODE = True
        set_debug_function(notices=True)
    server.serve_forever()