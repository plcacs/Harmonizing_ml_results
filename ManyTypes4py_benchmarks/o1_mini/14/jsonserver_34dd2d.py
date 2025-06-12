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
from typing import Any, Optional, Dict, Union, Tuple

try:
    import ujson as json
except ImportError:
    import json

PROJECT_ROOT: str = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(PROJECT_ROOT, 'anaconda_lib'))
from lib.path import log_directory
from jedi import set_debug_function
from lib.contexts import json_decode
from unix_socket import UnixSocketPath, get_current_umask
from handlers import ANACONDA_HANDLERS
from jedi import settings as jedi_settings
from lib.anaconda_handler import AnacondaHandler

DEBUG_MODE: bool = False
logger: logging.Logger = logging.getLogger('')
PY3: bool = True if sys.version_info >= (3,) else False


class JSONHandler(asynchat.async_chat):
    """Handles JSON messages from a client"""

    server: 'JSONServer'
    rbuffer: list

    def __init__(self, sock: socket.socket, server: 'JSONServer') -> None:
        self.server = server
        self.rbuffer = []
        super().__init__(sock)
        self.set_terminator(b'\r\n' if PY3 else '\r\n')

    def return_back(self, data: Optional[Dict[str, Any]] = None, message: Optional[str] = None, uid: Optional[Any] = None) -> None:
        """Send data back to the client"""
        if data is not None or message is not None:
            if message is not None:
                print(message)
                data = '{0}\r\n'.format(json.dumps(data)) if data else message
            elif data is not None:
                data = '{0}\r\n'.format(json.dumps(data))
            data_bytes: bytes
            if PY3:
                data_bytes = bytes(data, 'utf8') if isinstance(data, str) else data
            else:
                data_bytes = data
            if DEBUG_MODE:
                print(f'About push back to ST3: {data_bytes}')
                logging.info(f'About push back to ST3: {data_bytes}')
            self.push(data_bytes)

    def collect_incoming_data(self, data: bytes) -> None:
        """Called when data is ready to be read"""
        self.rbuffer.append(data)

    def found_terminator(self) -> None:
        """Called when the terminator is found in the buffer"""
        message: Union[bytes, str] = b''.join(self.rbuffer) if PY3 else ''.join(self.rbuffer)
        self.rbuffer = []
        with json_decode(message) as data:
            if not data:
                logging.info('No data received in the handler')
                return
            if data.get('method') == 'check':
                logging.info('Check received')
                self.return_back({'message': 'Ok', 'uid': data.get('uid')})
                return
            self.server.last_call = time.time()
        if isinstance(data, dict):
            logging.info(f'client requests: {data.get("method")}')
            method: str = data.pop('method')
            uid: Any = data.pop('uid')
            vid: Optional[Any] = data.pop('vid', None)
            settings: Dict[str, Any] = data.pop('settings', {})
            handler_type: str = data.pop('handler')
            if DEBUG_MODE:
                print(f'Received method: {method}, handler: {handler_type}')
            try:
                self.handle_command(handler_type, method, uid, vid, settings, self.return_back)
            except Exception as error:
                logging.error(error)
                log_traceback()
                self.return_back({'success': False, 'uid': uid, 'vid': vid, 'error': str(error)})
        else:
            logging.error(f"client sent something that I don't understand: {data}")

    def handle_command(self, handler_type: str, method: str, uid: Any, vid: Optional[Any], settings: Dict[str, Any], callback) -> None:
        """Call the right commands handler"""
        if not AnacondaHandler._registry.initialized:
            AnacondaHandler._registry.initialize()
        handler = ANACONDA_HANDLERS.get(handler_type, AnacondaHandler.get_handler(handler_type))
        if DEBUG_MODE:
            print(f'{handler} handler retrieved from registry')
        handler(method, data, uid, vid, settings, callback, DEBUG_MODE).run()


class JSONServer(asyncore.dispatcher):
    """Asynchronous standard library TCP JSON server"""
    allow_reuse_address: bool = False
    request_queue_size: int = 5
    address_family: int = socket.AF_UNIX if platform.system().lower() == 'linux' else socket.AF_INET
    socket_type: int = socket.SOCK_STREAM

    address: Union[Tuple[str, int], str]
    handler: Any
    last_call: float
    logger: logging.Logger

    def __init__(self, address: Union[Tuple[str, int], str], handler: Any = JSONHandler) -> None:
        self.address = address
        self.handler = handler
        super().__init__()
        self.create_socket(self.address_family, self.socket_type)
        self.last_call = time.time()
        self.bind(self.address)
        if hasattr(socket, 'AF_UNIX') and self.address_family == socket.AF_UNIX:
            chmod(self.address, xor(511, get_current_umask()))
        logging.debug(f'bind: address={self.address}')
        self.listen(self.request_queue_size)
        logging.debug(f'listen: backlog={self.request_queue_size}')

    @property
    def fileno(self) -> int:
        return self.socket.fileno()

    def serve_forever(self) -> None:
        asyncore.loop()

    def shutdown(self) -> None:
        self.handle_close()

    def handle_accept(self) -> None:
        """Called when we accept an incoming connection"""
        try:
            sock, addr = self.accept()
            if addr:
                self.logger.info(f'Incoming connection from {repr(addr)}')
            else:
                self.logger.info('Incoming connection from unix socket')
            self.handler(sock, self)
        except Exception as e:
            logging.error(f'Error accepting connection: {e}')
            log_traceback()

    def handle_close(self) -> None:
        """Called when close"""
        logging.info('Closing the socket, server will be shutdown now...')
        self.close()


class Checker(threading.Thread):
    """Check that the ST3 PID already exists every delta seconds"""
    MAX_INACTIVITY: int = 1800

    server: JSONServer
    delta: int
    die: bool
    pid: int

    def __init__(self, server: JSONServer, pid: int, delta: int = 5) -> None:
        super().__init__()
        self.server = server
        self.delta = delta
        self.daemon = True
        self.die = False
        self.pid = pid

    def run(self) -> None:
        while not self.die:
            if time.time() - self.server.last_call > self.MAX_INACTIVITY:
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
            SYNCHRONIZE: int = 1048576
            WAIT_TIMEOUT: int = 258
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
        """Check for the ST3 pid"""
        if not self._isprocessrunning():
            self.server.logger.info(f'process {self.pid} does not exist stopping server...')
            self.die = True


def get_logger(path: str) -> logging.Logger:
    """Build file logger"""
    if not os.path.exists(path):
        os.makedirs(path)
    log: logging.Logger = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    hdlr: handlers.RotatingFileHandler = handlers.RotatingFileHandler(
        filename=join(path, 'anaconda_jsonserver.log'),
        maxBytes=10000000,
        backupCount=5,
        encoding='utf-8'
    )
    formatter: logging.Formatter = logging.Formatter('%(asctime)s: %(levelname)-8s: %(message)s')
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    return log


def log_traceback() -> None:
    """Just log the traceback"""
    logging.error(traceback.format_exc())


if __name__ == '__main__':
    WINDOWS: bool = os.name == 'nt'
    LINUX: bool = platform.system().lower() == 'linux'
    usage: str = 'usage: %prog -p <project> -e <extra_paths> port' if WINDOWS else 'usage: %prog -p <project> -e <extra_paths> ST3_PID'
    opt_parser: OptionParser = OptionParser(usage=usage)
    opt_parser.add_option('-p', '--project', action='store', dest='project', help='project name')
    opt_parser.add_option('-e', '--extra_paths', action='store', dest='extra_paths', help='extra paths (separated by comma) that should be added to sys.paths')
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
    server: Optional[JSONServer] = None
    try:
        if not LINUX:
            server = JSONServer(('localhost', port))  # type: ignore
        else:
            unix_socket_path = UnixSocketPath(options.project)  # type: ignore
            if not os.path.exists(dirname(unix_socket_path.socket)):
                os.makedirs(dirname(unix_socket_path.socket))
            if os.path.exists(unix_socket_path.socket):
                os.unlink(unix_socket_path.socket)
            server = JSONServer(unix_socket_path.socket)
        log_info_message: str = f'Anaconda Server started in {port or unix_socket_path.socket} for PID {PID} with cache dir {jedi_settings.cache_directory}'
        if options.extra_paths is not None:
            log_info_message += f' and extra paths {options.extra_paths}'
        logger.info(log_info_message)
    except Exception as error:
        log_traceback()
        logger.error(str(error))
        if server is not None:
            server.shutdown()
        sys.exit(-1)
    server.logger = logger
    if PID != 'DEBUG':
        checker = Checker(server, pid=int(PID), delta=1)  # type: ignore
        checker.start()
    else:
        logger.info('Anaconda Server started in DEBUG mode...')
        print('DEBUG MODE')
        DEBUG_MODE = True
        set_debug_function(notices=True)
    if server is not None:
        server.serve_forever()
