#!/usr/bin/env python
# -*- coding: utf8 -*-

# Copyright (C) 2013 - Oscar Campos <oscar.campos@member.fsf.org>
# This program is Free Software see LICENSE file for details

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
from typing import Dict, Any, Optional, Union, List, Callable, Tuple, Type, cast

# we use ujson if it's available on the target interpreter
try:
    import ujson as json
except ImportError:
    import json

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
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
    """Hadnles JSON messages from a client
    """

    def __init__(self, sock: socket.socket, server: 'JSONServer') -> None:
        self.server: 'JSONServer' = server
        self.rbuffer: List[bytes] = []
        asynchat.async_chat.__init__(self, sock)
        self.set_terminator(b"\r\n" if PY3 else "\r\n")

    def return_back(self, data: Optional[Dict[str, Any]]) -> None:
        """Send data back to the client
        """

        if data is not None:
            print(data)
            data_str: str = '{0}\r\n'.format(json.dumps(data))
            data_bytes: Union[bytes, str] = bytes(data_str, 'utf8') if PY3 else data_str

            if DEBUG_MODE is True:
                print('About push back to ST3: {0}'.format(data_bytes))
                logging.info('About push back to ST3: {0}'.format(data_bytes))
            self.push(data_bytes)

    def collect_incoming_data(self, data: bytes) -> None:
        """Called when data is ready to be read
        """

        self.rbuffer.append(data)

    def found_terminator(self) -> None:
        """Called when the terminator is found in the buffer
        """

        message: Union[bytes, str] = b''.join(self.rbuffer) if PY3 else ''.join(self.rbuffer)
        self.rbuffer = []

        with json_decode(message) as data:
            if not data:
                logging.info('No data received in the handler')
                return

            if data['method'] == 'check':
                logging.info('Check received')
                self.return_back({'message': 'Ok', 'uid': data['uid']})
                return

            self.server.last_call = time.time()

        if isinstance(data, dict):
            logging.info(
                'client requests: {0}'.format(data['method'])
            )

            method: str = data.pop('method')
            uid: str = data.pop('uid')
            vid: Optional[str] = data.pop('vid', None)
            settings: Dict[str, Any] = data.pop('settings', {})

            handler_type: str = data.pop('handler')
            if DEBUG_MODE is True:
                print('Received method: {0}, handler: {1}'.format(
                    method, handler_type)
                )
            try:
                self.handle_command(
                    handler_type, method, uid, vid, settings, data,
                )
            except Exception as error:
                logging.error(error)
                log_traceback()
                self.return_back({
                    'success': False, 'uid': uid,
                    'vid': vid, 'error': str(error)
                })
        else:
            logging.error(
                'client sent somethinf that I don\'t understand: {0}'.format(
                    data
                )
            )

    def handle_command(self, handler_type: str, method: str, uid: str, 
                       vid: Optional[str], settings: Dict[str, Any], 
                       data: Dict[str, Any]) -> None:
        """Call the right commands handler
        """

        # lazy initialization of anaconda plugins
        if not AnacondaHandler._registry.initialized:
            AnacondaHandler._registry.initialize()

        handler: Any = ANACONDA_HANDLERS.get(
            handler_type, AnacondaHandler.get_handler(handler_type))
        if DEBUG_MODE is True:
            print('{0} handler retrieved from registry'.format(handler))

        handler(
            method, data, uid, vid, settings, self.return_back, DEBUG_MODE,
        ).run()


class JSONServer(asyncore.dispatcher):
    """Asynchronous standard library TCP JSON server
    """

    allow_reuse_address: bool = False
    request_queue_size: int = 5
    if platform.system().lower() != 'linux':
        address_family: int = socket.AF_INET
    else:
        address_family: int = socket.AF_UNIX
    socket_type: int = socket.SOCK_STREAM

    def __init__(self, address: Union[str, Tuple[str, int]], 
                 handler: Type[JSONHandler] = JSONHandler) -> None:
        self.address: Union[str, Tuple[str, int]] = address
        self.handler: Type[JSONHandler] = handler
        self.logger: Optional[logging.Logger] = None
        self.last_call: float = time.time()

        asyncore.dispatcher.__init__(self)
        self.create_socket(self.address_family, self.socket_type)

        self.bind(self.address)
        if hasattr(socket, 'AF_UNIX') and \
                self.address_family == socket.AF_UNIX:
            # WSL 1903 fix
            chmod(cast(str, self.address), xor(0o777, get_current_umask()))
        logging.debug('bind: address=%s' % (address,))
        self.listen(self.request_queue_size)
        logging.debug('listen: backlog=%d' % (self.request_queue_size,))

    @property
    def fileno(self) -> int:
        return self.socket.fileno()

    def serve_forever(self) -> None:
        asyncore.loop()

    def shutdown(self) -> None:
        self.handle_close()

    def handle_accept(self) -> None:
        """Called when we accept and incoming connection
        """
        result = self.accept()
        if result is None:
            return
        sock, addr = result
        if self.logger:
            self.logger.info('Incoming connection from {0}'.format(
                repr(addr) or 'unix socket')
            )
        self.handler(sock, self)

    def handle_close(self) -> None:
        """Called when close
        """

        logging.info('Closing the socket, server will be shutdown now...')
        self.close()


class Checker(threading.Thread):
    """Check that the ST3 PID already exists every delta seconds
    """

    MAX_INACTIVITY: int = 1800  # 30 minutes in seconds

    def __init__(self, server: JSONServer, pid: int, delta: int = 5) -> None:
        threading.Thread.__init__(self)
        self.server: JSONServer = server
        self.delta: int = delta
        self.daemon: bool = True
        self.die: bool = False
        self.pid: int = int(pid)

    def run(self) -> None:
        while not self.die:
            if time.time() - self.server.last_call > self.MAX_INACTIVITY:
                # is now more than 30 minutes of inactivity
                if self.server.logger:
                    self.server.logger.info(
                        'detected inactivity for more than 30 minutes... '
                        'shuting down...'
                    )
                break

            self._check()
            if not self.die:
                time.sleep(self.delta)

        self.server.shutdown()

    if os.name == 'nt':
        def _isprocessrunning(self, timeout: int = MAX_INACTIVITY * 1000) -> bool:
            """Blocking until process has exited or timeout is reached.
            """
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            WAIT_TIMEOUT = 0x00000102
            hprocess = kernel32.OpenProcess(SYNCHRONIZE, False, self.pid)
            if hprocess == 0:
                return False
            ret = kernel32.WaitForSingleObject(hprocess, timeout)
            kernel32.CloseHandle(hprocess)
            return ret == WAIT_TIMEOUT
    else:
        def _isprocessrunning(self) -> bool:
            """Returning immediately whether process is running.
            """
            try:
                os.kill(self.pid, 0)
            except OSError:
                return False
            return True

    def _check(self) -> None:
        """Check for the ST3 pid
        """

        if not self._isprocessrunning():
            if self.server.logger:
                self.server.logger.info(
                    'process {0} does not exists stopping server...'.format(
                        self.pid
                    )
                )
            self.die = True


def get_logger(path: str) -> logging.Logger:
    """Build file logger
    """

    if not os.path.exists(path):
        os.makedirs(path)

    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    hdlr = handlers.RotatingFileHandler(
        filename=join(path, 'anaconda_jsonserver.log'),
        maxBytes=10000000,
        backupCount=5,
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s: %(message)s')
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    return log


def log_traceback() -> None:
    """Just log the traceback
    """

    logging.error(traceback.format_exc())


if __name__ == "__main__":

    WINDOWS: bool = os.name == 'nt'
    LINUX: bool = platform.system().lower() == 'linux'
    opt_parser = OptionParser(usage=(
        'usage: %prog -p <project> -e <extra_paths> port'
    )) if WINDOWS else OptionParser(usage=(
        "usage: %prog -p <project> -e <extra_paths> ST3_PID")
    )

    opt_parser.add_option(
        '-p', '--project', action='store', dest='project', help='project name'
    )

    opt_parser.add_option(
        '-e', '--extra_paths', action='store', dest='extra_paths',
        help='extra paths (separed by comma) that should be added to sys.paths'
    )

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
        jedi_settings.cache_directory = join(
            jedi_settings.cache_directory, options.project
        )
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
            server = JSONServer(('localhost', port or 0))
        else:
            unix_socket_path = UnixSocketPath(options.project)
            if not os.path.exists(dirname(unix_socket_path.socket)):
                os.makedirs(dirname(unix_socket_path.socket))
            if os.path.exists(unix_socket_path.socket):
                os.unlink(unix_socket_path.socket)
            server = JSONServer(unix_socket_path.socket)

        logger.info(
            'Anaconda Server started in {0} for '
            'PID {1} with cache dir {2}{3}'.format(
                port or unix_socket_path.socket, PID,
                jedi_settings.cache_directory,
                ' and extra paths {0}'.format(
                    options.extra_paths
                ) if options.extra_paths is not None else ''
            )
        )
    except Exception as error:
        log_traceback()
        logger.error(str(error))
        if server is not None:
            server.shutdown()
        sys.exit(-1)

    server.logger = logger

    # start PID checker thread
    if PID != 'DEBUG':
        checker = Checker(server, pid=int(PID), delta=1)
        checker.start()
    else:
        logger.info('Anaconda Server started in DEBUG mode...')
        print('DEBUG MODE')
        DEBUG_MODE = True
        set_debug_function(notices=True)

    # start the server
    server.serve_forever()
