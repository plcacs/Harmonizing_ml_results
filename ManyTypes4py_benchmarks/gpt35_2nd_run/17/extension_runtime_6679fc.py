from __future__ import annotations
import asyncio
import logging
import signal
import sys
from collections import deque
from time import time
from typing import Callable, Optional, Tuple, Union
if sys.version_info >= (3, 8):
    from typing import Literal
    ExtensionRuntimeError: Union[Literal['Terminated', 'Exited', 'MissingModule', 'MissingInternals', 'Incompatible', 'Invalid'], str]
else:
    ExtensionRuntimeError: str
from gi.repository import Gio, GLib
logger: logging.Logger = logging.getLogger()
ErrorHandlerCallback: Callable[[ExtensionRuntimeError, str], None]

class ExtensionRuntime:

    def __init__(self, ext_id: str, cmd: List[str], env: Optional[Dict[str, str]] = None, error_handler: Optional[ErrorHandlerCallback] = None) -> None:
        self.ext_id: str = ext_id
        self.error_handler: Optional[ErrorHandlerCallback] = error_handler
        self.recent_errors: deque[str] = deque(maxlen=1)
        self.start_time: float = time()
        launcher: Gio.SubprocessLauncher = Gio.SubprocessLauncher.new(Gio.SubprocessFlags.STDERR_PIPE)
        for env_name, env_value in (env or {}).items():
            launcher.setenv(env_name, env_value, True)
        self.subprocess: Gio.Subprocess = launcher.spawnv(cmd)
        error_input_stream: Gio.InputStream = self.subprocess.get_stderr_pipe()
        if not error_input_stream:
            err_msg: str = 'Subprocess must be created with Gio.SubprocessFlags.STDERR_PIPE'
            raise AssertionError(err_msg)
        self.error_stream: Gio.DataInputStream = Gio.DataInputStream.new(error_input_stream)
        logger.debug('Launched %s using Gio.Subprocess', self.ext_id)
        self.subprocess.wait_async(None, self.handle_exit)
        self.read_stderr_line()

    async def stop(self) -> None:
        """
        Terminates extension
        """
        logger.info('Terminating extension "%s"', self.ext_id)
        self.subprocess.send_signal(signal.SIGTERM)
        await asyncio.sleep(0.5)
        if self.subprocess.get_identifier():
            logger.info('Extension %s still running, sending SIGKILL', self.ext_id)
            self.subprocess.send_signal(signal.SIGKILL)

    def read_stderr_line(self) -> None:
        self.error_stream.read_line_async(GLib.PRIORITY_DEFAULT, None, self.handle_stderr)

    def handle_stderr(self, error_stream: Gio.DataInputStream, result: GLib.AsyncResult) -> None:
        output, _ = error_stream.read_line_finish_utf8(result)
        if output:
            print(output)
            self.recent_errors.append(output)
            self.read_stderr_line()

    def handle_exit(self, _subprocess: Gio.Subprocess, _result: GLib.AsyncResult) -> None:
        error_type, error_msg = self.extract_error()
        logger.error(error_msg)
        if self.error_handler:
            self.error_handler(error_type, error_msg)

    def extract_error(self) -> Tuple[str, str]:
        if self.subprocess.get_if_signaled():
            kill_signal: int = self.subprocess.get_term_sig()
            return ('Terminated', f'Extension "{self.ext_id}" was terminated with signal {kill_signal}')
        uptime_seconds: float = time() - self.start_time
        code: int = self.subprocess.get_exit_status()
        error_msg: str = '\n'.join(self.recent_errors)
        logger.error('Extension "%s" exited with an error: %s', self.ext_id, error_msg)
        if 'ModuleNotFoundError' in error_msg:
            package_name: str = error_msg.split("'")[1].split('.')[0]
            if package_name == 'ulauncher':
                return ('MissingInternals', error_msg)
            if package_name:
                return ('MissingModule', package_name)
        if uptime_seconds < 1:
            return ('Terminated', error_msg)
        error_msg = f'Extension "{self.ext_id}" exited with code {code} after {uptime_seconds} seconds.'
        return ('Exited', error_msg)
