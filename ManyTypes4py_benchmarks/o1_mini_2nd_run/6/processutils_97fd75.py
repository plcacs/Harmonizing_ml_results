import asyncio
import os
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from types import FrameType
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Optional,
    TextIO,
    Union,
    cast,
    overload,
)
import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream, TextSendStream
from typing_extensions import TypeAlias, TypeVar

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

TextSink: TypeAlias = Union[anyio.AsyncFile[AnyStr], TextIO, TextSendStream]
PrintFn: TypeAlias = Callable[[str], Any]
T = TypeVar("T", bound=Any)

if sys.platform == "win32":
    from ctypes import WINFUNCTYPE, c_int, c_uint, windll

    _windows_process_group_pids: set[int] = set()

    @WINFUNCTYPE(c_int, c_uint)
    def _win32_ctrl_handler(dwCtrlType: int) -> int:
        """
        A callback function for handling CTRL events cleanly on Windows. When called,
        this function will terminate all running win32 subprocesses the current
        process started in new process groups.
        """
        for pid in _windows_process_group_pids:
            try:
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            except OSError:
                pass
        return 0

    @dataclass(eq=False)
    class StreamReaderWrapper(anyio.abc.ByteReceiveStream):
        _stream: asyncio.StreamReader

        async def receive(self, max_bytes: int = 65536) -> bytes:
            data = await self._stream.read(max_bytes)
            if data:
                return data
            else:
                raise anyio.EndOfStream

        async def aclose(self) -> None:
            self._stream.feed_eof()

    @dataclass(eq=False)
    class StreamWriterWrapper(anyio.abc.ByteSendStream):
        _stream: asyncio.StreamWriter

        async def send(self, item: bytes) -> None:
            self._stream.write(item)
            await self._stream.drain()

        async def aclose(self) -> None:
            self._stream.close()
            await self._stream.wait_closed()

    @dataclass(eq=False)
    class Process(anyio.abc.Process):
        _process: asyncio.subprocess.Process
        _stdin: Optional[StreamWriterWrapper]
        _stdout: Optional[StreamReaderWrapper]
        _stderr: Optional[StreamReaderWrapper]

        async def aclose(self) -> None:
            if self._stdin:
                await self._stdin.aclose()
            if self._stdout:
                await self._stdout.aclose()
            if self._stderr:
                await self._stderr.aclose()
            await self.wait()

        async def wait(self) -> int:
            return await self._process.wait()

        def terminate(self) -> None:
            self._process.terminate()

        def kill(self) -> None:
            self._process.kill()

        def send_signal(self, sig: signal.Signals) -> None:
            self._process.send_signal(sig)

        @property
        def pid(self) -> int:
            return self._process.pid

        @property
        def returncode(self) -> Optional[int]:
            return self._process.returncode

        @property
        def stdin(self) -> Optional[StreamWriterWrapper]:
            return self._stdin

        @property
        def stdout(self) -> Optional[StreamReaderWrapper]:
            return self._stdout

        @property
        def stderr(self) -> Optional[StreamReaderWrapper]:
            return self._stderr

    async def _open_anyio_process(
        command: Union[str, list[str]],
        *,
        stdin: Optional[int] = None,
        stdout: Optional[int] = None,
        stderr: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        start_new_session: bool = False,
        **kwargs: Any,
    ) -> Process:
        """
        Open a subprocess and return a `Process` object.

        Args:
            command: The command to run
            kwargs: Additional arguments to pass to `asyncio.create_subprocess_exec`

        Returns:
            A `Process` object
        """
        if isinstance(command, list):
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
                start_new_session=start_new_session,
                **kwargs,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
                start_new_session=start_new_session,
                **kwargs,
            )
        return Process(
            _process=process,
            _stdin=StreamWriterWrapper(process.stdin) if process.stdin else None,
            _stdout=StreamReaderWrapper(process.stdout) if process.stdout else None,
            _stderr=StreamReaderWrapper(process.stderr) if process.stderr else None,
        )

@asynccontextmanager
async def open_process(
    command: Union[list[str], str],
    **kwargs: Any,
) -> AsyncGenerator[anyio.abc.Process, None]:
    """
    Like `anyio.open_process` but with:
    - Support for Windows command joining
    - Termination of the process on exception during yield
    - Forced cleanup of process resources during cancellation
    """
    if not TYPE_CHECKING:
        if not isinstance(command, list):
            raise TypeError(
                f"The command passed to open process must be a list. You passed the command'{command}', which is type '{type(command)}'."
            )
    if sys.platform == "win32":
        if isinstance(command, list):
            command_str = " ".join(command)
        else:
            command_str = cast(str, command)
        process = await _open_anyio_process(command_str, **kwargs)
    else:
        process = await anyio.open_process(command, **kwargs)
    win32_process_group = False
    if (
        sys.platform == "win32"
        and "creationflags" in kwargs
        and kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    ):
        win32_process_group = True
        _windows_process_group_pids.add(process.pid)
        windll.kernel32.SetConsoleCtrlHandler(_win32_ctrl_handler, 1)
    try:
        async with process:
            yield process
    finally:
        try:
            process.terminate()
            if sys.platform == "win32" and win32_process_group:
                _windows_process_group_pids.remove(process.pid)
        except OSError:
            pass
        async with anyio.CancelScope(shield=True):
            await process.aclose()

@overload
async def run_process(
    command: Union[list[str], str],
    *,
    stream_output: Union[bool, tuple[TextIO, TextIO]] = ...,
    task_status: Optional[anyio.typing.TaskStatus] = ...,
    task_status_handler: Optional[Callable[[anyio.abc.Process], T]] = ...,
    **kwargs: Any,
) -> Optional[T]:
    ...

@overload
async def run_process(
    command: Union[list[str], str],
    *,
    stream_output: Union[bool, tuple[TextIO, TextIO]] = False,
    task_status: None = ...,
    task_status_handler: None = ...,
    **kwargs: Any,
) -> Optional[anyio.abc.Process]:
    ...

async def run_process(
    command: Union[list[str], str],
    *,
    stream_output: Union[bool, tuple[TextIO, TextIO]] = False,
    task_status: Optional[anyio.typing.TaskStatus] = None,
    task_status_handler: Optional[Callable[[anyio.abc.Process], T]] = None,
    **kwargs: Any,
) -> Optional[Union[T, anyio.abc.Process]]:
    """
    Like `anyio.run_process` but with:

    - Use of our `open_process` utility to ensure resources are cleaned up
    - Simple `stream_output` support to connect the subprocess to the parent stdout/err
    - Support for submission with `TaskGroup.start` marking as 'started' after the
        process has been created. When used, the PID is returned to the task status.

    """
    if stream_output is True:
        stream_output_cast: Union[bool, tuple[TextIO, TextIO]] = (sys.stdout, sys.stderr)
    else:
        stream_output_cast = stream_output
    async with open_process(
        command,
        stdout=subprocess.PIPE if stream_output_cast else subprocess.DEVNULL,
        stderr=subprocess.PIPE if stream_output_cast else subprocess.DEVNULL,
        **kwargs,
    ) as process:
        if task_status is not None:
            if task_status_handler:
                value: T = task_status_handler(process)
            else:
                value = cast(T, process.pid)
            task_status.started(value)
        if stream_output_cast:
            await consume_process_output(
                process,
                stdout_sink=cast(tuple[TextIO, TextIO], stream_output_cast)[0]
                if isinstance(stream_output_cast, tuple)
                else sys.stdout,
                stderr_sink=cast(tuple[TextIO, TextIO], stream_output_cast)[1]
                if isinstance(stream_output_cast, tuple)
                else sys.stderr,
            )
        await process.wait()
    return process.returncode

async def consume_process_output(
    process: anyio.abc.Process,
    stdout_sink: Optional[TextIO] = None,
    stderr_sink: Optional[TextIO] = None,
) -> None:
    async with anyio.create_task_group() as tg:
        if process.stdout is not None:
            tg.start_soon(stream_text, TextReceiveStream(process.stdout), stdout_sink)
        if process.stderr is not None:
            tg.start_soon(stream_text, TextReceiveStream(process.stderr), stderr_sink)

async def stream_text(
    source: TextReceiveStream,
    *sinks: Optional[TextSink],
) -> None:
    wrapped_sinks: list[Union[anyio.AsyncFile[AnyStr], TextSendStream]] = [
        anyio.wrap_file(cast(IO[str], sink))
        if isinstance(sink, IO) and hasattr(sink, "write") and hasattr(sink, "flush")
        else sink
        for sink in sinks
        if sink is not None
    ]
    async for item in source:
        for sink in wrapped_sinks:
            if isinstance(sink, TextSendStream):
                await sink.send(item)
            elif isinstance(sink, anyio.AsyncFile):
                await sink.write(item)
                await sink.flush()

def _register_signal(signum: signal.Signals, handler: Callable[..., None]) -> None:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signum, handler)

def forward_signal_handler(
    pid: int,
    signum: signal.Signals,
    *signums: signal.Signals,
    process_name: str,
    print_fn: PrintFn,
) -> None:
    """Forward subsequent signum events (e.g. interrupts) to respective signums."""
    if not signums:
        raise ValueError("At least one signal must be provided")

    current_signal: signal.Signals = signums[0]
    future_signums: tuple[signal.Signals, ...] = signums[1:]
    original_handler: Optional[Callable[..., Any]] = None
    avoid_infinite_recursion: bool = signum == current_signal and pid == os.getpid()
    if avoid_infinite_recursion:
        original_handler = signal.getsignal(current_signal)

    def handler(*args: Any, **kwargs: Any) -> None:
        signal_name = getattr(signum, "name", str(signum))
        current_signal_name = getattr(current_signal, "name", str(current_signal))
        print_fn(
            f"Received {signal_name}. Sending {current_signal_name} to {process_name} (PID {pid})..."
        )
        if avoid_infinite_recursion and original_handler:
            signal.signal(current_signal, original_handler)
        os.kill(pid, current_signal)
        if future_signums:
            forward_signal_handler(
                pid,
                *future_signums,
                process_name=process_name,
                print_fn=print_fn,
            )

    _register_signal(signum, handler)

def setup_signal_handlers_server(
    pid: int,
    process_name: str,
    print_fn: PrintFn,
) -> None:
    """Handle interrupts of the server gracefully."""
    setup_handler = partial(
        forward_signal_handler, pid, process_name=process_name, print_fn=print_fn
    )
    if sys.platform == "win32":
        setup_handler(signal.SIGINT, signal.CTRL_BREAK_EVENT)
    else:
        setup_handler(signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
        setup_handler(signal.SIGTERM, signal.SIGTERM, signal.SIGKILL)

def setup_signal_handlers_agent(
    pid: int,
    process_name: str,
    print_fn: PrintFn,
) -> None:
    """Handle interrupts of the agent gracefully."""
    setup_handler = partial(
        forward_signal_handler, pid, process_name=process_name, print_fn=print_fn
    )
    if sys.platform == "win32":
        setup_handler(signal.SIGINT, signal.CTRL_BREAK_EVENT)
    else:
        setup_handler(signal.SIGINT, signal.SIGINT, signal.SIGKILL)
        setup_handler(signal.SIGTERM, signal.SIGINT, signal.SIGKILL)

def setup_signal_handlers_worker(
    pid: int,
    process_name: str,
    print_fn: PrintFn,
) -> None:
    """Handle interrupts of workers gracefully."""
    setup_handler = partial(
        forward_signal_handler, pid, process_name=process_name, print_fn=print_fn
    )
    if sys.platform == "win32":
        setup_handler(signal.SIGINT, signal.CTRL_BREAK_EVENT)
    else:
        setup_handler(signal.SIGINT, signal.SIGINT, signal.SIGKILL)
        setup_handler(signal.SIGTERM, signal.SIGINT, signal.SIGKILL)

def get_sys_executable() -> str:
    if os.name == "nt":
        executable_path = f'"{sys.executable}"'
    else:
        executable_path = sys.executable
    return executable_path
