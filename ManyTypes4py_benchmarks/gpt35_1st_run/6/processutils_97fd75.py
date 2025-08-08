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
from typing import IO, TYPE_CHECKING, Any, AnyStr, Callable, Optional, TextIO, Union, cast, overload
import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream, TextSendStream
from typing_extensions import TypeAlias, TypeVar

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

TextSink = Union[anyio.AsyncFile[AnyStr], TextIO, TextSendStream]
PrintFn = Callable[[str], object]
T = TypeVar('T', covariant=True)

if sys.platform == 'win32':
    from ctypes import WINFUNCTYPE, c_int, c_uint, windll
    _windows_process_group_pids = set()

    @WINFUNCTYPE(c_int, c_uint)
    def _win32_ctrl_handler(dwCtrlType: c_uint) -> c_int:
        ...

    @dataclass(eq=False)
    class StreamReaderWrapper(anyio.abc.ByteReceiveStream):

        async def receive(self, max_bytes: int = 65536) -> AnyStr:
            ...

        async def aclose(self) -> None:
            ...

    @dataclass(eq=False)
    class StreamWriterWrapper(anyio.abc.ByteSendStream):

        async def send(self, item: AnyStr) -> None:
            ...

        async def aclose(self) -> None:
            ...

    @dataclass(eq=False)
    class Process(anyio.abc.Process):

        async def aclose(self) -> None:
            ...

        async def wait(self) -> int:
            ...

        def terminate(self) -> None:
            ...

        def kill(self) -> None:
            ...

        def send_signal(self, signal: int) -> None:
            ...

        @property
        def pid(self) -> int:
            ...

        @property
        def returncode(self) -> Optional[int]:
            ...

        @property
        def stdin(self) -> Optional[StreamWriterWrapper]:
            ...

        @property
        def stdout(self) -> Optional[StreamReaderWrapper]:
            ...

        @property
        def stderr(self) -> Optional[StreamReaderWrapper]:
            ...

    async def _open_anyio_process(command: Union[str, List[str]], *, stdin: Optional[IO[AnyStr]] = None, stdout: Optional[IO[AnyStr]] = None, stderr: Optional[IO[AnyStr]] = None, cwd: Optional[str] = None, env: Optional[Mapping[str, str]] = None, start_new_session: bool = False, **kwargs: Any) -> Process:
        ...

@asynccontextmanager
async def open_process(command: Union[str, List[str]], **kwargs: Any) -> AsyncGenerator[Process, None]:
    ...

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = ..., task_status: Any, task_status_handler: Callable[[Process], T], **kwargs: Any) -> T:
    ...

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = ..., task_status: Any, task_status_handler: Optional[Callable[[Process], T]] = None, **kwargs: Any) -> T:
    ...

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = False, task_status: Optional[Any] = None, task_status_handler: Optional[Callable[[Process], T]] = None, **kwargs: Any) -> T:
    ...

async def run_process(command: Union[str, List[str]], *, stream_output: bool = False, task_status: Optional[Any] = None, task_status_handler: Optional[Callable[[Process], T]] = None, **kwargs: Any) -> Process:
    ...

async def consume_process_output(process: Process, stdout_sink: Optional[TextSink] = None, stderr_sink: Optional[TextSink] = None) -> None:
    ...

async def stream_text(source: TextReceiveStream, *sinks: TextSink) -> None:
    ...

def _register_signal(signum: int, handler: Callable) -> None:
    ...

def forward_signal_handler(pid: int, signum: int, *signums: int, process_name: str, print_fn: PrintFn) -> None:
    ...

def setup_signal_handlers_server(pid: int, process_name: str, print_fn: PrintFn) -> None:
    ...

def setup_signal_handlers_agent(pid: int, process_name: str, print_fn: PrintFn) -> None:
    ...

def setup_signal_handlers_worker(pid: int, process_name: str, print_fn: PrintFn) -> None:
    ...

def get_sys_executable() -> str:
    ...
