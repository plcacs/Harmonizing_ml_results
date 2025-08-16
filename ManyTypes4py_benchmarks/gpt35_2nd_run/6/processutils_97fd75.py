from typing import Any, Callable, IO, TextIO, Union, TypeVar, overload

TextSink = Union[anyio.AsyncFile[AnyStr], TextIO, TextSendStream]
PrintFn = Callable[[str], object]
T = TypeVar('T', covariant=True)

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = ..., task_status: Any = ..., task_status_handler: Any = ..., **kwargs: Any) -> Any:
    ...

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = ..., task_status: Any = ..., task_status_handler: Any = None, **kwargs: Any) -> Any:
    ...

@overload
async def run_process(command: Union[str, List[str]], *, stream_output: bool = False, task_status: Any = None, task_status_handler: Any = None, **kwargs: Any) -> Any:
    ...

async def run_process(command: Union[str, List[str]], *, stream_output: bool = False, task_status: Any = None, task_status_handler: Any = None, **kwargs: Any) -> Any:
    ...

async def consume_process_output(process: Process, stdout_sink: Optional[TextSink] = None, stderr_sink: Optional[TextSink] = None) -> None:
    ...

async def stream_text(source: TextReceiveStream, *sinks: TextSink) -> None:
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
