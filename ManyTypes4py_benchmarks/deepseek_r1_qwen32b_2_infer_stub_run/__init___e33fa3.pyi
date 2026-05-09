import subprocess
import queue
import weakref
import collections
import os
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    AnyStr,
    Optional,
    Type,
    TypeVar,
    overload,
    Dict,
    TextIO,
    Iterable,
    Iterator,
    Partial,
    Protocol,
    runtime_checkable,
)

_MAIN_PATH: str = ...
PICKLE_PROTOCOL: int = ...

def _GeneralizedPopen(*args: Any, **kwargs: Any) -> subprocess.Popen:
    ...

def _enqueue_output(out: TextIO, queue_: queue.Queue) -> None:
    ...

def _add_stderr_to_debug(stderr_queue: queue.Queue) -> None:
    ...

def _get_function(name: str) -> Callable:
    ...

class _InferenceStateProcess:
    def __init__(self, inference_state: weakref.ref) -> None:
        ...

    def get_or_create_access_handle(self, obj: Any) -> Any:
        ...

    def get_access_handle(self, id_: int) -> Any:
        ...

    def set_access_handle(self, handle: Any) -> None:
        ...

class InferenceStateSameProcess(_InferenceStateProcess):
    def __getattr__(self, name: str) -> Partial:
        ...

class InferenceStateSubprocess(_InferenceStateProcess):
    def __init__(self, inference_state: weakref.ref, compiled_subprocess: 'CompiledSubprocess') -> None:
        ...

    def __getattr__(self, name: str) -> Callable:
        ...

    def _convert_access_handles(self, obj: Any) -> Any:
        ...

    def __del__(self) -> None:
        ...

class CompiledSubprocess:
    is_crashed: bool

    def __init__(self, executable: str, env_vars: Optional[Dict[str, str]] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def _get_process(self) -> subprocess.Popen:
        ...

    def run(self, inference_state: weakref.ref, function: Callable, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        ...

    def get_sys_path(self) -> List[str]:
        ...

    def _kill(self) -> None:
        ...

    def _send(self, inference_state_id: int, function: Callable, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        ...

    def delete_inference_state(self, inference_state_id: int) -> None:
        ...

class Listener:
    def __init__(self) -> None:
        ...

    def _get_inference_state(self, function: Callable, inference_state_id: int) -> Any:
        ...

    def _run(self, inference_state_id: int, function: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        ...

    def listen(self) -> None:
        ...

class AccessHandle:
    def __init__(self, subprocess: CompiledSubprocess, access: Any, id_: int) -> None:
        ...

    def add_subprocess(self, subprocess: CompiledSubprocess) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> int:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __getattr__(self, name: str) -> Partial:
        ...

    def _workaround(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def _cached_results(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...