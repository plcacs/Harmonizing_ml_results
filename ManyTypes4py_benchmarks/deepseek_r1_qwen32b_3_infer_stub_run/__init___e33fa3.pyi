"""
Stub file for __init___e33fa3 module
"""

import collections
import os
import subprocess
import weakref
from threading import Thread
from queue import Queue
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
)
from jedi.inference import InferenceState

Process = subprocess.Popen

class _InferenceStateProcess:
    _handles: Dict[int, Any]
    _inference_state_weakref: weakref.ref[Any]

    def __init__(self, inference_state: weakref.ref[Any]) -> None:
        ...

    def get_or_create_access_handle(self, obj: Any) -> Any:
        ...

    def get_access_handle(self, id_: int) -> Any:
        ...

    def set_access_handle(self, handle: Any) -> None:
        ...

class InferenceStateSameProcess(_InferenceStateProcess):
    def __getattr__(self, name: str) -> Callable:
        ...

class InferenceStateSubprocess(_InferenceStateProcess):
    _used: bool
    _compiled_subprocess: 'CompiledSubprocess'

    def __init__(self, inference_state: weakref.ref[Any], compiled_subprocess: 'CompiledSubprocess') -> None:
        ...

    def __getattr__(self, name: str) -> Callable:
        ...

    def _convert_access_handles(self, obj: Any) -> Any:
        ...

    def __del__(self) -> None:
        ...

class CompiledSubprocess:
    is_crashed: bool
    _inference_state_deletion_queue: collections.deque[int]

    def __init__(self, executable: str, env_vars: Optional[Dict[str, str]] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def _get_process(self) -> Process:
        ...

    def run(self, inference_state: weakref.ref[Any], function: Callable, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        ...

    def get_sys_path(self) -> List[str]:
        ...

    def _kill(self) -> None:
        ...

    def _send(self, inference_state_id: Optional[int], function: Optional[Callable], args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        ...

    def delete_inference_state(self, inference_state_id: int) -> None:
        ...

class Listener:
    _inference_states: Dict[int, InferenceState]

    def __init__(self) -> None:
        ...

    def _get_inference_state(self, function: Callable, inference_state_id: int) -> InferenceState:
        ...

    def _run(self, inference_state_id: Optional[int], function: Optional[Callable], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        ...

    def listen(self) -> None:
        ...

class AccessHandle:
    access: Any
    _subprocess: _InferenceStateProcess
    id: int

    def __init__(self, subprocess: _InferenceStateProcess, access: Any, id_: int) -> None:
        ...

    def add_subprocess(self, subprocess: _InferenceStateProcess) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> int:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __getattr__(self, name: str) -> Callable:
        ...

    def _workaround(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def _cached_results(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...

def _GeneralizedPopen(*args: Any, **kwargs: Any) -> Process:
    ...

def _enqueue_output(out: Any, queue_: Queue[bytes]) -> None:
    ...

def _add_stderr_to_debug(stderr_queue: Queue[bytes]) -> None:
    ...

def _get_function(name: str) -> Callable:
    ...

def _cleanup_process(process: Process, thread: Thread) -> None:
    ...