"""
Makes it possible to do the compiled analysis in a subprocess. This has two
goals:

1. Making it safer - Segfaults and RuntimeErrors as well as stdout/stderr can
   be ignored and dealt with.
2. Make it possible to handle different Python versions as well as virtualenvs.
"""

import subprocess
import io
import queue
import functools
import weakref
from jedi._compatibility import pickle_dump, pickle_load
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, SignatureParam
from jedi.api.exceptions import InternalError
from jedi.inference import InferenceState

__all__ = [
    '_GeneralizedPopen', '_enqueue_output', '_add_stderr_to_debug',
    '_get_function', '_cleanup_process', '_InferenceStateProcess',
    'InferenceStateSameProcess', 'InferenceStateSubprocess', 'CompiledSubprocess',
    'Listener', 'AccessHandle'
]

def _GeneralizedPopen(*args: Any, **kwargs: Any) -> subprocess.Popen:
    ...

def _enqueue_output(out: io.BufferedReader, queue_: queue.Queue) -> None:
    ...

def _add_stderr_to_debug(stderr_queue: queue.Queue) -> None:
    ...

def _get_function(name: str) -> functools.partial:
    ...

class _InferenceStateProcess:
    def __init__(self, inference_state: weakref.ref) -> None:
        ...

    def get_or_create_access_handle(self, obj: Any) -> AccessHandle:
        ...

    def get_access_handle(self, id_: int) -> AccessHandle:
        ...

    def set_access_handle(self, handle: AccessHandle) -> None:
        ...

class InferenceStateSameProcess(_InferenceStateProcess):
    def __getattr__(self, name: str) -> functools.partial:
        ...

class InferenceStateSubprocess(_InferenceStateProcess):
    def __init__(self, inference_state: weakref.ref, compiled_subprocess: 'CompiledSubprocess') -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def _convert_access_handles(self, obj: Any) -> Any:
        ...

    def __del__(self) -> None:
        ...

class CompiledSubprocess:
    is_crashed: bool

    def __init__(self, executable: str, env_vars: dict | None = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @memoize_method
    def _get_process(self) -> subprocess.Popen:
        ...

    def run(self, inference_state: weakref.ref, function: callable, args: tuple = (), kwargs: dict = {}) -> Any:
        ...

    def get_sys_path(self) -> list[str]:
        ...

    def _kill(self) -> None:
        ...

    def _send(self, inference_state_id: int | None, function: callable | None, args: tuple = (), kwargs: dict = {}) -> Any:
        ...

    def delete_inference_state(self, inference_state_id: int) -> None:
        ...

class Listener:
    def __init__(self) -> None:
        ...

    def _get_inference_state(self, function: callable, inference_state_id: int) -> InferenceState:
        ...

    def _run(self, inference_state_id: int | None, function: callable | None, args: tuple, kwargs: dict) -> Any:
        ...

    def listen(self) -> None:
        ...

class AccessHandle:
    def __init__(self, subprocess: _InferenceStateProcess, access: DirectObjectAccess, id_: int) -> None:
        ...

    def add_subprocess(self, subprocess: _InferenceStateProcess) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> int:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __getattr__(self, name: str) -> functools.partial:
        ...

    def _workaround(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...

    @memoize_method
    def _cached_results(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...