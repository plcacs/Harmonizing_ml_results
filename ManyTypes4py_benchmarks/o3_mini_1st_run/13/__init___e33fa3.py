#!/usr/bin/env python3
"""
Makes it possible to do the compiled analysis in a subprocess. This has two
goals:

1. Making it safer - Segfaults and RuntimeErrors as well as stdout/stderr can
   be ignored and dealt with.
2. Make it possible to handle different Python versions as well as virtualenvs.
"""

import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from typing import Any, Callable, Deque, Dict, IO, List, Optional, Tuple, Union

from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, SignatureParam
from jedi.api.exceptions import InternalError

_MAIN_PATH: str = os.path.join(os.path.dirname(__file__), '__main__.py')
PICKLE_PROTOCOL: int = 4


def _GeneralizedPopen(*args: Any, **kwargs: Any) -> subprocess.Popen:
    if os.name == 'nt':
        try:
            CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
        except AttributeError:
            CREATE_NO_WINDOW = 134217728
        kwargs['creationflags'] = CREATE_NO_WINDOW
    kwargs['close_fds'] = 'posix' in sys.builtin_module_names
    return subprocess.Popen(*args, **kwargs)


def _enqueue_output(out: IO[bytes], queue_: "queue.Queue[bytes]") -> None:
    for line in iter(out.readline, b''):
        queue_.put(line)


def _add_stderr_to_debug(stderr_queue: "queue.Queue[bytes]") -> None:
    while True:
        try:
            line: bytes = stderr_queue.get_nowait()
            line_decoded: str = line.decode('utf-8', 'replace')
            debug.warning('stderr output: %s' % line_decoded.rstrip('\n'))
        except queue.Empty:
            break


def _get_function(name: str) -> Any:
    return getattr(functions, name)


def _cleanup_process(process: subprocess.Popen, thread: Thread) -> None:
    try:
        process.kill()
        process.wait()
    except OSError:
        pass
    thread.join()
    for stream in [process.stdin, process.stdout, process.stderr]:
        try:
            stream.close()
        except OSError:
            pass


class _InferenceStateProcess:
    def __init__(self, inference_state: Any) -> None:
        self._inference_state_weakref: "weakref.ReferenceType[Any]" = weakref.ref(inference_state)
        self._inference_state_id: int = id(inference_state)
        self._handles: Dict[int, "AccessHandle"] = {}

    def get_or_create_access_handle(self, obj: Any) -> "AccessHandle":
        id_: int = id(obj)
        try:
            return self.get_access_handle(id_)
        except KeyError:
            access: Any = DirectObjectAccess(self._inference_state_weakref(), obj)
            handle: AccessHandle = AccessHandle(self, access, id_)
            self.set_access_handle(handle)
            return handle

    def get_access_handle(self, id_: int) -> "AccessHandle":
        return self._handles[id_]

    def set_access_handle(self, handle: "AccessHandle") -> None:
        self._handles[handle.id] = handle


class InferenceStateSameProcess(_InferenceStateProcess):
    """
    Basically just an easy access to functions.py. It has the same API
    as InferenceStateSubprocess and does the same thing without using a subprocess.
    This is necessary for the Interpreter process.
    """
    def __getattr__(self, name: str) -> Callable[..., Any]:
        return partial(_get_function(name), self._inference_state_weakref())


class InferenceStateSubprocess(_InferenceStateProcess):
    def __init__(self, inference_state: Any, compiled_subprocess: "CompiledSubprocess") -> None:
        super().__init__(inference_state)
        self._used: bool = False
        self._compiled_subprocess: CompiledSubprocess = compiled_subprocess

    def __getattr__(self, name: str) -> Callable[..., Any]:
        func: Callable[..., Any] = _get_function(name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._used = True
            result: Any = self._compiled_subprocess.run(self._inference_state_weakref(), func, args=args, kwargs=kwargs)
            return self._convert_access_handles(result)
        return wrapper

    def _convert_access_handles(self, obj: Any) -> Any:
        if isinstance(obj, SignatureParam):
            return SignatureParam(*self._convert_access_handles(tuple(obj)))
        elif isinstance(obj, tuple):
            return tuple((self._convert_access_handles(o) for o in obj))
        elif isinstance(obj, list):
            return [self._convert_access_handles(o) for o in obj]
        elif isinstance(obj, AccessHandle):
            try:
                obj = self.get_access_handle(obj.id)
            except KeyError:
                obj.add_subprocess(self)
                self.set_access_handle(obj)
        elif isinstance(obj, AccessPath):
            return AccessPath(self._convert_access_handles(obj.accesses))
        return obj

    def __del__(self) -> None:
        if self._used and (not self._compiled_subprocess.is_crashed):
            self._compiled_subprocess.delete_inference_state(self._inference_state_id)


class CompiledSubprocess:
    is_crashed: bool = False

    def __init__(self, executable: str, env_vars: Optional[Dict[str, str]] = None) -> None:
        self._executable: str = executable
        self._env_vars: Optional[Dict[str, str]] = env_vars
        self._inference_state_deletion_queue: Deque[int] = collections.deque()
        self._cleanup_callable: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        pid: int = os.getpid()
        return '<%s _executable=%r, is_crashed=%r, pid=%r>' % (self.__class__.__name__, self._executable, self.is_crashed, pid)

    @memoize_method
    def _get_process(self) -> subprocess.Popen:
        debug.dbg('Start environment subprocess %s', self._executable)
        parso_path: str = sys.modules['parso'].__file__  # type: ignore
        args: Tuple[Any, ...] = (self._executable, _MAIN_PATH, os.path.dirname(os.path.dirname(parso_path)),
                                  '.'.join((str(x) for x in sys.version_info[:3])))
        process: subprocess.Popen = _GeneralizedPopen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self._env_vars)
        self._stderr_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stderr_thread: Thread = t = Thread(target=_enqueue_output, args=(process.stderr, self._stderr_queue))
        t.daemon = True
        t.start()
        self._cleanup_callable = weakref.finalize(self, _cleanup_process, process, t)
        return process

    def run(self, inference_state: Any, function: Callable[..., Any], args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        while True:
            try:
                inference_state_id: int = self._inference_state_deletion_queue.pop()
            except IndexError:
                break
            else:
                self._send(inference_state_id, None)
        assert callable(function)
        return self._send(id(inference_state), function, args, kwargs)

    def get_sys_path(self) -> Any:
        return self._send(None, functions.get_sys_path, (), {})

    def _kill(self) -> None:
        self.is_crashed = True
        self._cleanup_callable()

    def _send(self, inference_state_id: Optional[int], function: Optional[Callable[..., Any]], args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Any:
        if self.is_crashed:
            raise InternalError('The subprocess %s has crashed.' % self._executable)
        data: Tuple[Optional[int], Optional[Callable[..., Any]], Tuple[Any, ...], Dict[str, Any]] = (inference_state_id, function, args, kwargs)
        try:
            pickle_dump(data, self._get_process().stdin, PICKLE_PROTOCOL)
        except BrokenPipeError:
            self._kill()
            raise InternalError('The subprocess %s was killed. Maybe out of memory?' % self._executable)
        try:
            is_exception, tb, result = pickle_load(self._get_process().stdout)
        except EOFError as eof_error:
            try:
                stderr: str = self._get_process().stderr.read().decode('utf-8', 'replace')
            except Exception as exc:
                stderr = '<empty/not available (%r)>' % exc
            self._kill()
            _add_stderr_to_debug(self._stderr_queue)
            raise InternalError('The subprocess %s has crashed (%r, stderr=%s).' % (self._executable, eof_error, stderr))
        _add_stderr_to_debug(self._stderr_queue)
        if is_exception:
            result.args = (tb,)
            raise result
        return result

    def delete_inference_state(self, inference_state_id: int) -> None:
        """
        Currently we are not deleting inference_state instantly. They only get
        deleted once the subprocess is used again. It would probably a better
        solution to move all of this into a thread. However, the memory usage
        of a single inference_state shouldn't be that high.
        """
        self._inference_state_deletion_queue.append(inference_state_id)


class Listener:
    def __init__(self) -> None:
        self._inference_states: Dict[int, Any] = {}
        self._process: _InferenceStateProcess = _InferenceStateProcess(Listener)

    def _get_inference_state(self, function: Callable[..., Any], inference_state_id: int) -> Any:
        from jedi.inference import InferenceState  # type: ignore
        try:
            inference_state: Any = self._inference_states[inference_state_id]
        except KeyError:
            from jedi import InterpreterEnvironment  # type: ignore
            inference_state = InferenceState(project=None, environment=InterpreterEnvironment())
            self._inference_states[inference_state_id] = inference_state
        return inference_state

    def _run(self, inference_state_id: Optional[int], function: Optional[Callable[..., Any]], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if inference_state_id is None:
            return function(*args, **kwargs)  # type: ignore
        elif function is None:
            del self._inference_states[inference_state_id]
        else:
            inference_state: Any = self._get_inference_state(function, inference_state_id)
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, AccessHandle):
                    args[i] = inference_state.compiled_subprocess.get_access_handle(arg.id)
            for key, value in kwargs.items():
                if isinstance(value, AccessHandle):
                    kwargs[key] = inference_state.compiled_subprocess.get_access_handle(value.id)
            return function(inference_state, *args, **kwargs)
        return None

    def listen(self) -> None:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        stdin = sys.stdin
        stdout_buffer = stdout.buffer  # type: IO[bytes]
        stdin_buffer = stdin.buffer  # type: IO[bytes]
        while True:
            try:
                payload: Any = pickle_load(stdin_buffer)
            except EOFError:
                exit(0)
            try:
                result: Tuple[bool, Optional[str], Any] = (False, None, self._run(*payload))
            except Exception as e:
                result = (True, traceback.format_exc(), e)
            pickle_dump(result, stdout_buffer, PICKLE_PROTOCOL)


class AccessHandle:
    def __init__(self, subprocess_obj: Any, access: Any, id_: int) -> None:
        self.access: Any = access
        self._subprocess: "CompiledSubprocess" = subprocess_obj  # type: ignore
        self.id: int = id_

    def add_subprocess(self, subprocess_obj: "CompiledSubprocess") -> None:
        self._subprocess = subprocess_obj

    def __repr__(self) -> str:
        try:
            detail: Any = self.access
        except AttributeError:
            detail = '#' + str(self.id)
        return '<%s of %s>' % (self.__class__.__name__, detail)

    def __getstate__(self) -> int:
        return self.id

    def __setstate__(self, state: int) -> None:
        self.id = state

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name in ('id', 'access') or name.startswith('_'):
            raise AttributeError('Something went wrong with unpickling')
        return partial(self._workaround, name)

    def _workaround(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        TODO Currently we're passing slice objects around. This should not
        happen. They are also the only unhashable objects that we're passing
        around.
        """
        if args and isinstance(args[0], slice):
            return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)  # type: ignore
        return self._cached_results(name, *args, **kwargs)

    @memoize_method
    def _cached_results(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)  # type: ignore
