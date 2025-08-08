import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, SignatureParam
from jedi.api.exceptions import InternalError

_MAIN_PATH: str = os.path.join(os.path.dirname(__file__), '__main__.py')
PICKLE_PROTOCOL: int = 4

def _GeneralizedPopen(*args, **kwargs) -> subprocess.Popen:
    ...

def _enqueue_output(out: subprocess.Popen.stdout, queue_: queue.Queue) -> None:
    ...

def _add_stderr_to_debug(stderr_queue: queue.Queue) -> None:
    ...

def _get_function(name: str):
    ...

def _cleanup_process(process: subprocess.Popen, thread: Thread) -> None:
    ...

class _InferenceStateProcess:
    ...

class InferenceStateSameProcess(_InferenceStateProcess):
    ...

class InferenceStateSubprocess(_InferenceStateProcess):
    ...

class CompiledSubprocess:
    ...

class Listener:
    ...

class AccessHandle:
    ...
