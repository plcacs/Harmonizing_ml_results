import warnings
import threading
import queue
import numpy as np
from nevergrad.common.errors import NevergradError
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter

class StopOptimizerThread(Exception):
    pass

class TooManyAskError(NevergradError):
    pass

class _MessagingThread(threading.Thread):
    def __init__(self, caller, *args, **kwargs):
        super().__init__()
        self.messages_ask: queue.Queue = queue.Queue()
        self.messages_tell: queue.Queue = queue.Queue()
        self.call_count: int = 0
        self.error: Exception = None
        self._caller = caller
        self._args = args
        self._kwargs = kwargs
        self.output = None

    def run(self) -> None:
        ...

    def _fake_callable(self, *args) -> np.ndarray:
        ...

    def stop(self) -> None:
        ...

class MessagingThread:
    def __init__(self, caller, *args, **kwargs):
        ...

    def is_alive(self) -> bool:
        ...

    @property
    def output(self):
        ...

    @property
    def error(self):
        ...

    @property
    def messages_tell(self):
        ...

    @property
    def messages_ask(self):
        ...

    def stop(self) -> None:
        ...

    def __del__(self) -> None:
        ...

class RecastOptimizer(base.Optimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1):
        ...

    def get_optimization_function(self) -> tp.Callable:
        ...

    def _check_error(self) -> None:
        ...

    def _post_loss(self, candidate: p.Parameter, loss: float) -> float:
        ...

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        ...

    def _internal_provide_recommendation(self) -> tp.Optional[p.Parameter]:
        ...

    def __del__(self) -> None:
        ...

class SequentialRecastOptimizer(RecastOptimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int], num_workers: int = 1):
        ...

    def enable_pickling(self) -> None:
        ...

    def _internal_ask_candidate(self) -> p.Parameter:
        ...

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        ...

    def __getstate__(self) -> dict:
        ...

    def __setstate__(self, state: dict) -> None:
        ...

class BatchRecastOptimizer(RecastOptimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1):
        ...

    def _internal_ask_candidate(self) -> p.Parameter:
        ...

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        ...

    def minimize(self, objective_function, executor=None, batch_mode=False, verbosity=0, constraint_violation=None, max_time=None) -> None:
        ...

    def can_ask(self) -> bool:
        ...
