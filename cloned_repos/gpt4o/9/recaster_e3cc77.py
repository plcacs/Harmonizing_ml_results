import warnings
import threading
import queue
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common.errors import NevergradError
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter

class StopOptimizerThread(Exception):
    pass

class TooManyAskError(NevergradError):
    pass

class _MessagingThread(threading.Thread):
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.messages_ask: queue.Queue = queue.Queue()
        self.messages_tell: queue.Queue = queue.Queue()
        self.call_count: int = 0
        self.error: tp.Optional[Exception] = None
        self._caller: tp.Callable[..., tp.Any] = caller
        self._args: tp.Tuple[tp.Any, ...] = args
        self._kwargs: tp.Dict[str, tp.Any] = kwargs
        self.output: tp.Any = None

    def run(self) -> None:
        try:
            self.output = self._caller(self._fake_callable, *self._args, **self._kwargs)
        except StopOptimizerThread:
            self.messages_ask.put(ValueError('Optimization told to finish'))
        except Exception as e:
            self.messages_ask.put(e)
            self.error = e
        else:
            self.messages_ask.put(None)

    def _fake_callable(self, *args: tp.Any) -> tp.Any:
        self.call_count += 1
        self.messages_ask.put(args[0])
        candidate = self.messages_tell.get()
        if candidate is None:
            raise StopOptimizerThread()
        return candidate

    def stop(self) -> None:
        self.messages_tell.put(None)

class MessagingThread:
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        self._thread: _MessagingThread = _MessagingThread(caller, *args, **kwargs)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def output(self) -> tp.Any:
        return self._thread.output

    @property
    def error(self) -> tp.Optional[Exception]:
        return self._thread.error

    @property
    def messages_tell(self) -> queue.Queue:
        return self._thread.messages_tell

    @property
    def messages_ask(self) -> queue.Queue:
        return self._thread.messages_ask

    def stop(self) -> None:
        self._thread.stop()

    def __del__(self) -> None:
        self.stop()

class RecastOptimizer(base.Optimizer):
    recast: bool = True

    def __init__(self, parametrization: tp.Any, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._messaging_thread: tp.Optional[MessagingThread] = None

    def get_optimization_function(self) -> tp.Callable[..., tp.Any]:
        raise NotImplementedError('You should define your optimizer! Also, be very careful to avoid reference to this instance in the returned object')

    def _check_error(self) -> None:
        if self._messaging_thread is not None:
            if self._messaging_thread.error is not None:
                raise RuntimeError(f'Recast optimizer raised an error:\n{self._messaging_thread.error}') from self._messaging_thread.error

    def _post_loss(self, candidate: tp.Any, loss: float) -> float:
        return loss

    def _internal_tell_not_asked(self, candidate: tp.Any, loss: float) -> None:
        raise base.errors.TellNotAskedNotSupportedError

    def _internal_provide_recommendation(self) -> tp.Optional[tp.Any]:
        if self._messaging_thread is not None and self._messaging_thread.output is not None:
            return self._messaging_thread.output
        else:
            return None

    def __del__(self) -> None:
        if self._messaging_thread is not None:
            self._messaging_thread.stop()

class SequentialRecastOptimizer(RecastOptimizer):
    no_parallelization: bool = True

    def __init__(self, parametrization: tp.Any, budget: int, num_workers: int = 1) -> None:
        super().__init__(parametrization=parametrization, budget=budget, num_workers=num_workers)
        self._enable_pickling: bool = False
        self.replay_archive_tell: tp.List[tp.Any] = []

    def enable_pickling(self) -> None:
        if self.num_ask != 0:
            raise ValueError('Can only enable pickling before all\xa0asks.')
        self._enable_pickling = True

    def _internal_ask_candidate(self) -> tp.Any:
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        alive = self._messaging_thread.is_alive()
        if alive:
            point = self._messaging_thread.messages_ask.get()
            if isinstance(point, Exception):
                raise point
        if not alive or point is None:
            warnings.warn('Underlying optimizer has already converged, returning random points', base.errors.FinishedUnderlyingOptimizerWarning)
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        candidate = self.parametrization.spawn_child().set_standardized_data(point)
        return candidate

    def _internal_tell_candidate(self, candidate: tp.Any, loss: float) -> None:
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():
            self._check_error()
            return
        if self._enable_pickling:
            self.replay_archive_tell.append(candidate)
        self._messaging_thread.messages_tell.put(self._post_loss(candidate, loss))

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        if not self._enable_pickling:
            raise ValueError('If you want picklability you should have asked for it')
        thread = self._messaging_thread
        self._messaging_thread = None
        state = self.__dict__.copy()
        self._messaging_thread = thread
        return state

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        self.__dict__.update(state)
        if not self._enable_pickling:
            raise ValueError('Cannot unpickle the unpicklable')
        self._enable_pickling = False
        for i, candidate in enumerate(self.replay_archive_tell):
            new_candidate = self._internal_ask_candidate()
            norm = np.linalg.norm(new_candidate.get_standardized_data(reference=candidate))
            if norm > 1e-05:
                raise RuntimeError(f'Mismatch in replay at index {i} of {len(self.replay_archive_tell)}.')
            self._internal_tell_candidate(candidate, candidate.loss)
        if self.num_ask > self.num_tell:
            self._internal_ask_candidate()
        self._enable_pickling = True

class BatchRecastOptimizer(RecastOptimizer):
    def __init__(self, parametrization: tp.Any, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._current_batch: tp.List[tp.Any] = []
        self._batch_losses: tp.List[tp.Optional[float]] = []
        self._tell_counter: int = 0
        self.batch_size: int = 0
        self.indices: tp.Dict[str, int] = {}

    def _internal_ask_candidate(self) -> tp.Any:
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        if self._current_batch:
            return self._current_batch.pop()
        if not self.can_ask():
            raise TooManyAskError("You can't get a new batch until the old one has been fully told on. See docstring for more info.")
        alive = self._messaging_thread.is_alive()
        if alive:
            points = self._messaging_thread.messages_ask.get()
            if isinstance(points, Exception):
                raise points
        if not alive or points is None:
            warnings.warn('Underlying optimizer has already converged, returning random points', base.errors.FinishedUnderlyingOptimizerWarning)
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        self.batch_size = len(points)
        self._current_batch = [self.parametrization.spawn_child().set_standardized_data(point) for point in points]
        self._batch_losses = [None] * len(points)
        self.indices = {candidate.uid: i for i, candidate in enumerate(self._current_batch)}
        return self._current_batch.pop()

    def _internal_tell_candidate(self, candidate: tp.Any, loss: float) -> None:
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():
            self._check_error()
            return
        candidate_index = self.indices.pop(candidate.uid)
        self._batch_losses[candidate_index] = self._post_loss(candidate, loss)
        self._tell_counter += 1
        if self._tell_counter == self.batch_size:
            self._messaging_thread.messages_tell.put(np.array(self._batch_losses))
            self._batch_losses = []
            self._tell_counter = 0

    def minimize(self, objective_function: tp.Callable[..., float], executor: tp.Optional[tp.Any] = None, batch_mode: bool = False, verbosity: int = 0, constraint_violation: tp.Optional[tp.Callable[..., float]] = None, max_time: tp.Optional[float] = None) -> None:
        raise NotImplementedError("This optimizer isn't supported by the way minimize works by default.")

    def can_ask(self) -> bool:
        return len(self.indices) == 0 or len(self._current_batch) > 0
