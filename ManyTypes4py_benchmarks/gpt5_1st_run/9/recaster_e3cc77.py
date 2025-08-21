import warnings
import threading
import queue
from typing import Any, Callable, Dict, List, Optional
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
    """Thread that runs a function taking another function as input. Each call of the inner function
    adds the point given by the algorithm into the ask queue and then blocks until the main thread sends
    the result back into the tell queue.

    Note
    ----
    This thread must be overlaid into another MessagingThread  because:
    - the threading part should hold no reference from outside (otherwise the destructors may wait each other)
    - the destructor cannot be implemented, hence there is no way to stop the thread automatically
    """

    def __init__(self, caller: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.messages_ask: "queue.Queue[Any]" = queue.Queue()
        self.messages_tell: "queue.Queue[Any]" = queue.Queue()
        self.call_count: int = 0
        self.error: Optional[BaseException] = None
        self._caller: Callable[..., Any] = caller
        self._args: tuple[Any, ...] = args
        self._kwargs: Dict[str, Any] = kwargs
        self.output: Any = None

    def run(self) -> None:
        """Starts the thread and run the "caller" function argument on
        the fake callable, which posts messages and awaits for their answers.
        """
        try:
            self.output = self._caller(self._fake_callable, *self._args, **self._kwargs)
        except StopOptimizerThread:
            self.messages_ask.put(ValueError('Optimization told to finish'))
        except Exception as e:
            self.messages_ask.put(e)
            self.error = e
        else:
            self.messages_ask.put(None)

    def _fake_callable(self, *args: Any) -> Any:
        """
        Puts a new point into the ask queue to be evaluated on the
        main thread and blocks on get from tell queue until point
        is evaluated on main thread and placed into tell queue when
        it is then returned to the caller.
        """
        self.call_count += 1
        self.messages_ask.put(args[0])
        candidate = self.messages_tell.get()
        if candidate is None:
            raise StopOptimizerThread()
        return candidate

    def stop(self) -> None:
        """Notifies the thread that it must stop"""
        self.messages_tell.put(None)


class MessagingThread:
    """Encapsulate the inner thread, so that kill order is automatically called at deletion."""

    def __init__(self, caller: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._thread = _MessagingThread(caller, *args, **kwargs)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def output(self) -> Any:
        return self._thread.output

    @property
    def error(self) -> Optional[BaseException]:
        return self._thread.error

    @property
    def messages_tell(self) -> "queue.Queue[Any]":
        return self._thread.messages_tell

    @property
    def messages_ask(self) -> "queue.Queue[Any]":
        return self._thread.messages_ask

    def stop(self) -> None:
        self._thread.stop()

    def __del__(self) -> None:
        self.stop()


class RecastOptimizer(base.Optimizer):
    """Base class for ask and tell optimizer derived from implementations with no ask and tell interface.
    The underlying optimizer implementation is a function which is supposed to call directly the function
    to optimize. It is tricked into optimizing a "fake" function in a thread:
    - calls to the fake functions are returned by the "ask()" interface
    - return values of the fake functions are provided to the thread when calling "tell(x, value)"

    Note
    ----
    These implementations are not necessarily robust. More specifically, one cannot "tell" any
    point which was not "asked" before.

    An optimization is performed by a third-party library in a background thread. This communicates
    with the main thread using two queue objects. Specifically:

        messages_ask is filled by the background thread with a candidate (or batch of candidates)
        it wants evaluated for, or None if the background thread is over, or an Exception
        which needs to be raised to the user.

        messages_tell supplies the background thread with a value to return from the fake function.
        A value of None means the background thread is no longer relevant and should exit.
    """
    recast: bool = True

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._messaging_thread: Optional[MessagingThread] = None

    def get_optimization_function(self) -> Callable[[Callable[..., Any]], Any]:
        """Return an optimization procedure function (taking a function to optimize as input)

        Note
        ----
        This optimization procedure must be a function or an object which is completely
        independent from self, otherwise deletion of the optimizer may hang indefinitely.
        """
        raise NotImplementedError('You should define your optimizer! Also, be very careful to avoid  reference to this instance in the returned object')

    def _check_error(self) -> None:
        if self._messaging_thread is not None:
            if self._messaging_thread.error is not None:
                raise RuntimeError(f'Recast optimizer raised an error:\n{self._messaging_thread.error}') from self._messaging_thread.error

    def _post_loss(self, candidate: p.Parameter, loss: Any) -> Any:
        """
        Posts the value, and the thread will deal with it.
        """
        return loss

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: Any) -> None:
        raise base.errors.TellNotAskedNotSupportedError

    def _internal_provide_recommendation(self) -> Optional[Any]:
        """Returns the underlying optimizer output if provided (ie if the optimizer did finish)
        else the best pessimistic point.
        """
        if self._messaging_thread is not None and self._messaging_thread.output is not None:
            return self._messaging_thread.output
        else:
            return None

    def __del__(self) -> None:
        if self._messaging_thread is not None:
            self._messaging_thread.stop()


class SequentialRecastOptimizer(RecastOptimizer):
    """Recast Optimizer which cannot deal with parallelization

    There can only be one worker. Each ask must be followed by
    a tell.

    A simple usage is that you have a library which can minimize
    a function which returns a scalar.
    Just make an optimizer inheriting from this class, and inplement
    get_optimization_function to return a callable which runs the
    optimization, taking the objective as its only parameter. The
    callable must not have any references to the optimizer itself.
    (This avoids a reference cycle between the background thread and
    the optimizer, aiding cleanup.) It can have a weakref though.

    If you want your optimizer instance to be picklable, we have to
    store every candidate during optimization, which may use a lot
    of memory. This lets us replay the optimization when
    unpickling. We only do this if you ask for it. To enable:
        - The optimization must be reproducible, asking for the same
          candidates every time. If you need a seed from nevergrad's
          generator, you can't necessarily generate this again after
          unpickling. One solution is to store it in self, if it is
          not there yet, in the body of get_optimization_function.

          As in general in nevergrad, do not set the seed from the
          RNG in your own __init__ because it will cause surprises
          to anyone re-seeding your parametrization after init.
        - The user must call enable_pickling() after initializing
          the optimizer instance.
    """
    no_parallelization: bool = True

    def __init__(self, parametrization: IntOrParameter, budget: int, num_workers: int = 1) -> None:
        super().__init__(parametrization=parametrization, budget=budget, num_workers=num_workers)
        self._enable_pickling: bool = False
        self.replay_archive_tell: List[p.Parameter] = []

    def enable_pickling(self) -> None:
        """Make the optimizer store its history of tells, so
        that it can be serialized.
        """
        if self.num_ask != 0:
            raise ValueError('Can only enable pickling before all\xa0asks.')
        self._enable_pickling = True

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
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

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: Any) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():
            self._check_error()
            return
        if self._enable_pickling:
            self.replay_archive_tell.append(candidate)
        self._messaging_thread.messages_tell.put(self._post_loss(candidate, loss))

    def __getstate__(self) -> Dict[str, Any]:
        if not self._enable_pickling:
            raise ValueError('If you want picklability you should have asked for it')
        thread = self._messaging_thread
        self._messaging_thread = None
        state = self.__dict__.copy()
        self._messaging_thread = thread
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
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
    """Recast optimizer where points to evaluate are provided in batches
    and stored by the optimizer to be asked and told on. The fake_callable
    is only brought into action every 'batch size' number of asks and tells
    instead of every ask and tell. This opens up the optimizer to
    parallelism.

    Note
    ----
    You have to complete a batch before you start a new one so parallelism
    is only possible within batches i.e. if a batch size is 100 and you have
    done 100 asks, you must do 100 tells before you ask again but you could do
    those 100 asks and tells in parallel. To find out if you can perform an ask
    at any given time call self.can_ask.
    """

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._current_batch: List[p.Parameter] = []
        self._batch_losses: List[Any] = []
        self._tell_counter: int = 0
        self.batch_size: int = 0
        self.indices: Dict[str, int] = {}

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
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
        self._batch_losses = [None] * len(points)  # type: ignore[list-item]
        self.indices = {candidate.uid: i for i, candidate in enumerate(self._current_batch)}
        return self._current_batch.pop()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: Any) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
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

    def minimize(
        self,
        objective_function: Callable[..., Any],
        executor: Optional[Any] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
        constraint_violation: Optional[Callable[..., Any]] = None,
        max_time: Optional[float] = None,
    ) -> Any:
        raise NotImplementedError("This optimizer isn't supported by the way minimize works by default.")

    def can_ask(self) -> bool:
        """Returns whether the optimizer is able to perform another ask,
        either because there are points left in the current batch to ask
        or you are ready for a new batch (You have asked and told on every
        point in the last batch.)
        """
        return len(self.indices) == 0 or len(self._current_batch) > 0