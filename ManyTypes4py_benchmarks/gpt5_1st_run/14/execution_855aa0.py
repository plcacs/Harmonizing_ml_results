import heapq
import typing as tp
from collections import deque
from nevergrad.functions import ExperimentFunction


class MockedTimedJob:
    """Job returned by the MockedTimedExecutor, with the usual
    "done()" and "result()" methods.
    """

    def __init__(
        self,
        func: tp.Union[ExperimentFunction, tp.Callable[..., tp.Any]],
        args: tp.Tuple[tp.Any, ...],
        kwargs: tp.Dict[str, tp.Any],
        executor: "MockedTimedExecutor",
    ) -> None:
        self._executor: MockedTimedExecutor = executor
        self._time: float = executor.time
        self._func: tp.Union[ExperimentFunction, tp.Callable[..., tp.Any]] = func
        self._args: tp.Tuple[tp.Any, ...] = args
        self._kwargs: tp.Dict[str, tp.Any] = kwargs
        self._output: tp.Any = None
        self._delay: tp.Optional[float] = None
        self._done: bool = False
        self._is_read: bool = False

    @property
    def release_time(self) -> float:
        self.process()
        assert self._delay is not None
        return self._delay + self._time

    def done(self) -> bool:
        return self._executor.check_is_done(self)

    def process(self) -> None:
        if self._delay is None:
            self._output = self._func(*self._args, **self._kwargs)
            self._delay = 1.0
            if isinstance(self._func, ExperimentFunction):
                self._delay = max(
                    0, self._func.compute_pseudotime((self._args, self._kwargs), self._output)
                )

    def result(self) -> tp.Any:
        """Return the result if "done()" is true, and raises
        a RuntimeError otherwise.
        """
        self.process()
        if not self.done():
            raise RuntimeError("Asking result which is not ready")
        self._is_read = True
        self._executor.notify_read(self)
        return self._output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<value={self._output}, done={self._done}, read={self._is_read}>)"


class OrderedJobs(tp.NamedTuple):
    """Handle for sorting jobs by release_time (or submission order in case of tie)"""
    release_time: float
    order: int
    job: "MockedTimedJob"


class MockedTimedExecutor:
    """Executor that mocks a steady state, by only providing 1 job at a time which is done() while
    not having been "read" (i.e. "result()" method was not executed).
    This ensures we control the order of update of the optimizer for benchmarking.

    Additionally, "delays" can be provided by the function so that jobs are not "done()" by order of
    submission. To this end, callables must implement a "computation_time" method.
    """

    def __init__(self, batch_mode: bool = False) -> None:
        self.batch_mode: bool = batch_mode
        self._to_be_processed: tp.Deque[MockedTimedJob] = deque()
        self._steady_priority_queue: tp.List[OrderedJobs] = []
        self._order: int = 0
        self._time: float = 0.0

    @property
    def time(self) -> float:
        return self._time

    def submit(
        self,
        fn: tp.Union[ExperimentFunction, tp.Callable[..., tp.Any]],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> MockedTimedJob:
        job = MockedTimedJob(fn, args, kwargs, self)
        self._to_be_processed.append(job)
        return job

    def _process_submissions(self) -> None:
        """Process all submissions which have not been processed yet."""
        while self._to_be_processed:
            job: MockedTimedJob = self._to_be_processed[0]
            job.process()
            if not self.batch_mode:
                heapq.heappush(
                    self._steady_priority_queue, OrderedJobs(job.release_time, self._order, job)
                )
            self._to_be_processed.popleft()
            self._order += 1

    def check_is_done(self, job: MockedTimedJob) -> bool:
        """Called whenever "done" method is called on a job."""
        self._process_submissions()
        if self.batch_mode or job._is_read:
            return True
        else:
            return job is self._steady_priority_queue[0].job

    def notify_read(self, job: MockedTimedJob) -> None:
        """Called whenever a result is read, so as to activate the next result in line
        in case of steady mode, and to update executor time.
        """
        self._process_submissions()
        if not self.batch_mode:
            expected: OrderedJobs = self._steady_priority_queue[0]
            assert job is expected.job, "Only first job should be read"
            heapq.heappop(self._steady_priority_queue)
            if self._steady_priority_queue:
                self._steady_priority_queue[0].job._done = True
        self._time = max(self._time, job.release_time)