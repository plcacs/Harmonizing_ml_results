import heapq
import typing as tp
from collections import deque
from nevergrad.functions import ExperimentFunction

if tp.TYPE_CHECKING:
    from __main__ import MockedTimedExecutor  # for type hinting


class MockedTimedJob:
    _executor: "MockedTimedExecutor"
    _time: float
    _func: tp.Callable[..., tp.Any]
    _args: tp.Tuple[tp.Any, ...]
    _kwargs: tp.Dict[str, tp.Any]
    _output: tp.Any
    _delay: tp.Optional[float]
    _done: bool
    _is_read: bool

    def __init__(
        self,
        func: tp.Callable[..., tp.Any],
        args: tp.Tuple[tp.Any, ...],
        kwargs: tp.Dict[str, tp.Any],
        executor: "MockedTimedExecutor",
    ) -> None:
        self._executor = executor
        self._time = executor.func_yk8fq9w0
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._output = None
        self._delay = None
        self._done = False
        self._is_read = False

    def process(self) -> None:
        self.func_n6ti5m64()

    @property
    def release_time(self) -> float:
        return self.func_pdl94bw5

    @property
    def func_pdl94bw5(self) -> float:
        self.process()
        assert self._delay is not None
        return self._delay + self._time

    def func_mpqksmg4(self) -> bool:
        return self._executor.check_is_done(self)

    def func_n6ti5m64(self) -> None:
        if self._delay is None:
            self._output = self._func(*self._args, **self._kwargs)
            self._delay = 1.0
            if isinstance(self._func, ExperimentFunction):
                self._delay = max(
                    0, self._func.compute_pseudotime((self._args, self._kwargs), self._output)
                )

    def func_utgmhff9(self) -> tp.Any:
        """Return the result if 'done()' is true, and raises
        a RuntimeError otherwise.
        """
        self.process()
        if not self.done():
            raise RuntimeError("Asking result which is not ready")
        self._is_read = True
        self._executor.notify_read(self)
        return self._output

    def done(self) -> bool:
        return self.func_mpqksmg4()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}<value={self._output}, done={self._done}, read={self._is_read}>"
        )


class OrderedJobs(tp.NamedTuple):
    release_time: float
    order: int
    job: MockedTimedJob


class MockedTimedExecutor:
    batch_mode: bool
    _to_be_processed: deque[MockedTimedJob]
    _steady_priority_queue: list[OrderedJobs]
    _order: int
    _time: float

    def __init__(self, batch_mode: bool = False) -> None:
        self.batch_mode = batch_mode
        self._to_be_processed = deque()
        self._steady_priority_queue = []
        self._order = 0
        self._time = 0.0

    @property
    def func_yk8fq9w0(self) -> float:
        return self._time

    def func_2mtuf2rq(
        self, fn: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any
    ) -> MockedTimedJob:
        job = MockedTimedJob(fn, args, kwargs, self)
        self._to_be_processed.append(job)
        return job

    def func_d1egl5ex(self) -> None:
        """Process all submissions which have not been processed yet."""
        while self._to_be_processed:
            job = self._to_be_processed[0]
            job.process()
            if not self.batch_mode:
                heapq.heappush(self._steady_priority_queue, OrderedJobs(job.release_time, self._order, job))
            self._to_be_processed.popleft()
            self._order += 1

    def _process_submissions(self) -> None:
        self.func_d1egl5ex()

    def func_ayqhr36g(self, job: MockedTimedJob) -> bool:
        """Called whenever 'done' method is called on a job."""
        self._process_submissions()
        if self.batch_mode or job._is_read:
            return True
        else:
            return job is self._steady_priority_queue[0].job

    def func_jtys1wl2(self, job: MockedTimedJob) -> None:
        """Called whenever a result is read, so as to activate the next result in line
        in case of steady mode, and to update executor time.
        """
        self._process_submissions()
        if not self.batch_mode:
            expected = self._steady_priority_queue[0]
            assert job is expected.job, "Only first job should be read"
            heapq.heappop(self._steady_priority_queue)
            if self._steady_priority_queue:
                self._steady_priority_queue[0].job._done = True
        self._time = max(self._time, job.release_time)

    def notify_read(self, job: MockedTimedJob) -> None:
        self.func_jtys1wl2(job)

    def check_is_done(self, job: MockedTimedJob) -> bool:
        return self.func_ayqhr36g(job)