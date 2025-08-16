from typing import Any, ClassVar, Dict, Iterable, Optional, Type, Union, AsyncGenerator, Deque
from mode import Seconds
from mode.utils.typing import Counter
from .app import LiveCheck as _LiveCheck

class Case(Service):
    Runner: ClassVar[Type[TestRunner]
    active: bool
    status: State
    frequency: Optional[Seconds]
    warn_stalled_after: float
    probability: float
    max_consecutive_failures: int
    last_test_received: Optional[datetime]
    last_fail: Optional[datetime]
    max_history: int
    runtime_avg: Optional[float]
    latency_avg: Optional[float]
    frequency_avg: Optional[float]
    test_expires: timedelta
    realtime_logs: bool
    url_timeout_total: float
    url_timeout_connect: Optional[float]
    url_error_retries: int
    url_error_delay_min: float
    url_error_delay_backoff: float
    url_error_delay_max: float
    state_transition_delay: float
    consecutive_failures: int
    total_failures: int

    def __init__(self, *, app: _LiveCheck, name: str, probability: Optional[float] = None, warn_stalled_after: Optional[float] = None, active: Optional[bool] = None, signals: Optional[Iterable[BaseSignal]] = None, test_expires: Optional[Seconds] = None, frequency: Optional[Seconds] = None, realtime_logs: Optional[bool] = None, max_history: Optional[int] = None, max_consecutive_failures: Optional[int] = None, url_timeout_total: Optional[float] = None, url_timeout_connect: Optional[float] = None, url_error_retries: Optional[int] = None, url_error_delay_min: Optional[float] = None, url_error_delay_backoff: Optional[float] = None, url_error_delay_max: Optional[float] = None, **kwargs: Any) -> None:

    async def _sampler(self) -> None:

    async def _sample(self) -> None:

    @asynccontextmanager
    async def maybe_trigger(self, id: Optional[str] = None, *args: Any, **kwargs: Any) -> AsyncGenerator[Optional[TestExecution], None]:

    async def trigger(self, id: Optional[str] = None, *args: Any, **kwargs: Any) -> TestExecution:

    def _now(self) -> datetime:

    async def run(self, *test_args: Any, **test_kwargs: Any) -> None:

    async def resolve_signal(self, key: Any, event: SignalEvent) -> None:

    async def execute(self, test: TestExecution) -> None:

    async def on_test_start(self, runner: TestRunner) -> None:

    async def on_test_skipped(self, runner: TestRunner) -> None:

    async def on_test_failed(self, runner: TestRunner, exc: Exception) -> None:

    async def on_test_error(self, runner: TestRunner, exc: Exception) -> None:

    async def on_test_timeout(self, runner: TestRunner, exc: Exception) -> None:

    async def _set_test_error_state(self, state: State) -> None:

    def _set_pass_state(self, state: State) -> None:

    async def on_test_pass(self, runner: TestRunner) -> None:

    async def post_report(self, report: TestReport) -> None:

    async def _send_frequency(self) -> None:

    async def make_fake_request(self) -> None:

    async def _check_frequency(self) -> None:

    async def on_suite_fail(self, exc: SuiteFailed, new_state: State = State.FAIL) -> None:

    def _maybe_recover_from_failed_state(self) -> None:

    def _failed_longer_than(self, secs: float) -> bool:

    @property
    def seconds_since_last_fail(self) -> Optional[float]:

    async def get_url(self, url: URL, **kwargs: Any) -> Any:

    async def post_url(self, url: URL, **kwargs: Any) -> Any:

    async def url_request(self, method: str, url: URL, **kwargs: Any) -> Any:

    @property
    def current_test(self) -> Optional[TestExecution]:

    @property
    def current_execution(self) -> Optional[TestRunner]:

    @property
    def label(self) -> str:
