class Case(Service):
    """LiveCheck test case."""
    Runner: Type[TestRunner] = TestRunner
    active: bool = True
    status: State = State.INIT
    frequency: Optional[float] = None
    warn_stalled_after: float = 1800.0
    probability: float = 0.5
    max_consecutive_failures: int = 30
    last_test_received: Optional[float] = None
    last_fail: Optional[float] = None
    max_history: int = 100
    runtime_avg: Optional[float] = None
    latency_avg: Optional[float] = None
    frequency_avg: Optional[float] = None
    test_expires: timedelta = timedelta(hours=3)
    realtime_logs: bool = False
    url_timeout_total: float = 5 * 60.0
    url_timeout_connect: Optional[float] = None
    url_error_retries: int = 10
    url_error_delay_min: float = 0.5
    url_error_delay_backoff: float = 1.5
    url_error_delay_max: float = 5.0
    state_transition_delay: float = 60.0
    consecutive_failures: int = 0
    total_failures: int = 0

    def __init__(self, *, app: _LiveCheck, name: str, probability: Optional[float] = None, warn_stalled_after: Optional[float] = None, active: Optional[bool] = None, signals: Iterable[BaseSignal] = (), test_expires: Optional[float] = None, frequency: Optional[float] = None, realtime_logs: Optional[bool] = None, max_history: Optional[int] = None, max_consecutive_failures: Optional[int] = None, url_timeout_total: Optional[float] = None, url_timeout_connect: Optional[float] = None, url_error_retries: Optional[int] = None, url_error_delay_min: Optional[float] = None, url_error_delay_backoff: Optional[float] = None, url_error_delay_max: Optional[float] = None, **kwargs: Any):
        ...
