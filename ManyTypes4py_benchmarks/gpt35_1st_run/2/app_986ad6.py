    def on_stream_event_in(self, tp: TP, offset: int, stream: StreamT, event: EventT) -> None:

    def on_stream_event_out(self, tp: TP, offset: int, stream: StreamT, event: EventT, state: Any = None) -> None:

    def for_app(cls, app: AppT, *, prefix: str = 'livecheck-', web_port: int = 9999, test_topic_name: Optional[str] = None, bus_topic_name: Optional[str] = None, report_topic_name: Optional[str] = None, bus_concurrency: Optional[int] = None, test_concurrency: Optional[int] = None, send_reports: Optional[bool] = None, **kwargs: Any) -> 'LiveCheck':

    def __init__(self, id: str, *, test_topic_name: Optional[str] = None, bus_topic_name: Optional[str] = None, report_topic_name: Optional[str] = None, bus_concurrency: Optional[int] = None, test_concurrency: Optional[int] = None, send_reports: Optional[bool] = None, **kwargs: Any) -> None:

    def on_produce_attach_test_headers(self, sender: AppT, key: Any = None, value: Any = None, partition: int = None, timestamp: float = None, headers: List[Tuple[str, bytes]] = None, signal: Optional[BaseSignalT] = None, **kwargs: Any) -> None:

    def case(self, *, name: Optional[str] = None, probability: Optional[float] = None, warn_stalled_after: timedelta = timedelta(minutes=30), active: Optional[bool] = None, test_expires: Optional[timedelta] = None, frequency: Optional[Seconds] = None, max_history: Optional[int] = None, max_consecutive_failures: Optional[int] = None, url_timeout_total: Optional[float] = None, url_timeout_connect: Optional[float] = None, url_error_retries: Optional[int] = None, url_error_delay_min: Optional[float] = None, url_error_delay_backoff: Optional[float] = None, url_error_delay_max: Optional[float] = None, base: Type[Case] = Case) -> Callable[[Type], Type]:

    def add_case(self, case: Case) -> Case:

    async def post_report(self, report: TestReport) -> None:

    async def on_start(self) -> None:

    async def on_started(self) -> None:

    async def _populate_signals(self, events: Dict[str, SignalEvent]) -> None:

    async def _execute_tests(self, tests: Dict[str, TestExecution]) -> None:

    def _prepare_case_name(self, name: str) -> str:
