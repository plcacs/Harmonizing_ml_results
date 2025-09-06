    def __init__(self, args: Any, config: Any = None) -> None:
    def _init(self, reconfig: bool) -> None:
    def _notify(self, message: str) -> None:
    def run(self) -> None:
    def _worker(self, old_state: State) -> State:
    def _throttle(self, func: Callable, throttle_secs: float, timeframe: Any = None, timeframe_offset: float = 1.0, *args, **kwargs) -> Any:
    def _sleep(sleep_duration: float) -> None:
    def _process_stopped(self) -> None:
    def _process_running(self) -> None:
    def _reconfigure(self) -> None:
