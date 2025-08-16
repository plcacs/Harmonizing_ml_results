    def __init__(self, starttype: str = None, stoptype: str = None, startts: int = 0, stopts: int = 0) -> None:

    @property
    def startdt(self) -> datetime:

    @property
    def stopdt(self) -> datetime:

    @property
    def timerange_str(self) -> str:

    @property
    def start_fmt(self) -> str:

    @property
    def stop_fmt(self) -> str:

    def __eq__(self, other: 'TimeRange') -> bool:

    def subtract_start(self, seconds: int) -> None:

    def adjust_start_if_necessary(self, timeframe_secs: int, startup_candles: int, min_date: datetime) -> None:

    @classmethod
    def parse_timerange(cls, text: str) -> 'TimeRange':
