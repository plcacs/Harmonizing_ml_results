    def key(self, key: str) -> str:
    def incr(self, key: str) -> None:
    def decr(self, key: str) -> None:
    def timing(self, key: str, value: float) -> None:
    def gauge(self, key: str, value: float) -> None:

    def __init__(self, prefix: str = 'superset') -> None:
    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'superset', statsd_client: Optional[StatsClient] = None) -> None:
