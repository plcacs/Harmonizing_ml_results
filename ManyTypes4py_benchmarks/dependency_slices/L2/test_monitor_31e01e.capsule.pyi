from typing import Any

# === Internal dependency: faust ===
Event: Any
Stream: Any
Table: Any
Topic: Any

# === Internal dependency: faust.sensors.monitor ===
class TableState(KeywordReduce): ...
class Monitor(Sensor, KeywordReduce):
    def __init__(self, *, max_avg_history: int = ..., max_commit_latency_history: int = ..., max_send_latency_history: int = ..., max_assignment_latency_history: int = ..., messages_sent: int = ..., tables: MutableMapping[str, TableState] = ..., messages_active: int = ..., events_active: int = ..., messages_received_total: int = ..., messages_received_by_topic: Counter[str] = ..., events_total: int = ..., events_by_stream: Counter[StreamT] = ..., events_by_task: Counter[asyncio.Task] = ..., events_runtime: Deque[float] = ..., commit_latency: Deque[float] = ..., send_latency: Deque[float] = ..., assignment_latency: Deque[float] = ..., events_s: int = ..., messages_s: int = ..., events_runtime_avg: float = ..., topic_buffer_full: Counter[TP] = ..., rebalances: int = ..., rebalance_return_latency: Deque[float] = ..., rebalance_end_latency: Deque[float] = ..., rebalance_return_avg: float = ..., rebalance_end_avg: float = ..., time: Callable[[], float] = ..., http_response_codes: Counter[HTTPStatus] = ..., http_response_latency: Deque[float] = ..., http_response_latency_avg: float = ..., **kwargs: Any) -> None: ...
    async def _sampler(self) -> None: ...

# === Internal dependency: faust.transport.consumer ===
class Consumer(Service, ConsumerT): ...

# === Internal dependency: faust.transport.producer ===
class Producer(Service, ProducerT): ...

# === Internal dependency: faust.types ===
# re-export: from .tuples import Message
# re-export: from .tuples import TP

# === Third-party dependency: mode.utils.mocks ===
class Mock(unittest.mock.Mock):
    ...
class AsyncMock(unittest.mock.Mock):
    def __init__(self, *args: Any, name: str = ..., **kwargs: Any) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark