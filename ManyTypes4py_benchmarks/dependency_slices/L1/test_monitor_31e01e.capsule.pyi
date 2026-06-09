from typing import Any

# === Internal dependency: faust ===
Event: Any
Stream: Any
Table: Any
Topic: Any

# === Internal dependency: faust.sensors.monitor ===
class TableState(KeywordReduce): ...
class Monitor(Sensor, KeywordReduce):
    def __init__(self, *, max_avg_history=..., max_commit_latency_history=..., max_send_latency_history=..., max_assignment_latency_history=..., messages_sent=..., tables=..., messages_active=..., events_active=..., messages_received_total=..., messages_received_by_topic=..., events_total=..., events_by_stream=..., events_by_task=..., events_runtime=..., commit_latency=..., send_latency=..., assignment_latency=..., events_s=..., messages_s=..., events_runtime_avg=..., topic_buffer_full=..., rebalances=..., rebalance_return_latency=..., rebalance_end_latency=..., rebalance_return_avg=..., rebalance_end_avg=..., time=..., http_response_codes=..., http_response_latency=..., http_response_latency_avg=..., **kwargs): ...
    async def _sampler(self): ...

# === Internal dependency: faust.transport.consumer ===
class Consumer(Service, ConsumerT): ...

# === Internal dependency: faust.transport.producer ===
class Producer(Service, ProducerT): ...

# === Internal dependency: faust.types ===
from .tuples import Message
from .tuples import TP

# === Third-party dependency: mode.utils.mocks ===
class Mock(unittest.mock.Mock):
    ...
class AsyncMock(unittest.mock.Mock):
    def __init__(self, *args: Any, name: str = ..., **kwargs: Any) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark