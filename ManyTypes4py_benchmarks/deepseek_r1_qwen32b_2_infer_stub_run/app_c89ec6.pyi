import abc
import asyncio
import typing
from datetime import tzinfo
from typing import Any, AsyncIterable, Awaitable, Callable, ClassVar, ContextManager, Iterable, Mapping, MutableSequence, Optional, Pattern, Set, Tuple, Type, TypeVar, Union, no_type_check
import opentracing
from mode import Seconds, ServiceT, Signal, SupervisorStrategyT, SyncSignal
from mode.utils.futures import stampede
from mode.utils.objects import cached_property
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT
from mode.utils.typing import NoReturn
from .agents import AgentFun, AgentManagerT, AgentT, SinkT
from .assignor import PartitionAssignorT
from .codecs import CodecArg
from .core import HeadersArg, K, V
from .fixups import FixupT
from .router import RouterT
from .sensors import SensorDelegateT
from .serializers import RegistryT
from .streams import StreamT
from .tables import CollectionT, TableManagerT, TableT
from .topics import ChannelT, TopicT
from .transports import ConductorT, ConsumerT, ProducerT, TransportT
from .tuples import Message, MessageSentCallback, RecordMetadata, TP
from .web import CacheBackendT, HttpClientT, PageArg, ResourceOptions, View, ViewDecorator, Web
from .windows import WindowT

__all__ = ['TaskArg', 'AppT']

TaskArg = Union[Callable[['AppT'], Awaitable], Callable[[], Awaitable]]

class TracerT(abc.ABC):
    @property
    @abc.abstractmethod
    def default_tracer(self) -> Any: ...

    @abc.abstractmethod
    def trace(self, name: str, sample_rate: Optional[float] = ..., **extra_context: Any) -> ContextManager: ...

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> Any: ...

class BootStrategyT:
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool

    @abc.abstractmethod
    def __init__(self, app: Any, *, enable_web: Optional[bool] = ..., enable_kafka: bool = ..., enable_kafka_producer: Optional[bool] = ..., enable_kafka_consumer: Optional[bool] = ..., enable_sensors: bool = ...) -> None: ...

    @abc.abstractmethod
    def server(self) -> None: ...

    @abc.abstractmethod
    def client_only(self) -> None: ...

    @abc.abstractmethod
    def producer_only(self) -> None: ...

class AppT(ServiceT):
    finalized: bool
    configured: bool
    rebalancing: bool
    rebalancing_count: int
    unassigned: bool
    in_worker: bool
    on_configured: SyncSignal
    on_before_configured: SyncSignal
    on_after_configured: SyncSignal
    on_partitions_assigned: Signal
    on_partitions_revoked: Signal
    on_rebalance_complete: Signal
    on_before_shutdown: Signal
    on_worker_init: SyncSignal
    on_produce_message: SyncSignal
    tracer: Any
    on_startup_finished: Any

    @abc.abstractmethod
    def __init__(self, id: Any, *, monitor: Any, config_source: Optional[Any] = ..., **options: Any) -> None: ...

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: bool = ..., force: bool = ...) -> None: ...

    @abc.abstractmethod
    def finalize(self) -> None: ...

    @abc.abstractmethod
    def main(self) -> Awaitable[Any]: ...

    @abc.abstractmethod
    def worker_init(self) -> None: ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None: ...

    @abc.abstractmethod
    def discover(self, *extra_modules: Any, categories: Iterable[str] = ..., ignore: Iterable[str] = ...) -> None: ...

    @abc.abstractmethod
    def topic(self, *topics: str, pattern: Optional[Pattern] = ..., schema: Any = ..., key_type: Any = ..., value_type: Any = ..., key_serializer: CodecArg = ..., value_serializer: CodecArg = ..., partitions: Optional[int] = ..., retention: Optional[Seconds] = ..., compacting: Optional[bool] = ..., deleting: Optional[bool] = ..., replicas: Optional[int] = ..., acks: Union[bool, int] = ..., internal: bool = ..., config: Optional[Mapping] = ..., maxsize: Optional[int] = ..., allow_empty: bool = ..., has_prefix: bool = ..., loop: Optional[Any] = ...) -> TopicT: ...

    @abc.abstractmethod
    def channel(self, *, schema: Any = ..., key_type: Any = ..., value_type: Any = ..., maxsize: Optional[int] = ..., loop: Optional[Any] = ...) -> ChannelT: ...

    @abc.abstractmethod
    def agent(self, channel: Optional[Union[ChannelT, str]] = ..., *, name: Optional[str] = ..., concurrency: int = ..., supervisor_strategy: Optional[SupervisorStrategyT] = ..., sink: Optional[SinkT] = ..., isolated_partitions: bool = ..., use_reply_headers: bool = ..., **kwargs: Any) -> AgentT: ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Callable, *, on_leader: bool = ..., traced: bool = ...) -> Any: ...

    @abc.abstractmethod
    def timer(self, interval: Seconds, on_leader: bool = ..., traced: bool = ..., name: Optional[str] = ..., max_drift_correction: float = ...) -> Any: ...

    @abc.abstractmethod
    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = ..., on_leader: bool = ..., traced: bool = ...) -> Any: ...

    @abc.abstractmethod
    def service(self, cls: Any) -> Any: ...

    @abc.abstractmethod
    def stream(self, channel: Union[ChannelT, str], beacon: Optional[Any] = ..., **kwargs: Any) -> StreamT: ...

    @abc.abstractmethod
    def Table(self, name: str, *, default: Any = ..., window: Optional[WindowT] = ..., partitions: Optional[int] = ..., help: Optional[str] = ..., **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def GlobalTable(self, name: str, *, default: Any = ..., window: Optional[WindowT] = ..., partitions: Optional[int] = ..., help: Optional[str] = ..., **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def SetTable(self, name: str, *, window: Optional[WindowT] = ..., partitions: Optional[int] = ..., start_manager: bool = ..., help: Optional[str] = ..., **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def SetGlobalTable(self, name: str, *, window: Optional[WindowT] = ..., partitions: Optional[int] = ..., start_manager: bool = ..., help: Optional[str] = ..., **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def page(self, path: str, *, base: Type[View] = ..., cors_options: Optional[Mapping] = ..., name: Optional[str] = ...) -> ViewDecorator: ...

    @abc.abstractmethod
    def table_route(self, table: TableT, shard_param: Optional[str] = ..., *, query_param: Optional[str] = ..., match_info: Optional[str] = ..., exact_key: Optional[Any] = ...) -> Any: ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Optional[Any] = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def create_event(self, key: K, value: V, headers: HeadersArg, message: Message) -> Any: ...

    @abc.abstractmethod
    async def start_client(self) -> None: ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None: ...

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = ..., **extra_context: Any) -> ContextManager: ...

    @abc.abstractmethod
    async def send(self, channel: Union[ChannelT, str], key: Optional[K] = ..., value: Optional[V] = ..., partition: Optional[int] = ..., timestamp: Optional[float] = ..., headers: HeadersArg = ..., schema: Any = ..., key_serializer: CodecArg = ..., value_serializer: CodecArg = ..., callback: Optional[MessageSentCallback] = ...) -> Optional[RecordMetadata]: ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> Any: ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None: ...

    @abc.abstractmethod
    def is_leader(self) -> bool: ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Optional[int] = ..., *, clear_on_resume: bool = ..., loop: Optional[Any] = ...) -> ThrowableQueue: ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Web) -> None: ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None: ...

    @abc.abstractmethod
    def on_rebalance_return(self) -> None: ...

    @abc.abstractmethod
    def on_rebalance_end(self) -> None: ...

    @property
    def conf(self) -> Any: ...

    @conf.setter
    def conf(self, settings: Any) -> None: ...

    @property
    @abc.abstractmethod
    def transport(self) -> TransportT: ...

    @transport.setter
    def transport(self, transport: TransportT) -> None: ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> TransportT: ...

    @producer_transport.setter
    def producer_transport(self, transport: TransportT) -> None: ...

    @property
    @abc.abstractmethod
    def cache(self) -> CacheBackendT: ...

    @cache.setter
    def cache(self, cache: CacheBackendT) -> None: ...

    @property
    @abc.abstractmethod
    def producer(self) -> ProducerT: ...

    @property
    @abc.abstractmethod
    def consumer(self) -> ConsumerT: ...

    @cached_property
    @abc.abstractmethod
    def tables(self) -> TableManagerT: ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> RouterT: ...

    @property
    @abc.abstractmethod
    def monitor(self) -> _Monitor: ...

    @monitor.setter
    def monitor(self, value: _Monitor) -> None: ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> FlowControlEvent: ...

    @property
    @abc.abstractmethod
    def http_client(self) -> HttpClientT: ...

    @http_client.setter
    def http_client(self, client: HttpClientT) -> None: ...

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> PartitionAssignorT: ...

    @cached_property
    @abc.abstractmethod
    def router(self) -> RouterT: ...

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> RegistryT: ...

    @cached_property
    @abc.abstractmethod
    def web(self) -> Web: ...

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> bool: ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: opentracing.Span) -> None: ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span: ...