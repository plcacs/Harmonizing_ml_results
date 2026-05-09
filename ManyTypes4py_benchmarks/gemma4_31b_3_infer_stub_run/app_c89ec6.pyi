import abc
import asyncio
import typing
from typing import Any, AsyncIterable, Awaitable, Callable, Optional, Pattern, Set, Tuple, Type, TypeVar, Union
from mode import Seconds, ServiceT, Signal, SupervisorStrategyT, SyncSignal
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from .agents import AgentFun, AgentManagerT, AgentT, SinkT
from .assignor import PartitionAssignorT
from .codecs import CodecArg
from .core import HeadersArg, K, V
from .fixups import FixupT
from .router import RouterT
from .sensors import SensorDelegateT
from .serializers import RegistryT, SchemaT
from .streams import StreamT
from .tables import CollectionT, TableManagerT, TableT
from .topics import ChannelT, TopicT
from .transports import ConductorT, ConsumerT, ProducerT, TransportT
from .tuples import Message, MessageSentCallback, RecordMetadata, TP
from .web import CacheBackendT, HttpClientT, PageArg, ResourceOptions, View, ViewDecorator, Web
from .windows import WindowT

if typing.TYPE_CHECKING:
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck.app import LiveCheck as _LiveCheck
    from faust.sensors.monitor import Monitor as _Monitor
    from faust.worker import Worker as _Worker
    from .events import EventT as _EventT
    from .models import ModelArg as _ModelArg
    from .settings import Settings as _Settings

__all__ = ['TaskArg', 'AppT']

TaskArg = Union[Callable[['AppT'], Awaitable[Any]], Callable[[], Awaitable[Any]]]
_T = TypeVar('_T')

class TracerT(abc.ABC):
    @property
    @abc.abstractmethod
    def default_tracer(self) -> Any: ...

    @abc.abstractmethod
    def trace(self, name: str, sample_rate: Optional[float] = None, **extra_context: Any) -> Any: ...

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> Any: ...

class BootStrategyT(abc.ABC):
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool

    @abc.abstractmethod
    def __init__(
        self, 
        app: 'AppT', 
        *, 
        enable_web: Optional[bool] = None, 
        enable_kafka: bool = True, 
        enable_kafka_producer: Optional[bool] = None, 
        enable_kafka_consumer: Optional[bool] = None, 
        enable_sensors: bool = True
    ) -> None: ...

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
    tracer: Optional[TracerT]

    @abc.abstractmethod
    def __init__(self, id: str, *, monitor: Any, config_source: Optional[Any] = None, **options: Any) -> None: ...

    on_startup_finished: Optional[Callable[[], Awaitable[Any]]]

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None: ...

    @abc.abstractmethod
    def finalize(self) -> None: ...

    @abc.abstractmethod
    def main(self) -> None: ...

    @abc.abstractmethod
    def worker_init(self) -> None: ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None: ...

    @abc.abstractmethod
    def discover(self, *extra_modules: str, categories: Tuple[str, ...], ignore: Tuple[str, ...]) -> None: ...

    @abc.abstractmethod
    def topic(
        self, 
        *topics: str, 
        pattern: Optional[Pattern] = None, 
        schema: Optional[SchemaT] = None, 
        key_type: Optional[Type] = None, 
        value_type: Optional[Type] = None, 
        key_serializer: Optional[CodecArg] = None, 
        value_serializer: Optional[CodecArg] = None, 
        partitions: Optional[int] = None, 
        retention: Optional[Union[int, str]] = None, 
        compacting: Optional[bool] = None, 
        deleting: Optional[bool] = None, 
        replicas: Optional[int] = None, 
        acks: bool = True, 
        internal: bool = False, 
        config: Optional[typing.Mapping[str, Any]] = None, 
        maxsize: Optional[int] = None, 
        allow_empty: bool = False, 
        has_prefix: bool = False, 
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> TopicT: ...

    @abc.abstractmethod
    def channel(self, *, schema: Optional[SchemaT] = None, key_type: Optional[Type] = None, value_type: Optional[Type] = None, maxsize: Optional[int] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> ChannelT: ...

    @abc.abstractmethod
    def agent(self, channel: Optional[Union[TopicT, ChannelT]] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional[SupervisorStrategyT] = None, sink: Optional[SinkT] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> AgentT: ...

    @abc.abstractmethod
    def task(self, fun: Callable[..., Awaitable[Any]], *, on_leader: bool = False, traced: bool = True) -> Any: ...

    @abc.abstractmethod
    def timer(self, interval: Union[float, Seconds], on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1) -> Any: ...

    @abc.abstractmethod
    def crontab(self, cron_format: str, *, timezone: Optional[typing.Union[str, typing.Any]] = None, on_leader: bool = False, traced: bool = True) -> Any: ...

    @abc.abstractmethod
    def service(self, cls: Type[ServiceT]) -> Any: ...

    @abc.abstractmethod
    def stream(self, channel: Union[TopicT, ChannelT], beacon: Optional[Any] = None, **kwargs: Any) -> StreamT: ...

    @abc.abstractmethod
    def Table(self, name: str, *, default: Any = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def GlobalTable(self, name: str, *, default: Any = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def SetTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def SetGlobalTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT: ...

    @abc.abstractmethod
    def page(self, path: str, *, base: Type[View] = View, cors_options: Optional[ResourceOptions] = None, name: Optional[str] = None) -> View: ...

    @abc.abstractmethod
    def table_route(self, table: TableT, shard_param: Optional[str] = None, *, query_param: Optional[str] = None, match_info: Optional[Any] = None, exact_key: Optional[bool] = None) -> Any: ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Optional[Type] = None, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: Optional[HeadersArg] = None, message: Optional[Message] = None) -> _EventT: ...

    @abc.abstractmethod
    async def start_client(self) -> None: ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None: ...

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> Any: ...

    @abc.abstractmethod
    async def send(
        self, 
        channel: Union[TopicT, ChannelT], 
        key: Any = None, 
        value: Any = None, 
        partition: Optional[int] = None, 
        timestamp: Optional[float] = None, 
        headers: Optional[HeadersArg] = None, 
        schema: Optional[SchemaT] = None, 
        key_serializer: Optional[CodecArg] = None, 
        value_serializer: Optional[CodecArg] = None, 
        callback: Optional[MessageSentCallback] = None
    ) -> Any: ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck: ...

    @abc.abstractmethod
    async def maybe_start_producer(self) -> None: ...

    @abc.abstractmethod
    def is_leader(self) -> bool: ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Optional[int] = None, *, clear_on_resume: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> ThrowableQueue: ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> _Worker: ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Web) -> None: ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None: ...

    @abc.abstractmethod
    def on_rebalance_return(self) -> None: ...

    @abc.abstractmethod
    def on_rebalance_end(self) -> None: ...

    @property
    def conf(self) -> _Settings: ...
    @conf.setter
    def conf(self, settings: _Settings) -> None: ...

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

    @property
    @abc.abstractmethod
    def tables(self) -> TableManagerT: ...

    @property
    @abc.abstractmethod
    def topics(self) -> typing.Mapping[str, TopicT]: ...

    @property
    @abc.abstractmethod
    def monitor(self) -> _Monitor: ...
    @monitor.setter
    def monitor(self, value: _Monitor) -> None: ...

    @property
    @abc.abstractmethod
    def flow_control(self) -> FlowControlEvent: ...

    @property
    @abc.abstractmethod
    def http_client(self) -> HttpClientT: ...
    @http_client.setter
    def http_client(self, client: HttpClientT) -> None: ...

    @property
    @abc.abstractmethod
    def assignor(self) -> PartitionAssignorT: ...

    @property
    @abc.abstractmethod
    def router(self) -> RouterT: ...

    @property
    @abc.abstractmethod
    def serializers(self) -> RegistryT: ...

    @property
    @abc.abstractmethod
    def web(self) -> Web: ...

    @property
    @abc.abstractmethod
    def in_transaction(self) -> bool: ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: Any) -> None: ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> Any: ...