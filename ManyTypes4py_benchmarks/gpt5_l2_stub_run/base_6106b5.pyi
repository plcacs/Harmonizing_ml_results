from typing import Any, AsyncIterable, Awaitable, Callable, ContextManager, Iterable, List, Mapping, Optional, Pattern, Type, Union
import asyncio
from datetime import tzinfo
from mode import Seconds, Service, ServiceT, SupervisorStrategyT
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT
from faust.agents import AgentFun, AgentManager, AgentT, SinkT
from faust.channels import ChannelT
from faust.fixups import FixupT
from faust.sensors import Monitor, SensorDelegate
from faust.web.views import View
from faust.types.app import AppT, BootStrategyT, TaskArg, TracerT
from faust.types.codecs import CodecArg
from faust.types.core import HeadersArg, K, V
from faust.types.events import EventT
from faust.types.models import ModelArg
from faust.types.router import RouterT
from faust.types.serializers import RegistryT, SchemaT
from faust.types.streams import StreamT
from faust.types.tables import GlobalTableT, TableManagerT, TableT
from faust.types.topics import TopicT
from faust.types.transports import ConductorT, ConsumerT, ProducerT, TPorTopicSet, TransportT
from faust.types.tuples import Message, MessageSentCallback, RecordMetadata
from faust.types.web import CacheBackendT, HttpClientT, PageArg, Request, ResourceOptions, Response, Web
from faust.types.windows import WindowT
from faust.cli.base import AppCommand as _AppCommand
from faust.livecheck import LiveCheck as _LiveCheck
from faust.worker import Worker as _Worker

__all__: List[str] = ...
logger: Any = ...
APP_REPR_FINALIZED: str = ...
APP_REPR_UNFINALIZED: str = ...
SCAN_AGENT: str = ...
SCAN_COMMAND: str = ...
SCAN_PAGE: str = ...
SCAN_SERVICE: str = ...
SCAN_TASK: str = ...
SCAN_CATEGORIES: List[str] = ...
SCAN_IGNORE: List[Any] = ...
E_NEED_ORIGIN: str = ...
W_OPTION_DEPRECATED: str = ...
W_DEPRECATED_SHARD_PARAM: str = ...
TaskDecoratorRet = Union[Callable[[TaskArg], TaskArg], TaskArg]


class BootStrategy(BootStrategyT):
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool

    def __init__(
        self,
        app: AppT,
        *,
        enable_web: Optional[bool] = ...,
        enable_kafka: Optional[bool] = ...,
        enable_kafka_producer: Optional[bool] = ...,
        enable_kafka_consumer: Optional[bool] = ...,
        enable_sensors: Optional[bool] = ...,
    ) -> None: ...
    def server(self) -> Iterable[ServiceT]: ...
    def client_only(self) -> Iterable[ServiceT]: ...
    def producer_only(self) -> Iterable[ServiceT]: ...
    def sensors(self) -> Iterable[ServiceT]: ...
    def kafka_producer(self) -> Iterable[ServiceT]: ...
    def kafka_consumer(self) -> Iterable[ServiceT]: ...
    def kafka_client_consumer(self) -> Iterable[ServiceT]: ...
    def agents(self) -> Iterable[ServiceT]: ...
    def kafka_conductor(self) -> Iterable[ServiceT]: ...
    def web_server(self) -> Iterable[ServiceT]: ...
    def web_components(self) -> Iterable[ServiceT]: ...
    def tables(self) -> Iterable[ServiceT]: ...


class App(AppT, Service):
    SCAN_CATEGORIES: List[str]
    BootStrategy: Type[BootStrategy]
    Settings: Type['_Settings']
    client_only: bool
    producer_only: bool
    tracer: Optional[TracerT]
    agents: AgentManager
    sensors: SensorDelegate
    fixups: List[FixupT]
    boot_strategy: BootStrategy
    on_startup_finished: Optional[Callable[[], Awaitable[None]]]

    def __init__(
        self,
        id: str,
        *,
        monitor: Optional[Monitor] = ...,
        config_source: Any = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
        beacon: Any = ...,
        **options: Any
    ) -> None: ...
    def on_init_dependencies(self) -> Iterable[ServiceT]: ...
    async def on_first_start(self) -> None: ...
    async def on_start(self) -> None: ...
    async def on_started(self) -> None: ...
    async def on_started_init_extra_tasks(self) -> None: ...
    async def on_started_init_extra_services(self) -> None: ...
    async def on_init_extra_service(self, service: Union[Type[ServiceT], ServiceT]) -> ServiceT: ...
    def config_from_object(self, obj: Any, *, silent: bool = ..., force: bool = ...) -> None: ...
    def finalize(self) -> None: ...
    def worker_init(self) -> None: ...
    def worker_init_post_autodiscover(self) -> None: ...
    def discover(
        self,
        *extra_modules: str,
        categories: Optional[Iterable[str]] = ...,
        ignore: Iterable[Any] = ...,
    ) -> None: ...
    def main(self) -> None: ...
    def topic(
        self,
        *topics: str,
        pattern: Union[str, Pattern[str]] = ...,
        schema: SchemaT = ...,
        key_type: ModelArg = ...,
        value_type: ModelArg = ...,
        key_serializer: CodecArg = ...,
        value_serializer: CodecArg = ...,
        partitions: int = ...,
        retention: Seconds = ...,
        compacting: bool = ...,
        deleting: bool = ...,
        replicas: int = ...,
        acks: bool = ...,
        internal: bool = ...,
        config: Mapping[str, Any] = ...,
        maxsize: int = ...,
        allow_empty: bool = ...,
        has_prefix: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> TopicT: ...
    def channel(
        self,
        *,
        schema: SchemaT = ...,
        key_type: ModelArg = ...,
        value_type: ModelArg = ...,
        maxsize: int = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ChannelT: ...
    def agent(
        self,
        channel: Union[str, ChannelT] = ...,
        *,
        name: Optional[str] = ...,
        concurrency: int = ...,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = ...,
        sink: Optional[Iterable[SinkT]] = ...,
        isolated_partitions: bool = ...,
        use_reply_headers: bool = ...,
        **kwargs: Any
    ) -> Callable[[AgentFun], AgentT]: ...
    def actor(
        self,
        channel: Union[str, ChannelT] = ...,
        *,
        name: Optional[str] = ...,
        concurrency: int = ...,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = ...,
        sink: Optional[Iterable[SinkT]] = ...,
        isolated_partitions: bool = ...,
        use_reply_headers: bool = ...,
        **kwargs: Any
    ) -> Callable[[AgentFun], AgentT]: ...
    def task(self, fun: Optional[TaskArg] = ..., *, on_leader: bool = ..., traced: bool = ...) -> TaskDecoratorRet: ...
    def timer(
        self,
        interval: Seconds,
        on_leader: bool = ...,
        traced: bool = ...,
        name: Optional[str] = ...,
        max_drift_correction: float = ...,
    ) -> Callable[[TaskArg], TaskArg]: ...
    def crontab(
        self,
        cron_format: str,
        *,
        timezone: Optional[tzinfo] = ...,
        on_leader: bool = ...,
        traced: bool = ...,
    ) -> Callable[[TaskArg], TaskArg]: ...
    def service(self, cls: Type[ServiceT]) -> Type[ServiceT]: ...
    def is_leader(self) -> bool: ...
    def stream(self, channel: AsyncIterable, beacon: Optional[NodeT] = ..., **kwargs: Any) -> StreamT: ...
    def Table(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = ...,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        help: Optional[str] = ...,
        **kwargs: Any
    ) -> TableT: ...
    def GlobalTable(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = ...,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        help: Optional[str] = ...,
        **kwargs: Any
    ) -> TableT: ...
    def SetTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any
    ) -> TableT: ...
    def SetGlobalTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any
    ) -> TableT: ...
    def page(
        self,
        path: str,
        *,
        base: Optional[Type[View]] = ...,
        cors_options: Optional[Mapping[str, ResourceOptions]] = ...,
        name: Optional[str] = ...,
    ) -> Callable[[PageArg], Type[View]]: ...
    def table_route(
        self,
        table: TableT,
        shard_param: Optional[str] = ...,
        *,
        query_param: Optional[str] = ...,
        match_info: Optional[str] = ...,
        exact_key: Optional[str] = ...,
    ) -> Callable[[Callable[..., Awaitable[Response]]], Callable[..., Awaitable[Response]]]: ...
    def command(
        self,
        *options: Any,
        base: Optional[Type[_AppCommand]] = ...,
        **kwargs: Any
    ) -> Callable[[Callable[..., Any]], _AppCommand]: ...
    def create_event(self, key: K, value: V, headers: HeadersArg, message: Message) -> EventT: ...
    async def start_client(self) -> None: ...
    async def maybe_start_client(self) -> None: ...
    def trace(self, name: str, trace_enabled: bool = ..., **extra_context: Any) -> ContextManager[Any]: ...
    def traced(self, fun: Callable[..., Any], name: Optional[str] = ..., sample_rate: float = ..., **context: Any) -> Callable[..., Any]: ...
    async def send(
        self,
        channel: Union[str, ChannelT],
        key: K = ...,
        value: V = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
    ) -> RecordMetadata: ...
    @property
    def in_transaction(self) -> bool: ...
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck: ...
    async def maybe_start_producer(self) -> ServiceT: ...
    async def commit(self, topics: Optional[TPorTopicSet]) -> None: ...
    async def on_stop(self) -> None: ...
    def on_rebalance_start(self) -> None: ...
    def on_rebalance_return(self) -> None: ...
    def on_rebalance_end(self) -> None: ...
    def FlowControlQueue(
        self,
        maxsize: Optional[int] = ...,
        *,
        clear_on_resume: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ThrowableQueue: ...
    def Worker(self, **kwargs: Any) -> _Worker: ...
    def on_webserver_init(self, web: Web) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def conf(self) -> '_Settings': ...
    @conf.setter
    def conf(self, settings: '_Settings') -> None: ...
    @property
    def producer(self) -> ProducerT: ...
    @producer.setter
    def producer(self, producer: ProducerT) -> None: ...
    @property
    def consumer(self) -> ConsumerT: ...
    @consumer.setter
    def consumer(self, consumer: ConsumerT) -> None: ...
    @property
    def transport(self) -> TransportT: ...
    @transport.setter
    def transport(self, transport: TransportT) -> None: ...
    @property
    def producer_transport(self) -> TransportT: ...
    @producer_transport.setter
    def producer_transport(self, transport: TransportT) -> None: ...
    @property
    def cache(self) -> CacheBackendT: ...
    @cache.setter
    def cache(self, cache: CacheBackendT) -> None: ...
    @property
    def tables(self) -> TableManagerT: ...
    @property
    def topics(self) -> ConductorT: ...
    @property
    def monitor(self) -> Monitor: ...
    @monitor.setter
    def monitor(self, monitor: Monitor) -> None: ...
    @property
    def flow_control(self) -> FlowControlEvent: ...
    @property
    def http_client(self) -> HttpClientT: ...
    @http_client.setter
    def http_client(self, client: HttpClientT) -> None: ...
    @property
    def assignor(self) -> 'PartitionAssignorT': ...
    @property
    def router(self) -> RouterT: ...
    @property
    def web(self) -> Web: ...
    @property
    def serializers(self) -> RegistryT: ...
    @property
    def label(self) -> str: ...
    @property
    def shortlabel(self) -> str: ...