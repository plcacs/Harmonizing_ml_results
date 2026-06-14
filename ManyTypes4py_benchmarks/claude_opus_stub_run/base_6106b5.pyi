import asyncio
import typing
from datetime import tzinfo
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import opentracing
from mode import Seconds, Service, ServiceT, SupervisorStrategyT
from mode.utils.futures import stampede
from mode.utils.logging import flight_recorder
from mode.utils.objects import cached_property
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT

from faust.agents import AgentFun, AgentManager, AgentT, ReplyConsumer, SinkT
from faust.channels import Channel, ChannelT
from faust.fixups import FixupT
from faust.sensors import Monitor, SensorDelegate
from faust.web.views import View
from faust.types.app import AppT, BootStrategyT, TaskArg, TracerT
from faust.types.assignor import LeaderAssignorT, PartitionAssignorT
from faust.types.codecs import CodecArg
from faust.types.core import HeadersArg, K, V
from faust.types.enums import ProcessingGuarantee
from faust.types.events import EventT
from faust.types.models import ModelArg
from faust.types.router import RouterT
from faust.types.serializers import RegistryT, SchemaT
from faust.types.settings import Settings as _Settings
from faust.types.streams import StreamT
from faust.types.tables import CollectionT, GlobalTableT, TableManagerT, TableT
from faust.types.topics import TopicT
from faust.types.transports import ConductorT, ConsumerT, ProducerT, TPorTopicSet, TransportT
from faust.types.tuples import Message, MessageSentCallback, RecordMetadata, TP
from faust.types.web import (
    CacheBackendT,
    HttpClientT,
    PageArg,
    Request,
    ResourceOptions,
    Response,
    ViewDecorator,
    ViewHandlerFun,
    Web,
)
from faust.types.windows import WindowT
from ._attached import Attachments

if typing.TYPE_CHECKING:
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck import LiveCheck as _LiveCheck
    from faust.transport.consumer import Fetcher as _Fetcher
    from faust.worker import Worker as _Worker
else:
    class _AppCommand: ...
    class _LiveCheck: ...
    class _Fetcher: ...
    class _Worker: ...

__all__: List[str]

logger: Any

_T = TypeVar('_T')

APP_REPR_FINALIZED: str
APP_REPR_UNFINALIZED: str
SCAN_AGENT: str
SCAN_COMMAND: str
SCAN_PAGE: str
SCAN_SERVICE: str
SCAN_TASK: str
SCAN_CATEGORIES: List[str]
SCAN_IGNORE: List[Any]
E_NEED_ORIGIN: str
W_OPTION_DEPRECATED: str
W_DEPRECATED_SHARD_PARAM: str

TaskDecoratorRet = Union[Callable[[TaskArg], TaskArg], TaskArg]


class BootStrategy(BootStrategyT):
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool
    app: AppT

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
    def _chain(self, *arguments: Iterable[ServiceT]) -> Iterable[ServiceT]: ...
    def sensors(self) -> Iterable[ServiceT]: ...
    def kafka_producer(self) -> Iterable[ServiceT]: ...
    def _should_enable_kafka_producer(self) -> bool: ...
    def kafka_consumer(self) -> Iterable[ServiceT]: ...
    def _should_enable_kafka_consumer(self) -> bool: ...
    def kafka_client_consumer(self) -> Iterable[ServiceT]: ...
    def agents(self) -> Iterable[ServiceT]: ...
    def kafka_conductor(self) -> Iterable[ServiceT]: ...
    def web_server(self) -> Iterable[ServiceT]: ...
    def _should_enable_web(self) -> bool: ...
    def web_components(self) -> Iterable[ServiceT]: ...
    def tables(self) -> Iterable[ServiceT]: ...


class App(AppT, Service):
    SCAN_CATEGORIES: List[str]
    BootStrategy: Type[BootStrategyT]
    Settings: Type[_Settings]
    client_only: bool
    producer_only: bool
    _conf: Optional[_Settings]
    _config_source: Any
    _consumer: Optional[ConsumerT]
    _producer: Optional[ProducerT]
    _transport: Optional[TransportT]
    _producer_transport: Optional[TransportT]
    _cache: Optional[CacheBackendT]
    _monitor: Optional[Monitor]
    _http_client: Optional[HttpClientT]
    _extra_service_instances: Optional[List[ServiceT]]
    _assignment: Optional[Set[TP]]
    tracer: Optional[TracerT]
    _rebalancing_span: Optional[opentracing.Span]
    _rebalancing_sensor_state: Optional[Dict[str, Any]]
    agents: AgentManager
    sensors: SensorDelegate
    _attachments: Attachments
    _app_tasks: List[Callable[[], Awaitable[Any]]]
    on_startup_finished: Optional[Callable[[], Awaitable[None]]]
    _extra_services: List[Union[ServiceT, Type[ServiceT]]]
    fixups: List[FixupT]
    boot_strategy: BootStrategyT
    _default_options: Tuple[str, Dict[str, Any]]

    def __init__(
        self,
        id: str,
        *,
        monitor: Optional[Monitor] = ...,
        config_source: Any = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
        beacon: Optional[NodeT] = ...,
        **options: Any,
    ) -> None: ...
    def _init_signals(self) -> None: ...
    def _init_fixups(self) -> List[FixupT]: ...
    def on_init_dependencies(self) -> Iterable[ServiceT]: ...
    async def on_first_start(self) -> None: ...
    async def on_start(self) -> None: ...
    async def on_started(self) -> None: ...
    async def _wait_for_table_recovery_completed(self) -> bool: ...
    async def on_started_init_extra_tasks(self) -> None: ...
    async def on_started_init_extra_services(self) -> None: ...
    async def on_init_extra_service(self, service: Union[ServiceT, Type[ServiceT]]) -> ServiceT: ...
    def _prepare_subservice(self, service: Union[ServiceT, Type[ServiceT]]) -> ServiceT: ...
    def config_from_object(self, obj: Any, *, silent: bool = ..., force: bool = ...) -> None: ...
    def finalize(self) -> None: ...
    async def _maybe_close_http_client(self) -> None: ...
    def worker_init(self) -> None: ...
    def worker_init_post_autodiscover(self) -> None: ...
    def discover(
        self,
        *extra_modules: str,
        categories: Optional[List[str]] = ...,
        ignore: List[Any] = ...,
    ) -> None: ...
    def _on_autodiscovery_error(self, name: str) -> None: ...
    def _discovery_modules(self) -> List[str]: ...
    def main(self) -> None: ...
    def topic(
        self,
        *topics: str,
        pattern: Optional[Union[str, Pattern[str]]] = ...,
        schema: Optional[SchemaT] = ...,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        partitions: Optional[int] = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        replicas: Optional[int] = ...,
        acks: bool = ...,
        internal: bool = ...,
        config: Optional[Mapping[str, Any]] = ...,
        maxsize: Optional[int] = ...,
        allow_empty: bool = ...,
        has_prefix: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> TopicT: ...
    def channel(
        self,
        *,
        schema: Optional[SchemaT] = ...,
        key_type: Optional[ModelArg] = ...,
        value_type: Optional[ModelArg] = ...,
        maxsize: Optional[int] = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ChannelT: ...
    def agent(
        self,
        channel: Optional[Union[str, ChannelT]] = ...,
        *,
        name: Optional[str] = ...,
        concurrency: int = ...,
        supervisor_strategy: Optional[Type[SupervisorStrategyT]] = ...,
        sink: Optional[Iterable[SinkT]] = ...,
        isolated_partitions: bool = ...,
        use_reply_headers: bool = ...,
        **kwargs: Any,
    ) -> Callable[[AgentFun], AgentT]: ...
    actor = agent
    async def _on_agent_error(self, agent: AgentT, exc: BaseException) -> None: ...
    def task(
        self,
        fun: Optional[TaskArg] = ...,
        *,
        on_leader: bool = ...,
        traced: bool = ...,
    ) -> TaskDecoratorRet: ...
    def _task(
        self,
        fun: TaskArg,
        on_leader: bool = ...,
        traced: bool = ...,
    ) -> TaskArg: ...
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
    def stream(
        self,
        channel: Union[AsyncIterable[Any], Iterable[Any], ChannelT],
        beacon: Optional[NodeT] = ...,
        **kwargs: Any,
    ) -> StreamT: ...
    def Table(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = ...,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> TableT: ...
    def GlobalTable(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = ...,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> GlobalTableT: ...
    def SetTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> TableT: ...
    def SetGlobalTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> TableT: ...
    def page(
        self,
        path: str,
        *,
        base: Optional[Type[View]] = ...,
        cors_options: Optional[Mapping[str, ResourceOptions]] = ...,
        name: Optional[str] = ...,
    ) -> Callable[[Any], Type[View]]: ...
    def table_route(
        self,
        table: CollectionT,
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
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Any]: ...
    def create_event(
        self,
        key: K,
        value: V,
        headers: HeadersArg,
        message: Message,
    ) -> EventT: ...
    async def start_client(self) -> None: ...
    async def maybe_start_client(self) -> None: ...
    def trace(
        self,
        name: str,
        trace_enabled: bool = ...,
        **extra_context: Any,
    ) -> ContextManager[Any]: ...
    def traced(
        self,
        fun: Callable[..., Any],
        name: Optional[str] = ...,
        sample_rate: float = ...,
        **context: Any,
    ) -> Callable[..., Any]: ...
    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span: ...
    async def send(
        self,
        channel: Union[str, ChannelT],
        key: Optional[K] = ...,
        value: Optional[V] = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
    ) -> Awaitable[RecordMetadata]: ...
    @cached_property
    def in_transaction(self) -> bool: ...
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck: ...
    @stampede
    async def maybe_start_producer(self) -> ProducerT: ...
    async def commit(self, topics: TPorTopicSet) -> bool: ...
    async def on_stop(self) -> None: ...
    async def _producer_flush(self, logger: Any) -> None: ...
    async def _stop_consumer(self) -> None: ...
    async def _consumer_wait_empty(self, consumer: ConsumerT, logger: Any) -> None: ...
    def on_rebalance_start(self) -> None: ...
    def _span_add_default_tags(self, span: opentracing.Span) -> None: ...
    def on_rebalance_return(self) -> None: ...
    def on_rebalance_end(self) -> None: ...
    async def _on_partitions_revoked(self, revoked: Set[TP]) -> None: ...
    async def _stop_fetcher(self) -> None: ...
    def _on_rebalance_when_stopped(self) -> None: ...
    async def _on_partitions_assigned(self, assigned: Set[TP]) -> None: ...
    def _update_assignment(self, assigned: Set[TP]) -> Tuple[Set[TP], Set[TP]]: ...
    def _new_producer(self) -> ProducerT: ...
    def _new_consumer(self) -> ConsumerT: ...
    def _new_conductor(self) -> ConductorT: ...
    def _new_transport(self) -> TransportT: ...
    def _new_producer_transport(self) -> TransportT: ...
    def _new_cache_backend(self) -> CacheBackendT: ...
    def FlowControlQueue(
        self,
        maxsize: Optional[int] = ...,
        *,
        clear_on_resume: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ThrowableQueue: ...
    def Worker(self, **kwargs: Any) -> _Worker: ...
    def on_webserver_init(self, web: Web) -> None: ...
    def _create_directories(self) -> None: ...
    def __repr__(self) -> str: ...
    def _configure(self, *, silent: bool = ...) -> None: ...
    def _load_settings(self, *, silent: bool = ...) -> _Settings: ...
    def _prepare_compat_settings(self, options: Dict[str, Any]) -> Dict[str, Any]: ...
    def _load_settings_from_source(self, source: Any, *, silent: bool = ...) -> Mapping[str, Any]: ...
    @property
    def conf(self) -> _Settings: ...
    @conf.setter
    def conf(self, settings: _Settings) -> None: ...
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
    @cached_property
    def tables(self) -> TableManagerT: ...
    @cached_property
    def topics(self) -> ConductorT: ...
    @property
    def monitor(self) -> Monitor: ...
    @monitor.setter
    def monitor(self, monitor: Monitor) -> None: ...
    @cached_property
    def _fetcher(self) -> _Fetcher: ...
    @cached_property
    def _reply_consumer(self) -> ReplyConsumer: ...
    @cached_property
    def flow_control(self) -> FlowControlEvent: ...
    @property
    def http_client(self) -> HttpClientT: ...
    @http_client.setter
    def http_client(self, client: HttpClientT) -> None: ...
    @cached_property
    def assignor(self) -> PartitionAssignorT: ...
    @cached_property
    def _leader_assignor(self) -> LeaderAssignorT: ...
    @cached_property
    def router(self) -> RouterT: ...
    @cached_property
    def web(self) -> Web: ...
    def _new_web(self) -> Web: ...
    @cached_property
    def serializers(self) -> RegistryT: ...
    @property
    def label(self) -> str: ...
    @property
    def shortlabel(self) -> str: ...