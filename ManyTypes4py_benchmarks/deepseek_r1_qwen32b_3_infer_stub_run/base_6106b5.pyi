"""Faust Application.

An app is an instance of the Faust library.
Everything starts here.

"""

from __future__ import annotations
from datetime import tzinfo
from functools import wraps
from itertools import chain
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
    cast,
    no_type_check,
)
import asyncio
import warnings
import opentracing
from mode import ServiceT, SupervisorStrategyT, Seconds
from mode.utils.contexts import nullcontext
from mode.utils.typing import NoReturn
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
from faust.types.web import CacheBackendT, HttpClientT, PageArg, Request, ResourceOptions, Response, View, ViewDecorator, ViewHandlerFun, Web
from faust.types.windows import WindowT

__all__: List[str] = ['App', 'BootStrategy']
logger: Any = ...

_T: TypeVar = TypeVar('_T')

class BootStrategy(BootStrategyT):
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool

    def __init__(self, app: App, *, enable_web: Optional[bool] = None, enable_kafka: Optional[bool] = None, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: Optional[bool] = None) -> None:
        ...

    def server(self) -> Iterable[ServiceT]:
        ...

    def client_only(self) -> Iterable[ServiceT]:
        ...

    def producer_only(self) -> Iterable[ServiceT]:
        ...

    def _chain(self, *arguments: Iterable[ServiceT]) -> Iterable[ServiceT]:
        ...

    def sensors(self) -> Iterable[ServiceT]:
        ...

    def kafka_producer(self) -> Iterable[ServiceT]:
        ...

    def _should_enable_kafka_producer(self) -> bool:
        ...

    def kafka_consumer(self) -> Iterable[ServiceT]:
        ...

    def _should_enable_kafka_consumer(self) -> bool:
        ...

    def kafka_client_consumer(self) -> Iterable[ServiceT]:
        ...

    def agents(self) -> Iterable[ServiceT]:
        ...

    def kafka_conductor(self) -> Iterable[ServiceT]:
        ...

    def web_server(self) -> Iterable[ServiceT]:
        ...

    def _should_enable_web(self) -> bool:
        ...

    def web_components(self) -> Iterable[ServiceT]:
        ...

    def tables(self) -> Iterable[ServiceT]:
        ...

class App(AppT, Service):
    SCAN_CATEGORIES: List[str] = ...
    BootStrategy: Type[BootStrategy] = ...
    Settings: Type[_Settings] = ...
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
    _rebalancing_sensor_state: Any

    def __init__(self, id: str, *, monitor: Optional[Monitor] = None, config_source: Any = None, loop: Optional[asyncio.AbstractEventLoop] = None, beacon: Any = None, **options: Any) -> None:
        ...

    def _init_signals(self) -> None:
        ...

    def _init_fixups(self) -> List[Any]:
        ...

    def on_init_dependencies(self) -> Iterable[ServiceT]:
        ...

    async def on_first_start(self) -> None:
        ...

    async def on_start(self) -> None:
        ...

    async def on_started(self) -> None:
        ...

    async def _wait_for_table_recovery_completed(self) -> bool:
        ...

    async def on_started_init_extra_tasks(self) -> None:
        ...

    async def on_started_init_extra_services(self) -> None:
        ...

    async def on_init_extra_service(self, service: ServiceT) -> ServiceT:
        ...

    def _prepare_subservice(self, service: Union[Type[ServiceT], ServiceT]) -> ServiceT:
        ...

    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None:
        ...

    def finalize(self) -> None:
        ...

    async def _maybe_close_http_client(self) -> None:
        ...

    def worker_init(self) -> None:
        ...

    def worker_init_post_autodiscover(self) -> None:
        ...

    def discover(self, *extra_modules: str, categories: Optional[List[str]] = None, ignore: Optional[List[Union[str, Pattern[str]]]] = None) -> None:
        ...

    def _on_autodiscovery_error(self, name: str) -> None:
        ...

    def _discovery_modules(self) -> List[str]:
        ...

    def main(self) -> None:
        ...

    def topic(self, *topics: str, pattern: Optional[Pattern[str]] = None, **kwargs: Any) -> TopicT:
        ...

    def channel(self, *, schema: Optional[SchemaT] = None, key_type: Optional[Type[K]] = None, value_type: Optional[Type[V]] = None, maxsize: Optional[int] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> ChannelT:
        ...

    def agent(self, channel: Optional[ChannelT] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional[SupervisorStrategyT] = None, sink: Optional[SinkT] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> Callable[[Callable[..., Awaitable]], AgentT]:
        ...

    actor: Callable[[Callable[..., Awaitable]], AgentT] = ...

    async def _on_agent_error(self, agent: AgentT, exc: BaseException) -> None:
        ...

    @no_type_check
    def task(self, fun: Optional[Callable[..., Awaitable]] = None, *, on_leader: bool = False, traced: bool = True) -> Union[Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]], Callable[..., Awaitable]]:
        ...

    def _task(self, fun: Callable[..., Awaitable], on_leader: bool = False, traced: bool = False) -> Callable[..., Awaitable]:
        ...

    @no_type_check
    def timer(self, interval: Seconds, on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
        ...

    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = None, on_leader: bool = False, traced: bool = True) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
        ...

    def service(self, cls: Type[ServiceT]) -> Type[ServiceT]:
        ...

    def is_leader(self) -> bool:
        ...

    def stream(self, channel: AsyncIterable, beacon: Optional[Any] = None, **kwargs: Any) -> StreamT:
        ...

    def Table(self, name: str, *, default: Optional[Callable[..., Any]] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT:
        ...

    def GlobalTable(self, name: str, *, default: Optional[Callable[..., Any]] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> GlobalTableT:
        ...

    def SetTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        ...

    def SetGlobalTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> GlobalTableT:
        ...

    def page(self, path: str, *, base: Type[View] = View, cors_options: Optional[ResourceOptions] = None, name: Optional[str] = None) -> Callable[[Callable[..., Awaitable]], Type[View]]:
        ...

    def table_route(self, table: str, shard_param: Optional[str] = None, *, query_param: Optional[str] = None, match_info: Optional[str] = None, exact_key: Optional[Any] = None) -> Callable[[Callable[[View, Request, *Any, **Any], Awaitable]], Callable[[View, Request, *Any, **Any], Awaitable]]:
        ...

    def command(self, *options: Any, base: Optional[Type[Any]] = None, **kwargs: Any) -> Callable[[Callable[..., Awaitable]], Any]:
        ...

    def create_event(self, key: K, value: V, headers: HeadersArg, message: Message) -> EventT:
        ...

    async def start_client(self) -> None:
        ...

    async def maybe_start_client(self) -> None:
        ...

    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> ContextManager:
        ...

    def traced(self, fun: Callable[..., Awaitable], name: Optional[str] = None, sample_rate: float = 1.0, **context: Any) -> Callable[..., Awaitable]:
        ...

    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span:
        ...

    async def send(self, channel: Union[ChannelT, str], key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: HeadersArg = None, schema: Optional[SchemaT] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, callback: Optional[MessageSentCallback] = None) -> RecordMetadata:
        ...

    @cached_property
    def in_transaction(self) -> bool:
        ...

    def LiveCheck(self, **kwargs: Any) -> Any:
        ...

    async def maybe_start_producer(self) -> Union[ProducerT, Any]:
        ...

    async def commit(self, topics: Optional[Set[TP]] = None) -> None:
        ...

    async def on_stop(self) -> None:
        ...

    async def _producer_flush(self, logger: Any) -> None:
        ...

    async def _stop_consumer(self) -> None:
        ...

    def on_rebalance_start(self) -> None:
        ...

    def _span_add_default_tags(self, span: opentracing.Span) -> None:
        ...

    def on_rebalance_return(self) -> None:
        ...

    def on_rebalance_end(self) -> None:
        ...

    async def _on_partitions_revoked(self, revoked: Set[TP]) -> None:
        ...

    async def _stop_fetcher(self) -> None:
        ...

    def _on_rebalance_when_stopped(self) -> None:
        ...

    async def _on_partitions_assigned(self, assigned: Set[TP]) -> None:
        ...

    def _update_assignment(self, assigned: Set[TP]) -> Tuple[Set[TP], Set[TP]]:
        ...

    def _new_producer(self) -> ProducerT:
        ...

    def _new_consumer(self) -> ConsumerT:
        ...

    def _new_conductor(self) -> ConductorT:
        ...

    def _new_transport(self) -> TransportT:
        ...

    def _new_producer_transport(self) -> TransportT:
        ...

    def _new_cache_backend(self) -> CacheBackendT:
        ...

    def FlowControlQueue(self, maxsize: Optional[int] = None, *, clear_on_resume: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> ThrowableQueue:
        ...

    def Worker(self, **kwargs: Any) -> Any:
        ...

    def on_webserver_init(self, web: Web) -> None:
        ...

    def _create_directories(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def _configure(self, *, silent: bool = False) -> None:
        ...

    def _load_settings(self, *, silent: bool = False) -> Dict[str, Any]:
        ...

    def _prepare_compat_settings(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def _load_settings_from_source(self, source: Any, *, silent: bool = False) -> Dict[str, Any]:
        ...

    @property
    def conf(self) -> _Settings:
        ...

    @conf.setter
    def conf(self, settings: _Settings) -> None:
        ...

    @property
    def producer(self) -> ProducerT:
        ...

    @producer.setter
    def producer(self, producer: ProducerT) -> None:
        ...

    @property
    def consumer(self) -> ConsumerT:
        ...

    @consumer.setter
    def consumer(self, consumer: ConsumerT) -> None:
        ...

    @property
    def transport(self) -> TransportT:
        ...

    @transport.setter
    def transport(self, transport: TransportT) -> None:
        ...

    @property
    def producer_transport(self) -> TransportT:
        ...

    @producer_transport.setter
    def producer_transport(self, transport: TransportT) -> None:
        ...

    @property
    def cache(self) -> CacheBackendT:
        ...

    @cache.setter
    def cache(self, cache: CacheBackendT) -> None:
        ...

    @cached_property
    def tables(self) -> TableManagerT:
        ...

    @cached_property
    def topics(self) -> ConductorT:
        ...

    @property
    def monitor(self) -> Monitor:
        ...

    @monitor.setter
    def monitor(self, monitor: Monitor) -> None:
        ...

    @cached_property
    def _fetcher(self) -> Any:
        ...

    @cached_property
    def _reply_consumer(self) -> ReplyConsumer:
        ...

    @cached_property
    def flow_control(self) -> FlowControlEvent:
        ...

    @property
    def http_client(self) -> HttpClientT:
        ...

    @http_client.setter
    def http_client(self, client: HttpClientT) -> None:
        ...

    @cached_property
    def assignor(self) -> PartitionAssignorT:
        ...

    @cached_property
    def _leader_assignor(self) -> LeaderAssignorT:
        ...

    @cached_property
    def router(self) -> RouterT:
        ...

    @cached_property
    def web(self) -> Web:
        ...

    def _new_web(self) -> Web:
        ...

    @cached_property
    def serializers(self) -> RegistryT:
        ...

    @property
    def label(self) -> str:
        ...

    @property
    def shortlabel(self) -> str:
        ...