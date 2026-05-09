"""Faust Application stub file."""

import asyncio
import re
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
    cast,
    no_type_check,
)
from mode import Seconds, Service, ServiceT, SupervisorStrategyT, want_seconds
from mode.utils.aiter import aiter
from mode.utils.collections import force_mapping
from mode.utils.contexts import nullcontext
from mode.utils.futures import stampede
from mode.utils.imports import import_from_cwd, smart_import
from mode.utils.logging import flight_recorder, get_logger
from mode.utils.objects import cached_property, qualname, shortlabel
from mode.utils.typing import NoReturn
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT
from faust import transport
from faust.agents import AgentFun, AgentManager, AgentT, ReplyConsumer, SinkT
from faust.channels import Channel, ChannelT
from faust.exceptions import ConsumerNotStarted, ImproperlyConfigured, SameNode
from faust.fixups import FixupT, fixups
from faust.sensors import Monitor, SensorDelegate
from faust.utils import cron, venusian
from faust.utils.tracing import call_with_trace, noop_span, operation_name_from_fun, set_current_span, traced_from_parent_span
from faust.web import drivers as web_drivers
from faust.web.cache import backends as cache_backends
from faust.web.views import View
from faust.types._env import STRICT
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
from faust.types.web import CacheBackendT, HttpClientT, PageArg, Request, ResourceOptions, Response, ViewDecorator, ViewHandlerFun, Web
from faust.types.windows import WindowT

__all__ = ['App', 'BootStrategy']

class BootStrategy(BootStrategyT):
    enable_kafka: bool
    enable_kafka_consumer: Optional[bool]
    enable_kafka_producer: Optional[bool]
    enable_web: Optional[bool]
    enable_sensors: bool

    def __init__(self, app: AppT, *, enable_web: Optional[bool] = None, enable_kafka: Optional[bool] = None, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: Optional[bool] = None) -> None:
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
    SCAN_CATEGORIES: List[str]
    BootStrategy: Type[BootStrategy]
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
    _rebalancing_sensor_state: Any

    def __init__(self, id: str, *, monitor: Optional[Monitor] = None, config_source: Any = None, loop: Optional[asyncio.AbstractEventLoop] = None, beacon: Optional[NodeT] = None, **options: Any) -> None:
        ...

    def _init_signals(self) -> None:
        ...

    def _init_fixups(self) -> List[FixupT]:
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

    async def on_init_extra_service(self, service: Union[Type[ServiceT], ServiceT]) -> ServiceT:
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

    def discover(self, *extra_modules: str, categories: Optional[List[str]] = None, ignore: List[Union[Pattern[str], str]] = [re.compile('test_.*').search, '.__main__']) -> None:
        ...

    def _on_autodiscovery_error(self, name: str) -> None:
        ...

    def _discovery_modules(self) -> List[str]:
        ...

    def main(self) -> None:
        ...

    def topic(self, *topics: str, pattern: Optional[Pattern[str]] = None, schema: Optional[SchemaT] = None, key_type: Optional[Type[K]] = None, value_type: Optional[Type[V]] = None, key_serializer: CodecArg = None, value_serializer: CodecArg = None, partitions: Optional[int] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Dict[str, Any]] = None, maxsize: Optional[int] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> TopicT:
        ...

    def channel(self, *, schema: Optional[SchemaT] = None, key_type: Optional[Type[K]] = None, value_type: Optional[Type[V]] = None, maxsize: Optional[int] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> ChannelT:
        ...

    def agent(self, channel: Optional[Union[ChannelT, str]] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional[SupervisorStrategyT] = None, sink: Optional[SinkT] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> Callable[[AgentFun], AgentT]:
        ...

    actor = agent

    async def _on_agent_error(self, agent: AgentT, exc: Exception) -> None:
        ...

    @no_type_check
    def task(self, fun: Optional[Callable] = None, *, on_leader: bool = False, traced: bool = True) -> Union[Callable[[Callable], TaskArg], TaskArg]:
        ...

    def _task(self, fun: Callable, on_leader: bool = False, traced: bool = False) -> TaskArg:
        ...

    @no_type_check
    def timer(self, interval: Seconds, on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1) -> Callable[[Callable], TaskArg]:
        ...

    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = None, on_leader: bool = False, traced: bool = True) -> Callable[[Callable], TaskArg]:
        ...

    def service(self, cls: Type[ServiceT]) -> Type[ServiceT]:
        ...

    def is_leader(self) -> bool:
        ...

    def stream(self, channel: AsyncIterable, beacon: Optional[NodeT] = None, **kwargs: Any) -> StreamT:
        ...

    def Table(self, name: str, *, default: Optional[Callable] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT:
        ...

    def GlobalTable(self, name: str, *, default: Optional[Callable] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> GlobalTableT:
        ...

    def SetTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        ...

    def SetGlobalTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> GlobalTableT:
        ...

    def page(self, path: str, *, base: Optional[Type[View]] = None, cors_options: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> Callable[[Union[Type[View], Callable]], Type[View]]:
        ...

    def table_route(self, table: TableT, shard_param: Optional[str] = None, *, query_param: Optional[str] = None, match_info: Optional[str] = None, exact_key: Optional[Any] = None) -> Callable[[Callable], Callable]:
        ...

    def command(self, *options: Any, base: Optional[Type[FixupT]] = None, **kwargs: Any) -> Callable[[Callable], FixupT]:
        ...

    def create_event(self, key: K, value: V, headers: HeadersArg, message: Message) -> EventT:
        ...

    async def start_client(self) -> None:
        ...

    async def maybe_start_client(self) -> None:
        ...

    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> ContextManager:
        ...

    def traced(self, fun: Callable, name: Optional[str] = None, sample_rate: float = 1.0, **context: Any) -> Callable:
        ...

    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span:
        ...

    async def send(self, channel: Union[ChannelT, str], key: Optional[K] = None, value: Optional[V] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: HeadersArg = None, schema: Optional[SchemaT] = None, key_serializer: CodecArg = None, value_serializer: CodecArg = None, callback: Optional[MessageSentCallback] = None) -> RecordMetadata:
        ...

    @cached_property
    def in_transaction(self) -> bool:
        ...

    def LiveCheck(self, **kwargs: Any) -> Any:
        ...

    @stampede
    async def maybe_start_producer(self) -> Union[ProducerT, Any]:
        ...

    async def commit(self, topics: Optional[Union[TopicT, str, Iterable[Union[TopicT, str]]]] = None) -> None:
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

    def Worker(self, **kwargs: Any) -> _Worker:
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
    def _fetcher(self) -> _Fetcher:
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