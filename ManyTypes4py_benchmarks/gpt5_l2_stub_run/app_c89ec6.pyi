from __future__ import annotations

import abc
import asyncio
from datetime import tzinfo
from typing import Any, Awaitable, Callable, ClassVar, ContextManager, Iterable, Mapping, Optional, Pattern, Type, TypeVar, Union

import opentracing
from mode import Seconds, ServiceT, Signal, SupervisorStrategyT, SyncSignal
from mode.utils.futures import stampede
from mode.utils.objects import cached_property
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT

from .agents import AgentT, SinkT
from .assignor import PartitionAssignorT
from .codecs import CodecArg
from .core import HeadersArg, K, V
from .router import RouterT
from .serializers import RegistryT, SchemaT as _SchemaT
from .streams import StreamT
from .tables import CollectionT, TableManagerT, TableT
from .topics import ChannelT, TopicT
from .transports import ConsumerT, ProducerT, TransportT
from .tuples import Message, MessageSentCallback, RecordMetadata
from .web import CacheBackendT, HttpClientT, PageArg, ResourceOptions, View, ViewDecorator, Web
from .windows import WindowT
from .events import EventT as _EventT
from .models import ModelArg as _ModelArg
from .settings import Settings as _Settings
from faust.cli.base import AppCommand as _AppCommand
from faust.livecheck.app import LiveCheck as _LiveCheck
from faust.sensors.monitor import Monitor as _Monitor
from faust.worker import Worker as _Worker

__all__ = ['TaskArg', 'AppT']

_T = TypeVar('_T')

TaskArg = Union[Callable[['AppT'], Awaitable[None]], Callable[[], Awaitable[None]]]


class TracerT(abc.ABC):
    @property
    @abc.abstractmethod
    def default_tracer(self) -> opentracing.Tracer: ...

    @abc.abstractmethod
    def trace(
        self, name: str, sample_rate: Optional[float] = ..., **extra_context: Any
    ) -> ContextManager[opentracing.Span]: ...

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> opentracing.Tracer: ...


class BootStrategyT:
    enable_kafka: ClassVar[bool] = True
    enable_kafka_consumer: ClassVar[Optional[bool]] = None
    enable_kafka_producer: ClassVar[Optional[bool]] = None
    enable_web: ClassVar[Optional[bool]] = None
    enable_sensors: ClassVar[bool] = True

    @abc.abstractmethod
    def __init__(
        self,
        app: 'AppT',
        *,
        enable_web: Optional[bool] = ...,
        enable_kafka: bool = ...,
        enable_kafka_producer: Optional[bool] = ...,
        enable_kafka_consumer: Optional[bool] = ...,
        enable_sensors: bool = ...,
    ) -> None: ...

    @abc.abstractmethod
    def server(self) -> ServiceT: ...

    @abc.abstractmethod
    def client_only(self) -> ServiceT: ...

    @abc.abstractmethod
    def producer_only(self) -> ServiceT: ...


class AppT(ServiceT):
    finalized: ClassVar[bool] = ...
    configured: ClassVar[bool] = ...
    rebalancing: ClassVar[bool] = ...
    rebalancing_count: ClassVar[int] = ...
    unassigned: ClassVar[bool] = ...
    in_worker: ClassVar[bool] = ...
    on_configured: ClassVar[SyncSignal] = ...
    on_before_configured: ClassVar[SyncSignal] = ...
    on_after_configured: ClassVar[SyncSignal] = ...
    on_partitions_assigned: ClassVar[Signal] = ...
    on_partitions_revoked: ClassVar[Signal] = ...
    on_rebalance_complete: ClassVar[Signal] = ...
    on_before_shutdown: ClassVar[Signal] = ...
    on_worker_init: ClassVar[SyncSignal] = ...
    on_produce_message: ClassVar[SyncSignal] = ...
    tracer: ClassVar[Optional[TracerT]] = ...
    on_startup_finished: Optional[asyncio.Event]

    @abc.abstractmethod
    def __init__(
        self,
        id: str,
        *,
        monitor: _Monitor,
        config_source: Optional[Any] = ...,
        **options: Any,
    ) -> None: ...

    @abc.abstractmethod
    def config_from_object(self, obj: object, *, silent: bool = ..., force: bool = ...) -> None: ...

    @abc.abstractmethod
    def finalize(self) -> None: ...

    @abc.abstractmethod
    def main(self) -> None: ...

    @abc.abstractmethod
    def worker_init(self) -> None: ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None: ...

    @abc.abstractmethod
    def discover(
        self,
        *extra_modules: str,
        categories: Iterable[str] = ...,
        ignore: Iterable[str] = ...,
    ) -> None: ...

    @abc.abstractmethod
    def topic(
        self,
        *topics: str,
        pattern: Optional[Pattern[str]] = ...,
        schema: Optional[_SchemaT] = ...,
        key_type: Optional[_ModelArg] = ...,
        value_type: Optional[_ModelArg] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        partitions: Optional[int] = ...,
        retention: Optional[Seconds] = ...,
        compacting: Optional[bool] = ...,
        deleting: Optional[bool] = ...,
        replicas: Optional[int] = ...,
        acks: Union[bool, int] = ...,
        internal: bool = ...,
        config: Optional[Mapping[str, Any]] = ...,
        maxsize: Optional[int] = ...,
        allow_empty: bool = ...,
        has_prefix: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> TopicT: ...

    @abc.abstractmethod
    def channel(
        self,
        *,
        schema: Optional[_SchemaT] = ...,
        key_type: Optional[_ModelArg] = ...,
        value_type: Optional[_ModelArg] = ...,
        maxsize: Optional[int] = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ChannelT: ...

    @abc.abstractmethod
    def agent(
        self,
        channel: Optional[ChannelT] = ...,
        *,
        name: Optional[str] = ...,
        concurrency: int = ...,
        supervisor_strategy: Optional[SupervisorStrategyT] = ...,
        sink: Optional[SinkT] = ...,
        isolated_partitions: bool = ...,
        use_reply_headers: bool = ...,
        **kwargs: Any,
    ) -> AgentT: ...

    @abc.abstractmethod
    def task(self, fun: TaskArg, *, on_leader: bool = ..., traced: bool = ...) -> AgentT: ...

    @abc.abstractmethod
    def timer(
        self,
        interval: Seconds,
        on_leader: bool = ...,
        traced: bool = ...,
        name: Optional[str] = ...,
        max_drift_correction: float = ...,
    ) -> AgentT: ...

    @abc.abstractmethod
    def crontab(
        self,
        cron_format: str,
        *,
        timezone: Optional[tzinfo] = ...,
        on_leader: bool = ...,
        traced: bool = ...,
    ) -> AgentT: ...

    @abc.abstractmethod
    def service(self, cls: Type[ServiceT]) -> ServiceT: ...

    @abc.abstractmethod
    def stream(self, channel: ChannelT, beacon: Optional[NodeT] = ..., **kwargs: Any) -> StreamT: ...

    @abc.abstractmethod
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

    @abc.abstractmethod
    def GlobalTable(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = ...,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> TableT: ...

    @abc.abstractmethod
    def SetTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> CollectionT: ...

    @abc.abstractmethod
    def SetGlobalTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = ...,
        partitions: Optional[int] = ...,
        start_manager: bool = ...,
        help: Optional[str] = ...,
        **kwargs: Any,
    ) -> CollectionT: ...

    @abc.abstractmethod
    def page(
        self,
        path: PageArg,
        *,
        base: Type[View] = ...,
        cors_options: Optional[ResourceOptions] = ...,
        name: Optional[str] = ...,
    ) -> ViewDecorator: ...

    @abc.abstractmethod
    def table_route(
        self,
        table: TableT,
        shard_param: Optional[str] = ...,
        *,
        query_param: Optional[str] = ...,
        match_info: Optional[str] = ...,
        exact_key: Optional[bool] = ...,
    ) -> ViewDecorator: ...

    @abc.abstractmethod
    def command(
        self, *options: Any, base: Optional[Type[_AppCommand]] = ..., **kwargs: Any
    ) -> Union[_AppCommand, Callable[..., _AppCommand]]: ...

    @abc.abstractmethod
    def create_event(
        self, key: K, value: V, headers: HeadersArg, message: Message
    ) -> _EventT: ...

    @abc.abstractmethod
    async def start_client(self) -> None: ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None: ...

    @abc.abstractmethod
    def trace(
        self, name: str, trace_enabled: bool = ..., **extra_context: Any
    ) -> ContextManager[opentracing.Span]: ...

    @abc.abstractmethod
    async def send(
        self,
        channel: ChannelT,
        key: Optional[K] = ...,
        value: Optional[V] = ...,
        partition: Optional[int] = ...,
        timestamp: Optional[float] = ...,
        headers: Optional[HeadersArg] = ...,
        schema: Optional[_SchemaT] = ...,
        key_serializer: Optional[CodecArg] = ...,
        value_serializer: Optional[CodecArg] = ...,
        callback: Optional[MessageSentCallback] = ...,
    ) -> RecordMetadata: ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck: ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None: ...

    @abc.abstractmethod
    def is_leader(self) -> bool: ...

    @abc.abstractmethod
    def FlowControlQueue(
        self,
        maxsize: Optional[int] = ...,
        *,
        clear_on_resume: bool = ...,
        loop: Optional[asyncio.AbstractEventLoop] = ...,
    ) -> ThrowableQueue: ...

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

    @cached_property
    @abc.abstractmethod
    def tables(self) -> TableManagerT: ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> Mapping[str, TopicT]: ...

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
    def in_transaction(self) -> ContextManager[None]: ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: opentracing.Span) -> None: ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span: ...