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

if typing.TYPE_CHECKING:
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck.app import LiveCheck as _LiveCheck
    from faust.sensors.monitor import Monitor as _Monitor
    from faust.worker import Worker as _Worker
    from .events import EventT as _EventT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
    from .settings import Settings as _Settings
else:

    class _AppCommand:
        ...

    class _SchemaT:
        ...

    class _LiveCheck:
        ...

    class _Monitor:
        ...

    class _Worker:
        ...

    class _EventT:
        ...

    class _ModelArg:
        ...

    class _Settings:
        ...


__all__ = ['TaskArg', 'AppT']
TaskArg = Union[Callable[['AppT'], Awaitable[Any]], Callable[[], Awaitable[Any]]]
_T = TypeVar('_T')


class TracerT(abc.ABC):

    @property
    @abc.abstractmethod
    def default_tracer(self) -> Any:
        ...

    @abc.abstractmethod
    def trace(self, name: str, sample_rate: Optional[float] = None, **extra_context: Any) -> Any:
        ...

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> Any:
        ...


class BootStrategyT:
    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    @abc.abstractmethod
    def __init__(
        self,
        app: Any,
        *,
        enable_web: Optional[bool] = None,
        enable_kafka: bool = True,
        enable_kafka_producer: Optional[bool] = None,
        enable_kafka_consumer: Optional[bool] = None,
        enable_sensors: bool = True
    ) -> None:
        ...

    @abc.abstractmethod
    def server(self) -> Any:
        ...

    @abc.abstractmethod
    def client_only(self) -> Any:
        ...

    @abc.abstractmethod
    def producer_only(self) -> Any:
        ...


class AppT(ServiceT):
    """Abstract type for the Faust application.

    See Also:
        :class:`faust.App`.
    """
    finalized: bool = False
    configured: bool = False
    rebalancing: bool = False
    rebalancing_count: int = 0
    unassigned: bool = False
    in_worker: bool = False
    on_configured: SyncSignal = SyncSignal()
    on_before_configured: SyncSignal = SyncSignal()
    on_after_configured: SyncSignal = SyncSignal()
    on_partitions_assigned: Signal = Signal()
    on_partitions_revoked: Signal = Signal()
    on_rebalance_complete: Signal = Signal()
    on_before_shutdown: Signal = Signal()
    on_worker_init: SyncSignal = SyncSignal()
    on_produce_message: SyncSignal = SyncSignal()
    tracer: Optional[Any] = None

    @abc.abstractmethod
    def __init__(self, id: Any, *, monitor: Any, config_source: Optional[Any] = None, **options: Any) -> None:
        self.on_startup_finished: Optional[Any] = None

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None:
        ...

    @abc.abstractmethod
    def finalize(self) -> None:
        ...

    @abc.abstractmethod
    def main(self) -> None:
        ...

    @abc.abstractmethod
    def worker_init(self) -> None:
        ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None:
        ...

    @abc.abstractmethod
    def discover(self, *extra_modules: Any, categories: Tuple[str, ...] = ('a', 'b', 'c'), ignore: Tuple[str, ...] = ('foo', 'bar')) -> None:
        ...

    @abc.abstractmethod
    def topic(
        self,
        *topics: str,
        pattern: Optional[Any] = None,
        schema: Optional[Any] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        partitions: Optional[int] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        replicas: Optional[int] = None,
        acks: Union[bool, int] = True,
        internal: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        maxsize: Optional[int] = None,
        allow_empty: bool = False,
        has_prefix: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> TopicT:
        ...

    @abc.abstractmethod
    def channel(
        self,
        *,
        schema: Optional[Any] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        maxsize: Optional[int] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> ChannelT:
        ...

    @abc.abstractmethod
    def agent(
        self,
        channel: Optional[ChannelT] = None,
        *,
        name: Optional[str] = None,
        concurrency: int = 1,
        supervisor_strategy: Optional[SupervisorStrategyT] = None,
        sink: Optional[SinkT] = None,
        isolated_partitions: bool = False,
        use_reply_headers: bool = True,
        **kwargs: Any
    ) -> AgentFun:
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Callable[..., Any], *, on_leader: bool = False, traced: bool = True) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def timer(
        self,
        interval: Seconds,
        on_leader: bool = False,
        traced: bool = True,
        name: Optional[str] = None,
        max_drift_correction: float = 0.1
    ) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def crontab(
        self,
        cron_format: str,
        *,
        timezone: Optional[Union[str, tzinfo]] = None,
        on_leader: bool = False,
        traced: bool = True
    ) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def service(self, cls: Type[Any]) -> Any:
        ...

    @abc.abstractmethod
    def stream(self, channel: ChannelT, beacon: Optional[Any] = None, **kwargs: Any) -> StreamT:
        ...

    @abc.abstractmethod
    def Table(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = None,
        window: Optional[WindowT] = None,
        partitions: Optional[int] = None,
        help: Optional[str] = None,
        **kwargs: Any
    ) -> TableT:
        ...

    @abc.abstractmethod
    def GlobalTable(
        self,
        name: str,
        *,
        default: Optional[Callable[[], Any]] = None,
        window: Optional[WindowT] = None,
        partitions: Optional[int] = None,
        help: Optional[str] = None,
        **kwargs: Any
    ) -> TableT:
        ...

    @abc.abstractmethod
    def SetTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = None,
        partitions: Optional[int] = None,
        start_manager: bool = False,
        help: Optional[str] = None,
        **kwargs: Any
    ) -> CollectionT:
        ...

    @abc.abstractmethod
    def SetGlobalTable(
        self,
        name: str,
        *,
        window: Optional[WindowT] = None,
        partitions: Optional[int] = None,
        start_manager: bool = False,
        help: Optional[str] = None,
        **kwargs: Any
    ) -> CollectionT:
        ...

    @abc.abstractmethod
    def page(
        self,
        path: str,
        *,
        base: Type[View] = View,
        cors_options: Optional[Any] = None,
        name: Optional[str] = None
    ) -> ViewDecorator:
        ...

    @abc.abstractmethod
    def table_route(
        self,
        table: TableT,
        shard_param: Optional[str] = None,
        *,
        query_param: Optional[str] = None,
        match_info: Optional[str] = None,
        exact_key: Optional[Any] = None
    ) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Optional[Type[Any]] = None, **kwargs: Any) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: HeadersArg, message: Message) -> _EventT:
        ...

    @abc.abstractmethod
    async def start_client(self) -> None:
        ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None:
        ...

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> Any:
        ...

    @abc.abstractmethod
    async def send(
        self,
        channel: ChannelT,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        partition: Optional[int] = None,
        timestamp: Optional[float] = None,
        headers: Optional[HeadersArg] = None,
        schema: Optional[CodecArg] = None,
        key_serializer: Optional[Any] = None,
        value_serializer: Optional[Any] = None,
        callback: Optional[MessageSentCallback] = None
    ) -> Union[RecordMetadata, NoReturn]:
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck:
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None:
        ...

    @abc.abstractmethod
    def is_leader(self) -> bool:
        ...

    @abc.abstractmethod
    def FlowControlQueue(
        self,
        maxsize: Optional[int] = None,
        *,
        clear_on_resume: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> ThrowableQueue:
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> _Worker:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Web) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_return(self) -> None:
        ...

    @abc.abstractmethod
    def on_rebalance_end(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def conf(self) -> _Settings:
        ...

    @conf.setter
    @abc.abstractmethod
    def conf(self, settings: _Settings) -> None:
        ...

    @property
    @abc.abstractmethod
    def transport(self) -> TransportT:
        ...

    @transport.setter
    @abc.abstractmethod
    def transport(self, transport: TransportT) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> TransportT:
        ...

    @producer_transport.setter
    @abc.abstractmethod
    def producer_transport(self, transport: TransportT) -> None:
        ...

    @property
    @abc.abstractmethod
    def cache(self) -> CacheBackendT:
        ...

    @cache.setter
    @abc.abstractmethod
    def cache(self, cache: CacheBackendT) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer(self) -> ProducerT:
        ...

    @property
    @abc.abstractmethod
    def consumer(self) -> ConsumerT:
        ...

    @cached_property
    @abc.abstractmethod
    def tables(self) -> Mapping[str, TableT]:
        ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> Mapping[str, TopicT]:
        ...

    @property
    @abc.abstractmethod
    def monitor(self) -> Optional[_Monitor]:
        ...

    @monitor.setter
    @abc.abstractmethod
    def monitor(self, value: _Monitor) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> FlowControlEvent:
        return FlowControlEvent(loop=self.loop)  # type: ignore

    @property
    @abc.abstractmethod
    def http_client(self) -> HttpClientT:
        ...

    @http_client.setter
    @abc.abstractmethod
    def http_client(self, client: HttpClientT) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> PartitionAssignorT:
        ...

    @cached_property
    @abc.abstractmethod
    def router(self) -> RouterT:
        ...

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> RegistryT:
        ...

    @cached_property
    @abc.abstractmethod
    def web(self) -> Web:
        ...

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> bool:
        ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: Any) -> None:
        ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> Any:
        ...