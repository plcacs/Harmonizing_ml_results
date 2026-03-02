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
TaskArg = Union[Callable[['AppT'], Awaitable], Callable[[], Awaitable]]
_T = TypeVar('_T')

class TracerT(abc.ABC):
    """Abstract base class for a tracer."""

    @property
    @abc.abstractmethod
    def default_tracer(self) -> opentracing.Tracer:
        """Returns the default tracer instance."""

    @abc.abstractmethod
    def trace(self, name: str, sample_rate: Optional[float] = None, **extra_context: Any) -> opentracing.Span:
        """Starts a new span with the given name."""

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> opentracing.Tracer:
        """Returns the tracer for the given service name."""

class BootStrategyT:
    """Abstract base class for a boot strategy."""

    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    @abc.abstractmethod
    def __init__(self, app: AppT, *, enable_web: Optional[bool] = None, enable_kafka: bool = True, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: bool = True):
        ...

    @abc.abstractmethod
    def server(self) -> AppT:
        """Returns the server instance."""

    @abc.abstractmethod
    def client_only(self) -> AppT:
        """Returns the client-only instance."""

    @abc.abstractmethod
    def producer_only(self) -> AppT:
        """Returns the producer-only instance."""

class AppT(ServiceT):
    """Abstract type for the Faust application."""

    finalized: bool = False
    configured: bool = False
    rebalancing: bool = False
    rebalancing_count: int = 0
    unassigned: bool = False
    in_worker: bool = False
    on_configured: SyncSignal
    on_before_configured: SyncSignal
    on_after_configured: SyncSignal
    on_partitions_assigned: Signal
    on_partitions_revoked: Signal
    on_rebalance_complete: Signal
    on_before_shutdown: Signal
    on_worker_init: SyncSignal
    on_produce_message: SyncSignal
    tracer: Optional[opentracing.Tracer] = None

    @abc.abstractmethod
    def __init__(self, id: str, *, monitor: _Monitor, config_source: Optional[Any] = None, **options: Any):
        ...

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False):
        """Configures the application from a given object."""

    @abc.abstractmethod
    def finalize(self):
        """Finalizes the application."""

    @abc.abstractmethod
    def main(self):
        """Runs the application."""

    @abc.abstractmethod
    def worker_init(self):
        """Initializes the worker."""

    @abc.abstractmethod
    def worker_init_post_autodiscover(self):
        """Initializes the worker after autodiscovery."""

    @abc.abstractmethod
    def discover(self, *extra_modules: Any, categories: Tuple[str, ...] = ('a', 'b', 'c'), ignore: Tuple[str, ...] = ('foo', 'bar')):
        """Discovers and initializes the application."""

    @abc.abstractmethod
    def topic(self, *topics: str, pattern: Optional[str] = None, schema: Optional[Any] = None, key_type: Optional[Any] = None, value_type: Optional[Any] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, partitions: Optional[int] = None, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Any] = None, maxsize: Optional[int] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional[asyncio.BaseEventLoop] = None) -> TopicT:
        """Creates a topic."""

    @abc.abstractmethod
    def channel(self, *, schema: Optional[Any] = None, key_type: Optional[Any] = None, value_type: Optional[Any] = None, maxsize: Optional[int] = None, loop: Optional[asyncio.BaseEventLoop] = None) -> ChannelT:
        """Creates a channel."""

    @abc.abstractmethod
    def agent(self, channel: Optional[ChannelT] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional[Any] = None, sink: Optional[Any] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> AgentT:
        """Creates an agent."""

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Callable[[], Awaitable], *, on_leader: bool = False, traced: bool = True) -> Awaitable:
        """Runs a task."""

    @abc.abstractmethod
    def timer(self, interval: Seconds, on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1):
        """Creates a timer."""

    @abc.abstractmethod
    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = None, on_leader: bool = False, traced: bool = True):
        """Creates a crontab."""

    @abc.abstractmethod
    def service(self, cls: Type[_Worker]) -> _Worker:
        """Creates a service."""

    @abc.abstractmethod
    def stream(self, channel: ChannelT, beacon: Optional[Any] = None, **kwargs: Any) -> StreamT:
        """Creates a stream."""

    @abc.abstractmethod
    def Table(self, name: str, *, default: Optional[Any] = None, window: Optional[Any] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Creates a table."""

    @abc.abstractmethod
    def GlobalTable(self, name: str, *, default: Optional[Any] = None, window: Optional[Any] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Creates a global table."""

    @abc.abstractmethod
    def SetTable(self, name: str, *, window: Optional[Any] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Creates a set table."""

    @abc.abstractmethod
    def SetGlobalTable(self, name: str, *, window: Optional[Any] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Creates a set global table."""

    @abc.abstractmethod
    def page(self, path: str, *, base: Type[View] = View, cors_options: Optional[Any] = None, name: Optional[str] = None) -> View:
        """Creates a page."""

    @abc.abstractmethod
    def table_route(self, table: TableT, shard_param: Optional[str] = None, *, query_param: Optional[str] = None, match_info: Optional[Any] = None, exact_key: Optional[str] = None) -> str:
        """Creates a table route."""

    @abc.abstractmethod
    def command(self, *options: Any, base: Optional[Any] = None, **kwargs: Any) -> _AppCommand:
        """Creates a command."""

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: Mapping[str, Any], message: Message) -> EventT:
        """Creates an event."""

    @abc.abstractmethod
    async def start_client(self) -> None:
        """Starts the client."""

    @abc.abstractmethod
    async def maybe_start_client(self) -> None:
        """Maybe starts the client."""

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> opentracing.Span:
        """Traces the application."""

    @abc.abstractmethod
    async def send(self, channel: ChannelT, key: Optional[Any] = None, value: Optional[Any] = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[Mapping[str, Any]] = None, schema: Optional[Any] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, callback: Optional[MessageSentCallback] = None) -> Message:
        """Sends a message."""

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> _LiveCheck:
        """Creates a LiveCheck instance."""

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None:
        """Maybe starts the producer."""

    @abc.abstractmethod
    def is_leader(self) -> bool:
        """Checks if the application is a leader."""

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Optional[int] = None, *, clear_on_resume: bool = False, loop: Optional[asyncio.BaseEventLoop] = None) -> ThrowableQueue:
        """Creates a flow control queue."""

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> _Worker:
        """Creates a worker."""

    @abc.abstractmethod
    def on_webserver_init(self, web: Web) -> None:
        """Initializes the web server."""

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        """Starts the rebalance."""

    @abc.abstractmethod
    def on_rebalance_return(self) -> None:
        """Returns from the rebalance."""

    @abc.abstractmethod
    def on_rebalance_end(self) -> None:
        """Ends the rebalance."""

    @property
    def conf(self) -> _Settings:
        """Returns the configuration."""

    @conf.setter
    def conf(self, settings: _Settings) -> None:
        """Sets the configuration."""

    @property
    @abc.abstractmethod
    def transport(self) -> TransportT:
        """Returns the transport."""

    @transport.setter
    def transport(self, transport: TransportT) -> None:
        """Sets the transport."""

    @property
    @abc.abstractmethod
    def producer_transport(self) -> TransportT:
        """Returns the producer transport."""

    @producer_transport.setter
    def producer_transport(self, transport: TransportT) -> None:
        """Sets the producer transport."""

    @property
    @abc.abstractmethod
    def cache(self) -> CacheBackendT:
        """Returns the cache."""

    @cache.setter
    def cache(self, cache: CacheBackendT) -> None:
        """Sets the cache."""

    @property
    @abc.abstractmethod
    def producer(self) -> ProducerT:
        """Returns the producer."""

    @property
    @abc.abstractmethod
    def consumer(self) -> ConsumerT:
        """Returns the consumer."""

    @cached_property
    @abc.abstractmethod
    def tables(self) -> CollectionT:
        """Returns the tables."""

    @cached_property
    @abc.abstractmethod
    def topics(self) -> CollectionT:
        """Returns the topics."""

    @property
    @abc.abstractmethod
    def monitor(self) -> _Monitor:
        """Returns the monitor."""

    @monitor.setter
    def monitor(self, value: _Monitor) -> None:
        """Sets the monitor."""

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> FlowControlEvent:
        """Returns the flow control."""

    @property
    @abc.abstractmethod
    def http_client(self) -> HttpClientT:
        """Returns the HTTP client."""

    @http_client.setter
    def http_client(self, client: HttpClientT) -> None:
        """Sets the HTTP client."""

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> PartitionAssignorT:
        """Returns the assignor."""

    @cached_property
    @abc.abstractmethod
    def router(self) -> RouterT:
        """Returns the router."""

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> RegistryT:
        """Returns the serializers."""

    @cached_property
    @abc.abstractmethod
    def web(self) -> Web:
        """Returns the web instance."""

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> bool:
        """Checks if the application is in a transaction."""

    @abc.abstractmethod
    def _span_add_default_tags(self, span: opentracing.Span) -> None:
        """Adds default tags to a span."""

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> opentracing.Span:
        """Starts a span from rebalancing."""
