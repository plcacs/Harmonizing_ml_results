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
    def default_tracer(self) -> 'TracerT':
        """Get the default tracer."""

    @abc.abstractmethod
    def trace(self, name: str, sample_rate: Optional[float] = None, **extra_context: Any) -> 'Span':
        """Trace a function call."""

    @abc.abstractmethod
    def get_tracer(self, service_name: str) -> 'TracerT':
        """Get a tracer for a service."""

class BootStrategyT:
    """Abstract base class for a boot strategy."""

    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    @abc.abstractmethod
    def __init__(self, app: 'AppT', *, enable_web: Optional[bool] = None, enable_kafka: bool = True, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: bool = True):
        ...

    @abc.abstractmethod
    def server(self) -> 'BootStrategyT':
        """Run the server."""

    @abc.abstractmethod
    def client_only(self) -> 'BootStrategyT':
        """Run the client only."""

    @abc.abstractmethod
    def producer_only(self) -> 'BootStrategyT':
        """Run the producer only."""

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
    tracer: Optional['TracerT']

    @abc.abstractmethod
    def __init__(self, id: str, *, monitor: 'Monitor', config_source: Optional['ConfigSource'] = None, **options: Any):
        ...

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None:
        """Configure the application from an object."""

    @abc.abstractmethod
    def finalize(self) -> None:
        """Finalize the application."""

    @abc.abstractmethod
    def main(self) -> None:
        """Run the application."""

    @abc.abstractmethod
    def worker_init(self) -> None:
        """Initialize the worker."""

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None:
        """Initialize the worker post-autodiscover."""

    @abc.abstractmethod
    def discover(self, *extra_modules: Any, categories: Tuple[str, ...] = ('a', 'b', 'c'), ignore: Tuple[str, ...] = ('foo', 'bar')) -> None:
        """Discover the application."""

    @abc.abstractmethod
    def topic(self, *topics: str, pattern: Optional[str] = None, schema: Optional['SchemaT'] = None, key_type: Optional['TypeT'] = None, value_type: Optional['TypeT'] = None, key_serializer: Optional['CodecArg'] = None, value_serializer: Optional['CodecArg'] = None, partitions: Optional[int] = None, retention: Optional[int] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Any] = None, maxsize: Optional[int] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional['EventLoop'] = None) -> 'TopicT':
        """Create a topic."""

    @abc.abstractmethod
    def channel(self, *, schema: Optional['SchemaT'] = None, key_type: Optional['TypeT'] = None, value_type: Optional['TypeT'] = None, maxsize: Optional[int] = None, loop: Optional['EventLoop'] = None) -> 'ChannelT':
        """Create a channel."""

    @abc.abstractmethod
    def agent(self, channel: Optional['ChannelT'] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional['SupervisorStrategyT'] = None, sink: Optional['SinkT'] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> 'AgentT':
        """Create an agent."""

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Callable, *, on_leader: bool = False, traced: bool = True) -> None:
        """Run a task."""

    @abc.abstractmethod
    def timer(self, interval: Seconds, on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1) -> None:
        """Create a timer."""

    @abc.abstractmethod
    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = None, on_leader: bool = False, traced: bool = True) -> None:
        """Create a crontab."""

    @abc.abstractmethod
    def service(self, cls: Type['ServiceT']) -> 'ServiceT':
        """Create a service."""

    @abc.abstractmethod
    def stream(self, channel: 'ChannelT', beacon: Optional['Beacon'] = None, **kwargs: Any) -> 'StreamT':
        """Create a stream."""

    @abc.abstractmethod
    def Table(self, name: str, *, default: Optional[Any] = None, window: Optional['WindowT'] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> 'TableT':
        """Create a table."""

    @abc.abstractmethod
    def GlobalTable(self, name: str, *, default: Optional[Any] = None, window: Optional['WindowT'] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> 'GlobalTableT':
        """Create a global table."""

    @abc.abstractmethod
    def SetTable(self, name: str, *, window: Optional['WindowT'] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> 'SetTableT':
        """Create a set table."""

    @abc.abstractmethod
    def SetGlobalTable(self, name: str, *, window: Optional['WindowT'] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> 'SetGlobalTableT':
        """Create a set global table."""

    @abc.abstractmethod
    def page(self, path: str, *, base: Type['View'] = View, cors_options: Optional['ResourceOptions'] = None, name: Optional[str] = None) -> 'PageT':
        """Create a page."""

    @abc.abstractmethod
    def table_route(self, table: 'TableT', shard_param: Optional[str] = None, *, query_param: Optional[str] = None, match_info: Optional[str] = None, exact_key: Optional[str] = None) -> None:
        """Route a table."""

    @abc.abstractmethod
    def command(self, *options: Any, base: Optional['AppCommand'] = None, **kwargs: Any) -> 'AppCommand':
        """Create a command."""

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: Mapping[str, Any], message: 'Message') -> 'EventT':
        """Create an event."""

    @abc.abstractmethod
    async def start_client(self) -> None:
        """Start the client."""

    @abc.abstractmethod
    async def maybe_start_client(self) -> None:
        """Maybe start the client."""

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> 'Span':
        """Trace a function call."""

    @abc.abstractmethod
    async def send(self, channel: 'ChannelT', key: Optional[Any] = None, value: Optional[Any] = None, partition: Optional[int] = None, timestamp: Optional[int] = None, headers: Optional[Mapping[str, Any]] = None, schema: Optional['SchemaT'] = None, key_serializer: Optional['CodecArg'] = None, value_serializer: Optional['CodecArg'] = None, callback: Optional['MessageSentCallback'] = None) -> None:
        """Send a message."""

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> 'LiveCheck':
        """Create a live check."""

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None:
        """Maybe start the producer."""

    @abc.abstractmethod
    def is_leader(self) -> bool:
        """Check if the application is the leader."""

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Optional[int] = None, *, clear_on_resume: bool = False, loop: Optional['EventLoop'] = None) -> 'FlowControlQueueT':
        """Create a flow control queue."""

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> 'Worker':
        """Create a worker."""

    @abc.abstractmethod
    def on_webserver_init(self, web: 'Web') -> None:
        """Initialize the web server."""

    @abc.abstractmethod
    def on_rebalance_start(self) -> None:
        """Start the rebalance."""

    @abc.abstractmethod
    def on_rebalance_return(self) -> None:
        """Return from the rebalance."""

    @abc.abstractmethod
    def on_rebalance_end(self) -> None:
        """End the rebalance."""

    @property
    def conf(self) -> 'Settings':
        ...

    @conf.setter
    def conf(self, settings: 'Settings') -> None:
        ...

    @property
    @abc.abstractmethod
    def transport(self) -> 'TransportT':
        """Get the transport."""

    @transport.setter
    def transport(self, transport: 'TransportT') -> None:
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> 'TransportT':
        """Get the producer transport."""

    @producer_transport.setter
    def producer_transport(self, transport: 'TransportT') -> None:
        ...

    @property
    @abc.abstractmethod
    def cache(self) -> 'CacheBackendT':
        """Get the cache."""

    @cache.setter
    def cache(self, cache: 'CacheBackendT') -> None:
        ...

    @property
    @abc.abstractmethod
    def producer(self) -> 'ProducerT':
        """Get the producer."""

    @property
    @abc.abstractmethod
    def consumer(self) -> 'ConsumerT':
        """Get the consumer."""

    @cached_property
    @abc.abstractmethod
    def tables(self) -> 'TableManagerT':
        """Get the tables."""

    @cached_property
    @abc.abstractmethod
    def topics(self) -> 'TopicManagerT':
        """Get the topics."""

    @property
    @abc.abstractmethod
    def monitor(self) -> 'Monitor':
        """Get the monitor."""

    @monitor.setter
    def monitor(self, value: 'Monitor') -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> 'FlowControlEvent':
        """Get the flow control."""

    @property
    @abc.abstractmethod
    def http_client(self) -> 'HttpClientT':
        """Get the HTTP client."""

    @http_client.setter
    def http_client(self, client: 'HttpClientT') -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> 'PartitionAssignorT':
        """Get the assignor."""

    @cached_property
    @abc.abstractmethod
    def router(self) -> 'RouterT':
        """Get the router."""

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> 'RegistryT':
        """Get the serializers."""

    @cached_property
    @abc.abstractmethod
    def web(self) -> 'Web':
        """Get the web."""

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> bool:
        """Check if in transaction."""

    @abc.abstractmethod
    def _span_add_default_tags(self, span: 'Span') -> None:
        """Add default tags to a span."""

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> 'Span':
        """Start a span from rebalancing."""
