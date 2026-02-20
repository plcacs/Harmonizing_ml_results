import abc
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
    Iterable,
    Mapping,
    MutableSequence,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

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
from .web import (
    CacheBackendT,
    HttpClientT,
    PageArg,
    ResourceOptions,
    View,
    ViewDecorator,
    Web,
)
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
    class _AppCommand: ...     # noqa
    class _SchemaT: ...        # noqa
    class _LiveCheck: ...      # noqa
    class _Monitor: ...        # noqa
    class _Worker: ...         # noqa
    class _EventT: ...         # noqa
    class _ModelArg: ...       # noqa
    class _Settings: ...       # noqa

__all__ = [
    'TaskArg',
    'AppT',
]

TaskArg = Union[Callable[['AppT'], Awaitable], Callable[[], Awaitable]]
_T = TypeVar('_T')


class TracerT(abc.ABC):

    @property
    @abc.abstractmethod
    def default_tracer(self) -> opentracing.Tracer:
        ...

    @abc.abstractmethod
    def trace(self, name,
              sample_rate = None,
              **extra_context) -> ContextManager:
        ...

    @abc.abstractmethod
    def get_tracer(self, service_name) -> opentracing.Tracer:
        ...


class BootStrategyT:
    app: 'AppT'

    enable_kafka: bool = True
    # We want these to take default from `enable_kafka`
    # attribute, but still want to allow subclasses to define
    # them like this:
    #   class MyBoot(BootStrategy):
    #       enable_kafka_consumer = False
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None

    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    @abc.abstractmethod
    def __init__(self, app, *,
                 enable_web = None,
                 enable_kafka = True,
                 enable_kafka_producer = None,
                 enable_kafka_consumer: bool = None,
                 enable_sensors = True) -> None:
        ...

    @abc.abstractmethod
    def server(self) -> Iterable[ServiceT]:
        ...

    @abc.abstractmethod
    def client_only(self) -> Iterable[ServiceT]:
        ...

    @abc.abstractmethod
    def producer_only(self) -> Iterable[ServiceT]:
        ...


class AppT(ServiceT):
    """Abstract type for the Faust application.

    See Also:
        :class:`faust.App`.
    """
    Settings: ClassVar[Type[_Settings]]

    BootStrategy: ClassVar[Type[BootStrategyT]]
    boot_strategy: BootStrategyT

    #: Set to true when the app is finalized (can read configuration).
    finalized: bool = False

    #: Set to true when the app has read configuration.
    configured: bool = False

    #: Set to true if the worker is currently rebalancing.
    rebalancing: bool = False
    rebalancing_count: int = 0

    #: Set to true if the assignment is empty
    # This flag is set by App._on_partitions_assigned
    unassigned: bool = False

    #: Set to true when app is executing within a worker instance.
    # This flag is set in faust/worker.py
    in_worker: bool = False

    on_configured: SyncSignal[_Settings] = SyncSignal()
    on_before_configured: SyncSignal = SyncSignal()
    on_after_configured: SyncSignal = SyncSignal()
    on_partitions_assigned: Signal[Set[TP]] = Signal()
    on_partitions_revoked: Signal[Set[TP]] = Signal()
    on_rebalance_complete: Signal = Signal()
    on_before_shutdown: Signal = Signal()
    on_worker_init: SyncSignal = SyncSignal()
    on_produce_message: SyncSignal = SyncSignal()

    client_only: bool
    producer_only: bool

    agents: AgentManagerT
    sensors: SensorDelegateT

    fixups: MutableSequence[FixupT]

    tracer: Optional[TracerT] = None

    #: Original id argument + kwargs passed to App.__init__
    _default_options: Tuple[str, Mapping[str, Any]]

    @abc.abstractmethod
    def __init__(self,
                 id: str,
                 *,
                 monitor,
                 config_source = None,
                 **options) -> None:
        self.on_startup_finished: Optional[Callable] = None

    @abc.abstractmethod
    def config_from_object(self,
                           obj: Any,
                           *,
                           silent = False,
                           force = False) -> None:
        ...

    @abc.abstractmethod
    def finalize(self) -> None:
        ...

    @abc.abstractmethod
    def main(self) -> NoReturn:
        ...

    @abc.abstractmethod
    def worker_init(self) -> None:
        ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> None:
        ...

    @abc.abstractmethod
    def discover(self,
                 *extra_modules,
                 categories = ('a', 'b', 'c'),
                 ignore = ('foo', 'bar')) -> None:
        ...

    @abc.abstractmethod
    def topic(self,
              *topics,
              pattern = None,
              schema = None,
              key_type = None,
              value_type = None,
              key_serializer = None,
              value_serializer = None,
              partitions = None,
              retention: Seconds = None,
              compacting = None,
              deleting = None,
              replicas = None,
              acks = True,
              internal = False,
              config = None,
              maxsize = None,
              allow_empty = False,
              has_prefix = False,
              loop = None) -> TopicT:
        ...

    @abc.abstractmethod
    def channel(self,
                *,
                schema = None,
                key_type = None,
                value_type = None,
                maxsize = None,
                loop = None) -> ChannelT:
        ...

    @abc.abstractmethod
    def agent(self,
              channel = None,
              *,
              name: str = None,
              concurrency = 1,
              supervisor_strategy = None,
              sink = None,
              isolated_partitions = False,
              use_reply_headers = True,
              **kwargs) -> Callable[[AgentFun[_T]], AgentT[_T]]:
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: TaskArg, *,
             on_leader = False,
             traced = True) -> Callable:
        ...

    @abc.abstractmethod
    def timer(self, interval,
              on_leader = False,
              traced = True,
              name = None,
              max_drift_correction = 0.1) -> Callable:
        ...

    @abc.abstractmethod
    def crontab(self, cron_format, *,
                timezone = None,
                on_leader = False,
                traced: bool = True) -> Callable:
        ...

    @abc.abstractmethod
    def service(self, cls) -> Type[ServiceT]:
        ...

    @abc.abstractmethod
    def stream(self,
               channel,
               beacon = None,
               **kwargs) -> StreamT:
        ...

    @abc.abstractmethod
    def Table(self,
              name,
              *,
              default = None,
              window = None,
              partitions: int = None,
              help = None,
              **kwargs) -> TableT:
        ...

    @abc.abstractmethod
    def GlobalTable(self,
                    name,
                    *,
                    default = None,
                    window = None,
                    partitions = None,
                    help = None,
                    **kwargs) -> TableT:
        ...

    @abc.abstractmethod
    def SetTable(self,
                 name,
                 *,
                 window = None,
                 partitions: int = None,
                 start_manager = False,
                 help = None,
                 **kwargs) -> TableT:
        ...

    @abc.abstractmethod
    def SetGlobalTable(self,
                       name,
                       *,
                       window = None,
                       partitions = None,
                       start_manager = False,
                       help = None,
                       **kwargs) -> TableT:
        ...

    @abc.abstractmethod
    def page(self, path, *,
             base = View,
             cors_options = None,
             name = None) -> Callable[[PageArg], Type[View]]:
        ...

    @abc.abstractmethod
    def table_route(self, table: CollectionT,
                    shard_param: str = None,
                    *,
                    query_param = None,
                    match_info = None,
                    exact_key = None) -> ViewDecorator:
        ...

    @abc.abstractmethod
    def command(self,
                *options,
                base = None,
                **kwargs) -> Callable[[Callable], Type[_AppCommand]]:
        ...

    @abc.abstractmethod
    def create_event(self,
                     key,
                     value,
                     headers,
                     message) -> _EventT:
        ...

    @abc.abstractmethod
    async def start_client(self) -> None:
        ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None:
        ...

    @abc.abstractmethod
    def trace(self,
              name: str,
              trace_enabled = True,
              **extra_context: Any) -> ContextManager:
        ...

    @abc.abstractmethod
    async def send(
            self,
            channel: Union[ChannelT, str],
            key = None,
            value = None,
            partition = None,
            timestamp = None,
            headers = None,
            schema = None,
            key_serializer = None,
            value_serializer = None,
            callback = None) -> Awaitable[RecordMetadata]:
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs) -> _LiveCheck:
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> ProducerT:
        ...

    @abc.abstractmethod
    def is_leader(self) -> bool:
        ...

    @abc.abstractmethod
    def FlowControlQueue(
            self,
            maxsize = None,
            *,
            clear_on_resume = False,
            loop = None) -> ThrowableQueue:
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs) -> _Worker:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web) -> None:
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
    def conf(self) -> _Settings:
        ...

    @conf.setter
    def conf(self, settings) -> None:
        ...

    @property
    @abc.abstractmethod
    def transport(self) -> TransportT:
        ...

    @transport.setter
    def transport(self, transport) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> TransportT:
        ...

    @producer_transport.setter
    def producer_transport(self, transport) -> None:
        ...

    @property
    @abc.abstractmethod
    def cache(self) -> CacheBackendT:
        ...

    @cache.setter
    def cache(self, cache) -> None:
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
    def tables(self) -> TableManagerT:
        ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> ConductorT:
        ...

    @property
    @abc.abstractmethod
    def monitor(self) -> _Monitor:
        ...

    @monitor.setter
    def monitor(self, value) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> FlowControlEvent:
        return FlowControlEvent(loop=self.loop)

    @property
    @abc.abstractmethod
    def http_client(self) -> HttpClientT:
        ...

    @http_client.setter
    def http_client(self, client) -> None:
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
    def _span_add_default_tags(self, span) -> None:
        ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name) -> opentracing.Span:
        ...
