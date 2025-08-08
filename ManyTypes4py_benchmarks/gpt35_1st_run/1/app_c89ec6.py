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
    def __init__(self, app: Any, *, enable_web: Optional[bool] = None, enable_kafka: bool = True, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: bool = True) -> None:
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
    tracer: Any = None

    @abc.abstractmethod
    def __init__(self, id: Any, *, monitor: Any, config_source: Any = None, **options: Any) -> None:
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
    def discover(self, *extra_modules: Any, categories: Tuple[str, str, str] = ('a', 'b', 'c'), ignore: Tuple[str, str] = ('foo', 'bar')) -> None:
        ...

    @abc.abstractmethod
    def topic(self, *topics: Any, pattern: Any = None, schema: Any = None, key_type: Any = None, value_type: Any = None, key_serializer: Any = None, value_serializer: Any = None, partitions: Any = None, retention: Any = None, compacting: Any = None, deleting: Any = None, replicas: Any = None, acks: bool = True, internal: bool = False, config: Any = None, maxsize: Any = None, allow_empty: bool = False, has_prefix: bool = False, loop: Any = None) -> None:
        ...

    @abc.abstractmethod
    def channel(self, *, schema: Any = None, key_type: Any = None, value_type: Any = None, maxsize: Any = None, loop: Any = None) -> None:
        ...

    @abc.abstractmethod
    def agent(self, channel: Any = None, *, name: Any = None, concurrency: int = 1, supervisor_strategy: Any = None, sink: Any = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Callable, *, on_leader: bool = False, traced: bool = True) -> None:
        ...

    @abc.abstractmethod
    def timer(self, interval: Any, on_leader: bool = False, traced: bool = True, name: Any = None, max_drift_correction: float = 0.1) -> None:
        ...

    @abc.abstractmethod
    def crontab(self, cron_format: Any, *, timezone: Any = None, on_leader: bool = False, traced: bool = True) -> None:
        ...

    @abc.abstractmethod
    def service(self, cls: Any) -> None:
        ...

    @abc.abstractmethod
    def stream(self, channel: Any, beacon: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def Table(self, name: Any, *, default: Any = None, window: Any = None, partitions: Any = None, help: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def GlobalTable(self, name: Any, *, default: Any = None, window: Any = None, partitions: Any = None, help: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def SetTable(self, name: Any, *, window: Any = None, partitions: Any = None, start_manager: bool = False, help: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def SetGlobalTable(self, name: Any, *, window: Any = None, partitions: Any = None, start_manager: bool = False, help: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def page(self, path: Any, *, base: Any = View, cors_options: Any = None, name: Any = None) -> None:
        ...

    @abc.abstractmethod
    def table_route(self, table: Any, shard_param: Any = None, *, query_param: Any = None, match_info: Any = None, exact_key: Any = None) -> None:
        ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Any = None, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: Any, message: Any) -> None:
        ...

    @abc.abstractmethod
    async def start_client(self) -> None:
        ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> None:
        ...

    @abc.abstractmethod
    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> None:
        ...

    @abc.abstractmethod
    async def send(self, channel: Any, key: Any = None, value: Any = None, partition: Any = None, timestamp: Any = None, headers: Any = None, schema: Any = None, key_serializer: Any = None, value_serializer: Any = None, callback: Any = None) -> None:
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> None:
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) -> None:
        ...

    @abc.abstractmethod
    def is_leader(self) -> None:
        ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Any = None, *, clear_on_resume: bool = False, loop: Any = None) -> None:
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Any) -> None:
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
    def conf(self) -> Any:
        ...

    @conf.setter
    def conf(self, settings: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def transport(self) -> Any:
        ...

    @transport.setter
    def transport(self, transport: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> Any:
        ...

    @producer_transport.setter
    def producer_transport(self, transport: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def cache(self) -> Any:
        ...

    @cache.setter
    def cache(self, cache: Any) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer(self) -> Any:
        ...

    @property
    @abc.abstractmethod
    def consumer(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def tables(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> Any:
        ...

    @property
    @abc.abstractmethod
    def monitor(self) -> Any:
        ...

    @monitor.setter
    def monitor(self, value: Any) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self) -> Any:
        return FlowControlEvent(loop=self.loop)

    @property
    @abc.abstractmethod
    def http_client(self) -> Any:
        ...

    @http_client.setter
    def http_client(self, client: Any) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def router(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def web(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> Any:
        ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: Any) -> None:
        ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: str) -> None:
        ...
