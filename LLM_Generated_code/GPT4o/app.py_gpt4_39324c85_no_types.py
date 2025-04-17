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
    def default_tracer(self):
        ...

    @abc.abstractmethod
    def trace(self, name, sample_rate=None, **extra_context: Any):
        ...

    @abc.abstractmethod
    def get_tracer(self, service_name):
        ...


class BootStrategyT:
    app: 'AppT'
    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    @abc.abstractmethod
    def __init__(self, app, *, enable_web: bool=None, enable_kafka: bool=
        True, enable_kafka_producer: bool=None, enable_kafka_consumer: bool
        =None, enable_sensors: bool=True):
        ...

    @abc.abstractmethod
    def server(self):
        ...

    @abc.abstractmethod
    def client_only(self):
        ...

    @abc.abstractmethod
    def producer_only(self):
        ...


class AppT(ServiceT):
    """Abstract type for the Faust application.

    See Also:
        :class:`faust.App`.
    """
    Settings: ClassVar[Type[_Settings]]
    BootStrategy: ClassVar[Type[BootStrategyT]]
    boot_strategy: BootStrategyT
    finalized: bool = False
    configured: bool = False
    rebalancing: bool = False
    rebalancing_count: int = 0
    unassigned: bool = False
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
    _default_options: Tuple[str, Mapping[str, Any]]

    @abc.abstractmethod
    def __init__(self, id, *, monitor: _Monitor, config_source: Any=None,
        **options: Any):
        self.on_startup_finished: Optional[Callable] = None

    @abc.abstractmethod
    def config_from_object(self, obj, *, silent: bool=False, force: bool=False
        ):
        ...

    @abc.abstractmethod
    def finalize(self):
        ...

    @abc.abstractmethod
    def main(self):
        ...

    @abc.abstractmethod
    def worker_init(self):
        ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self):
        ...

    @abc.abstractmethod
    def discover(self, *extra_modules: str, categories: Iterable[str]=('a',
        'b', 'c'), ignore: Iterable[Any]=('foo', 'bar')):
        ...

    @abc.abstractmethod
    def topic(self, *topics: str, pattern: Union[str, Pattern]=None, schema:
        _SchemaT=None, key_type: _ModelArg=None, value_type: _ModelArg=None,
        key_serializer: CodecArg=None, value_serializer: CodecArg=None,
        partitions: int=None, retention: Seconds=None, compacting: bool=
        None, deleting: bool=None, replicas: int=None, acks: bool=True,
        internal: bool=False, config: Mapping[str, Any]=None, maxsize: int=
        None, allow_empty: bool=False, has_prefix: bool=False, loop:
        asyncio.AbstractEventLoop=None):
        ...

    @abc.abstractmethod
    def channel(self, *, schema: _SchemaT=None, key_type: _ModelArg=None,
        value_type: _ModelArg=None, maxsize: int=None, loop: asyncio.
        AbstractEventLoop=None):
        ...

    @abc.abstractmethod
    def agent(self, channel=None, *, name: str=None, concurrency: int=1,
        supervisor_strategy: Type[SupervisorStrategyT]=None, sink: Iterable
        [SinkT]=None, isolated_partitions: bool=False, use_reply_headers:
        bool=True, **kwargs: Any):
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun, *, on_leader: bool=False, traced: bool=True):
        ...

    @abc.abstractmethod
    def timer(self, interval, on_leader=False, traced=True, name=None,
        max_drift_correction=0.1):
        ...

    @abc.abstractmethod
    def crontab(self, cron_format, *, timezone: tzinfo=None, on_leader:
        bool=False, traced: bool=True):
        ...

    @abc.abstractmethod
    def service(self, cls):
        ...

    @abc.abstractmethod
    def stream(self, channel, beacon=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def Table(self, name, *, default: Callable[[], Any]=None, window:
        WindowT=None, partitions: int=None, help: str=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def GlobalTable(self, name, *, default: Callable[[], Any]=None, window:
        WindowT=None, partitions: int=None, help: str=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def SetTable(self, name, *, window: WindowT=None, partitions: int=None,
        start_manager: bool=False, help: str=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def SetGlobalTable(self, name, *, window: WindowT=None, partitions: int
        =None, start_manager: bool=False, help: str=None, **kwargs: Any):
        ...

    @abc.abstractmethod
    def page(self, path, *, base: Type[View]=View, cors_options: Mapping[
        str, ResourceOptions]=None, name: str=None):
        ...

    @abc.abstractmethod
    def table_route(self, table, shard_param=None, *, query_param: str=None,
        match_info: str=None, exact_key: str=None):
        ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Type[_AppCommand]=None, **kwargs:
        Any):
        ...

    @abc.abstractmethod
    def create_event(self, key, value, headers, message):
        ...

    @abc.abstractmethod
    async def start_client(self) ->None:
        ...

    @abc.abstractmethod
    async def maybe_start_client(self) ->None:
        ...

    @abc.abstractmethod
    def trace(self, name, trace_enabled=True, **extra_context: Any):
        ...

    @abc.abstractmethod
    async def send(self, channel: Union[ChannelT, str], key: K=None, value:
        V=None, partition: int=None, timestamp: float=None, headers:
        HeadersArg=None, schema: _SchemaT=None, key_serializer: CodecArg=
        None, value_serializer: CodecArg=None, callback:
        MessageSentCallback=None) ->Awaitable[RecordMetadata]:
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any):
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self) ->ProducerT:
        ...

    @abc.abstractmethod
    def is_leader(self):
        ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize=None, *, clear_on_resume: bool=False,
        loop: asyncio.AbstractEventLoop=None):
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any):
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web):
        ...

    @abc.abstractmethod
    def on_rebalance_start(self):
        ...

    @abc.abstractmethod
    def on_rebalance_return(self):
        ...

    @abc.abstractmethod
    def on_rebalance_end(self):
        ...

    @property
    def conf(self):
        ...

    @conf.setter
    def conf(self, settings):
        ...

    @property
    @abc.abstractmethod
    def transport(self):
        ...

    @transport.setter
    def transport(self, transport):
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self):
        ...

    @producer_transport.setter
    def producer_transport(self, transport):
        ...

    @property
    @abc.abstractmethod
    def cache(self):
        ...

    @cache.setter
    def cache(self, cache):
        ...

    @property
    @abc.abstractmethod
    def producer(self):
        ...

    @property
    @abc.abstractmethod
    def consumer(self):
        ...

    @cached_property
    @abc.abstractmethod
    def tables(self):
        ...

    @cached_property
    @abc.abstractmethod
    def topics(self):
        ...

    @property
    @abc.abstractmethod
    def monitor(self):
        ...

    @monitor.setter
    def monitor(self, value):
        ...

    @cached_property
    @abc.abstractmethod
    def flow_control(self):
        return FlowControlEvent(loop=self.loop)

    @property
    @abc.abstractmethod
    def http_client(self):
        ...

    @http_client.setter
    def http_client(self, client):
        ...

    @cached_property
    @abc.abstractmethod
    def assignor(self):
        ...

    @cached_property
    @abc.abstractmethod
    def router(self):
        ...

    @cached_property
    @abc.abstractmethod
    def serializers(self):
        ...

    @cached_property
    @abc.abstractmethod
    def web(self):
        ...

    @cached_property
    @abc.abstractmethod
    def in_transaction(self):
        ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span):
        ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name):
        ...
