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
    def trace(self, name, sample_rate=None, **extra_context):
        ...

    @abc.abstractmethod
    def get_tracer(self, service_name):
        ...

class BootStrategyT:
    enable_kafka = True
    enable_kafka_consumer = None
    enable_kafka_producer = None
    enable_web = None
    enable_sensors = True

    @abc.abstractmethod
    def __init__(self, app, *, enable_web=None, enable_kafka=True, enable_kafka_producer=None, enable_kafka_consumer=None, enable_sensors=True):
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
    finalized = False
    configured = False
    rebalancing = False
    rebalancing_count = 0
    unassigned = False
    in_worker = False
    on_configured = SyncSignal()
    on_before_configured = SyncSignal()
    on_after_configured = SyncSignal()
    on_partitions_assigned = Signal()
    on_partitions_revoked = Signal()
    on_rebalance_complete = Signal()
    on_before_shutdown = Signal()
    on_worker_init = SyncSignal()
    on_produce_message = SyncSignal()
    tracer = None

    @abc.abstractmethod
    def __init__(self, id, *, monitor, config_source=None, **options):
        self.on_startup_finished = None

    @abc.abstractmethod
    def config_from_object(self, obj, *, silent=False, force=False):
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
    def discover(self, *extra_modules, categories=('a', 'b', 'c'), ignore=('foo', 'bar')):
        ...

    @abc.abstractmethod
    def topic(self, *topics, pattern=None, schema=None, key_type=None, value_type=None, key_serializer=None, value_serializer=None, partitions=None, retention=None, compacting=None, deleting=None, replicas=None, acks=True, internal=False, config=None, maxsize=None, allow_empty=False, has_prefix=False, loop=None):
        ...

    @abc.abstractmethod
    def channel(self, *, schema=None, key_type=None, value_type=None, maxsize=None, loop=None):
        ...

    @abc.abstractmethod
    def agent(self, channel=None, *, name=None, concurrency=1, supervisor_strategy=None, sink=None, isolated_partitions=False, use_reply_headers=True, **kwargs):
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun, *, on_leader=False, traced=True):
        ...

    @abc.abstractmethod
    def timer(self, interval, on_leader=False, traced=True, name=None, max_drift_correction=0.1):
        ...

    @abc.abstractmethod
    def crontab(self, cron_format, *, timezone=None, on_leader=False, traced=True):
        ...

    @abc.abstractmethod
    def service(self, cls):
        ...

    @abc.abstractmethod
    def stream(self, channel, beacon=None, **kwargs):
        ...

    @abc.abstractmethod
    def Table(self, name, *, default=None, window=None, partitions=None, help=None, **kwargs):
        ...

    @abc.abstractmethod
    def GlobalTable(self, name, *, default=None, window=None, partitions=None, help=None, **kwargs):
        ...

    @abc.abstractmethod
    def SetTable(self, name, *, window=None, partitions=None, start_manager=False, help=None, **kwargs):
        ...

    @abc.abstractmethod
    def SetGlobalTable(self, name, *, window=None, partitions=None, start_manager=False, help=None, **kwargs):
        ...

    @abc.abstractmethod
    def page(self, path, *, base=View, cors_options=None, name=None):
        ...

    @abc.abstractmethod
    def table_route(self, table, shard_param=None, *, query_param=None, match_info=None, exact_key=None):
        ...

    @abc.abstractmethod
    def command(self, *options, base=None, **kwargs):
        ...

    @abc.abstractmethod
    def create_event(self, key, value, headers, message):
        ...

    @abc.abstractmethod
    async def start_client(self):
        ...

    @abc.abstractmethod
    async def maybe_start_client(self):
        ...

    @abc.abstractmethod
    def trace(self, name, trace_enabled=True, **extra_context):
        ...

    @abc.abstractmethod
    async def send(self, channel, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None):
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs):
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self):
        ...

    @abc.abstractmethod
    def is_leader(self):
        ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize=None, *, clear_on_resume=False, loop=None):
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs):
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