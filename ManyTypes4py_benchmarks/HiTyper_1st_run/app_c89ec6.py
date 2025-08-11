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
    def default_tracer(self) -> None:
        ...

    @abc.abstractmethod
    def trace(self, name: Union[int, str, dict], sample_rate: Union[None, int, str, dict]=None, **extra_context) -> None:
        ...

    @abc.abstractmethod
    def get_tracer(self, service_name: Union[str, int, list[str]]) -> None:
        ...

class BootStrategyT:
    enable_kafka = True
    enable_kafka_consumer = None
    enable_kafka_producer = None
    enable_web = None
    enable_sensors = True

    @abc.abstractmethod
    def __init__(self, app: Union[bool, types.topics.ChannelT, typing.Sequence[str]], *, enable_web: Union[None, bool, types.topics.ChannelT, typing.Sequence[str]]=None, enable_kafka: bool=True, enable_kafka_producer: Union[None, bool, types.topics.ChannelT, typing.Sequence[str]]=None, enable_kafka_consumer: Union[None, bool, types.topics.ChannelT, typing.Sequence[str]]=None, enable_sensors: bool=True) -> None:
        ...

    @abc.abstractmethod
    def server(self) -> None:
        ...

    @abc.abstractmethod
    def client_only(self) -> None:
        ...

    @abc.abstractmethod
    def producer_only(self) -> None:
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
    def __init__(self, id, *, monitor, config_source=None, **options) -> None:
        self.on_startup_finished = None

    @abc.abstractmethod
    def config_from_object(self, obj: bool, *, silent: bool=False, force: bool=False) -> None:
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
    def discover(self, *extra_modules, categories: tuple[typing.Text]=('a', 'b', 'c') -> None, ignore=('foo', 'bar')):
        ...

    @abc.abstractmethod
    def topic(self, *topics, pattern: Union[None, bool]=None, schema: Union[None, bool]=None, key_type: Union[None, bool]=None, value_type: Union[None, bool]=None, key_serializer: Union[None, bool]=None, value_serializer: Union[None, bool]=None, partitions: Union[None, bool]=None, retention: Union[None, bool]=None, compacting: Union[None, bool]=None, deleting: Union[None, bool]=None, replicas: Union[None, bool]=None, acks: bool=True, internal: bool=False, config: Union[None, bool]=None, maxsize: Union[None, bool]=None, allow_empty: bool=False, has_prefix: bool=False, loop: Union[None, bool]=None) -> None:
        ...

    @abc.abstractmethod
    def channel(self, *, schema: Union[None, typing.Hashable, bool]=None, key_type: Union[None, typing.Hashable, bool]=None, value_type: Union[None, typing.Hashable, bool]=None, maxsize: Union[None, typing.Hashable, bool]=None, loop: Union[None, typing.Hashable, bool]=None) -> None:
        ...

    @abc.abstractmethod
    def agent(self, channel: Union[None, bool, str]=None, *, name: Union[None, bool, str]=None, concurrency: int=1, supervisor_strategy: Union[None, bool, str]=None, sink: Union[None, bool, str]=None, isolated_partitions: bool=False, use_reply_headers: bool=True, **kwargs) -> None:
        ...

    @abc.abstractmethod
    @no_type_check
    def task(self, fun: Union[bool, faustypes.app.TaskArg, typing.Callable], *, on_leader: bool=False, traced: bool=True) -> None:
        ...

    @abc.abstractmethod
    def timer(self, interval: Union[bool, float, str], on_leader: bool=False, traced: bool=True, name: Union[None, bool, float, str]=None, max_drift_correction: float=0.1) -> None:
        ...

    @abc.abstractmethod
    def crontab(self, cron_format: Union[bool, str, datetime.tzinfo], *, timezone: Union[None, bool, str, datetime.tzinfo]=None, on_leader: bool=False, traced: bool=True) -> None:
        ...

    @abc.abstractmethod
    def service(self, cls: Union[str, typing.Type]) -> None:
        ...

    @abc.abstractmethod
    def stream(self, channel: Union[str, int, list[str]], beacon: Union[None, str, int, list[str]]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def Table(self, name: Union[bool, str, None], *, default: Union[None, bool, str]=None, window: Union[None, bool, str]=None, partitions: Union[None, bool, str]=None, help: Union[None, bool, str]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def GlobalTable(self, name: Union[bool, str, None, list[dict[str, typing.Any]]], *, default: Union[None, bool, str, list[dict[str, typing.Any]]]=None, window: Union[None, bool, str, list[dict[str, typing.Any]]]=None, partitions: Union[None, bool, str, list[dict[str, typing.Any]]]=None, help: Union[None, bool, str, list[dict[str, typing.Any]]]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def SetTable(self, name: Union[bool, str], *, window: Union[None, bool, str]=None, partitions: Union[None, bool, str]=None, start_manager: bool=False, help: Union[None, bool, str]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def SetGlobalTable(self, name: Union[str, bool], *, window: Union[None, str, bool]=None, partitions: Union[None, str, bool]=None, start_manager: bool=False, help: Union[None, str, bool]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def page(self, path: Union[str, typing.Mapping, typing.Type], *, base: Any=View, cors_options: Union[None, str, typing.Mapping, typing.Type]=None, name: Union[None, str, typing.Mapping, typing.Type]=None) -> None:
        ...

    @abc.abstractmethod
    def table_route(self, table: Union[str, faustypes.tables.CollectionT, dict], shard_param: Union[None, str, faustypes.tables.CollectionT, dict]=None, *, query_param: Union[None, str, faustypes.tables.CollectionT, dict]=None, match_info: Union[None, str, faustypes.tables.CollectionT, dict]=None, exact_key: Union[None, str, faustypes.tables.CollectionT, dict]=None) -> None:
        ...

    @abc.abstractmethod
    def command(self, *options, base: Union[None, str, T, dict]=None, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def create_event(self, key: Union[str, bytes, None, faustypes.HeadersArg], value: Union[str, bytes, None, faustypes.HeadersArg], headers: Union[str, bytes, None, faustypes.HeadersArg], message: Union[str, bytes, None, faustypes.HeadersArg]) -> None:
        ...

    @abc.abstractmethod
    async def start_client(self):
        ...

    @abc.abstractmethod
    async def maybe_start_client(self):
        ...

    @abc.abstractmethod
    def trace(self, name: Union[int, str, dict], trace_enabled=True, **extra_context) -> None:
        ...

    @abc.abstractmethod
    async def send(self, channel, key=None, value=None, partition=None, timestamp=None, headers=None, schema=None, key_serializer=None, value_serializer=None, callback=None):
        ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs) -> None:
        ...

    @stampede
    @abc.abstractmethod
    async def maybe_start_producer(self):
        ...

    @abc.abstractmethod
    def is_leader(self) -> None:
        ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Union[None, bool, tuples.FutureMessage, tuple]=None, *, clear_on_resume: bool=False, loop: Union[None, bool, tuples.FutureMessage, tuple]=None) -> None:
        ...

    @abc.abstractmethod
    def Worker(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Union[faustypes.web.Web, str, abilian.app.Application]) -> None:
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
    def conf(self) -> None:
        ...

    @conf.setter
    def conf(self, settings) -> None:
        ...

    @property
    @abc.abstractmethod
    def transport(self) -> None:
        ...

    @transport.setter
    def transport(self, transport) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> None:
        ...

    @producer_transport.setter
    def producer_transport(self, transport) -> None:
        ...

    @property
    @abc.abstractmethod
    def cache(self) -> None:
        ...

    @cache.setter
    def cache(self, cache) -> None:
        ...

    @property
    @abc.abstractmethod
    def producer(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def consumer(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def tables(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def topics(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def monitor(self) -> None:
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
    def http_client(self) -> None:
        ...

    @http_client.setter
    def http_client(self, client) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def assignor(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def router(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def serializers(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def web(self) -> None:
        ...

    @cached_property
    @abc.abstractmethod
    def in_transaction(self) -> None:
        ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: Union[list[dict[str, typing.Any]], deeplearning.ml4pl.models.epoch.Type]) -> None:
        ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: Union[str, None]) -> None:
        ...