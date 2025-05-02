"""Faust Application.

An app is an instance of the Faust library.
Everything starts here.

"""
import asyncio
import importlib
import inspect
import re
import sys
import typing
import warnings
from datetime import tzinfo
from functools import wraps
from itertools import chain
from typing import (
    Any, AsyncIterable, Awaitable, Callable, ClassVar, ContextManager, Dict,
    Iterable, Iterator, List, Mapping, MutableMapping, MutableSequence,
    Optional, Pattern, Set, Tuple, Type, TypeVar, Union, cast, no_type_check
)
import opentracing
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
from faust.utils.tracing import (
    call_with_trace, noop_span, operation_name_from_fun, set_current_span,
    traced_from_parent_span
)
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
from faust.types.transports import (
    ConductorT, ConsumerT, ProducerT, TPorTopicSet, TransportT
)
from faust.types.tuples import Message, MessageSentCallback, RecordMetadata, TP
from faust.types.web import (
    CacheBackendT, HttpClientT, PageArg, Request, ResourceOptions, Response,
    ViewDecorator, ViewHandlerFun, Web
)
from faust.types.windows import WindowT
from ._attached import Attachments

if typing.TYPE_CHECKING:
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck import LiveCheck as _LiveCheck
    from faust.transport.consumer import Fetcher as _Fetcher
    from faust.worker import Worker as _Worker
else:
    class _AppCommand: ...
    class _LiveCheck: ...
    class _Fetcher: ...
    class _Worker: ...

__all__ = ['App', 'BootStrategy']
logger = get_logger(__name__)
_T = TypeVar('_T')
APP_REPR_FINALIZED = '\n<{name}({c.id}): {c.broker} {s.state} agents({agents}) {id:#x}>\n'.strip()
APP_REPR_UNFINALIZED = '\n<{name}: <non-finalized> {id:#x}>\n'.strip()
SCAN_AGENT = 'faust.agent'
SCAN_COMMAND = 'faust.command'
SCAN_PAGE = 'faust.page'
SCAN_SERVICE = 'faust.service'
SCAN_TASK = 'faust.task'
SCAN_CATEGORIES = [SCAN_AGENT, SCAN_COMMAND, SCAN_PAGE, SCAN_SERVICE, SCAN_TASK]
SCAN_IGNORE = [re.compile('test_.*').search, '.__main__']
E_NEED_ORIGIN = '\n`origin` argument to faust.App is mandatory when autodiscovery enabled.\n\nThis parameter sets the canonical path to the project package,\nand describes how a user, or program can find it on the command-line when using\nthe `faust -A project` option.  It\'s also used as the default package\nto scan when autodiscovery is enabled.\n\nIf your app is defined in a module: ``project/app.py``, then the\norigin will be "project":\n\n    # file: project/app.py\n    import faust\n\n    app = faust.App(\n        id=\'myid\',\n        origin=\'project\',\n    )\n'
W_OPTION_DEPRECATED = 'Argument {old!r} is deprecated and scheduled for removal in Faust 1.0.\n\nPlease use {new!r} instead.\n'
W_DEPRECATED_SHARD_PARAM = 'The second argument to `@table_route` is deprecated,\nplease use the `query_param` keyword argument instead.\n'
TaskDecoratorRet = Union[Callable[[TaskArg], TaskArg], TaskArg]

class BootStrategy(BootStrategyT):
    """App startup strategy."""
    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    def __init__(self, app: AppT, *,
                 enable_web: Optional[bool] = None,
                 enable_kafka: Optional[bool] = None,
                 enable_kafka_producer: Optional[bool] = None,
                 enable_kafka_consumer: Optional[bool] = None,
                 enable_sensors: Optional[bool] = None) -> None:
        self.app = app
        if enable_kafka is not None:
            self.enable_kafka = enable_kafka
        if enable_kafka_producer is not None:
            self.enable_kafka_producer = enable_kafka_producer
        if enable_kafka_consumer is not None:
            self.enable_kafka_consumer = enable_kafka_consumer
        if enable_web is not None:
            self.enable_web = enable_web
        if enable_sensors is not None:
            self.enable_sensors = enable_sensors

    def server(self) -> Iterable[ServiceT]:
        """Return services to start when app is in default mode."""
        return self._chain(
            self.sensors(),
            self.kafka_producer(),
            self.web_server(),
            self.kafka_consumer(),
            self.agents(),
            self.kafka_conductor(),
            self.tables(),
        )

    def client_only(self) -> Iterable[ServiceT]:
        """Return services to start when app is in client_only mode."""
        app = cast(App, self.app)
        return self._chain(
            self.kafka_producer(),
            self.kafka_client_consumer(),
            self.kafka_conductor(),
            [app._fetcher],
        )

    def producer_only(self) -> Iterable[ServiceT]:
        """Return services to start when app is in producer_only mode."""
        return self._chain(
            self.web_server(),
            self.kafka_producer(),
        )

    def _chain(self, *arguments: Iterable[ServiceT]) -> Iterable[ServiceT]:
        return cast(Iterable[ServiceT], chain.from_iterable(arguments))

    def sensors(self) -> List[ServiceT]:
        """Return list of services required to start sensors."""
        if self.enable_sensors:
            return self.app.sensors
        return []

    def kafka_producer(self) -> List[ServiceT]:
        """Return list of services required to start Kafka producer."""
        if self._should_enable_kafka_producer():
            return [self.app.producer]
        return []

    def _should_enable_kafka_producer(self) -> bool:
        if self.enable_kafka_producer is None:
            return self.enable_kafka
        return self.enable_kafka_producer

    def kafka_consumer(self) -> List[ServiceT]:
        """Return list of services required to start Kafka consumer."""
        if self._should_enable_kafka_consumer():
            app = cast(App, self.app)
            return [self.app.consumer, app._leader_assignor, app._reply_consumer]
        return []

    def _should_enable_kafka_consumer(self) -> bool:
        if self.enable_kafka_consumer is None:
            return self.enable_kafka
        return self.enable_kafka_consumer

    def kafka_client_consumer(self) -> List[ServiceT]:
        """Return list of services required to start Kafka client consumer."""
        app = cast(App, self.app)
        return [app.consumer, app._reply_consumer]

    def agents(self) -> List[ServiceT]:
        """Return list of services required to start agents."""
        return [self.app.agents]

    def kafka_conductor(self) -> List[ServiceT]:
        """Return list of services required to start Kafka conductor."""
        if self._should_enable_kafka_consumer():
            return [self.app.topics]
        return []

    def web_server(self) -> List[ServiceT]:
        """Return list of web-server services."""
        if self._should_enable_web():
            return list(self.web_components()) + [self.app.web]
        return []

    def _should_enable_web(self) -> bool:
        if self.enable_web is None:
            return self.app.conf.web_enabled
        return self.enable_web

    def web_components(self) -> List[ServiceT]:
        """Return list of web-related services (excluding web server)."""
        return [self.app.cache]

    def tables(self) -> List[ServiceT]:
        """Return list of table-related services."""
        if self._should_enable_kafka_consumer():
            return [self.app.tables]
        return []

class App(AppT, Service):
    """Faust Application."""
    SCAN_CATEGORIES: ClassVar[List[str]] = list(SCAN_CATEGORIES)
    BootStrategy: ClassVar[Type[BootStrategyT]] = BootStrategy
    Settings: ClassVar[Type[_Settings]] = _Settings
    
    client_only: bool = False
    producer_only: bool = False
    _conf: Optional[_Settings] = None
    _config_source: Any = None
    _consumer: Optional[ConsumerT] = None
    _producer: Optional[ProducerT] = None
    _transport: Optional[TransportT] = None
    _producer_transport: Optional[TransportT] = None
    _cache: Optional[CacheBackendT] = None
    _monitor: Optional[Monitor] = None
    _http_client: Optional[HttpClientT] = None
    _extra_service_instances: Optional[List[ServiceT]] = None
    _assignment: Optional[Set[TP]] = None
    tracer: Optional[TracerT] = None
    _rebalancing_span: Optional[Any] = None
    _rebalancing_sensor_state: Optional[Any] = None

    def __init__(self, id: str, *,
                 monitor: Optional[Monitor] = None,
                 config_source: Any = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 beacon: Optional[ServiceT] = None,
                 **options: Any) -> None:
        self._default_options = (id, options)
        self.agents = AgentManager(self)
        self.sensors = SensorDelegate(self)
        self._attachments = Attachments(self)
        self._monitor = monitor
        self._app_tasks: List[Callable[[], Awaitable[None]]] = []
        self.on_startup_finished: Optional[Callable[[], Awaitable[None]]] = None
        self._extra_services: List[Type[ServiceT]] = []
        self._config_source = config_source
        self._init_signals()
        self.fixups = self._init_fixups()
        self.boot_strategy = self.BootStrategy(self)
        Service.__init__(self, loop=loop, beacon=beacon)

    def _init_signals(self) -> None:
        self.on_before_configured = self.on_before_configured.with_default_sender(self)
        self.on_configured = self.on_configured.with_default_sender(self)
        self.on_after_configured = self.on_after_configured.with_default_sender(self)
        self.on_partitions_assigned = self.on_partitions_assigned.with_default_sender(self)
        self.on_partitions_revoked = self.on_partitions_revoked.with_default_sender(self)
        self.on_worker_init = self.on_worker_init.with_default_sender(self)
        self.on_rebalance_complete = self.on_rebalance_complete.with_default_sender(self)
        self.on_before_shutdown = self.on_before_shutdown.with_default_sender(self)
        self.on_produce_message = self.on_produce_message.with_default_sender(self)

    def _init_fixups(self) -> List[FixupT]:
        return list(fixups(self))

    def on_init_dependencies(self) -> Iterable[ServiceT]:
        """Return list of additional service dependencies."""
        self.monitor.beacon.reattach(self.beacon)
        self.monitor.loop = self.loop
        self.sensors.add(self.monitor)
        if self.producer_only:
            return self.boot_strategy.producer_only()
        elif self.client_only:
            return self.boot_strategy.client_only()
        else:
            return self.boot_strategy.server()

    async def on_first_start(self) -> None:
        """Call first time app starts in this process."""
        self._create_directories()

    async def on_start(self) -> None:
        """Call every time app start/restarts."""
        self.finalize()
        self.topics.beacon.reattach(self.consumer.beacon)
        if self.conf.debug:
            logger.warning('!!! DEBUG is enabled -- disable for production environments')

    async def on_started(self) -> None:
        """Call when app is fully started."""
        if not await self._wait_for_table_recovery_completed():
            await self.on_started_init_extra_tasks()
            await self.on_started_init_extra_services()
            if self.on_startup_finished:
                await self.on_startup_finished()

    async def _wait_for_table_recovery_completed(self) -> bool:
        return await self.tables.wait_until_recovery_completed()

    async def on_started_init_extra_tasks(self) -> None:
        """Call when started to start additional tasks."""
        for task in self._app_tasks:
            self.add_future(task())

    async def on_started_init_extra_services(self) -> None:
        """Call when initializing extra services at startup."""
        if self._extra_service_instances is None:
            self._extra_service_instances = [
                await self.on_init_extra_service(service)
                for service in self._extra_services
            ]

    async def on_init_extra_service(self, service: Union[Type[ServiceT], ServiceT]) -> ServiceT:
        """Call when adding user services to this app."""
        s = self._prepare_subservice(service)
        await self.add_runtime_dependency(s)
        return s

    def _prepare_subservice(self, service: Union[Type[ServiceT], ServiceT]) -> ServiceT:
        if inspect.isclass(service):
            return cast(Type[ServiceT], service)(loop=self.loop, beacon=self.beacon)
        else:
            return cast(ServiceT, service)

    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None:
        """Read configuration from object."""
        self._config_source = obj
        if self.finalized or self.configured:
            self.Settings._warn_already_configured()
        if force or self.configured:
            self._conf = None
            self._configure(silent=silent)

    def finalize(self) -> None:
        """Finalize app configuration."""
        if not self.finalized:
            self.finalized = True
            id = self.conf.id
            if not id:
                raise ImproperlyConfigured('App requires an id!')

    async def _maybe_close_http_client(self) -> None:
        if self._http_client:
            await self._http_client.close()

    def worker_init(self) -> None:
        """Init worker/CLI commands."""
        for fixup in self.fixups:
            fixup.on_worker_init()

    def worker_init_post_autodiscover(self) -> None:
        """Init worker after autodiscover."""
        self.web.init_server()
        self.on_worker_init.send()

    def discover(self, *extra_modules: str,
                 categories: Optional[List[str]] = None,
                 ignore: List[Any] = SCAN_IGNORE) -> None:
        """Discover decorators in packages."""
        if categories is None:
            categories = self.SCAN_CATEGORIES
        modules = set(self._discovery_modules())
        modules |= set(extra_modules)
        for fixup in self.fixups:
            modules |= set(fixup.autodiscover_modules())
        if modules:
            scanner = venusian.Scanner()
            for name in modules:
                try:
                    module = importlib.import_module(name)
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        f'Unknown module {name} in App.conf.autodiscover list')
                scanner.scan(
                    module, ignore=ignore, categories=tuple(categories),
                    onerror=self._on_autodiscovery_error)

    def _on_autodiscovery_error(self, name: str