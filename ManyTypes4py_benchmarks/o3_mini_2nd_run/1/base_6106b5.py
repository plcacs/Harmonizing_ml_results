#!/usr/bin/env python3
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
from typing import Any, AsyncIterable, Awaitable, Callable, ClassVar, ContextManager, Dict, Iterable, Iterator, List, Mapping, MutableMapping, MutableSequence, Optional, Pattern, Set, Tuple, Type, TypeVar, Union, cast, no_type_check, overload, NoReturn

import opentracing
from mode import Seconds, Service, ServiceT, SupervisorStrategyT, want_seconds
from mode.utils.aiter import aiter
from mode.utils.collections import force_mapping
from mode.utils.contexts import nullcontext
from mode.utils.futures import stampede
from mode.utils.imports import import_from_cwd, smart_import
from mode.utils.logging import flight_recorder, get_logger
from mode.utils.objects import cached_property, qualname, shortlabel
from mode.utils.typing import NoReturn as TypingNoReturn
from mode.utils.queues import FlowControlEvent, ThrowableQueue
from mode.utils.types.trees import NodeT
from faust import transport
from faust.agents import AgentFun, AgentManager, AgentT, ReplyConsumer, SinkT
from faust.channels import Channel, ChannelT
from faust.exceptions import ConsumerNotStarted, ImproperlyConfigured, SameNode
from faust.fixups import FixupT, fixups
from faust.sensors import Monitor, SensorDelegate
from faust.utils import cron, venusian
from faust.utils.tracing import call_with_trace, noop_span, operation_name_from_fun, set_current_span, traced_from_parent_span
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
from faust.types.transports import ConductorT, ConsumerT, ProducerT, TPorTopicSet, TransportT
from faust.types.tuples import Message, MessageSentCallback, RecordMetadata, TP
from faust.types.web import CacheBackendT, HttpClientT, PageArg, Request, ResourceOptions, Response, ViewDecorator, ViewHandlerFun, Web
from faust.types.windows import WindowT
from ._attached import Attachments

if typing.TYPE_CHECKING:
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck import LiveCheck as _LiveCheck
    from faust.transport.consumer import Fetcher as _Fetcher
    from faust.worker import Worker as _Worker
else:

    class _AppCommand:
        ...

    class _LiveCheck:
        ...

    class _Fetcher:
        ...

    class _Worker:
        ...

__all__ = ['App', 'BootStrategy']
logger = get_logger(__name__)
_T = TypeVar('_T')

APP_REPR_FINALIZED: str = '\n<{name}({c.id}): {c.broker} {s.state} agents({agents}) {id:#x}>\n'.strip()
APP_REPR_UNFINALIZED: str = '\n<{name}: <non-finalized> {id:#x}>\n'.strip()
SCAN_AGENT: str = 'faust.agent'
SCAN_COMMAND: str = 'faust.command'
SCAN_PAGE: str = 'faust.page'
SCAN_SERVICE: str = 'faust.service'
SCAN_TASK: str = 'faust.task'
SCAN_CATEGORIES: List[str] = [SCAN_AGENT, SCAN_COMMAND, SCAN_PAGE, SCAN_SERVICE, SCAN_TASK]
SCAN_IGNORE: List[Union[Callable[[str], Any], str]] = [re.compile('test_.*').search, '.__main__']
E_NEED_ORIGIN: str = (
    "\n`origin` argument to faust.App is mandatory when autodiscovery enabled.\n\n"
    "This parameter sets the canonical path to the project package,\n"
    "and describes how a user, or program can find it on the command-line when using\n"
    "the `faust -A project` option.  It's also used as the default package\n"
    "to scan when autodiscovery is enabled.\n\n"
    "If your app is defined in a module: ``project/app.py``, then the\n"
    "origin will be \"project\":\n\n"
    "    # file: project/app.py\n"
    "    import faust\n\n"
    "    app = faust.App(\n"
    "        id='myid',\n"
    "        origin='project',\n"
    "    )\n"
)
W_OPTION_DEPRECATED: str = (
    'Argument {old!r} is deprecated and scheduled for removal in Faust 1.0.\n\n'
    'Please use {new!r} instead.\n'
)
W_DEPRECATED_SHARD_PARAM: str = (
    'The second argument to `@table_route` is deprecated,\n'
    'please use the `query_param` keyword argument instead.\n'
)
TaskDecoratorRet = Union[Callable[[TaskArg], TaskArg], TaskArg]


class BootStrategy(BootStrategyT):
    """App startup strategy.

    The startup strategy defines the graph of services
    to start when the Faust worker for an app starts.
    """
    enable_kafka: bool = True
    enable_kafka_consumer: Optional[bool] = None
    enable_kafka_producer: Optional[bool] = None
    enable_web: Optional[bool] = None
    enable_sensors: bool = True

    def __init__(self, app: AppT, *, enable_web: Optional[bool] = None, enable_kafka: Optional[bool] = None, enable_kafka_producer: Optional[bool] = None, enable_kafka_consumer: Optional[bool] = None, enable_sensors: Optional[bool] = None) -> None:
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
        return self._chain(self.sensors(), self.kafka_producer(), self.web_server(), self.kafka_consumer(), self.agents(), self.kafka_conductor(), self.tables())

    def client_only(self) -> Iterable[ServiceT]:
        """Return services to start when app is in client_only mode."""
        app = cast(App, self.app)
        return self._chain(self.kafka_producer(), self.kafka_client_consumer(), self.kafka_conductor(), [app._fetcher])

    def producer_only(self) -> Iterable[ServiceT]:
        """Return services to start when app is in producer_only mode."""
        return self._chain(self.web_server(), self.kafka_producer())

    def _chain(self, *arguments: Iterable[ServiceT]) -> Iterable[ServiceT]:
        return cast(Iterable[ServiceT], chain.from_iterable(arguments))

    def sensors(self) -> Iterable[ServiceT]:
        """Return list of services required to start sensors."""
        if self.enable_sensors:
            return self.app.sensors  # type: ignore
        return []

    def kafka_producer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka producer."""
        if self._should_enable_kafka_producer():
            return [self.app.producer]  # type: ignore
        return []

    def _should_enable_kafka_producer(self) -> bool:
        if self.enable_kafka_producer is None:
            return self.enable_kafka
        return self.enable_kafka_producer

    def kafka_consumer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka consumer."""
        if self._should_enable_kafka_consumer():
            app = cast(App, self.app)
            return [app.consumer, app._leader_assignor, app._reply_consumer]  # type: ignore
        return []

    def _should_enable_kafka_consumer(self) -> bool:
        if self.enable_kafka_consumer is None:
            return self.enable_kafka
        return self.enable_kafka_consumer

    def kafka_client_consumer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka client consumer."""
        app = cast(App, self.app)
        return [app.consumer, app._reply_consumer]  # type: ignore

    def agents(self) -> Iterable[ServiceT]:
        """Return list of services required to start agents."""
        return [self.app.agents]  # type: ignore

    def kafka_conductor(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka conductor."""
        if self._should_enable_kafka_consumer():
            return [self.app.topics]  # type: ignore
        return []

    def web_server(self) -> Iterable[ServiceT]:
        """Return list of web-server services."""
        if self._should_enable_web():
            return list(self.web_components()) + [self.app.web]
        return []

    def _should_enable_web(self) -> bool:
        if self.enable_web is None:
            return self.app.conf.web_enabled  # type: ignore
        return self.enable_web

    def web_components(self) -> Iterable[ServiceT]:
        """Return list of web-related services (excluding web server)."""
        return [self.app.cache]  # type: ignore

    def tables(self) -> Iterable[ServiceT]:
        """Return list of table-related services."""
        if self._should_enable_kafka_consumer():
            return [self.app.tables]  # type: ignore
        return []


class App(AppT, Service):
    """Faust Application.

    Arguments:
        id (str): Application ID.

    Keyword Arguments:
        loop (asyncio.AbstractEventLoop): optional event loop to use.

    See Also:
        :ref:`application-configuration` -- for supported keyword arguments.

    """
    SCAN_CATEGORIES: ClassVar[List[str]] = list(SCAN_CATEGORIES)
    BootStrategy: ClassVar[Type[BootStrategy]] = BootStrategy
    Settings: ClassVar[Type[_Settings]] = _Settings
    client_only: bool = False
    producer_only: bool = False
    _conf: Optional[_Settings] = None
    _config_source: Optional[Any] = None
    _consumer: Optional[Any] = None
    _producer: Optional[Any] = None
    _transport: Optional[Any] = None
    _producer_transport: Optional[Any] = None
    _cache: Optional[Any] = None
    _monitor: Optional[Monitor] = None
    _http_client: Optional[HttpClientT] = None
    _extra_service_instances: Optional[List[ServiceT]] = None
    _assignment: Optional[Set[Any]] = None
    tracer: Optional[TracerT] = None
    _rebalancing_span: Optional[Any] = None
    _rebalancing_sensor_state: Optional[Any] = None

    def __init__(self, id: str, *, monitor: Optional[Any] = None, config_source: Optional[Any] = None, loop: Optional[asyncio.AbstractEventLoop] = None, beacon: Optional[Any] = None, **options: Any) -> None:
        self._default_options: Tuple[str, Dict[str, Any]] = (id, options)
        self.agents: AgentManager = AgentManager(self)
        self.sensors: SensorDelegate = SensorDelegate(self)
        self._attachments: Attachments = Attachments(self)
        self._monitor = monitor
        self._app_tasks: List[Callable[[], Awaitable[Any]]] = []
        self.on_startup_finished: Optional[Callable[[], Awaitable[Any]]] = None
        self._extra_services: List[Any] = []
        self._config_source = config_source
        self._init_signals()
        self.fixups: List[FixupT] = self._init_fixups()
        self.boot_strategy: BootStrategy = self.BootStrategy(self)
        Service.__init__(self, loop=loop, beacon=beacon)

    def _init_signals(self) -> None:
        self.on_before_configured = self.on_before_configured.with_default_sender(self)  # type: ignore
        self.on_configured = self.on_configured.with_default_sender(self)  # type: ignore
        self.on_after_configured = self.on_after_configured.with_default_sender(self)  # type: ignore
        self.on_partitions_assigned = self.on_partitions_assigned.with_default_sender(self)  # type: ignore
        self.on_partitions_revoked = self.on_partitions_revoked.with_default_sender(self)  # type: ignore
        self.on_worker_init = self.on_worker_init.with_default_sender(self)  # type: ignore
        self.on_rebalance_complete = self.on_rebalance_complete.with_default_sender(self)  # type: ignore
        self.on_before_shutdown = self.on_before_shutdown.with_default_sender(self)  # type: ignore
        self.on_produce_message = self.on_produce_message.with_default_sender(self)  # type: ignore

    def _init_fixups(self) -> List[FixupT]:
        return list(fixups(self))

    def on_init_dependencies(self) -> Iterable[ServiceT]:
        """Return list of additional service dependencies."""
        self.monitor.beacon.reattach(self.beacon)  # type: ignore
        self.monitor.loop = self.loop  # type: ignore
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
        self.topics.beacon.reattach(self.consumer.beacon)  # type: ignore
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
        return await self.tables.wait_until_recovery_completed()  # type: ignore

    async def on_started_init_extra_tasks(self) -> None:
        """Call when started to start additional tasks."""
        for task in self._app_tasks:
            self.add_future(task())

    async def on_started_init_extra_services(self) -> None:
        """Call when initializing extra services at startup."""
        if self._extra_service_instances is None:
            self._extra_service_instances = [await self.on_init_extra_service(service) for service in self._extra_services]

    async def on_init_extra_service(self, service: Any) -> ServiceT:
        """Call when adding user services to this app."""
        s: ServiceT = self._prepare_subservice(service)
        await self.add_runtime_dependency(s)
        return s

    def _prepare_subservice(self, service: Any) -> ServiceT:
        if inspect.isclass(service):
            return cast(Type[ServiceT], service)(loop=self.loop, beacon=self.beacon)
        else:
            return cast(ServiceT, service)

    def config_from_object(self, obj: Any, *, silent: bool = False, force: bool = False) -> None:
        """Read configuration from object.
        
        Object is either an actual object or the name of a module to import.
        """
        self._config_source = obj
        if self.finalized or self.configured:  # type: ignore
            self.Settings._warn_already_configured()  # type: ignore
        if force or self.configured:  # type: ignore
            self._conf = None
            self._configure(silent=silent)

    def finalize(self) -> None:
        """Finalize app configuration."""
        if not self.finalized:  # type: ignore
            self.finalized = True  # type: ignore
            id_value = self.conf.id
            if not id_value:
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

    def discover(self, *extra_modules: str, categories: Optional[Any] = None, ignore: Any = SCAN_IGNORE) -> None:
        """Discover decorators in packages."""
        if categories is None:
            categories = self.SCAN_CATEGORIES
        modules: Set[str] = set(self._discovery_modules())
        modules |= set(extra_modules)
        for fixup in self.fixups:
            modules |= set(fixup.autodiscover_modules())
        if modules:
            scanner = venusian.Scanner()
            for name in modules:
                try:
                    module = importlib.import_module(name)
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(f'Unknown module {name} in App.conf.autodiscover list')
                scanner.scan(module, ignore=ignore, categories=tuple(categories), onerror=self._on_autodiscovery_error)

    def _on_autodiscovery_error(self, name: str) -> None:
        logger.warning('Autodiscovery importing module %r raised error: %r', name, sys.exc_info()[1], exc_info=True)

    def _discovery_modules(self) -> List[str]:
        modules: List[str] = []
        autodiscover = self.conf.autodiscover  # type: ignore
        if autodiscover:
            if isinstance(autodiscover, bool):
                if self.conf.origin is None:
                    raise ImproperlyConfigured(E_NEED_ORIGIN)
            elif callable(autodiscover):
                modules.extend(cast(Callable[[], Iterator[str]], autodiscover)())
            else:
                modules.extend(autodiscover)
            if self.conf.origin:  # type: ignore
                modules.append(self.conf.origin)  # type: ignore
        return modules

    def main(self) -> NoReturn:
        """Execute the :program:`faust` umbrella command using this app."""
        from faust.cli.faust import cli
        self.finalize()
        self.worker_init()
        if self.conf.autodiscover:  # type: ignore
            self.discover()
        self.worker_init_post_autodiscover()
        cli(app=self)
        raise SystemExit(3451)

    def topic(self, *topics: str, pattern: Optional[str] = None, schema: Optional[Any] = None, key_type: Optional[Type[K]] = None, value_type: Optional[Type[V]] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, partitions: Optional[int] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Mapping[str, Any]] = None, maxsize: Optional[int] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional[Any] = None) -> TopicT:
        """Create topic description."""
        return cast(TopicT, self.conf.Topic(self, topics=topics, pattern=pattern, schema=schema, key_type=key_type, value_type=value_type, key_serializer=key_serializer, value_serializer=value_serializer, partitions=partitions, retention=retention, compacting=compacting, deleting=deleting, replicas=replicas, acks=acks, internal=internal, config=config, allow_empty=allow_empty, has_prefix=has_prefix, loop=loop))  # type: ignore

    def channel(self, *, schema: Optional[Any] = None, key_type: Optional[Type[K]] = None, value_type: Optional[Type[V]] = None, maxsize: Optional[int] = None, loop: Optional[asyncio.AbstractEventLoop] = None) -> ChannelT:
        """Create new channel."""
        return Channel(self, schema=schema, key_type=key_type, value_type=value_type, maxsize=maxsize, loop=loop)

    def agent(self, channel: Optional[ChannelT] = None, *, name: Optional[str] = None, concurrency: int = 1, supervisor_strategy: Optional[SupervisorStrategyT] = None, sink: Optional[Union[SinkT, List[SinkT]]] = None, isolated_partitions: bool = False, use_reply_headers: bool = True, **kwargs: Any) -> Callable[[AgentFun], AgentT]:
        """Create Agent from async def function."""
        def _inner(fun: AgentFun) -> AgentT:
            agent: AgentT = cast(AgentT, self.conf.Agent(fun, name=name, app=self, channel=channel, concurrency=concurrency, supervisor_strategy=supervisor_strategy, sink=sink, isolated_partitions=isolated_partitions, on_error=self._on_agent_error, use_reply_headers=use_reply_headers, help=fun.__doc__, **kwargs))  # type: ignore
            self.agents[agent.name] = agent
            self.topics.beacon.add(agent)  # type: ignore
            venusian.attach(agent, category=SCAN_AGENT)
            return agent
        return _inner

    actor = agent

    async def _on_agent_error(self, agent: AgentT, exc: Exception) -> None:
        if self._consumer:
            try:
                await self._consumer.on_task_error(exc)
            except MemoryError:
                raise
            except Exception as exc_inner:
                self.log.exception('Consumer error callback raised: %r', exc_inner)

    @no_type_check
    def task(self, fun: Optional[Callable[..., Awaitable[Any]]] = None, *, on_leader: bool = False, traced: bool = True) -> Union[Callable[[TaskArg], TaskArg], TaskArg]:
        """Define an async def function to be started with the app."""
        def _inner(fun_inner: Callable[..., Awaitable[Any]]) -> TaskArg:
            return self._task(fun_inner, on_leader=on_leader, traced=traced)
        return _inner(fun) if fun is not None else _inner

    def _task(self, fun: Callable[..., Awaitable[Any]], on_leader: bool = False, traced: bool = False) -> TaskArg:
        app = self

        @wraps(fun)
        async def _wrapped() -> Any:
            should_run: bool = app.is_leader() if on_leader else True
            if should_run:
                with self.trace(shortlabel(fun), trace_enabled=traced):
                    if inspect.signature(fun).parameters:
                        task_takes_app: Callable[[AppT], Awaitable[Any]] = fun  # type: ignore
                        return await task_takes_app(app)
                    else:
                        task: Callable[[], Awaitable[Any]] = fun  # type: ignore
                        return await task()
        venusian.attach(_wrapped, category=SCAN_TASK)
        self._app_tasks.append(_wrapped)
        return _wrapped

    @no_type_check
    def timer(self, interval: Seconds, on_leader: bool = False, traced: bool = True, name: Optional[str] = None, max_drift_correction: float = 0.1) -> Callable[[Callable[..., Awaitable[Any]]], TaskArg]:
        """Define an async def function to be run at periodic intervals."""
        interval_s: float = want_seconds(interval)

        def _inner(fun: Callable[..., Awaitable[Any]]) -> TaskArg:
            timer_name: str = name or qualname(fun)

            @wraps(fun)
            async def around_timer(*args: Any, **kwargs: Any) -> None:
                async for _ in self.itertimer(interval_s, name=timer_name, max_drift_correction=max_drift_correction):
                    should_run: bool = not on_leader or self.is_leader()
                    if should_run:
                        with self.trace(shortlabel(fun), trace_enabled=traced):
                            await fun(*args, **kwargs)
            return cast(TaskArg, self.task(around_timer, traced=False))
        return _inner

    def crontab(self, cron_format: str, *, timezone: Optional[tzinfo] = None, on_leader: bool = False, traced: bool = True) -> Callable[[Callable[..., Awaitable[Any]]], TaskArg]:
        """Define periodic task using Crontab description."""
        def _inner(fun: Callable[..., Awaitable[Any]]) -> TaskArg:
            @wraps(fun)
            async def cron_starter(*args: Any, **kwargs: Any) -> None:
                _tz: Any = self.conf.timezone if timezone is None else timezone  # type: ignore
                while not self.should_stop:
                    await self.sleep(cron.secs_for_next(cron_format, _tz))
                    if not self.should_stop:
                        should_run: bool = not on_leader or self.is_leader()
                        if should_run:
                            with self.trace(shortlabel(fun), trace_enabled=traced):
                                await fun(*args, **kwargs)
            return cast(TaskArg, self.task(cron_starter, traced=False))
        return _inner

    def service(self, cls: Type[ServiceT]) -> Type[ServiceT]:
        """Decorate :class:`mode.Service` to be started with the app."""
        venusian.attach(cls, category=SCAN_SERVICE)
        self._extra_services.append(cls)
        return cls

    def is_leader(self) -> bool:
        """Return :const:`True` if we are in leader worker process."""
        return self._leader_assignor.is_leader()  # type: ignore

    def stream(self, channel: Any, beacon: Optional[Any] = None, **kwargs: Any) -> StreamT:
        """Create new stream from channel/topic/iterable/async iterable."""
        return cast(StreamT, self.conf.Stream(app=self, channel=aiter(channel) if channel is not None else None, beacon=beacon or self.beacon, **kwargs))  # type: ignore

    def Table(self, name: str, *, default: Optional[Callable[[], Any]] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Define new table."""
        table = self.tables.add(cast(TableT, self.conf.Table(self, name=name, default=default, beacon=self.tables.beacon, partitions=partitions, help=help, **kwargs)))  # type: ignore
        return cast(TableT, table.using_window(window) if window else table)

    def GlobalTable(self, name: str, *, default: Optional[Callable[[], Any]] = None, window: Optional[WindowT] = None, partitions: Optional[int] = None, help: Optional[str] = None, **kwargs: Any) -> GlobalTableT:
        """Define new global table."""
        gtable = self.tables.add(cast(GlobalTableT, self.conf.GlobalTable(self, name=name, default=default, beacon=self.tables.beacon, partitions=partitions, standby_buffer_size=1, is_global=True, help=help, **kwargs)))  # type: ignore
        return cast(GlobalTableT, gtable.using_window(window) if window else gtable)

    def SetTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Table of sets."""
        table = self.tables.add(cast(TableT, self.conf.SetTable(self, name=name, beacon=self.tables.beacon, partitions=partitions, start_manager=start_manager, help=help, **kwargs)))  # type: ignore
        return cast(TableT, table.using_window(window) if window else table)

    def SetGlobalTable(self, name: str, *, window: Optional[WindowT] = None, partitions: Optional[int] = None, start_manager: bool = False, help: Optional[str] = None, **kwargs: Any) -> TableT:
        """Table of sets (global)."""
        table = self.tables.add(cast(TableT, self.conf.SetGlobalTable(self, name=name, beacon=self.tables.beacon, partitions=partitions, start_manager=start_manager, help=help, **kwargs)))  # type: ignore
        return cast(TableT, table.using_window(window) if window else table)

    def page(self, path: str, *, base: Optional[Type[View]] = None, cors_options: Optional[Any] = None, name: Optional[str] = None) -> Callable[[Union[Callable[..., Any], Type[View]]], Union[Type[View], View]]:
        """Decorate view to be included in the web server."""
        view_base: Type[View] = base if base is not None else View

        def _decorator(fun: Union[Callable[..., Any], Type[View]]) -> Union[Type[View], View]:
            view: Optional[Union[Type[View], View]] = None
            if inspect.isclass(fun):
                view = cast(Type[View], fun)
                if not issubclass(view, View):
                    raise TypeError('When decorating class, it must be subclass of View')
            if view is None:
                view = view_base.from_handler(cast(ViewHandlerFun, fun))
            view.view_name = name or view.__name__  # type: ignore
            view.view_path = path  # type: ignore
            self.web.add_view(view, cors_options=cors_options)
            venusian.attach(view, category=SCAN_PAGE)
            return view
        return _decorator

    def table_route(self, table: TableT, shard_param: Optional[Any] = None, *, query_param: Optional[Any] = None, match_info: Optional[str] = None, exact_key: Optional[Any] = None) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorate view method to route request to table key destination."""
        def _decorator(fun: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            _query_param = query_param
            if shard_param is not None:
                warnings.warn(DeprecationWarning(W_DEPRECATED_SHARD_PARAM))
                if query_param:
                    raise TypeError('Cannot specify shard_param and query_param')
                _query_param = shard_param
            if _query_param is None and match_info is None and (exact_key is None):
                raise TypeError('Need one of query_param, shard_param, or exact key')

            @wraps(fun)
            async def get(view: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                if exact_key:
                    key = exact_key
                elif match_info:
                    key = request.match_info[match_info]
                elif _query_param:
                    key = request.query[_query_param]
                else:
                    raise Exception('cannot get here')
                try:
                    return await self.router.route_req(table.name, key, view.web, request)
                except SameNode:
                    return await fun(view, request, *args, **kwargs)
            return get
        return _decorator

    def command(self, *options: Any, base: Optional[Any] = None, **kwargs: Any) -> Callable[[Callable[..., Awaitable[Any]]], Any]:
        """Decorate ``async def`` function to be used as CLI command."""
        if base is None:
            from faust.cli import base as cli_base
            _base = cli_base.AppCommand
        else:
            _base = base

        def _inner(fun: Callable[..., Awaitable[Any]]) -> Any:
            cmd = _base.from_handler(*options, **kwargs)(fun)
            venusian.attach(cmd, category=SCAN_COMMAND)
            return cmd
        return _inner

    def create_event(self, key: Any, value: Any, headers: Any, message: Any) -> EventT:
        """Create new :class:`faust.Event` object."""
        event = self.conf.Event(self, key, value, headers, message)  # type: ignore
        return cast(EventT, event)

    async def start_client(self) -> None:
        """Start the app in Client-Only mode necessary for RPC requests."""
        self.client_only = True
        await self.maybe_start()
        self.consumer.stop_flow()  # type: ignore
        await self.topics.wait_for_subscriptions()  # type: ignore
        await self.topics.on_client_only_start()  # type: ignore
        self.consumer.resume_flow()  # type: ignore
        self.flow_control.resume()

    async def maybe_start_client(self) -> None:
        """Start the app in Client-Only mode if not started as Server."""
        if not self.started:  # type: ignore
            await self.start_client()

    def trace(self, name: str, trace_enabled: bool = True, **extra_context: Any) -> ContextManager[Any]:
        """Return new trace context to trace operation using OpenTracing."""
        if self.tracer is None or not trace_enabled:
            return nullcontext()
        else:
            return self.tracer.trace(name=name, **extra_context)  # type: ignore

    def traced(self, fun: Callable[..., Any], name: Optional[str] = None, sample_rate: float = 1.0, **context: Any) -> Callable[..., Any]:
        """Decorate function to be traced using the OpenTracing API."""
        assert fun
        operation: str = name or operation_name_from_fun(fun)

        @wraps(fun)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            span = self.trace(operation, sample_rate=sample_rate, **context)
            return call_with_trace(span, fun, None, *args, **kwargs)
        return wrapped

    def _start_span_from_rebalancing(self, name: str) -> ContextManager[Any]:
        rebalancing_span = self._rebalancing_span
        if rebalancing_span is not None and self.tracer is not None:
            category = f'{self.conf.name}-_faust'
            span = self.tracer.get_tracer(category).start_span(operation_name=name, child_of=rebalancing_span)  # type: ignore
            self._span_add_default_tags(span)
            set_current_span(span)
            return span
        else:
            return noop_span()

    async def send(self, channel: Union[str, ChannelT], key: Any = None, value: Any = None, partition: Optional[int] = None, timestamp: Optional[float] = None, headers: Optional[Union[Mapping[str, Any], Iterable[Tuple[str, Any]]]] = None, schema: Optional[Any] = None, key_serializer: Optional[Any] = None, value_serializer: Optional[Any] = None, callback: Optional[MessageSentCallback] = None) -> Any:
        """Send event to channel/topic."""
        if isinstance(channel, str):
            chan: TopicT = self.topic(channel)
        else:
            chan = channel  # type: ignore
        return await chan.send(key=key, value=value, partition=partition, timestamp=timestamp, headers=headers, schema=schema, key_serializer=key_serializer, value_serializer=value_serializer, callback=callback)

    @cached_property
    def in_transaction(self) -> bool:
        """Return :const:`True` if stream is using transactions."""
        return self.in_worker and self.conf.processing_guarantee == ProcessingGuarantee.EXACTLY_ONCE  # type: ignore

    def LiveCheck(self, **kwargs: Any) -> _LiveCheck:
        """Return new LiveCheck instance testing features for this app."""
        from faust.livecheck import LiveCheck
        return LiveCheck.for_app(self, **kwargs)

    @stampede
    async def maybe_start_producer(self) -> Any:
        """Ensure producer is started."""
        if self.in_transaction:
            return self.consumer.transactions  # type: ignore
        else:
            producer = self.producer
            await producer.maybe_start()
            return producer

    async def commit(self, topics: Optional[Any]) -> Any:
        """Commit offset for acked messages in specified topics'."""
        return await self.topics.commit(topics)  # type: ignore

    async def on_stop(self) -> None:
        """Call when application stops."""
        await self._stop_consumer()
        await self.on_before_shutdown.send()  # type: ignore
        await self._producer_flush(self.log)
        await self._maybe_close_http_client()

    async def _producer_flush(self, logger: Any) -> None:
        if self._producer is not None:
            logger.info('Flush producer buffer...')
            await self._producer.flush()

    async def _stop_consumer(self) -> None:
        if self._consumer is not None:
            consumer = self._consumer
            try:
                assignment = consumer.assignment()
            except ConsumerNotStarted:
                pass
            else:
                if assignment:
                    self.tables.on_partitions_revoked(assignment)  # type: ignore
                    consumer.stop_flow()
                    self.flow_control.suspend()
                    consumer.pause_partitions(assignment)
                    self.flow_control.clear()
                    await self._stop_fetcher()
                    await self._consumer_wait_empty(consumer, self.log)

    async def _consumer_wait_empty(self, consumer: Any, logger: Any) -> None:
        if self.conf.stream_wait_empty:  # type: ignore
            logger.info('Wait for streams...')
            await consumer.wait_empty()

    def on_rebalance_start(self) -> None:
        """Call when rebalancing starts."""
        self.rebalancing = True  # type: ignore
        self.rebalancing_count += 1  # type: ignore
        self._rebalancing_sensor_state = self.sensors.on_rebalance_start(self)
        if self.tracer:
            category = f'{self.conf.name}-_faust'
            tracer = self.tracer.get_tracer(category)  # type: ignore
            self._rebalancing_span = tracer.start_span(operation_name='rebalance', tags={'rebalancing_count': self.rebalancing_count})  # type: ignore
        self.tables.on_rebalance_start()  # type: ignore

    def _span_add_default_tags(self, span: Any) -> None:
        span.set_tag('faust_app', self.conf.name)  # type: ignore
        span.set_tag('faust_id', self.conf.id)  # type: ignore

    def on_rebalance_return(self) -> None:
        sensor_state = self._rebalancing_sensor_state
        if not sensor_state:
            self.log.warning('Missing sensor state for rebalance #%s', self.rebalancing_count)
        else:
            self.sensors.on_rebalance_return(self, sensor_state)

    def on_rebalance_end(self) -> None:
        """Call when rebalancing is done."""
        self.rebalancing = False  # type: ignore
        if self._rebalancing_span:
            self._rebalancing_span.finish()
        self._rebalancing_span = None
        sensor_state = self._rebalancing_sensor_state
        try:
            if not sensor_state:
                self.log.warning('Missing sensor state for rebalance #%s', self.rebalancing_count)
            else:
                self.sensors.on_rebalance_end(self, sensor_state)
        finally:
            self._rebalancing_sensor_state = None

    async def _on_partitions_revoked(self, revoked: Any) -> None:
        """Handle revocation of topic partitions."""
        if self.should_stop:
            return self._on_rebalance_when_stopped()
        session_timeout: float = self.conf.broker_session_timeout * 0.95  # type: ignore
        T = traced_from_parent_span()
        with flight_recorder(self.log, timeout=session_timeout) as on_timeout:
            consumer = self.consumer
            try:
                self.log.dev('ON PARTITIONS REVOKED')
                T(self.tables.on_partitions_revoked)(revoked)  # type: ignore
                assignment = consumer.assignment()
                if assignment:
                    on_timeout.info('flow_control.suspend()')
                    T(consumer.stop_flow)()
                    T(self.flow_control.suspend)()
                    on_timeout.info('consumer.pause_partitions')
                    T(consumer.pause_partitions)(assignment)
                    on_timeout.info('flow_control.clear()')
                    T(self.flow_control.clear)()
                    await T(self._consumer_wait_empty)(consumer, on_timeout)
                    await T(self._producer_flush)(on_timeout)
                    if self.in_transaction:
                        await T(consumer.transactions.on_partitions_revoked)(revoked)
                else:
                    self.log.dev('ON P. REVOKED NOT COMMITTING: NO ASSIGNMENT')
                on_timeout.info('+send signal: on_partitions_revoked')
                await T(self.on_partitions_revoked.send)(revoked)  # type: ignore
                on_timeout.info('-send signal: on_partitions_revoked')
            except Exception as exc:
                on_timeout.info('on partitions revoked crashed: %r', exc)
                await self.crash(exc)

    async def _stop_fetcher(self) -> None:
        await self._fetcher.stop()  # type: ignore
        self._fetcher.service_reset()  # type: ignore

    def _on_rebalance_when_stopped(self) -> None:
        self.consumer.close()  # type: ignore

    async def _on_partitions_assigned(self, assigned: Any) -> None:
        """Handle new topic partition assignment."""
        if self.should_stop:
            return self._on_rebalance_when_stopped()
        T = traced_from_parent_span()
        session_timeout: float = self.conf.broker_session_timeout * 0.95  # type: ignore
        self.unassigned = not assigned  # type: ignore
        revoked, newly_assigned = self._update_assignment(assigned)
        await asyncio.sleep(0)
        with flight_recorder(self.log, timeout=session_timeout) as on_timeout:
            consumer = self.consumer
            try:
                on_timeout.info('agents.on_rebalance()')
                await T(self.agents.on_rebalance, revoked=revoked, newly_assigned=newly_assigned)(revoked, newly_assigned)  # type: ignore
                on_timeout.info('topics.wait_for_subscriptions()')
                await T(self.topics.maybe_wait_for_subscriptions)()  # type: ignore
                on_timeout.info('consumer.pause_partitions()')
                T(consumer.pause_partitions)(assigned)
                on_timeout.info('topics.on_partitions_assigned()')
                await T(self.topics.on_partitions_assigned)(assigned)  # type: ignore
                on_timeout.info('transactions.on_rebalance()')
                if self.in_transaction:
                    await T(consumer.transactions.on_rebalance)(assigned, revoked, newly_assigned)
                on_timeout.info('tables.on_rebalance()')
                await asyncio.sleep(0)
                await T(self.tables.on_rebalance)(assigned, revoked, newly_assigned)  # type: ignore
                on_timeout.info('+send signal: on_partitions_assigned')
                await T(self.on_partitions_assigned.send)(assigned)  # type: ignore
                on_timeout.info('-send signal: on_partitions_assigned')
            except Exception as exc:
                on_timeout.info('on partitions assigned crashed: %r', exc)
                await self.crash(exc)

    def _update_assignment(self, assigned: Any) -> Tuple[Set[Any], Set[Any]]:
        if self._assignment is not None:
            revoked = self._assignment - assigned
            newly_assigned = assigned - self._assignment
        else:
            revoked = set()
            newly_assigned = assigned
        self._assignment = assigned
        return (revoked, newly_assigned)

    def _new_producer(self) -> Any:
        return self.transport.create_producer(beacon=self.beacon)

    def _new_consumer(self) -> Any:
        return self.transport.create_consumer(callback=self.topics.on_message, on_partitions_revoked=self._on_partitions_revoked, on_partitions_assigned=self._on_partitions_assigned, beacon=self.beacon)

    def _new_conductor(self) -> Any:
        return self.transport.create_conductor(beacon=None)

    def _new_producer_transport(self) -> Any:
        return transport.by_url(self.conf.broker_producer[0])(self.conf.broker_producer, self, loop=self.loop)

    def _new_cache_backend(self) -> Any:
        return cache_backends.by_url(self.conf.cache)(self, self.conf.cache, loop=self.loop)

    def FlowControlQueue(self, maxsize: Optional[int] = None, *, clear_on_resume: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None) -> ThrowableQueue:
        """Like :class:`asyncio.Queue`, but can be suspended/resumed."""
        return ThrowableQueue(maxsize=maxsize, flow_control=self.flow_control, clear_on_resume=clear_on_resume, loop=loop or self.loop)

    def Worker(self, **kwargs: Any) -> _Worker:
        """Return application worker instance."""
        worker = self.conf.Worker(self, **kwargs)  # type: ignore
        return cast(_Worker, worker)

    def on_webserver_init(self, web: Any) -> None:
        """Call when the Web server is initializing."""
        ...

    def _create_directories(self) -> None:
        self.conf.datadir.mkdir(exist_ok=True)  # type: ignore
        self.conf.appdir.mkdir(exist_ok=True)  # type: ignore
        self.conf.tabledir.mkdir(exist_ok=True)  # type: ignore

    def __repr__(self) -> str:
        if self._conf:
            return APP_REPR_FINALIZED.format(name=type(self).__name__, s=self, c=self.conf, agents=self.agents, id=id(self))
        else:
            return APP_REPR_UNFINALIZED.format(name=type(self).__name__, id=id(self))

    def _configure(self, *, silent: bool = False) -> None:
        self.on_before_configured.send()  # type: ignore
        conf = self._load_settings(silent=silent)
        self.on_configured.send(conf)  # type: ignore
        self._conf, self.configured = (conf, True)  # type: ignore
        self.on_after_configured.send()  # type: ignore

    def _load_settings(self, *, silent: bool = False) -> _Settings:
        changes: Dict[str, Any] = {}
        appid, defaults = self._default_options
        if self._config_source:
            changes = self._load_settings_from_source(self._config_source, silent=silent)
        conf_dict = {**defaults, **changes}
        return self.Settings(appid, **self._prepare_compat_settings(conf_dict))  # type: ignore

    def _prepare_compat_settings(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        COMPAT_OPTIONS: Dict[str, str] = {
            'client_id': 'broker_client_id',
            'commit_interval': 'broker_commit_interval',
            'create_reply_topic': 'reply_create_topic',
            'num_standby_replicas': 'table_standby_replicas',
            'default_partitions': 'topic_partitions',
            'replication_factor': 'topic_replication_factor'
        }
        for old, new in COMPAT_OPTIONS.items():
            val = options.get(new)
            try:
                options[new] = options[old]
            except KeyError:
                pass
            else:
                if val is not None:
                    raise ImproperlyConfigured(f'Cannot use both compat option {old!r} and {new!r}')
                warnings.warn(FutureWarning(W_OPTION_DEPRECATED.format(old=old, new=new)))
        return options

    def _load_settings_from_source(self, source: Any, *, silent: bool = False) -> Mapping[str, Any]:
        if isinstance(source, str):
            try:
                source = smart_import(source, imp=import_from_cwd)
            except (AttributeError, ImportError):
                if not silent:
                    raise
                return {}
        return force_mapping(source)

    @property
    def conf(self) -> _Settings:
        """Application configuration."""
        if not self.finalized and STRICT:  # type: ignore
            raise ImproperlyConfigured('App configuration accessed before app.finalize()')
        if self._conf is None:
            self._configure()
        return cast(_Settings, self._conf)

    @conf.setter
    def conf(self, settings: _Settings) -> None:
        self._conf = settings

    @property
    def producer(self) -> Any:
        """Message producer."""
        if self._producer is None:
            self._producer = self._new_producer()
        return self._producer

    @producer.setter
    def producer(self, producer: Any) -> None:
        self._producer = producer

    @property
    def consumer(self) -> Any:
        """Message consumer."""
        if self._consumer is None:
            self._consumer = self._new_consumer()
        return self._consumer

    @consumer.setter
    def consumer(self, consumer: Any) -> None:
        self._consumer = consumer

    @property
    def transport(self) -> Any:
        """Consumer message transport."""
        if self._transport is None:
            self._transport = self._new_transport()
        return self._transport

    @transport.setter
    def transport(self, transport: Any) -> None:
        self._transport = transport

    @property
    def producer_transport(self) -> Any:
        """Producer message transport."""
        if self._producer_transport is None:
            self._producer_transport = self._new_producer_transport()
        return self._producer_transport

    @producer_transport.setter
    def producer_transport(self, transport: Any) -> None:
        self._producer_transport = transport

    @property
    def cache(self) -> Any:
        """Cache backend."""
        if self._cache is None:
            self._cache = self._new_cache_backend()
        return self._cache

    @cache.setter
    def cache(self, cache: Any) -> None:
        self._cache = cache

    @cached_property
    def tables(self) -> TableManagerT:
        """Map of available tables, and the table manager service."""
        manager = self.conf.TableManager(app=self, loop=self.loop, beacon=self.beacon)  # type: ignore
        return cast(TableManagerT, manager)

    @cached_property
    def topics(self) -> Any:
        """Topic Conductor."""
        return self._new_conductor()

    @property
    def monitor(self) -> Monitor:
        """Monitor keeps stats about what's going on inside the worker."""
        if self._monitor is None:
            self._monitor = cast(Monitor, self.conf.Monitor(loop=self.loop, beacon=self.beacon))  # type: ignore
        return self._monitor

    @monitor.setter
    def monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor

    @cached_property
    def _fetcher(self) -> _Fetcher:
        """Fetcher helps Kafka Consumer retrieve records in topics."""
        return cast(_Fetcher, self.transport.Fetcher(self, loop=self.loop, beacon=self.consumer.beacon))  # type: ignore

    @cached_property
    def _reply_consumer(self) -> ReplyConsumer:
        """Kafka Consumer that consumes agent replies."""
        return ReplyConsumer(self, loop=self.loop, beacon=self.beacon)

    @cached_property
    def flow_control(self) -> FlowControlEvent:
        """Flow control of streams."""
        return FlowControlEvent(loop=self.loop)

    @property
    def http_client(self) -> HttpClientT:
        """HTTP client Session."""
        if self._http_client is None:
            client = self.conf.HttpClient()  # type: ignore
            self._http_client = cast(HttpClientT, client)
        return self._http_client

    @http_client.setter
    def http_client(self, client: HttpClientT) -> None:
        self._http_client = client

    @cached_property
    def assignor(self) -> PartitionAssignorT:
        """Partition Assignor."""
        assignor = self.conf.PartitionAssignor(self, replicas=self.conf.table_standby_replicas)  # type: ignore
        return cast(PartitionAssignorT, assignor)

    @cached_property
    def _leader_assignor(self) -> LeaderAssignorT:
        """Leader assignor."""
        assignor = self.conf.LeaderAssignor(self, loop=self.loop, beacon=self.beacon)  # type: ignore
        return cast(LeaderAssignorT, assignor)

    @cached_property
    def router(self) -> RouterT:
        """Find the node partitioned data belongs to."""
        router = self.conf.Router(self)  # type: ignore
        return cast(RouterT, router)

    @cached_property
    def web(self) -> Web:
        """Web driver."""
        return self._new_web()

    def _new_web(self) -> Any:
        return web_drivers.by_url(self.conf.web)(self)  # type: ignore

    @cached_property
    def serializers(self) -> RegistryT:
        """Return serializer registry."""
        self.finalize()
        serializers = self.conf.Serializers(key_serializer=self.conf.key_serializer, value_serializer=self.conf.value_serializer)  # type: ignore
        return cast(RegistryT, serializers)

    @property
    def label(self) -> str:
        """Return human readable description of application."""
        return f'{self.shortlabel}: {self.conf.id}@{self.conf.broker}'  # type: ignore

    @property
    def shortlabel(self) -> str:
        """Return short description of application."""
        return type(self).__name__
