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
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
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
from faust.agents import (
    AgentFun,
    AgentManager,
    AgentT,
    ReplyConsumer,
    SinkT,
)
from faust.channels import Channel, ChannelT
from faust.exceptions import ConsumerNotStarted, ImproperlyConfigured, SameNode
from faust.fixups import FixupT, fixups
from faust.sensors import Monitor, SensorDelegate
from faust.utils import cron, venusian
from faust.utils.tracing import (
    call_with_trace,
    noop_span,
    operation_name_from_fun,
    set_current_span,
    traced_from_parent_span,
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
    ConductorT,
    ConsumerT,
    ProducerT,
    TPorTopicSet,
    TransportT,
)
from faust.types.tuples import (
    Message,
    MessageSentCallback,
    RecordMetadata,
    TP,
)
from faust.types.web import (
    CacheBackendT,
    HttpClientT,
    PageArg,
    Request,
    ResourceOptions,
    Response,
    ViewDecorator,
    ViewHandlerFun,
    Web,
)
from faust.types.windows import WindowT

from ._attached import Attachments

if typing.TYPE_CHECKING:  # pragma: no cover
    from faust.cli.base import AppCommand as _AppCommand
    from faust.livecheck import LiveCheck as _LiveCheck
    from faust.transport.consumer import Fetcher as _Fetcher
    from faust.worker import Worker as _Worker
else:
    class _AppCommand: ...  # noqa
    class _LiveCheck: ...   # noqa
    class _Fetcher: ...     # noqa
    class _Worker: ...      # noqa

__all__ = ['App', 'BootStrategy']

logger = get_logger(__name__)

_T = TypeVar('_T')

#: Format string for ``repr(app)``.
APP_REPR_FINALIZED = '''
<{name}({c.id}): {c.broker} {s.state} agents({agents}) {id:#x}>
'''.strip()

APP_REPR_UNFINALIZED = '''
<{name}: <non-finalized> {id:#x}>
'''.strip()

# Venusian (pypi): This is used for "autodiscovery" of user code,
# CLI commands, and much more.
# Named after same concept from Django: the Django Admin autodiscover function
# that finds custom admin configuration in ``{app}/admin.py`` modules.

SCAN_AGENT = 'faust.agent'
SCAN_COMMAND = 'faust.command'
SCAN_PAGE = 'faust.page'
SCAN_SERVICE = 'faust.service'
SCAN_TASK = 'faust.task'

#: Default decorator categories for :pypi`venusian` to scan for when
#: autodiscovering things like @app.agent decorators.
SCAN_CATEGORIES: Iterable[str] = [
    SCAN_AGENT,
    SCAN_COMMAND,
    SCAN_PAGE,
    SCAN_SERVICE,
    SCAN_TASK,
]

#: List of regular expressions for :pypi:`venusian` that acts as a filter
#: for modules that :pypi:`venusian` should ignore when autodiscovering
#: decorators.
SCAN_IGNORE: Iterable[Any] = [
    re.compile('test_.*').search,
    '.__main__',
]

E_NEED_ORIGIN = '''
`origin` argument to faust.App is mandatory when autodiscovery enabled.

This parameter sets the canonical path to the project package,
and describes how a user, or program can find it on the command-line when using
the `faust -A project` option.  It's also used as the default package
to scan when autodiscovery is enabled.

If your app is defined in a module: ``project/app.py``, then the
origin will be "project":

    # file: project/app.py
    import faust

    app = faust.App(
        id='myid',
        origin='project',
    )
'''

W_OPTION_DEPRECATED = '''\
Argument {old!r} is deprecated and scheduled for removal in Faust 1.0.

Please use {new!r} instead.
'''

W_DEPRECATED_SHARD_PARAM = '''\
The second argument to `@table_route` is deprecated,
please use the `query_param` keyword argument instead.
'''

# @app.task decorator may be called in several ways:
#
# 1) Without parens:
#    @app.task
#    async def foo():
#
# 2) With parens:
#   @app.task()
#
# 3) With parens and arguments
#   @app.task(on_leader=True)
#
# This means @app.task attempts to do the right thing depending
# on how it's used. All the frameworks do this, but we have to also type it.
TaskDecoratorRet = Union[
    Callable[[TaskArg], TaskArg],
    TaskArg,
]


class BootStrategy(BootStrategyT):
    """App startup strategy.

    The startup strategy defines the graph of services
    to start when the Faust worker for an app starts.
    """

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

    def __init__(self, app: AppT, *,
                 enable_web: bool = None,
                 enable_kafka: bool = None,
                 enable_kafka_producer: bool = None,
                 enable_kafka_consumer: bool = None,
                 enable_sensors: bool = None) -> None:
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
            # Sensors (Sensor): always start first and stop last.
            self.sensors(),
            # Producer: always stop after Consumer.
            self.kafka_producer(),
            # Web
            self.web_server(),
            # Consumer: always stop after Conductor
            self.kafka_consumer(),
            # AgentManager starts agents (app.agents)
            self.agents(),
            # Conductor (transport.Conductor))
            self.kafka_conductor(),
            # Table Manager (app.TableManager)
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

    def sensors(self) -> Iterable[ServiceT]:
        """Return list of services required to start sensors."""
        if self.enable_sensors:
            return self.app.sensors
        return []

    def kafka_producer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka producer."""
        if self._should_enable_kafka_producer():
            return [self.app.producer]
        return []

    def _should_enable_kafka_producer(self) -> bool:
        if self.enable_kafka_producer is None:
            return self.enable_kafka
        return self.enable_kafka_producer

    def kafka_consumer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka consumer."""
        if self._should_enable_kafka_consumer():
            app = cast(App, self.app)
            return [
                self.app.consumer,
                # Leader Assignor (assignor.LeaderAssignor)
                app._leader_assignor,
                # Reply Consumer (ReplyConsumer)
                app._reply_consumer,
            ]
        return []

    def _should_enable_kafka_consumer(self) -> bool:
        if self.enable_kafka_consumer is None:
            return self.enable_kafka
        return self.enable_kafka_consumer

    def kafka_client_consumer(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka client consumer."""
        app = cast(App, self.app)
        return [
            app.consumer,
            app._reply_consumer,
        ]

    def agents(self) -> Iterable[ServiceT]:
        """Return list of services required to start agents."""
        return [self.app.agents]

    def kafka_conductor(self) -> Iterable[ServiceT]:
        """Return list of services required to start Kafka conductor."""
        if self._should_enable_kafka_consumer():
            return [self.app.topics]
        return []

    def web_server(self) -> Iterable[ServiceT]:
        """Return list of web-server services."""
        if self._should_enable_web():
            return list(self.web_components()) + [self.app.web]
        return []

    def _should_enable_web(self) -> bool:
        if self.enable_web is None:
            return self.app.conf.web_enabled
        return self.enable_web

    def web_components(self) -> Iterable[ServiceT]:
        """Return list of web-related services (excluding web server)."""
        return [self.app.cache]

    def tables(self) -> Iterable[ServiceT]:
        """Return list of table-related services."""
        if self._should_enable_kafka_consumer():
            return [self.app.tables]
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

    BootStrategy = BootStrategy
    Settings = _Settings

    #: Set this to True if app should only start the services required to
    #: operate as an RPC client (producer and simple reply consumer).
    client_only = False

    #: Set this to True if app should run without consumer/tables.
    producer_only = False

    #: Source of configuration: ``app.conf`` (when configured)
    _conf: Optional[_Settings] = None

    #: Original configuration source object.
    _config_source: Any = None

    # Default consumer instance.
    _consumer: Optional[ConsumerT] = None

    # Default producer instance.
    _producer: Optional[ProducerT] = None

    # Consumer transport is created on demand:
    # use `.transport` property.
    _transport: Optional[TransportT] = None

    # Producer transport is created on demand:
    # use `.producer_transport` property.
    _producer_transport: Optional[TransportT] = None

    # Cache is created on demand: use `.cache` property.
    _cache: Optional[CacheBackendT] = None

    # Monitor is created on demand: use `.monitor` property.
    _monitor: Optional[Monitor] = None

    # @app.task decorator adds asyncio tasks to be started
    # with the app here.
    _app_tasks: MutableSequence[Callable[[], Awaitable]]

    _http_client: Optional[HttpClientT] = None

    _extra_services: List[Type[ServiceT]]
    _extra_service_instances: Optional[List[ServiceT]] = None

    # See faust/app/_attached.py
    _attachments: Attachments

    #: Current assignment
    _assignment: Optional[Set[TP]] = None

    #: Optional tracing support.
    tracer: Optional[TracerT] = None
    _rebalancing_span: Optional[opentracing.Span] = None
    _rebalancing_sensor_state: Optional[Dict] = None

    def __init__(self,
                 id: str,
                 *,
                 monitor: Monitor = None,
                 config_source: Any = None,
                 loop: asyncio.AbstractEventLoop = None,
                 beacon: NodeT = None,
                 **options: Any) -> None:
        # This is passed to the configuration in self.conf
        self._default_options = (id, options)

        # The agent manager manages all agents.
        self.agents = AgentManager(self)

        # Sensors monitor Faust using a standard sensor API.
        self.sensors = SensorDelegate(self)

        # this is a local hack we use until we have proper
        # transactions support in the Python Kafka Client
        # and can have "exactly once" semantics.
        self._attachments = Attachments(self)

        # "The Monitor" is a special sensor that provides detailed stats
        # for the web server.
        self._monitor = monitor

        # Any additional asyncio.Task's specified using @app.task decorator.
        self._app_tasks = []

        # Called as soon as the a worker is fully operational.
        self.on_startup_finished: Optional[Callable] = None

        # Any additional services added using the @app.service decorator.
        self._extra_services = []

        # The configuration source object/module passed to ``config_by_object``
        # for introspectio purposes.
        self._config_source = config_source

        # create default sender for signals such as self.on_configured
        self._init_signals()

        # initialize fixups (automatically applied extensions,
        # such as Django integration).
        self.fixups = self._init_fixups()

        self.boot_strategy = self.BootStrategy(self)

        Service.__init__(self, loop=loop, beacon=beacon)

    def _init_signals(self) -> None:
        # Signals in Faust are the same as in Django, but asynchronous by
        # default (:class:`mode.SyncSignal` is the normal ``def`` version)).
        #
        # Signals in Faust are usually local to the app instance::
        #
        #  @app.on_before_configured.connect  # <-- only sent by this app
        #  def on_before_configured(self):
        #    ...
        #
        # In Django signals are usually global, and an easter-egg
        # provides this in Faust also::
        #
        #    V---- NOTE upper case A in App
        #   @App.on_before_configured.connect  # <-- sent by ALL apps
        #   def on_before_configured(app):
        #
        # Note: Signals are local-only, and cannot do anything to other
        # processes or machines.
        self.on_before_configured = (
            self.on_before_configured.with_default_sender(self))
        self.on_configured = self.on_configured.with_default_sender(self)
        self.on_after_configured = (
            self.on_after_configured.with_default_sender(self))
        self.on_partitions_assigned = (
            self.on_partitions_assigned.with_default_sender(self))
        self.on_partitions_revoked = (
            self.on_partitions_revoked.with_default_sender(self))
        self.on_worker_init = self.on_worker_init.with_default_sender(self)
        self.on_rebalance_complete = (
            self.on_rebalance_complete.with_default_sender(self))
        self.on_before_shutdown = (
            self.on_before_shutdown.with_default_sender(self))
        self.on_produce_message = (
            self.on_produce_message.with_default_sender(self))

    def _init_fixups(self) -> MutableSequence[FixupT]:
        # Returns list of "fixups"
        # Fixups are small additional patches we apply when certain
        # platforms or frameworks are present.
        #
        # One example is the Django fixup, responsible for Django integration
        # whenever the DJANGO_SETTINGS_MODULE environment variable is
        # set. See faust/fixups/django.py, it's not complicated - using
        # setuptools entry points you can very easily create extensions that
        # are automatically enabled by installing a PyPI package with
        # `pip install myname`.
        return list(fixups(self))

    def on_init_dependencies(self) -> Iterable[ServiceT]:
        """Return list of additional service dependencies.

        The services returned will be started with the
        app when the app starts.
        """
        # Add the main Monitor sensor.
        # The beacon is also reattached in case the monitor
        # was created by the user.
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

        # This makes it so that the topic conductor is a child
        # of consumer in the (pretty) dependency graph.
        self.topics.beacon.reattach(self.consumer.beacon)

        if self.conf.debug:
            logger.warning(
                '!!! DEBUG is enabled -- disable for production environments')

    async def on_started(self) -> None:
        """Call when app is fully started."""
        # Wait for table recovery to complete (returns True if app stopped)
        if not await self._wait_for_table_recovery_completed():
            # Add all asyncio.Tasks, like timers, etc.
            await self.on_started_init_extra_tasks()

            # Start user-provided services.
            await self.on_started_init_extra_services()

            # Call the app-is-fully-started callback used by Worker
            # to print the "ready" message that signals to the user that
            # the worker is ready to start processing.
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
            # instantiate the services added using the @app.service decorator.
            self._extra_service_instances = [
                await self.on_init_extra_service(service)
                for service in self._extra_services
            ]

    async def on_init_extra_service(
            self, service: Union[ServiceT, Type[ServiceT]]) -> ServiceT:
        """Call when adding user services to this app."""
        s: ServiceT = self._prepare_subservice(service)
        # start the service now, or when the app is started.
        await self.add_runtime_dependency(s)
        return s

    def _prepare_subservice(
            self, service: Union[ServiceT, Type[ServiceT]]) -> ServiceT:
        if inspect.isclass(service):
            return cast(Type[ServiceT], service)(
                loop=self.loop,
                beacon=self.beacon,
            )
        else:
            return cast(ServiceT, service)

    def config_from_object(self,
                           obj: Any,
                           *,
                           silent: bool = False,
                           force: bool = False) -> None:
        """Read configuration from object.

        Object is either an actual object or the name of a module to import.

        Examples:
            >>> app.config_from_object('myproj.faustconfig')

            >>> from myproj import faustconfig
            >>> app.config_from_object(faustconfig)

        Arguments:
            silent (bool): If true then import errors will be ignored.
            force (bool): Force reading configuration immediately.
                By default the configuration will be read only when required.
        """
        self._config_source = obj
        if self.finalized or self.configured:
            self.Settings._warn_already_configured()
        if force or self.configured:
            self._conf = None
            self._configure(silent=silent)

    def finalize(self) -> None:
        """Finalize app configuration."""
        # Finalization signals that the application have been configured
        # and is ready to use.

        # If you access configuration before an explicit call to
        # ``app.finalize()`` you will get an error.
        # The ``app.main`` entry point and the ``faust -A app`` command
        # both will automatically finalize the app for you.
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
        # This init is called by the `faust worker` command.
        for fixup in self.fixups:
            fixup.on_worker_init()

    def worker_init_post_autodiscover(self) -> None:
        """Init worker after autodiscover."""
        self.web.init_server()
        self.on_worker_init.send()

    def discover(self,
                 *extra_modules: str,
                 categories: Iterable[str] = None,
                 ignore: Iterable[Any] = SCAN_IGNORE) -> None:
        """Discover decorators in packages."""
        # based on autodiscovery in Django,
        # but finds @app.agent decorators, and so on.
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
                    module,
                    ignore=ignore,
                    categories=tuple(categories),
                    onerror=self._on_autodiscovery_error,
                )

    def _on_autodiscovery_error(self, name: str) -> None:
        logger.warning('Autodiscovery importing module %r raised error: %r',
                       name, sys.exc_info()[1], exc_info=True)

    def _discovery_modules(self) -> List[str]:
        modules: List[str] = []
        autodiscover = self.conf.autodiscover
        if autodiscover:
            if isinstance(autodiscover, bool):
                if self.conf.origin is None:
                    raise ImproperlyConfigured(E_NEED_ORIGIN)
            elif callable(autodiscover):
                modules.extend(
                    cast(Callable[[], Iterator[str]], autodiscover)())
            else:
                modules.extend(autodiscover)
            if self.conf.origin:
                modules.append(self.conf.origin)
        return modules

    def main(self) -> NoReturn:
        """Execute the :program:`faust` umbrella command using this app."""
        from faust.cli.faust import cli
        self.finalize()
        self.worker_init()
        if self.conf.autodiscover:
            self.discover()
        self.worker_init_post_autodiscover()
        cli(app=self)
        raise SystemExit(3451)  # for mypy: NoReturn

    def topic(self,
              *topics: str,
              pattern: Union[str, Pattern] = None,
              schema: SchemaT = None,
              key_type: ModelArg = None,
              value_type: ModelArg = None,
              key_serializer: CodecArg = None,
              value_serializer: CodecArg = None,
              partitions: int = None,
              retention: Seconds = None,
              compacting: bool = None,
              deleting: bool = None,
              replicas: int = None,
              acks: bool = True,
              internal: bool = False,
              config: Mapping[str, Any] = None,
              maxsize: int = None,
              allow_empty: bool = False,
              has_prefix: bool = False,
              loop: asyncio.AbstractEventLoop = None) -> TopicT:
        """Create topic description.

        Topics are named channels (for example a Kafka topic),
        that exist on a server.  To make an ephemeral local communication
        channel use: :meth:`channel`.

        See Also:
            :class:`faust.topics.Topic`
        """
        return cast(TopicT, self.conf.Topic(  # type: ignore
            self,
            topics=topics,
            pattern=pattern,
            schema=schema,
            key_type=key_type,
            value_type=value_type,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            partitions=partitions,
            retention=retention,
            compacting=compacting,
            deleting=deleting,
            replicas=replicas,
            acks=acks,
            internal=internal,
            config=config,
            allow_empty=allow_empty,
            has_prefix=has_prefix,
            loop=loop,
        ))

    def channel(self,
                *,
                schema: SchemaT = None,
                key_type: ModelArg = None,
                value_type: ModelArg = None,
                maxsize: int = None,
                loop: asyncio.AbstractEventLoop = None) -> ChannelT:
        """Create new channel.

        By default this will create an in-memory channel
        used for intra-process communication, but in practice
        channels can be backed by any transport (network or even means
        of inter-process communication).

        See Also:
            :class:`faust.channels.Channel`.
        """
        return Channel(
            self,
            schema=schema,
            key_type=key_type,
            value_type=value_type,
            maxsize=maxsize,
            loop=loop,
        )

    def agent(self,
              channel: Union[str, ChannelT[_T]] = None,
              *,
              name: str = None,
              concurrency: int = 1,
              supervisor_strategy: Type[SupervisorStrategyT] = None,
              sink: Iterable[SinkT] = None,
              isolated_partitions: bool = False,
              use_reply_headers: bool = True,
              **kwargs: Any) -> Callable[[AgentFun[_T]], AgentT[_T]]:
        """Create Agent from async def function.

        It can be a regular async function::

            @app.agent()
            async def my_agent(stream):
                async for number in stream:
                    print(f'Received: {number!r}')

        Or it can be an async iterator that yields values.
        These values can be used as the reply in an RPC-style call,
        or for sinks: callbacks that forward events to
        other agents/topics/statsd, and so on::

            @app.agent(sink=[log_topic])
            async def my_agent(requests):
                async for number in requests:
                    yield number * 2

        """
        def _inner(fun: AgentFun[_T]) -> AgentT[_T]:
            agent = cast(AgentT, self.conf.Agent(  # type: ignore
                fun,
                name=name,
                app=self,
                channel=channel,
                concurrency=concurrency,
                supervisor_strategy=supervisor_strategy,
                sink=sink,
                isolated_partitions=isolated_partitions,
                on_error=self._on_agent_error,
                use_reply_headers=use_reply_headers,
                help=fun.__doc__,
                **kwargs))
            self.agents[agent.name] = agent
            # This connects the agent to the topic conductor
            # to make the graph more pretty.
            self.topics.beacon.add(agent)
            venusian.attach(agent, category=SCAN_AGENT)
            return agent

        return _inner

    actor = agent  # XXX Compatibility alias: REMOVE FOR 1.0

    async def _on_agent_error(self, agent: AgentT, exc: BaseException) -> None:
        # See agent-errors in docs/userguide/agents.rst
        if self._consumer:
            try:
                await self._consumer.on_task_error(exc)
            except MemoryError:
                raise
            except Exception as exc:
                self.log.exception('Consumer error callback raised: %r', exc)

    @no_type_check
    def task(self,
             fun: TaskArg = None,
             *,
             on_leader: bool = False,
             traced: bool = True) -> TaskDecoratorRet:
        """Define an async def function to be started with the app.

        This is like :meth:`timer` but a one-shot task only
        executed at worker startup (after recovery and the worker is
        fully ready for operation).

        The function may take zero, or one argument.
        If the target function takes an argument, the ``app`` argument
        is passed::

            >>> @app.task
            >>> async def on_startup(app):
            ...    print('STARTING UP: %r' % (app,))

        Nullary functions are also supported::

            >>> @app.task
            >>> async def on_startup():
            ...     print('STARTING UP')
        """
        def _inner(fun: TaskArg) -> TaskArg:
            return self._task(fun, on_leader=on_leader, traced=traced)
        return _inner(fun) if fun is not None else _inner

    def _task(self, fun: TaskArg,
              on_leader: bool = False,
              traced: bool = False,
              ) -> TaskArg:
        app = self

        @wraps(fun)
        async def _wrapped() -> None:
            should_run = app.is_leader() if on_leader else True
            if should_run:
                with self.trace(shortlabel(fun), trace_enabled=traced):
                    # pass app only if decorated function takes an argument
                    if inspect.signature(fun).parameters:
                        task_takes_app = cast(Callable[[AppT], Awaitable], fun)
                        return await task_takes_app(app)
                    else:
                        task = cast(Callable[[], Awaitable], fun)
                        return await task()

        venusian.attach(_wrapped, category=SCAN_TASK)
        self._app_tasks.append(_wrapped)
        return _wrapped

    @no_type_check
    def timer(self, interval: Seconds,
              on_leader: bool = False,
              traced: bool = True,
              name: str = None,
              max_drift_correction: float = 0.1) -> Callable:
        """Define an async def function to be run at periodic intervals.

        Like :meth:`task`, but executes periodically until the worker
        is shut down.

        This decorator takes an async function and adds it to a
        list of timers started with the app.

        Arguments:
            interval (Seconds): How often the timer executes in seconds.

            on_leader (bool) = False: Should the timer only run on the leader?

        Example:
            >>> @app.timer(interval=10.0)
            >>> async def every_10_seconds():
            ...     print('TEN SECONDS JUST PASSED')


            >>> app.timer(interval=5.0, on_leader=True)
            >>> async def every_5_seconds():
            ...     print('FIVE SECONDS JUST PASSED. ALSO, I AM THE LEADER!')
        """
        interval_s = want_seconds(interval)

        def _inner(fun: TaskArg) -> TaskArg:
            timer_name = name or qualname(fun)

            @wraps(fun)
            async def around_timer(*args: Any) -> None:
                async for sleep_time in self.itertimer(
                        interval_s,
                        name=timer_name,
                        max_drift_correction=max_drift_correction):
                    should_run = not on_leader or self.is_leader()
                    if should_run:
                        with self.trace(shortlabel(fun),
                                        trace_enabled=traced):
                            await fun(*args)

            # If you call @app.task without parents the return value is:
            #    Callable[[TaskArg], TaskArg]
            # but we always call @app.task() - with parens, so return value is
            # always TaskArg.
            return cast(TaskArg, self.task