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
    client_only: bool = False

    #: Set this to True if app should run without consumer/tables.
    producer_only: bool = False

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
    _rebalancing_sensor_state: Optional[Dict[str, Any]] = None

    def __init__(self,
                 id: str,
                 *,
                 monitor: Optional[Monitor] = None,
                 config_source: Any = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 beacon: Optional[NodeT] = None,
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
            self.on_partitions