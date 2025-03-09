import logging
import os
import socket
import ssl
import typing
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)
from uuid import uuid4

from mode import SupervisorStrategyT
from mode.utils.imports import SymbolArg, symbol_by_name
from mode.utils.logging import Severity
from mode.utils.times import Seconds, want_seconds
from yarl import URL

from faust.types._env import DATADIR, WEB_BIND, WEB_PORT, WEB_TRANSPORT
from faust.types.agents import AgentT
from faust.types.assignor import LeaderAssignorT, PartitionAssignorT
from faust.types.auth import CredentialsArg, CredentialsT
from faust.types.codecs import CodecArg
from faust.types.events import EventT
from faust.types.enums import ProcessingGuarantee
from faust.types.router import RouterT
from faust.types.sensors import SensorT
from faust.types.serializers import RegistryT, SchemaT
from faust.types.streams import StreamT
from faust.types.transports import PartitionerT, SchedulingStrategyT
from faust.types.tables import GlobalTableT, TableManagerT, TableT
from faust.types.topics import TopicT
from faust.types.web import HttpClientT, ResourceOptions

from . import base
from . import params
from . import sections
from .params import BrokerArg, URLArg

if typing.TYPE_CHECKING:
    from faust.types.worker import Worker as _WorkerT
else:
    class _WorkerT: ...      # noqa

# XXX mypy borks if we do `from faust import __version__`
faust_version: str = symbol_by_name('faust:__version__')

AutodiscoverArg = Union[
    bool,
    Iterable[str],
    Callable[[], Iterable[str]],
]

class Settings(base.SettingsRegistry):
    NODE_HOSTNAME: ClassVar[str] = socket.gethostname()
    DEFAULT_BROKER_URL: ClassVar[str] = 'kafka://localhost:9092'

    _id: str
    _name: str
    _version: int

    #: Environment.
    #: Defaults to :data:`os.environ`.
    env: Mapping[str, str]

    def __init__(
            self,
            id: str, *,
            # Common settings:
            autodiscover: AutodiscoverArg = None,
            datadir: typing.Union[str, Path] = None,
            tabledir: typing.Union[str, Path] = None,
            debug: bool = None,
            env_prefix: str = None,
            id_format: str = None,
            origin: str = None,
            timezone: typing.Union[str, tzinfo] = None,
            version: int = None,
            # Agent settings:
            agent_supervisor: SymbolArg[Type[SupervisorStrategyT]] = None,
            # Broker settings:
            broker: BrokerArg = None,
            broker_consumer: BrokerArg = None,
            broker_producer: BrokerArg = None,
            broker_api_version: str = None,
            broker_check_crcs: bool = None,
            broker_client_id: str = None,
            broker_commit_every: int = None,
            broker_commit_interval: Seconds = None,
            broker_commit_livelock_soft_timeout: Seconds = None,
            broker_credentials: CredentialsArg = None,
            broker_heartbeat_interval: Seconds = None,
            broker_max_poll_interval: Seconds = None,
            broker_max_poll_records: int = None,
            broker_rebalance_timeout: Seconds = None,
            broker_request_timeout: Seconds = None,
            broker_session_timeout: Seconds = None,
            ssl_context: ssl.SSLContext = None,
            # Consumer settings:
            consumer_api_version: str = None,
            consumer_max_fetch_size: int = None,
            consumer_auto_offset_reset: str = None,
            consumer_group_instance_id: str = None,
            # Topic serialization settings:
            key_serializer: CodecArg = None,
            value_serializer: CodecArg = None,
            # Logging settings:
            logging_config: Mapping = None,
            loghandlers: List[logging.Handler] = None,
            # Producer settings:
            producer_acks: int = None,
            producer_api_version: str = None,
            producer_compression_type: str = None,
            producer_linger_ms: int = None,
            producer_max_batch_size: int = None,
            producer_max_request_size: int = None,
            producer_partitioner: SymbolArg[PartitionerT] = None,
            producer_request_timeout: Seconds = None,
            # RPC settings:
            reply_create_topic: bool = None,
            reply_expires: Seconds = None,
            reply_to: str = None,
            reply_to_prefix: str = None,
            # Stream settings:
            processing_guarantee: Union[str, ProcessingGuarantee] = None,
            stream_buffer_maxsize: int = None,
            stream_processing_timeout: Seconds = None,
            stream_publish_on_commit: bool = None,
            stream_recovery_delay: Seconds = None,
            stream_wait_empty: bool = None,
            # Table settings:
            store: URLArg = None,
            table_cleanup_interval: Seconds = None,
            table_key_index_size: int = None,
            table_standby_replicas: int = None,
            # Topic settings:
            topic_allow_declare: bool = None,
            topic_disable_leader: bool = None,
            topic_partitions: int = None,
            topic_replication_factor: int = None,
            # Web server settings:
            cache: URLArg = None,
            canonical_url: URLArg = None,
            web: URLArg = None,
            web_bind: str = None,
            web_cors_options: typing.Mapping[str, ResourceOptions] = None,
            web_enabled: bool = None,
            web_host: str = None,
            web_in_thread: bool = None,
            web_port: int = None,
            web_transport: URLArg = None,
            # Worker settings:
            worker_redirect_stdouts: bool = None,
            worker_redirect_stdouts_level: Severity = None,
            # Extension settings:
            Agent: SymbolArg[Type[AgentT]] = None,
            ConsumerScheduler: SymbolArg[Type[SchedulingStrategyT]] = None,
            Event: SymbolArg[Type[EventT]] = None,
            Schema: SymbolArg[Type[SchemaT]] = None,
            Stream: SymbolArg[Type[StreamT]] = None,
            Table: SymbolArg[Type[TableT]] = None,
            SetTable: SymbolArg[Type[TableT]] = None,
            GlobalTable: SymbolArg[Type[GlobalTableT]] = None,
            SetGlobalTable: SymbolArg[Type[GlobalTableT]] = None,
            TableManager: SymbolArg[Type[TableManagerT]] = None,
            Serializers: SymbolArg[Type[RegistryT]] = None,
            Worker: SymbolArg[Type[_WorkerT]] = None,
            PartitionAssignor: SymbolArg[Type[PartitionAssignorT]] = None,
            LeaderAssignor: SymbolArg[Type[LeaderAssignorT]] = None,
            Router: SymbolArg[Type[RouterT]] = None,
            Topic: SymbolArg[Type[TopicT]] = None,
            HttpClient: SymbolArg[Type[HttpClientT]] = None,
            Monitor: SymbolArg[Type[SensorT]] = None,
            # Deprecated settings:
            stream_ack_cancelled_tasks: bool = None,
            stream_ack_exceptions: bool = None,
            url: URLArg = None,
            **kwargs: Any) -> None:
        ...  # replaced by __init_subclass__ in BaseSettings

    def on_init(self, id: str, **kwargs: Any) -> None:
        # version is required for the id
        # and the id is required as a component in several default
        # setting values so we hack this in here to make sure
        # it's set.
        self._init_env_prefix(**kwargs)
        self._version = kwargs.get('version', 1)
        self.id = id

    def _init_env_prefix(self,
                         env: Mapping[str, str] = None,
                         env_prefix: str = None,
                         **kwargs: Any) -> None:
        if env is None:
            env = os.environ
        self.env = env
        env_name = self.SETTINGS['env_prefix'].env_name
        if env_name is not None:
            prefix_from_env = self.env.get(env_name)
            # prioritize environment
            if prefix_from_env is not None:
                self._env_prefix = prefix_from_env
            else:
                # then use provided argument
                if env_prefix is not None:
                    self._env_prefix = env_prefix

    def getenv(self, env_name: str) -> Any:
        if self._env_prefix:
            env_name = self._env_prefix.rstrip('_') + '_' + env_name
        return self.env.get(env_name)

    def relative_to_appdir(self, path: Path) -> Path:
        """Prepare app directory path.

        If path is absolute the path is returned as-is,
        but if path is relative it will be assumed to belong
        under the app directory.
        """
        return path if path.is_absolute() else self.appdir / path

    def data_directory_for_version(self, version: int) -> Path:
        """Return the directory path for data belonging to specific version."""
        return self.datadir / f'v{version}'

    def find_old_versiondirs(self) -> Iterable[Path]:
        for version in reversed(range(0, self.version)):
            path = self.data_directory_for_version(version)
            if path.is_dir():
                yield path

    @property
    def name(self) -> str:
        # name is a read-only property
        return self._name

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, name: str) -> None:
        self._name = name
        self._id = self._prepare_id(name)  # id is name+version

    def _prepare_id(self, id: str) -> str:
        if self.version > 1:
            return self.id_format.format(id=id, self=self)
        return id

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.id}>'

    @property
    def appdir(self) -> Path:
        return self.data_directory_for_version(self.version)

    # This is an example new setting having type ``str``
    @sections.Common.setting(
        params.Str,
        env_name='ENVIRONMENT_VARIABLE_NAME',
        version_removed='1.0',  # this disables the setting
    )
    def MY_SETTING(self) -> str:
        """My custom setting.

        To contribute new settings you only have to define a new
        setting decorated attribute here.

        Look at the other settings for examples.

        Remember that once you've added the setting you must
        also render the configuration reference:

        .. sourcecode:: console

            $ make configref
        """

    @sections.Common.setting(
        params.Param[AutodiscoverArg, AutodiscoverArg],
        default=False,
    )
    def autodiscover(self) -> AutodiscoverArg:
        """Automatic discovery of agents, tasks, timers, views and commands.

        Faust has an API to add different :mod:`asyncio` services and other
        user extensions, such as "Agents", HTTP web views,
        command-line commands, and timers to your Faust workers.
        These can be defined in any module, so to discover them at startup,
        the worker needs to traverse packages looking for them.

        .. warning::

            The autodiscovery functionality uses the :pypi:`Venusian` library
            to scan wanted packages for ``@app.agent``, ``@app.page``,
            ``@app.command``, ``@app.task`` and ``@app.timer`` decorators,
            but to do so, it's required to traverse the package path and import
            every module in it.

            Importing random modules like this can be dangerous so make sure
            you follow Python programming best practices. Do not start
            threads; perform network I/O; do test monkey-patching for mocks or
            similar, as a side effect of importing a module.  If you encounter
            a case such as this then please find a way to perform your
            action in a lazy manner.

        .. warning::

            If the above warning is something you cannot fix, or if it's out
            of your control, then please set ``autodiscover=False`` and make
            sure the worker imports all modules where your
            decorators are defined.

        The value for this argument can be:

        ``bool``
            If ``App(autodiscover=True)`` is set, the autodiscovery will
            scan the package name described in the ``origin`` attribute.

            The ``origin`` attribute is automatically set when you start
            a worker using the :program:`faust` command line program,
            for example:

            .. sourcecode:: console

                faust -A example.simple worker

            The :option:`-A <faust -A>`, option specifies the app, but you
            can also create a shortcut entry point by calling ``app.main()``:

            .. sourcecode:: python

                if __name__ == '__main__':
                    app.main()

            Then you can start the :program:`faust` program by executing for
            example ``python myscript.py worker --loglevel=INFO``, and it
            will use the correct application.

        ``Sequence[str]``
            The argument can also be a list of packages to scan::

                app = App(..., autodiscover=['proj_orders', 'proj_accounts'])

        ``Callable[[], Sequence[str]]``
            The argument can also be a function returning a list of packages
            to scan::

                def get_all_packages_to_scan():
                    return ['proj_orders', 'proj_accounts']

                app = App(..., autodiscover=get_all_packages_to_scan)

        ``False``
            If everything you need is in a self-contained module, or you
            import the stuff you need manually, just set ``autodiscover``
            to False and don't worry about it :-)

        .. admonition:: Django

            When using :pypi:`Django` and the :envvar:`DJANGO_SETTINGS_MODULE`
            environment variable is set, the Faust app will scan all packages
            found in the ``INSTALLED_APPS`` setting.

            If you're using Django you can use this to scan for
            agents/pages/commands in all packages
            defined in ``INSTALLED_APPS``.

            Faust will automatically detect that you're using Django
            and do the right thing if you do::

                app = App(..., autodiscover=True)

            It will find agents and other decorators in all of the
            reusable Django applications. If you want to manually control
            what packages are traversed, then provide a list::

                app = App(..., autodiscover=['package1', 'package2'])

            or if you want exactly :const:`None` packages to be traversed,
            then provide a False:

                app = App(.., autodiscover=False)

            which is the default, so you can simply omit the argument.

        .. tip::

            For manual control over autodiscovery, you can also call the
            :meth:`@discover` method manually.
        """

    @sections.Common.setting(
        params.Path,
        env_name='APP_DATADIR',
        default=DATADIR,
        related_cli_options={'faust': '--datadir'},
    )
    def datadir(self, path: Path) -> Path:
        """Application data directory.

        The directory in which this instance stores the data used by
        local tables, etc.

        .. seealso::

            - The data directory can also be set using the
              :option:`faust --datadir` option, from the command-line,
              so there is usually no reason to provide a default value
              when creating the app.
        """

    @datadir.on_get_value  # type: ignore
    def _prepare_datadir(self, path: Path) -> Path:
        # allow expanding variables in path
        return Path(str(path).format(conf=self))

    @sections.Common.setting(
        params.Path,
        #: This path will be treated as relative to datadir,
        #: unless the provided poth is absolute.
        default='tables',
        env_name='APP_TABLEDIR',
    )
    def tabledir(self) -> Path:
        """Application table data directory.

        The directory in which this instance stores local table data.
        Usually you will want to configure the :setting:`datadir` setting,
        but if you want to store tables separately you can configure this one.

        If the path provided is relative (it has no leading slash), then the
        path will be considered to be relative to the :setting:`datadir`
        setting.
        """

    @tabledir.on_get_value  # type: ignore
    def _prepare_tabledir(self, path: Path) -> Path:
        return self.relative_to_appdir(path)

    @sections.Common.setting(
        params.Bool,
        env_name='APP_DEBUG',
        default=False,
        related_cli_options={'faust': '--debug'},
    )
    def debug(self) -> bool:
        """Use in development to expose sensor information endpoint.


        .. tip::

            If you want to enable the sensor statistics endpoint in production,
            without enabling the :setting:`debug` setting, you can do so
            by adding the following code:

            .. sourcecode:: python

                app.web.blueprints.add(
                    '/stats/', 'faust.web.apps.stats:blueprint')
        """

    @sections.Common.setting(
        params.Str,
        env_name='APP_ENV_PREFIX',
        version_introduced='1.11',
        default=None,
        ignore_default=True,
    )
    def env_prefix(self) -> str:
        """Environment variable prefix.

        When configuring Faust by environent variables,
        this adds a common prefix to all Faust environment value names.
        """

    @sections.Common.setting(
        params.Str,
        env_name='APP_ID_FORMAT',
        default='{id}-v{self.version}',
    )
    def id_format(self) -> str:
        """Application ID format template.

        The format string used to generate the final :setting:`id` value
        by combining it with the :setting:`version` parameter.
        """

    @sections.Common.setting(
        params.Str,
        default=None,
    )
    def origin(self) -> str:
        """The reverse path used to find the app.

        For example if the app is located in::

            from myproj.app import app

        Then the ``origin`` should be ``"myproj.app"``.

        The :program:`faust worker` program will try to automatically set
        the origin, but if you are having problems with auto generated names
        then you can set origin manually.
        """

    @sections.Common.setting(
        params.Timezone,
        version_introduced='1.4',
        env_name='TIMEZONE',
        default=timezone.utc,
    )
    def timezone(self) -> tzinfo:
        """Project timezone.

        The timezone used for date-related functionality such as cronjobs.
        """

    @sections.Common.setting(
        params.Int,
        env_name='APP_VERSION',
        default=1,
        min_value=1,
    )
    def version(self) -> int:
        """App version.

        Version of the app, that when changed will create a new isolated
        instance of the application. The first version is 1,
        the second version is 2, and so on.

        .. admonition:: Source topics will not be affected by a version change.

            Faust applications will use two kinds of topics: source topics, and
            internally managed topics. The source topics are declared by the
            producer, and we do not have the opportunity to modify any
            configuration settings, like number of partitions for a source
            topic; we may only consume from them. To mark a topic as internal,
            use: ``app.topic(..., internal=True)``.
        """

    @sections.Agent.setting(
        params.Symbol(Type[SupervisorStrategyT]),
        env_name='AGENT_SUPERVISOR',
        default='mode.OneForOneSupervisor',
    )
    def agent_supervisor(self) -> Type[SupervisorStrategyT]:
        """Default agent supervisor type.

        An agent may start multiple instances (actors) when
        the concurrency setting is higher than one (e.g.
        ``@app.agent(concurrency=2)``).

        Multiple instances of the same agent are considered to be in the same
        supervisor group.

        The default supervisor is the :class:`mode.OneForOneSupervisor`:
        if an instance in the group crashes, we restart that instance only.

        These are the supervisors supported:

        + :class:`mode.OneForOneSupervisor`

            If an instance in the group crashes we restart only that instance.

        + :class:`mode.OneForAllSupervisor`

            If an instance in the group crashes we restart the whole group.

        + :class:`mode.CrashingSupervisor`

            If an instance in the group crashes we stop the whole application,
            and exit so that the Operating System supervisor can restart us.

        + :class:`mode.ForfeitOneForOneSupervisor`

            If an instance in the group crashes we give up on that instance
            and never restart it again (until the program is restarted).

        + :class:`mode.ForfeitOneForAllSupervisor`

            If an instance in the group crashes we stop all instances
            in the group and never restarted them again (until the program is
            restarted).
        """

    @sections.Common.setting(
        params.Seconds,
        env_name='BLOCKING_TIMEOUT',
        default=None,
        related_cli_options={'faust': '--blocking-timeout'},
    )
    def blocking_timeout(self) -> Optional[float]:
        """Blocking timeout (in seconds).

        When specified the worker will start a periodic signal based
        timer that only triggers when the loop has been blocked
        for a time exceeding this timeout.

        This is the most safe way to detect blocking, but could have
        adverse effects on libraries that do not automatically
        retry interrupted system calls.

        Python itself does retry all interrupted system calls
        since version 3.5 (see :pep:`475`), but this might not
        be the case with C extensions added to the worker by the user.

        The blocking detector is a background thread
        that periodically wakes up to either arm a timer, or cancel
        an already armed timer. In pseudocode:

        .. sourcecode:: python

            while True:
                # cancel previous alarm and arm new alarm
                signal.signal(signal.SIGALRM, on_alarm)
                signal.setitimer(signal.ITIMER_REAL, blocking_timeout)
                # sleep to wakeup just before the timeout
                await asyncio.sleep(blocking_timeout * 0.96)

            def on_alarm(signum, frame):
                logger.warning('Blocking detected: ...')

        If the sleep does not wake up in time the alarm signal
        will be sent to the process and a traceback will be logged.
        """

    @sections.Common.setting(
        params.BrokerList,
        env_name='BROKER_URL',
    )
    def broker(self) -> List[URL]:
        """Broker URL, or a list of alternative broker URLs.

        Faust needs the URL of a "transport" to send and receive messages.

        Currently, the only supported production transport is ``kafka://``.
        This uses the :pypi:`aiokafka` client under the hood, for consuming and
        producing messages.

        You can specify multiple hosts at the same time by separating them
        using the semi-comma:

        .. sourcecode:: text

            kafka://kafka1.example.com:9092;kafka2.example.com:9092

        Which in actual code looks like this:

        .. sourcecode:: python

            BROKERS = 'kafka://kafka1.example.com:9092;kafka2.example.com:9092'
            app = faust.App(
                'id',
                broker=BROKERS,
            )

        You can also pass a list of URLs:

        .. sourcecode:: python

            app = faust.App(
                'id',
                broker=['kafka://kafka1.example.com:9092',
                        'kafka://kafka2.example.com:9092'],
            )

        .. seealso::

            You can configure the transport used for consuming and producing
            separately, by setting the :setting:`broker_consumer` and
            :setting:`broker_producer` settings.

            This setting is used as the default.

        **Available Transports**

        - ``kafka://``

            Alias to ``aiokafka://``

        - ``aiokafka://``

            The recommended transport using the :pypi:`aiokafka` client.

            Limitations: None

        - ``confluent://``

            Experimental transport using the :pypi:`confluent-kafka` client.

            Limitations: Does not do sticky partition assignment (not
                suitable for tables), and do not create any necessary internal
                topics (you have to create them manually).
        """

    @broker.on_set_default  # type: ignore
    def _prepare_broker(self) -> BrokerArg:
        return self._url or self.DEFAULT_BROKER_URL

    @sections.Broker.setting(
        params.BrokerList,
        version_introduced='1.7',
        env_name='BROKER_CONSUMER_URL',
        default_alias='broker',
    )
    def broker_consumer(self) -> List[URL]:
        """Consumer broker URL.

        You can use this setting to configure the transport used for
        producing and consuming separately.

        If not set the value found in :setting:`broker` will be used.
        """

    @sections.Broker.setting(
        params.BrokerList,
        version_introduced='1.7',
        env_name='BROKER_PRODUCER_URL',
        default_alias='broker',
    )
    def broker_producer(self) -> List[URL]:
        """Producer broker URL.

        You can use this setting to configure the transport used for
        producing and consuming separately.

        If not set the value found in :setting:`broker` will be used.
        """

    @sections.Broker.setting(
        params.Str,
        version_introduced='1.10',
        env_name='BROKER_API_VERSION',
        #: Default broker API version.
        #: Used as default for
        #:     + :setting:`broker_api_version`,
        #:     + :setting:`consumer_api_version`,
        #:     + :setting:`producer_api_version',
        default='auto',
    )
    def broker_api_version(self) -> str:
        """Broker API version,.

        This setting is also the default for :setting:`consumer_api_version`,
        and :setting:`producer_api_version`.

        Negotiate producer protocol version.

        The default value - "auto" means use the latest version supported
        by both client and server.

        Any other version set means you are requesting a specific version of
        the protocol.

        Example Kafka uses:

        **Disable sending headers for all messages produced**

        Kafka headers support was added in Kafka 0.11, so you can specify
        ``broker_api_version="0.10"`` to remove the headers from messages.
        """

    @sections.Broker.setting(
        params.Bool,
        env_name='BROKER_CHECK_CRCS',
        default=True,
    )
    def broker_check_crcs(self) -> bool:
        """Broker CRC check.

        Automatically check the CRC32 of the records consumed.
        """

    @sections.Broker.setting(
        params.Str,
        env_name='BROKER_CLIENT_ID',
        default=f'faust-{faust_version}',
    )
    def broker_client_id(self) -> str:
        """Broker client ID.

        There is rarely any reason to configure this setting.

        The client id is used to identify the software used, and is not usually
        configured by the user.
        """

    @sections.Broker.setting(
        params.UnsignedInt,
        env_name='BROKER_COMMIT_EVERY',
        default=10_000,
    )
    def broker_commit_every(self) -> int:
        """Broker commit message frequency.

        Commit offset every n messages.

        See also :setting:`broker_commit_interval`, which is how frequently
        we commit on a timer when there are few messages being received.
        """

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_INTERVAL',
        default=2.8,
    )
    def broker_commit_interval(self) -> float:
        """Broker commit time frequency.

        How often we commit messages that have been
        fully processed (:term:`acked`).
        """

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT',
        default=want_seconds(timedelta(minutes=5)),
    )
    def broker_commit_livelock_soft_timeout(self) -> float:
        """Commit livelock timeout.

        How long time it takes before we warn that the Kafka commit offset has
        not advanced (only when processing messages).
        """

    @sections.Common.setting(
        params.Credentials,
        version_introduced='1.5',
        env_name='BROKER_CREDENTIALS',
        default=None,
    )
    def broker_credentials(self) -> CredentialsT:
        """Broker authentication mechanism.

        Specify the authentication mechanism to use when connecting to
        the broker.

        The default is to not use any authentication.

        SASL Authentication
            You can enable SASL authentication via plain text:

            .. sourcecode:: python

                app = faust.App(
                    broker_credentials=faust.SASLCredentials(
                        username='x',
                        password='y',
                    ))

            .. warning::

                Do not use literal strings when specifying passwords in
                production, as they can remain visible in stack traces.

                Instead the best practice is to get the password from
                a configuration file, or from the environment:

                .. sourcecode:: python

                    BROKER_USERNAME = os.environ.get('BROKER_USERNAME')
                    BROKER_PASSWORD = os.environ.get('BROKER_PASSWORD')

                    app = faust.App(
                        broker_credentials=faust.SASLCredentials(
                            username=BROKER_USERNAME,
                            password=BROKER_PASSWORD,
                        ))

        GSSAPI Authentication
            GSSAPI authentication over plain text:

            .. sourcecode:: python

                app = faust.App(
                    broker_credentials=faust.GSSAPICredentials(
                        kerberos_service_name='faust',
                        kerberos_domain_name='example.com',
                    ),
                )

            GSSAPI authentication over SSL:

            .. sourcecode:: python

                import ssl
                ssl_context = ssl.create_default_context(
                    purpose=ssl.Purpose.SERVER_AUTH, cafile='ca.pem')
                ssl_context.load_cert_chain(
                    'client.cert', keyfile='client.key')

                app = faust.App(
                    broker_credentials=faust.GSSAPICredentials(
                        kerberos_service_name='faust',
                        kerberos_domain_name='example.com',
                        ssl_context=ssl_context,
                    ),
                )

        SSL Authentication
            Provide an SSL context for the Kafka broker connections.

            This allows Faust to use a secure SSL/TLS connection for the
            Kafka connections and enabling certificate-based authentication.

            .. sourcecode:: python

                import ssl

                ssl_context = ssl.create_default_context(
                    purpose=ssl.Purpose.SERVER_AUTH, cafile='ca.pem')
                ssl_context.load_cert_chain(
                    'client.cert', keyfile='client.key')
                app = faust.App(..., broker_credentials=ssl_context)
        """

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.0.11',
        env_name='BROKER_HEARTBEAT_INTERVAL',
        default=3.0,
    )
    def broker_heartbeat_interval(self) -> float:
        """Broker heartbeat interval.

        How often we send heartbeats to the broker, and also how often
        we expect to receive heartbeats from the broker.

        If any of these time out, you should increase this setting.
        """

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.7',
        env_name='BROKER_MAX_POLL_INTERVAL',
        default=1000.0,
    )
    def broker_max_poll_interval(self) -> float:
        """Broker max poll interval.

        The maximum allowed time (in seconds) between calls to consume messages
        If this interval is exceeded the consumer
        is considered failed and the group will rebalance in order to reassign
        the partitions to another consumer group member. If API methods block
        waiting for messages, that time does not count against this timeout.

        See `KIP-62`_ for technical details.

        .. _`KIP-62`:
            https://cwiki.apache.org/confluence/display/KAFKA/KIP-62%3A+Allow+consumer+to+send+heartbeats+from+a+background+thread
        """

    @sections.Broker.setting(
        params.UnsignedInt,
        version_introduced='1.4',
        env_name='BROKER_MAX_POLL_RECORDS',
        default=None,
        allow_none=True,
    )
    def broker_max_poll_records(self) -> Optional[int]:
        """Broker max poll records.

        The maximum number of records returned in a single call to ``poll()``.
        If you find that your application needs more time to process
        messages you may want to adjust :setting:`broker_max_poll_records`
        to tune the number of records that must be handled on every
        loop iteration.
        """

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.10',
        env_name='BROKER_REBALANCE_TIMEOUT',
        default=60.0,
    )
    def broker_rebalance_timeout(self) -> float:
        """Broker rebalance timeout.

        How long to wait for a node to finish rebalancing before the broker
        will consider it dysfunctional and remove it from the cluster.

        Increase this if you experience the cluster being in a state of
        constantly rebalancing, but make sure you also increase the
        :setting:`broker_heartbeat_interval` at the same time.

        .. note::

            The session timeout must not be greater than the
            :setting:`broker_request_timeout`.
        """

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.4',
        env_name='BROKER_REQUEST_TIMEOUT',
        default=90.0,
    )
    def broker_request_timeout(self) -> float:
        """Kafka client request timeout.

        .. note::

            The request timeout must not be less than the
            :setting:`broker_session_timeout`.
        """

    @sections.Broker.setting(
        params.Seconds,
        version_int