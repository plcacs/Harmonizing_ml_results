import logging
import os
import socket
import ssl
import typing
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, List, Mapping, Optional, Type, Union
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
    class _WorkerT:
        ...


faust_version: str = symbol_by_name('faust:__version__')
AutodiscoverArg = Union[bool, Iterable[str], Callable[[], Iterable[str]]]


class Settings(base.SettingsRegistry):
    NODE_HOSTNAME: str = socket.gethostname()
    DEFAULT_BROKER_URL: str = 'kafka://localhost:9092'

    def __init__(
        self,
        id: str,
        *,
        autodiscover: Optional[AutodiscoverArg] = None,
        datadir: Optional[Union[str, Path]] = None,
        tabledir: Optional[Union[str, Path]] = None,
        debug: Optional[bool] = None,
        env_prefix: Optional[str] = None,
        id_format: Optional[str] = None,
        origin: Optional[str] = None,
        timezone: Optional[tzinfo] = None,
        version: Optional[Any] = None,
        agent_supervisor: Optional[Any] = None,
        broker: Optional[Any] = None,
        broker_consumer: Optional[Any] = None,
        broker_producer: Optional[Any] = None,
        broker_api_version: Optional[str] = None,
        broker_check_crcs: Optional[bool] = None,
        broker_client_id: Optional[str] = None,
        broker_commit_every: Optional[int] = None,
        broker_commit_interval: Optional[Seconds] = None,
        broker_commit_livelock_soft_timeout: Optional[Seconds] = None,
        broker_credentials: Optional[CredentialsT] = None,
        broker_heartbeat_interval: Optional[Seconds] = None,
        broker_max_poll_interval: Optional[Seconds] = None,
        broker_max_poll_records: Optional[int] = None,
        broker_rebalance_timeout: Optional[Seconds] = None,
        broker_request_timeout: Optional[Seconds] = None,
        broker_session_timeout: Optional[Seconds] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        consumer_api_version: Optional[str] = None,
        consumer_max_fetch_size: Optional[int] = None,
        consumer_auto_offset_reset: Optional[str] = None,
        consumer_group_instance_id: Optional[str] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        logging_config: Optional[Mapping[str, Any]] = None,
        loghandlers: Optional[List[Any]] = None,
        producer_acks: Optional[int] = None,
        producer_api_version: Optional[str] = None,
        producer_compression_type: Optional[str] = None,
        producer_linger_ms: Optional[int] = None,
        producer_max_batch_size: Optional[int] = None,
        producer_max_request_size: Optional[int] = None,
        producer_partitioner: Optional[Union[PartitionerT, SymbolArg]] = None,
        producer_request_timeout: Optional[Seconds] = None,
        reply_create_topic: Optional[bool] = None,
        reply_expires: Optional[Seconds] = None,
        reply_to: Optional[str] = None,
        reply_to_prefix: Optional[str] = None,
        processing_guarantee: Optional[ProcessingGuarantee] = None,
        stream_buffer_maxsize: Optional[int] = None,
        stream_processing_timeout: Optional[Seconds] = None,
        stream_publish_on_commit: Optional[bool] = None,
        stream_recovery_delay: Optional[Seconds] = None,
        stream_wait_empty: Optional[bool] = None,
        store: Optional[str] = None,
        table_cleanup_interval: Optional[Seconds] = None,
        table_key_index_size: Optional[int] = None,
        table_standby_replicas: Optional[int] = None,
        topic_allow_declare: Optional[bool] = None,
        topic_disable_leader: Optional[bool] = None,
        topic_partitions: Optional[int] = None,
        topic_replication_factor: Optional[int] = None,
        cache: Optional[str] = None,
        canonical_url: Optional[str] = None,
        web: Optional[Union[str, URL]] = None,
        web_bind: Optional[str] = None,
        web_cors_options: Optional[Mapping[str, ResourceOptions]] = None,
        web_enabled: Optional[bool] = None,
        web_host: Optional[str] = None,
        web_in_thread: Optional[bool] = None,
        web_port: Optional[int] = None,
        web_transport: Optional[str] = None,
        worker_redirect_stdouts: Optional[bool] = None,
        worker_redirect_stdouts_level: Optional[Severity] = None,
        Agent: Optional[Union[Type[AgentT], str]] = None,
        ConsumerScheduler: Optional[Union[Type[SchedulingStrategyT], str]] = None,
        Event: Optional[Union[Type[EventT], str]] = None,
        Schema: Optional[Union[Type[SchemaT], str]] = None,
        Stream: Optional[Union[Type[StreamT], str]] = None,
        Table: Optional[Union[Type[TableT], str]] = None,
        SetTable: Optional[Union[Type[TableT], str]] = None,
        GlobalTable: Optional[Union[Type[GlobalTableT], str]] = None,
        SetGlobalTable: Optional[Union[Type[GlobalTableT], str]] = None,
        TableManager: Optional[Union[Type[TableManagerT], str]] = None,
        Serializers: Optional[Union[Type[RegistryT], str]] = None,
        Worker: Optional[Union[Type[_WorkerT], str]] = None,
        PartitionAssignor: Optional[Union[Type[PartitionAssignorT], str]] = None,
        LeaderAssignor: Optional[Union[Type[LeaderAssignorT], str]] = None,
        Router: Optional[Union[Type[RouterT], str]] = None,
        Topic: Optional[Union[Type[TopicT], str]] = None,
        HttpClient: Optional[Union[Type[HttpClientT], str]] = None,
        Monitor: Optional[Union[Type[SensorT], str]] = None,
        stream_ack_cancelled_tasks: Optional[bool] = None,
        stream_ack_exceptions: Optional[bool] = None,
        url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        ...

    def on_init(self, id: str, **kwargs: Any) -> None:
        self._init_env_prefix(**kwargs)
        self._version = kwargs.get('version', 1)
        self.id = id

    def _init_env_prefix(self, env: Optional[Mapping[str, str]] = None, env_prefix: Optional[str] = None, **kwargs: Any) -> None:
        if env is None:
            env = os.environ  # type: Mapping[str, str]
        self.env: Mapping[str, str] = env
        env_name: Optional[str] = self.SETTINGS['env_prefix'].env_name  # type: ignore
        if env_name is not None:
            prefix_from_env: Optional[str] = self.env.get(env_name)
            if prefix_from_env is not None:
                self._env_prefix = prefix_from_env
            elif env_prefix is not None:
                self._env_prefix = env_prefix

    def getenv(self, env_name: str) -> Optional[str]:
        if hasattr(self, '_env_prefix') and self._env_prefix:
            env_name = self._env_prefix.rstrip('_') + '_' + env_name
        return self.env.get(env_name)

    def relative_to_appdir(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        else:
            return self.appdir / path

    def data_directory_for_version(self, version: int) -> Path:
        return self.datadir / f'v{version}'

    def find_old_versiondirs(self) -> Iterable[Path]:
        for version in reversed(range(0, self.version)):
            path: Path = self.data_directory_for_version(version)
            if path.is_dir():
                yield path

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, name: str) -> None:
        self._name = name
        self._id = self._prepare_id(name)

    def _prepare_id(self, id: str) -> str:
        if self.version > 1:
            return self.id_format.format(id=id, self=self)
        return id

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.id}>'

    @property
    def appdir(self) -> Path:
        return self.data_directory_for_version(self.version)

    @sections.Common.setting(params.Str, env_name='ENVIRONMENT_VARIABLE_NAME', version_removed='1.0')
    def MY_SETTING(self) -> Any:
        """
        My custom setting.
        """
        ...

    @sections.Common.setting(params.Param[AutodiscoverArg, AutodiscoverArg], default=False)
    def autodiscover(self) -> AutodiscoverArg:
        """
        Automatic discovery of agents, tasks, timers, views and commands.
        """
        ...

    @sections.Common.setting(params.Path, env_name='APP_DATADIR', default=DATADIR, related_cli_options={'faust': '--datadir'})
    def datadir(self, path: Union[str, Path]) -> Path:
        """
        Application data directory.
        """
        ...

    @datadir.on_get_value
    def _prepare_datadir(self, path: Union[str, Path]) -> Path:
        return Path(str(path).format(conf=self))

    @sections.Common.setting(params.Path, default='tables', env_name='APP_TABLEDIR')
    def tabledir(self) -> Path:
        """
        Application table data directory.
        """
        ...

    @tabledir.on_get_value
    def _prepare_tabledir(self, path: Union[str, Path]) -> Path:
        return self.relative_to_appdir(Path(path))

    @sections.Common.setting(params.Bool, env_name='APP_DEBUG', default=False, related_cli_options={'faust': '--debug'})
    def debug(self) -> bool:
        """
        Use in development to expose sensor information endpoint.
        """
        ...

    @sections.Common.setting(params.Str, env_name='APP_ENV_PREFIX', version_introduced='1.11', default=None, ignore_default=True)
    def env_prefix(self) -> Optional[str]:
        """
        Environment variable prefix.
        """
        ...

    @sections.Common.setting(params.Str, env_name='APP_ID_FORMAT', default='{id}-v{self.version}')
    def id_format(self) -> str:
        """
        Application ID format template.
        """
        ...

    @sections.Common.setting(params.Str, default=None)
    def origin(self) -> Optional[str]:
        """
        The reverse path used to find the app.
        """
        ...

    @sections.Common.setting(params.Timezone, version_introduced='1.4', env_name='TIMEZONE', default=timezone.utc)
    def timezone(self) -> tzinfo:
        """
        Project timezone.
        """
        ...

    @sections.Common.setting(params.Int, env_name='APP_VERSION', default=1, min_value=1)
    def version(self) -> int:
        """
        App version.
        """
        ...

    @sections.Agent.setting(params.Symbol(Type[SupervisorStrategyT]), env_name='AGENT_SUPERVISOR', default='mode.OneForOneSupervisor')
    def agent_supervisor(self) -> Union[Type[SupervisorStrategyT], str]:
        """
        Default agent supervisor type.
        """
        ...

    @sections.Common.setting(params.Seconds, env_name='BLOCKING_TIMEOUT', default=None, related_cli_options={'faust': '--blocking-timeout'})
    def blocking_timeout(self) -> Optional[Seconds]:
        """
        Blocking timeout (in seconds).
        """
        ...

    @sections.Broker.setting(params.BrokerList, env_name='BROKER_URL')
    def broker(self) -> Any:
        """
        Broker URL, or a list of alternative broker URLs.
        """
        ...

    @broker.on_set_default
    def _prepare_broker(self) -> Any:
        return self._url or self.DEFAULT_BROKER_URL

    @sections.Broker.setting(params.BrokerList, version_introduced='1.7', env_name='BROKER_CONSUMER_URL', default_alias='broker')
    def broker_consumer(self) -> Any:
        """
        Consumer broker URL.
        """
        ...

    @sections.Broker.setting(params.BrokerList, version_introduced='1.7', env_name='BROKER_PRODUCER_URL', default_alias='broker')
    def broker_producer(self) -> Any:
        """
        Producer broker URL.
        """
        ...

    @sections.Broker.setting(params.Str, version_introduced='1.10', env_name='BROKER_API_VERSION', default='auto')
    def broker_api_version(self) -> str:
        """
        Broker API version.
        """
        ...

    @sections.Broker.setting(params.Bool, env_name='BROKER_CHECK_CRCS', default=True)
    def broker_check_crcs(self) -> bool:
        """
        Broker CRC check.
        """
        ...

    @sections.Broker.setting(params.Str, env_name='BROKER_CLIENT_ID', default=f'faust-{faust_version}')
    def broker_client_id(self) -> str:
        """
        Broker client ID.
        """
        ...

    @sections.Broker.setting(params.UnsignedInt, env_name='BROKER_COMMIT_EVERY', default=10000)
    def broker_commit_every(self) -> int:
        """
        Broker commit message frequency.
        """
        ...

    @sections.Broker.setting(params.Seconds, env_name='BROKER_COMMIT_INTERVAL', default=2.8)
    def broker_commit_interval(self) -> Seconds:
        """
        Broker commit time frequency.
        """
        ...

    @sections.Broker.setting(params.Seconds, env_name='BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT', default=want_seconds(timedelta(minutes=5)))
    def broker_commit_livelock_soft_timeout(self) -> Seconds:
        """
        Commit livelock timeout.
        """
        ...

    @sections.Common.setting(params.Credentials, version_introduced='1.5', env_name='BROKER_CREDENTIALS', default=None)
    def broker_credentials(self) -> Optional[CredentialsT]:
        """
        Broker authentication mechanism.
        """
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.0.11', env_name='BROKER_HEARTBEAT_INTERVAL', default=3.0)
    def broker_heartbeat_interval(self) -> Seconds:
        """
        Broker heartbeat interval.
        """
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.7', env_name='BROKER_MAX_POLL_INTERVAL', default=1000.0)
    def broker_max_poll_interval(self) -> Seconds:
        """
        Broker max poll interval.
        """
        ...

    @sections.Broker.setting(params.UnsignedInt, version_introduced='1.4', env_name='BROKER_MAX_POLL_RECORDS', default=None, allow_none=True)
    def broker_max_poll_records(self) -> Optional[int]:
        """
        Broker max poll records.
        """
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.10', env_name='BROKER_REBALANCE_TIMEOUT', default=60.0)
    def broker_rebalance_timeout(self) -> Seconds:
        """
        Broker rebalance timeout.
        """
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.4', env_name='BROKER_REQUEST_TIMEOUT', default=90.0)
    def broker_request_timeout(self) -> Seconds:
        """
        Kafka client request timeout.
        """
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.0.11', env_name='BROKER_SESSION_TIMEOUT', default=60.0)
    def broker_session_timeout(self) -> Seconds:
        """
        Broker session timeout.
        """
        ...

    @sections.Common.setting(params.SSLContext, default=None)
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        SSL configuration.
        """
        ...

    @sections.Consumer.setting(params.Str, version_introduced='1.10', env_name='CONSUMER_API_VERSION', default_alias='broker_api_version')
    def consumer_api_version(self) -> str:
        """
        Consumer API version.
        """
        ...

    @sections.Consumer.setting(params.UnsignedInt, version_introduced='1.4', env_name='CONSUMER_MAX_FETCH_SIZE', default=1024 ** 2)
    def consumer_max_fetch_size(self) -> int:
        """
        Consumer max fetch size.
        """
        ...

    @sections.Consumer.setting(params.Str, version_introduced='1.5', env_name='CONSUMER_AUTO_OFFSET_RESET', default='earliest')
    def consumer_auto_offset_reset(self) -> str:
        """
        Consumer auto offset reset.
        """
        ...

    @sections.Consumer.setting(params.Str, version_introduced='2.1', env_name='CONSUMER_GROUP_INSTANCE_ID', default=None)
    def consumer_group_instance_id(self) -> Optional[str]:
        """
        Consumer group instance id.
        """
        ...

    @sections.Serialization.setting(params.Codec, env_name='APP_KEY_SERIALIZER', default='raw')
    def key_serializer(self) -> Union[str, CodecArg]:
        """
        Default key serializer.
        """
        ...

    @sections.Serialization.setting(params.Codec, env_name='APP_VALUE_SERIALIZER', default='json')
    def value_serializer(self) -> Union[str, CodecArg]:
        """
        Default value serializer.
        """
        ...

    @sections.Common.setting(params.Dict[Any], version_introduced='1.5')
    def logging_config(self) -> Mapping[str, Any]:
        """
        Logging dictionary configuration.
        """
        ...

    @sections.Common.setting(params.LogHandlers)
    def loghandlers(self) -> List[Any]:
        """
        List of custom logging handlers.
        """
        ...

    @sections.Producer.setting(params.Int, env_name='PRODUCER_ACKS', default=-1, number_aliases={'all': -1})
    def producer_acks(self) -> int:
        """
        Producer Acks.
        """
        ...

    @sections.Producer.setting(params.Str, version_introduced='1.5.3', env_name='PRODUCER_API_VERSION', default_alias='broker_api_version')
    def producer_api_version(self) -> str:
        """
        Producer API version.
        """
        ...

    @sections.Producer.setting(params.Str, env_name='PRODUCER_COMPRESSION_TYPE', default=None)
    def producer_compression_type(self) -> Optional[str]:
        """
        Producer compression type.
        """
        ...

    @sections.Producer.setting(params.Seconds, env_name='PRODUCER_LINGER')
    def producer_linger(self) -> Seconds:
        """
        Producer batch linger configuration.
        """
        ...

    @producer_linger.on_set_default
    def _prepare_producer_linger(self) -> float:
        return float(self._producer_linger_ms) / 1000.0

    @sections.Producer.setting(params.UnsignedInt, env_name='PRODUCER_MAX_BATCH_SIZE', default=16384)
    def producer_max_batch_size(self) -> int:
        """
        Producer max batch size.
        """
        ...

    @sections.Producer.setting(params.UnsignedInt, env_name='PRODUCER_MAX_REQUEST_SIZE', default=1000000)
    def producer_max_request_size(self) -> int:
        """
        Producer maximum request size.
        """
        ...

    @sections.Producer.setting(params._Symbol[PartitionerT, Optional[PartitionerT]], version_introduced='1.2', default=None)
    def producer_partitioner(self) -> Optional[Union[PartitionerT, SymbolArg]]:
        """
        Producer partitioning strategy.
        """
        ...

    @sections.Producer.setting(params.Seconds, version_introduced='1.4', env_name='PRODUCER_REQUEST_TIMEOUT', default=1200.0)
    def producer_request_timeout(self) -> Seconds:
        """
        Producer request timeout.
        """
        ...

    @sections.RPC.setting(params.Bool, env_name='APP_REPLY_CREATE_TOPIC', default=False)
    def reply_create_topic(self) -> bool:
        """
        Automatically create reply topics.
        """
        ...

    @sections.RPC.setting(params.Seconds, env_name='APP_REPLY_EXPIRES', default=want_seconds(timedelta(days=1)))
    def reply_expires(self) -> Seconds:
        """
        RPC reply expiry time in seconds.
        """
        ...

    @sections.RPC.setting(params.Str)
    def reply_to(self) -> str:
        """
        Reply to address.
        """
        ...

    @reply_to.on_set_default
    def _prepare_reply_to_default(self) -> str:
        return f'{self.reply_to_prefix}{uuid4()}'

    @sections.RPC.setting(params.Str, env_name='APP_REPLY_TO_PREFIX', default='f-reply-')
    def reply_to_prefix(self) -> str:
        """
        Reply address topic name prefix.
        """
        ...

    @sections.Common.setting(params.Enum(ProcessingGuarantee), version_introduced='1.5', env_name='PROCESSING_GUARANTEE', default=ProcessingGuarantee.AT_LEAST_ONCE)
    def processing_guarantee(self) -> ProcessingGuarantee:
        """
        The processing guarantee that should be used.
        """
        ...

    @sections.Stream.setting(params.UnsignedInt, env_name='STREAM_BUFFER_MAXSIZE', default=4096)
    def stream_buffer_maxsize(self) -> int:
        """
        Stream buffer maximum size.
        """
        ...

    @sections.Stream.setting(params.Seconds, version_introduced='1.10', env_name='STREAM_PROCESSING_TIMEOUT', default=5 * 60.0)
    def stream_processing_timeout(self) -> Seconds:
        """
        Stream processing timeout.
        """
        ...

    @sections.Stream.setting(params.Bool, default=False)
    def stream_publish_on_commit(self) -> bool:
        """
        Stream delay producing until commit time.
        """
        ...

    @sections.Stream.setting(params.Seconds, version_introduced='1.3', version_changed={'1.5.3': 'Disabled by default.'}, env_name='STREAM_RECOVERY_DELAY', default=0.0)
    def stream_recovery_delay(self) -> Seconds:
        """
        Stream recovery delay.
        """
        ...

    @sections.Stream.setting(params.Bool, env_name='STREAM_WAIT_EMPTY', default=True)
    def stream_wait_empty(self) -> bool:
        """
        Stream wait empty.
        """
        ...

    @sections.Common.setting(params.URL, env_name='APP_STORE', default='memory://')
    def store(self) -> str:
        """
        Table storage backend URL.
        """
        ...

    @sections.Table.setting(params.Seconds, env_name='TABLE_CLEANUP_INTERVAL', default=30.0)
    def table_cleanup_interval(self) -> Seconds:
        """
        Table cleanup interval.
        """
        ...

    @sections.Table.setting(params.UnsignedInt, version_introduced='1.7', env_name='TABLE_KEY_INDEX_SIZE', default=1000)
    def table_key_index_size(self) -> int:
        """
        Table key index size.
        """
        ...

    @sections.Table.setting(params.UnsignedInt, env_name='TABLE_STANDBY_REPLICAS', default=1)
    def table_standby_replicas(self) -> int:
        """
        Table standby replicas.
        """
        ...

    @sections.Topic.setting(params.Bool, version_introduced='1.5', env_name='TOPIC_ALLOW_DECLARE', default=True)
    def topic_allow_declare(self) -> bool:
        """
        Allow creating new topics.
        """
        ...

    @sections.Topic.setting(params.Bool, version_introduced='1.7', env_name='TOPIC_DISABLE_LEADER', default=False)
    def topic_disable_leader(self) -> bool:
        """
        Disable leader election topic.
        """
        ...

    @sections.Topic.setting(params.UnsignedInt, env_name='TOPIC_PARTITIONS', default=8)
    def topic_partitions(self) -> int:
        """
        Topic partitions.
        """
        ...

    @sections.Topic.setting(params.UnsignedInt, env_name='TOPIC_REPLICATION_FACTOR', default=1)
    def topic_replication_factor(self) -> int:
        """
        Topic replication factor.
        """
        ...

    @sections.Common.setting(params.URL, version_introduced='1.2', env_name='CACHE_URL', default='memory://')
    def cache(self) -> str:
        """
        Cache backend URL.
        """
        ...

    @sections.WebServer.setting(params.URL, version_introduced='1.2', default='aiohttp://')
    def web(self) -> Union[str, URL]:
        """
        Web server driver to use.
        """
        ...

    @sections.WebServer.setting(params.Str, version_introduced='1.2', env_name='WEB_BIND', default=WEB_BIND, related_cli_options={'faust worker': ['--web-bind']})
    def web_bind(self) -> str:
        """
        Web network interface binding mask.
        """
        ...

    @sections.WebServer.setting(params.Dict[ResourceOptions], version_introduced='1.5')
    def web_cors_options(self) -> Mapping[str, ResourceOptions]:
        """
        Cross Origin Resource Sharing options.
        """
        ...

    @sections.WebServer.setting(params.Bool, version_introduced='1.2', env_name='APP_WEB_ENABLED', default=True, related_cli_options={'faust worker': ['--with-web']})
    def web_enabled(self) -> bool:
        """
        Enable/disable internal web server.
        """
        ...

    @sections.WebServer.setting(params.Str, version_introduced='1.2', env_name='WEB_HOST', default_template='{conf.NODE_HOSTNAME}', related_cli_options={'faust worker': ['--web-host']})
    def web_host(self) -> str:
        """
        Web server host name.
        """
        ...

    @sections.WebServer.setting(params.Bool, version_introduced='1.5', default=False)
    def web_in_thread(self) -> bool:
        """
        Run the web server in a separate thread.
        """
        ...

    @sections.WebServer.setting(params.Port, version_introduced='1.2', env_name='WEB_PORT', default=WEB_PORT, related_cli_options={'faust worker': ['--web-port']})
    def web_port(self) -> int:
        """
        Web server port.
        """
        ...

    @sections.WebServer.setting(params.URL, version_introduced='1.2', default=WEB_TRANSPORT, related_cli_options={'faust worker': ['--web-transport']})
    def web_transport(self) -> str:
        """
        Network transport used for the web server.
        """
        ...

    @sections.WebServer.setting(params.URL, default_template='http://{conf.web_host}:{conf.web_port}', env_name='NODE_CANONICAL_URL', related_cli_options={'faust worker': ['--web-host', '--web-port']}, related_settings=[web_host, web_port])
    def canonical_url(self) -> str:
        """
        Node specific canonical URL.
        """
        ...

    @sections.Worker.setting(params.Bool, env_name='WORKER_REDIRECT_STDOUTS', default=True)
    def worker_redirect_stdouts(self) -> bool:
        """
        Redirecting standard outputs.
        """
        ...

    @sections.Worker.setting(params.Severity, env_name='WORKER_REDIRECT_STDOUTS_LEVEL', default='WARN')
    def worker_redirect_stdouts_level(self) -> Severity:
        """
        Level used when redirecting standard outputs.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[AgentT]), default='faust:Agent')
    def Agent(self) -> Union[Type[AgentT], str]:
        """
        Agent class type.
        """
        ...

    @sections.Consumer.setting(params.Symbol(Type[SchedulingStrategyT]), version_introduced='1.5', default='faust.transport.utils:DefaultSchedulingStrategy')
    def ConsumerScheduler(self) -> Union[Type[SchedulingStrategyT], str]:
        """
        Consumer scheduler class.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[EventT]), default='faust:Event')
    def Event(self) -> Union[Type[EventT], str]:
        """
        Event class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[SchemaT]), default='faust:Schema')
    def Schema(self) -> Union[Type[SchemaT], str]:
        """
        Schema class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[StreamT]), default='faust:Stream')
    def Stream(self) -> Union[Type[StreamT], str]:
        """
        Stream class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[TableT]), default='faust:Table')
    def Table(self) -> Union[Type[TableT], str]:
        """
        Table class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[TableT]), default='faust:SetTable')
    def SetTable(self) -> Union[Type[TableT], str]:
        """
        SetTable extension table.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[GlobalTableT]), default='faust:GlobalTable')
    def GlobalTable(self) -> Union[Type[GlobalTableT], str]:
        """
        GlobalTable class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[GlobalTableT]), default='faust:SetGlobalTable')
    def SetGlobalTable(self) -> Union[Type[GlobalTableT], str]:
        """
        SetGlobalTable class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[TableManagerT]), default='faust.tables:TableManager')
    def TableManager(self) -> Union[Type[TableManagerT], str]:
        """
        Table manager class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[RegistryT]), default='faust.serializers:Registry')
    def Serializers(self) -> Union[Type[RegistryT], str]:
        """
        Serializer registry class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[_WorkerT]), default='faust.worker:Worker')
    def Worker(self) -> Union[Type[_WorkerT], str]:
        """
        Worker class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[PartitionAssignorT]), default='faust.assignor:PartitionAssignor')
    def PartitionAssignor(self) -> Union[Type[PartitionAssignorT], str]:
        """
        Partition assignor class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[LeaderAssignorT]), default='faust.assignor:LeaderAssignor')
    def LeaderAssignor(self) -> Union[Type[LeaderAssignorT], str]:
        """
        Leader assignor class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[RouterT]), default='faust.app.router:Router')
    def Router(self) -> Union[Type[RouterT], str]:
        """
        Router class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[TopicT]), default='faust:Topic')
    def Topic(self) -> Union[Type[TopicT], str]:
        """
        Topic class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[HttpClientT]), default='aiohttp.client:ClientSession')
    def HttpClient(self) -> Union[Type[HttpClientT], str]:
        """
        Http client class type.
        """
        ...

    @sections.Extension.setting(params.Symbol(Type[SensorT]), default='faust.sensors:Monitor')
    def Monitor(self) -> Union[Type[SensorT], str]:
        """
        Monitor sensor class type.
        """
        ...

    @sections.Stream.setting(params.Bool, default=True, version_deprecated='1.0', deprecation_reason='no longer has any effect')
    def stream_ack_cancelled_tasks(self) -> bool:
        """
        Deprecated setting has no effect.
        """
        ...

    @sections.Stream.setting(params.Bool, default=True, version_deprecated='1.0', deprecation_reason='no longer has any effect')
    def stream_ack_exceptions(self) -> bool:
        """
        Deprecated setting has no effect.
        """
        ...

    @sections.Producer.setting(params.UnsignedInt, env_name='PRODUCER_LINGER_MS', version_deprecated='1.11', deprecation_reason='use producer_linger in seconds instead.', default=0)
    def producer_linger_ms(self) -> int:
        """
        Deprecated setting, please use producer_linger instead.
        """
        ...

    @sections.Common.setting(params.URL, default=None, version_deprecated=1.0, deprecation_reason='Please use "broker" setting instead')
    def url(self) -> Optional[str]:
        """
        Backward compatibility alias to broker.
        """
        ...