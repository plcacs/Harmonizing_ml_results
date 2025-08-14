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
    _env_prefix: Optional[str] = None
    _url: Optional[URLArg] = None
    _producer_linger_ms: int = 0

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
        self.on_init(id, **locals())

    def on_init(self, id: str, **kwargs: Any) -> None:
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
            if prefix_from_env is not None:
                self._env_prefix = prefix_from_env
            else:
                if env_prefix is not None:
                    self._env_prefix = env_prefix

    def getenv(self, env_name: str) -> Any:
        if self._env_prefix:
            env_name = self._env_prefix.rstrip('_') + '_' + env_name
        return self.env.get(env_name)

    def relative_to_appdir(self, path: Path) -> Path:
        return path if path.is_absolute() else self.appdir / path

    def data_directory_for_version(self, version: int) -> Path:
        return self.datadir / f'v{version}'

    def find_old_versiondirs(self) -> Iterable[Path]:
        for version in reversed(range(0, self.version)):
            path = self.data_directory_for_version(version)
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

    @sections.Common.setting(
        params.Str,
        env_name='ENVIRONMENT_VARIABLE_NAME',
        version_removed='1.0',
    )
    def MY_SETTING(self) -> str:
        pass

    @sections.Common.setting(
        params.Param[AutodiscoverArg, AutodiscoverArg],
        default=False,
    )
    def autodiscover(self) -> AutodiscoverArg:
        pass

    @sections.Common.setting(
        params.Path,
        env_name='APP_DATADIR',
        default=DATADIR,
        related_cli_options={'faust': '--datadir'},
    )
    def datadir(self) -> Path:
        pass

    @datadir.on_get_value
    def _prepare_datadir(self, path: Path) -> Path:
        return Path(str(path).format(conf=self))

    @sections.Common.setting(
        params.Path,
        default='tables',
        env_name='APP_TABLEDIR',
    )
    def tabledir(self) -> Path:
        pass

    @tabledir.on_get_value
    def _prepare_tabledir(self, path: Path) -> Path:
        return self.relative_to_appdir(path)

    @sections.Common.setting(
        params.Bool,
        env_name='APP_DEBUG',
        default=False,
        related_cli_options={'faust': '--debug'},
    )
    def debug(self) -> bool:
        pass

    @sections.Common.setting(
        params.Str,
        env_name='APP_ENV_PREFIX',
        version_introduced='1.11',
        default=None,
        ignore_default=True,
    )
    def env_prefix(self) -> Optional[str]:
        pass

    @sections.Common.setting(
        params.Str,
        env_name='APP_ID_FORMAT',
        default='{id}-v{self.version}',
    )
    def id_format(self) -> str:
        pass

    @sections.Common.setting(
        params.Str,
        default=None,
    )
    def origin(self) -> Optional[str]:
        pass

    @sections.Common.setting(
        params.Timezone,
        version_introduced='1.4',
        env_name='TIMEZONE',
        default=timezone.utc,
    )
    def timezone(self) -> tzinfo:
        pass

    @sections.Common.setting(
        params.Int,
        env_name='APP_VERSION',
        default=1,
        min_value=1,
    )
    def version(self) -> int:
        pass

    @sections.Agent.setting(
        params.Symbol(Type[SupervisorStrategyT]),
        env_name='AGENT_SUPERVISOR',
        default='mode.OneForOneSupervisor',
    )
    def agent_supervisor(self) -> Type[SupervisorStrategyT]:
        pass

    @sections.Common.setting(
        params.Seconds,
        env_name='BLOCKING_TIMEOUT',
        default=None,
        related_cli_options={'faust': '--blocking-timeout'},
    )
    def blocking_timeout(self) -> Optional[float]:
        pass

    @sections.Common.setting(
        params.BrokerList,
        env_name='BROKER_URL',
    )
    def broker(self) -> List[URL]:
        pass

    @broker.on_set_default
    def _prepare_broker(self) -> BrokerArg:
        return self._url or self.DEFAULT_BROKER_URL

    @sections.Broker.setting(
        params.BrokerList,
        version_introduced='1.7',
        env_name='BROKER_CONSUMER_URL',
        default_alias='broker',
    )
    def broker_consumer(self) -> List[URL]:
        pass

    @sections.Broker.setting(
        params.BrokerList,
        version_introduced='1.7',
        env_name='BROKER_PRODUCER_URL',
        default_alias='broker',
    )
    def broker_producer(self) -> List[URL]:
        pass

    @sections.Broker.setting(
        params.Str,
        version_introduced='1.10',
        env_name='BROKER_API_VERSION',
        default='auto',
    )
    def broker_api_version(self) -> str:
        pass

    @sections.Broker.setting(
        params.Bool,
        env_name='BROKER_CHECK_CRCS',
        default=True,
    )
    def broker_check_crcs(self) -> bool:
        pass

    @sections.Broker.setting(
        params.Str,
        env_name='BROKER_CLIENT_ID',
        default=f'faust-{faust_version}',
    )
    def broker_client_id(self) -> str:
        pass

    @sections.Broker.setting(
        params.UnsignedInt,
        env_name='BROKER_COMMIT_EVERY',
        default=10_000,
    )
    def broker_commit_every(self) -> int:
        pass

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_INTERVAL',
        default=2.8,
    )
    def broker_commit_interval(self) -> float:
        pass

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT',
        default=want_seconds(timedelta(minutes=5)),
    )
    def broker_commit_livelock_soft_timeout(self) -> float:
        pass

    @sections.Common.setting(
        params.Credentials,
        version_introduced='1.5',
        env_name='BROKER_CREDENTIALS',
        default=None,
    )
    def broker_credentials(self) -> Optional[CredentialsT]:
        pass

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.0.11',
        env_name='BROKER_HEARTBEAT_INTERVAL',
        default=3.0,
    )
    def broker_heartbeat_interval(self) -> float:
        pass

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.7',
        env_name='BROKER_MAX_POLL_INTERVAL',
        default=1000.0,
    )
    def broker_max_poll_interval(self) -> float:
        pass

    @sections.Broker.setting(
        params.UnsignedInt,
        version_introduced='1.4',
        env_name='BROKER_MAX_POLL_RECORDS',
        default=None,
        allow_none=True,
    )
    def broker_max_poll_records(self) -> Optional[int]:
        pass

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.10',
        env_name='BROKER_REBALANCE_TIMEOUT',
        default=60.0,
    )
    def broker_rebalance_timeout(self) -> float:
        pass

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.4',
        env_name='BROKER_REQUEST_TIMEOUT',
        default=90.0,
    )
    def broker_request_timeout(self) -> float:
        pass

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.0.11',
        env_name='BROKER_SESSION_TIMEOUT',
        default=60.0,
    )
    def broker_session_timeout(self) -> float:
        pass

    @sections.Common.setting(
        params.SSLContext,
        default=None,
    )
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        pass

    @sections.Consumer.setting(
        params.Str,
        version_introduced='1.10',
        env_name='CONSUMER_API_VERSION',
        default_alias='broker_api_version',
    )
    def consumer_api_version(self) -> str:
        pass

    @sections.Consumer.setting(
        params.UnsignedInt,
        version_introduced='1.4',
        env_name