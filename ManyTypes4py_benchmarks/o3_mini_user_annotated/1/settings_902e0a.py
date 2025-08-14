#!/usr/bin/env python3
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
    class _WorkerT:  # type: ignore
        pass

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
    _env_prefix: Optional[str]
    _url: Optional[str]

    #: Environment.
    #: Defaults to :data:`os.environ`.
    env: Mapping[str, str]

    def __init__(
        self,
        id: str,
        *,
        autodiscover: AutodiscoverArg = None,
        datadir: Union[str, Path] = None,
        tabledir: Union[str, Path] = None,
        debug: bool = None,
        env_prefix: str = None,
        id_format: str = None,
        origin: str = None,
        timezone: Union[str, tzinfo] = None,
        version: int = None,
        agent_supervisor: SymbolArg[Type[SupervisorStrategyT]] = None,
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
        consumer_api_version: str = None,
        consumer_max_fetch_size: int = None,
        consumer_auto_offset_reset: str = None,
        consumer_group_instance_id: str = None,
        key_serializer: CodecArg = None,
        value_serializer: CodecArg = None,
        logging_config: Mapping = None,
        loghandlers: List[logging.Handler] = None,
        producer_acks: int = None,
        producer_api_version: str = None,
        producer_compression_type: str = None,
        producer_linger_ms: int = None,
        producer_max_batch_size: int = None,
        producer_max_request_size: int = None,
        producer_partitioner: SymbolArg[PartitionerT] = None,
        producer_request_timeout: Seconds = None,
        reply_create_topic: bool = None,
        reply_expires: Seconds = None,
        reply_to: str = None,
        reply_to_prefix: str = None,
        processing_guarantee: Union[str, ProcessingGuarantee] = None,
        stream_buffer_maxsize: int = None,
        stream_processing_timeout: Seconds = None,
        stream_publish_on_commit: bool = None,
        stream_recovery_delay: Seconds = None,
        stream_wait_empty: bool = None,
        store: URLArg = None,
        table_cleanup_interval: Seconds = None,
        table_key_index_size: int = None,
        table_standby_replicas: int = None,
        topic_allow_declare: bool = None,
        topic_disable_leader: bool = None,
        topic_partitions: int = None,
        topic_replication_factor: int = None,
        cache: URLArg = None,
        canonical_url: URLArg = None,
        web: URLArg = None,
        web_bind: str = None,
        web_cors_options: Mapping[str, ResourceOptions] = None,
        web_enabled: bool = None,
        web_host: str = None,
        web_in_thread: bool = None,
        web_port: int = None,
        web_transport: URLArg = None,
        worker_redirect_stdouts: bool = None,
        worker_redirect_stdouts_level: Severity = None,
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
        stream_ack_cancelled_tasks: bool = None,
        stream_ack_exceptions: bool = None,
        url: URLArg = None,
        **kwargs: Any
    ) -> None:
        self._env_prefix = None
        self._url = None
        ...  # replaced by __init_subclass__ in BaseSettings

    def on_init(self, id: str, **kwargs: Any) -> None:
        self._init_env_prefix(**kwargs)
        self._version = kwargs.get('version', 1)
        self.id = id

    def _init_env_prefix(self, env: Optional[Mapping[str, str]] = None, env_prefix: Optional[str] = None, **kwargs: Any) -> None:
        if env is None:
            env = os.environ
        self.env = env
        env_name: Optional[str] = self.SETTINGS['env_prefix'].env_name  # type: ignore
        if env_name is not None:
            prefix_from_env: Optional[str] = self.env.get(env_name)
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

    @sections.Common.setting(
        params.Str,
        env_name='ENVIRONMENT_VARIABLE_NAME',
        version_removed='1.0',
    )
    def MY_SETTING(self) -> str:
        ...

    @sections.Common.setting(
        params.Param[AutodiscoverArg, AutodiscoverArg],
        default=False,
    )
    def autodiscover(self) -> AutodiscoverArg:
        ...

    @sections.Common.setting(
        params.Path,
        env_name='APP_DATADIR',
        default=DATADIR,
        related_cli_options={'faust': '--datadir'},
    )
    def datadir(self, path: Path) -> Path:
        ...

    @datadir.on_get_value  # type: ignore
    def _prepare_datadir(self, path: Path) -> Path:
        return Path(str(path).format(conf=self))

    @sections.Common.setting(
        params.Path,
        default='tables',
        env_name='APP_TABLEDIR',
    )
    def tabledir(self) -> Path:
        ...

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
        ...

    @sections.Common.setting(
        params.Str,
        env_name='APP_ENV_PREFIX',
        version_introduced='1.11',
        default=None,
        ignore_default=True,
    )
    def env_prefix(self) -> str:
        ...

    @sections.Common.setting(
        params.Str,
        env_name='APP_ID_FORMAT',
        default='{id}-v{self.version}',
    )
    def id_format(self) -> str:
        ...

    @sections.Common.setting(
        params.Str,
        default=None,
    )
    def origin(self) -> str:
        ...

    @sections.Common.setting(
        params.Timezone,
        version_introduced='1.4',
        env_name='TIMEZONE',
        default=timezone.utc,
    )
    def timezone(self) -> tzinfo:
        ...

    @sections.Common.setting(
        params.Int,
        env_name='APP_VERSION',
        default=1,
        min_value=1,
    )
    def version(self) -> int:
        ...

    @sections.Agent.setting(
        params.Symbol(Type[SupervisorStrategyT]),
        env_name='AGENT_SUPERVISOR',
        default='mode.OneForOneSupervisor',
    )
    def agent_supervisor(self) -> Type[SupervisorStrategyT]:
        ...

    @sections.Common.setting(
        params.Seconds,
        env_name='BLOCKING_TIMEOUT',
        default=None,
        related_cli_options={'faust': '--blocking-timeout'},
    )
    def blocking_timeout(self) -> Optional[float]:
        ...

    @sections.Common.setting(
        params.BrokerList,
        env_name='BROKER_URL',
    )
    def broker(self) -> List[URL]:
        ...

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
        ...

    @sections.Broker.setting(
        params.BrokerList,
        version_introduced='1.7',
        env_name='BROKER_PRODUCER_URL',
        default_alias='broker',
    )
    def broker_producer(self) -> List[URL]:
        ...

    @sections.Broker.setting(
        params.Str,
        version_introduced='1.10',
        env_name='BROKER_API_VERSION',
        default='auto',
    )
    def broker_api_version(self) -> str:
        ...

    @sections.Broker.setting(
        params.Bool,
        env_name='BROKER_CHECK_CRCS',
        default=True,
    )
    def broker_check_crcs(self) -> bool:
        ...

    @sections.Broker.setting(
        params.Str,
        env_name='BROKER_CLIENT_ID',
        default=f'faust-{faust_version}',
    )
    def broker_client_id(self) -> str:
        ...

    @sections.Broker.setting(
        params.UnsignedInt,
        env_name='BROKER_COMMIT_EVERY',
        default=10_000,
    )
    def broker_commit_every(self) -> int:
        ...

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_INTERVAL',
        default=2.8,
    )
    def broker_commit_interval(self) -> float:
        ...

    @sections.Broker.setting(
        params.Seconds,
        env_name='BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT',
        default=want_seconds(timedelta(minutes=5)),
    )
    def broker_commit_livelock_soft_timeout(self) -> float:
        ...

    @sections.Common.setting(
        params.Credentials,
        version_introduced='1.5',
        env_name='BROKER_CREDENTIALS',
        default=None,
    )
    def broker_credentials(self) -> CredentialsT:
        ...

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.0.11',
        env_name='BROKER_HEARTBEAT_INTERVAL',
        default=3.0,
    )
    def broker_heartbeat_interval(self) -> float:
        ...

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.7',
        env_name='BROKER_MAX_POLL_INTERVAL',
        default=1000.0,
    )
    def broker_max_poll_interval(self) -> float:
        ...

    @sections.Broker.setting(
        params.UnsignedInt,
        version_introduced='1.4',
        env_name='BROKER_MAX_POLL_RECORDS',
        default=None,
        allow_none=True,
    )
    def broker_max_poll_records(self) -> Optional[int]:
        ...

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.10',
        env_name='BROKER_REBALANCE_TIMEOUT',
        default=60.0,
    )
    def broker_rebalance_timeout(self) -> float:
        ...

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.4',
        env_name='BROKER_REQUEST_TIMEOUT',
        default=90.0,
    )
    def broker_request_timeout(self) -> float:
        ...

    @sections.Broker.setting(
        params.Seconds,
        version_introduced='1.0.11',
        env_name='BROKER_SESSION_TIMEOUT',
        default=60.0,
    )
    def broker_session_timeout(self) -> float:
        ...

    @sections.Common.setting(
        params.SSLContext,
        default=None,
    )
    def ssl_context(self) -> ssl.SSLContext:
        ...

    @sections.Consumer.setting(
        params.Str,
        version_introduced='1.10',
        env_name='CONSUMER_API_VERSION',
        default_alias='broker_api_version',
    )
    def consumer_api_version(self) -> str:
        ...

    @sections.Consumer.setting(
        params.UnsignedInt,
        version_introduced='1.4',
        env_name='CONSUMER_MAX_FETCH_SIZE',
        default=1024 ** 2,
    )
    def consumer_max_fetch_size(self) -> int:
        ...

    @sections.Consumer.setting(
        params.Str,
        version_introduced='1.5',
        env_name='CONSUMER_AUTO_OFFSET_RESET',
        default='earliest',
    )
    def consumer_auto_offset_reset(self) -> str:
        ...

    @sections.Consumer.setting(
        params.Str,
        version_introduced='2.1',
        env_name='CONSUMER_GROUP_INSTANCE_ID',
        default=None,
    )
    def consumer_group_instance_id(self) -> str:
        ...

    @sections.Serialization.setting(
        params.Codec,
        env_name='APP_KEY_SERIALIZER',
        default='raw',
    )
    def key_serializer(self) -> CodecArg:
        ...

    @sections.Serialization.setting(
        params.Codec,
        env_name='APP_VALUE_SERIALIZER',
        default='json',
    )
    def value_serializer(self) -> CodecArg:
        ...

    @sections.Common.setting(
        params.Dict[Any],
        version_introduced='1.5',
    )
    def logging_config(self) -> Mapping[str, Any]:
        ...

    @sections.Common.setting(
        params.LogHandlers,
    )
    def loghandlers(self) -> List[logging.Handler]:
        ...

    @sections.Producer.setting(
        params.Int,
        env_name='PRODUCER_ACKS',
        default=-1,
        number_aliases={'all': -1},
    )
    def producer_acks(self) -> int:
        ...

    @sections.Producer.setting(
        params.Str,
        version_introduced='1.5.3',
        env_name='PRODUCER_API_VERSION',
        default_alias='broker_api_version',
    )
    def producer_api_version(self) -> str:
        ...

    @sections.Producer.setting(
        params.Str,
        env_name='PRODUCER_COMPRESSION_TYPE',
        default=None,
    )
    def producer_compression_type(self) -> str:
        ...

    @sections.Producer.setting(
        params.Seconds,
        env_name='PRODUCER_LINGER',
    )
    def producer_linger(self) -> Optional[float]:
        ...

    @producer_linger.on_set_default  # type: ignore
    def _prepare_producer_linger(self) -> float:
        return float(self._producer_linger_ms) / 1000.0

    @sections.Producer.setting(
        params.UnsignedInt,
        env_name='PRODUCER_MAX_BATCH_SIZE',
        default=16384,
    )
    def producer_max_batch_size(self) -> int:
        ...

    @sections.Producer.setting(
        params.UnsignedInt,
        env_name='PRODUCER_MAX_REQUEST_SIZE',
        default=1_000_000,
    )
    def producer_max_request_size(self) -> int:
        ...

    @sections.Producer.setting(
        params._Symbol[PartitionerT, Optional[PartitionerT]],
        version_introduced='1.2',
        default=None,
    )
    def producer_partitioner(self) -> Optional[PartitionerT]:
        ...

    @sections.Producer.setting(
        params.Seconds,
        version_introduced='1.4',
        env_name='PRODUCER_REQUEST_TIMEOUT',
        default=1200.0,
    )
    def producer_request_timeout(self) -> float:
        ...

    @sections.RPC.setting(
        params.Bool,
        env_name='APP_REPLY_CREATE_TOPIC',
        default=False,
    )
    def reply_create_topic(self) -> bool:
        ...

    @sections.RPC.setting(
        params.Seconds,
        env_name='APP_REPLY_EXPIRES',
        default=want_seconds(timedelta(days=1)),
    )
    def reply_expires(self) -> float:
        ...

    @sections.RPC.setting(
        params.Str,
    )
    def reply_to(self) -> str:
        ...

    @reply_to.on_set_default  # type: ignore
    def _prepare_reply_to_default(self) -> str:
        return f'{self.reply_to_prefix}{uuid4()}'

    @sections.RPC.setting(
        params.Str,
        env_name='APP_REPLY_TO_PREFIX',
        default='f-reply-',
    )
    def reply_to_prefix(self) -> str:
        ...

    @sections.Common.setting(
        params.Enum(ProcessingGuarantee),
        version_introduced='1.5',
        env_name='PROCESSING_GUARANTEE',
        default=ProcessingGuarantee.AT_LEAST_ONCE,
    )
    def processing_guarantee(self) -> ProcessingGuarantee:
        ...

    @sections.Stream.setting(
        params.UnsignedInt,
        env_name='STREAM_BUFFER_MAXSIZE',
        default=4096,
    )
    def stream_buffer_maxsize(self) -> int:
        ...

    @sections.Stream.setting(
        params.Seconds,
        version_introduced='1.10',
        env_name='STREAM_PROCESSING_TIMEOUT',
        default=5 * 60.0,
    )
    def stream_processing_timeout(self) -> float:
        ...

    @sections.Stream.setting(
        params.Bool,
        default=False,
    )
    def stream_publish_on_commit(self) -> bool:
        ...

    @sections.Stream.setting(
        params.Seconds,
        version_introduced='1.3',
        version_changed={'1.5.3': 'Disabled by default.'},
        env_name='STREAM_RECOVERY_DELAY',
        default=0.0,
    )
    def stream_recovery_delay(self) -> float:
        ...

    @sections.Stream.setting(
        params.Bool,
        env_name='STREAM_WAIT_EMPTY',
        default=True,
    )
    def stream_wait_empty(self) -> bool:
        ...

    @sections.Common.setting(
        params.URL,
        env_name='APP_STORE',
        default='memory://',
    )
    def store(self) -> URL:
        ...

    @sections.Table.setting(
        params.Seconds,
        env_name='TABLE_CLEANUP_INTERVAL',
        default=30.0,
    )
    def table_cleanup_interval(self) -> float:
        ...

    @sections.Table.setting(
        params.UnsignedInt,
        version_introduced='1.7',
        env_name='TABLE_KEY_INDEX_SIZE',
        default=1000,
    )
    def table_key_index_size(self) -> int:
        ...

    @sections.Table.setting(
        params.UnsignedInt,
        env_name='TABLE_STANDBY_REPLICAS',
        default=1,
    )
    def table_standby_replicas(self) -> int:
        ...

    @sections.Topic.setting(
        params.Bool,
        version_introduced='1.5',
        env_name='TOPIC_ALLOW_DECLARE',
        default=True,
    )
    def topic_allow_declare(self) -> bool:
        ...

    @sections.Topic.setting(
        params.Bool,
        version_introduced='1.7',
        env_name='TOPIC_DISABLE_LEADER',
        default=False,
    )
    def topic_disable_leader(self) -> bool:
        ...

    @sections.Topic.setting(
        params.UnsignedInt,
        env_name='TOPIC_PARTITIONS',
        default=8,
    )
    def topic_partitions(self) -> int:
        ...

    @sections.Topic.setting(
        params.UnsignedInt,
        env_name='TOPIC_REPLICATION_FACTOR',
        default=1,
    )
    def topic_replication_factor(self) -> int:
        ...

    @sections.Common.setting(
        params.URL,
        version_introduced='1.2',
        env_name='CACHE_URL',
        default='memory://',
    )
    def cache(self) -> URL:
        ...

    @sections.WebServer.setting(
        params.URL,
        version_introduced='1.2',
        default='aiohttp://',
    )
    def web(self) -> URL:
        ...

    @sections.WebServer.setting(
        params.Str,
        version_introduced='1.2',
        env_name='WEB_BIND',
        default=WEB_BIND,
        related_cli_options={'faust worker': ['--web-bind']},
    )
    def web_bind(self) -> str:
        ...

    @sections.WebServer.setting(
        params.Dict[ResourceOptions],
        version_introduced='1.5',
    )
    def web_cors_options(self) -> Mapping[str, ResourceOptions]:
        ...

    @sections.WebServer.setting(
        params.Bool,
        version_introduced='1.2',
        env_name='APP_WEB_ENABLED',
        default=True,
        related_cli_options={'faust worker': ['--with-web']},
    )
    def web_enabled(self) -> bool:
        ...

    @sections.WebServer.setting(
        params.Str,
        version_introduced='1.2',
        env_name='WEB_HOST',
        default_template='{conf.NODE_HOSTNAME}',
        related_cli_options={'faust worker': ['--web-host']},
    )
    def web_host(self) -> str:
        ...

    @sections.WebServer.setting(
        params.Bool,
        version_introduced='1.5',
        default=False,
    )
    def web_in_thread(self) -> bool:
        ...

    @sections.WebServer.setting(
        params.Port,
        version_introduced='1.2',
        env_name='WEB_PORT',
        default=WEB_PORT,
        related_cli_options={'faust worker': ['--web-port']},
    )
    def web_port(self) -> int:
        ...

    @sections.WebServer.setting(
        params.URL,
        version_introduced='1.2',
        default=WEB_TRANSPORT,
        related_cli_options={'faust worker': ['--web-transport']},
    )
    def web_transport(self) -> URL:
        ...

    @sections.WebServer.setting(
        params.URL,
        default_template='http://{conf.web_host}:{conf.web_port}',
        env_name='NODE_CANONICAL_URL',
        related_cli_options={'faust worker': ['--web-host', '--web-port']},
        related_settings=[web_host, web_port],
    )
    def canonical_url(self) -> URL:
        ...

    @sections.Worker.setting(
        params.Bool,
        env_name='WORKER_REDIRECT_STDOUTS',
        default=True,
    )
    def worker_redirect_stdouts(self) -> bool:
        ...

    @sections.Worker.setting(
        params.Severity,
        env_name='WORKER_REDIRECT_STDOUTS_LEVEL',
        default='WARN',
    )
    def worker_redirect_stdouts_level(self) -> Severity:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[AgentT]),
        default='faust:Agent',
    )
    def Agent(self) -> Type[AgentT]:
        ...

    @sections.Consumer.setting(
        params.Symbol(Type[SchedulingStrategyT]),
        version_introduced='1.5',
        default='faust.transport.utils:DefaultSchedulingStrategy',
    )
    def ConsumerScheduler(self) -> Type[SchedulingStrategyT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[EventT]),
        default='faust:Event',
    )
    def Event(self) -> Type[EventT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[SchemaT]),
        default='faust:Schema',
    )
    def Schema(self) -> Type[SchemaT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[StreamT]),
        default='faust:Stream',
    )
    def Stream(self) -> Type[StreamT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[TableT]),
        default='faust:Table',
    )
    def Table(self) -> Type[TableT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[TableT]),
        default='faust:SetTable',
    )
    def SetTable(self) -> Type[TableT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[GlobalTableT]),
        default='faust:GlobalTable',
    )
    def GlobalTable(self) -> Type[GlobalTableT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[GlobalTableT]),
        default='faust:SetGlobalTable',
    )
    def SetGlobalTable(self) -> Type[GlobalTableT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[TableManagerT]),
        default='faust.tables:TableManager',
    )
    def TableManager(self) -> Type[TableManagerT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[RegistryT]),
        default='faust.serializers:Registry',
    )
    def Serializers(self) -> Type[RegistryT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[_WorkerT]),
        default='faust.worker:Worker',
    )
    def Worker(self) -> Type[_WorkerT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[PartitionAssignorT]),
        default='faust.assignor:PartitionAssignor',
    )
    def PartitionAssignor(self) -> Type[PartitionAssignorT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[LeaderAssignorT]),
        default='faust.assignor:LeaderAssignor',
    )
    def LeaderAssignor(self) -> Type[LeaderAssignorT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[RouterT]),
        default='faust.app.router:Router',
    )
    def Router(self) -> Type[RouterT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[TopicT]),
        default='faust:Topic',
    )
    def Topic(self) -> Type[TopicT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[HttpClientT]),
        default='aiohttp.client:ClientSession',
    )
    def HttpClient(self) -> Type[HttpClientT]:
        ...

    @sections.Extension.setting(
        params.Symbol(Type[SensorT]),
        default='faust.sensors:Monitor',
    )
    def Monitor(self) -> Type[SensorT]:
        ...

    @sections.Stream.setting(
        params.Bool,
        default=True,
        version_deprecated='1.0',
        deprecation_reason='no longer has any effect',
    )
    def stream_ack_cancelled_tasks(self) -> bool:
        ...

    @sections.Stream.setting(
        params.Bool,
        default=True,
        version_deprecated='1.0',
        deprecation_reason='no longer has any effect',
    )
    def stream_ack_exceptions(self) -> bool:
        ...

    @sections.Producer.setting(
        params.UnsignedInt,
        env_name='PRODUCER_LINGER_MS',
        version_deprecated='1.11',
        deprecation_reason='use producer_linger in seconds instead.',
        default=0,
    )
    def producer_linger_ms(self) -> int:
        ...

    @sections.Common.setting(
        params.URL,
        default=None,
        version_deprecated=1.0,
        deprecation_reason='Please use "broker" setting instead',
    )
    def url(self) -> URL:
        ...
