import logging
import os
import socket
import ssl
import typing
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Type, Union
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
    NODE_HOSTNAME: ClassVar[str] = socket.gethostname()
    DEFAULT_BROKER_URL: ClassVar[str] = 'kafka://localhost:9092'

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
        version: Optional[int] = None,
        agent_supervisor: Optional[Union[str, Type[SupervisorStrategyT]]] = None,
        broker: Optional[BrokerArg] = None,
        broker_consumer: Optional[BrokerArg] = None,
        broker_producer: Optional[BrokerArg] = None,
        broker_api_version: Optional[str] = None,
        broker_check_crcs: Optional[bool] = None,
        broker_client_id: Optional[str] = None,
        broker_commit_every: Optional[int] = None,
        broker_commit_interval: Optional[float] = None,
        broker_commit_livelock_soft_timeout: Optional[float] = None,
        broker_credentials: Optional[CredentialsArg] = None,
        broker_heartbeat_interval: Optional[float] = None,
        broker_max_poll_interval: Optional[float] = None,
        broker_max_poll_records: Optional[int] = None,
        broker_rebalance_timeout: Optional[float] = None,
        broker_request_timeout: Optional[float] = None,
        broker_session_timeout: Optional[float] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        consumer_api_version: Optional[str] = None,
        consumer_max_fetch_size: Optional[int] = None,
        consumer_auto_offset_reset: Optional[str] = None,
        consumer_group_instance_id: Optional[str] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        loghandlers: Optional[List[logging.Handler]] = None,
        producer_acks: Optional[int] = None,
        producer_api_version: Optional[str] = None,
        producer_compression_type: Optional[str] = None,
        producer_linger_ms: Optional[int] = None,
        producer_max_batch_size: Optional[int] = None,
        producer_max_request_size: Optional[int] = None,
        producer_partitioner: Optional[Union[str, PartitionerT]] = None,
        producer_request_timeout: Optional[float] = None,
        reply_create_topic: Optional[bool] = None,
        reply_expires: Optional[float] = None,
        reply_to: Optional[str] = None,
        reply_to_prefix: Optional[str] = None,
        processing_guarantee: Optional[Union[str, ProcessingGuarantee]] = None,
        stream_buffer_maxsize: Optional[int] = None,
        stream_processing_timeout: Optional[float] = None,
        stream_publish_on_commit: Optional[bool] = None,
        stream_recovery_delay: Optional[float] = None,
        stream_wait_empty: Optional[bool] = None,
        store: Optional[Union[str, URL]] = None,
        table_cleanup_interval: Optional[float] = None,
        table_key_index_size: Optional[int] = None,
        table_standby_replicas: Optional[int] = None,
        topic_allow_declare: Optional[bool] = None,
        topic_disable_leader: Optional[bool] = None,
        topic_partitions: Optional[int] = None,
        topic_replication_factor: Optional[int] = None,
        cache: Optional[Union[str, URL]] = None,
        canonical_url: Optional[Union[str, URL]] = None,
        web: Optional[Union[str, URL]] = None,
        web_bind: Optional[str] = None,
        web_cors_options: Optional[Dict[str, ResourceOptions]] = None,
        web_enabled: Optional[bool] = None,
        web_host: Optional[str] = None,
        web_in_thread: Optional[bool] = None,
        web_port: Optional[int] = None,
        web_transport: Optional[Union[str, URL]] = None,
        worker_redirect_stdouts: Optional[bool] = None,
        worker_redirect_stdouts_level: Optional[Union[str, int]] = None,
        Agent: Optional[Union[str, Type[AgentT]]] = None,
        ConsumerScheduler: Optional[Union[str, Type[SchedulingStrategyT]]] = None,
        Event: Optional[Union[str, Type[EventT]]] = None,
        Schema: Optional[Union[str, Type[SchemaT]]] = None,
        Stream: Optional[Union[str, Type[StreamT]]] = None,
        Table: Optional[Union[str, Type[TableT]]] = None,
        SetTable: Optional[Union[str, Type[TableT]]] = None,
        GlobalTable: Optional[Union[str, Type[GlobalTableT]]] = None,
        SetGlobalTable: Optional[Union[str, Type[GlobalTableT]]] = None,
        TableManager: Optional[Union[str, Type[TableManagerT]]] = None,
        Serializers: Optional[Union[str, Type[RegistryT]]] = None,
        Worker: Optional[Union[str, Type[_WorkerT]]] = None,
        PartitionAssignor: Optional[Union[str, Type[PartitionAssignorT]]] = None,
        LeaderAssignor: Optional[Union[str, Type[LeaderAssignorT]]] = None,
        Router: Optional[Union[str, Type[RouterT]]] = None,
        Topic: Optional[Union[str, Type[TopicT]]] = None,
        HttpClient: Optional[Union[str, Type[HttpClientT]]] = None,
        Monitor: Optional[Union[str, Type[SensorT]]] = None,
        stream_ack_cancelled_tasks: Optional[bool] = None,
        stream_ack_exceptions: Optional[bool] = None,
        url: Optional[Union[str, URL]] = None,
        **kwargs: Any
    ) -> None:
        ...

    def on_init(self, id: str, **kwargs: Any) -> None:
        self._init_env_prefix(**kwargs)
        self._version = kwargs.get('version', 1)
        self.id = id

    def _init_env_prefix(self, env: Optional[Mapping[str, str]] = None, env_prefix: Optional[str] = None, **kwargs: Any) -> None:
        if env is None:
            env = os.environ
        self.env = env
        env_name = self.SETTINGS['env_prefix'].env_name
        if env_name is not None:
            prefix_from_env = self.env.get(env_name)
            if prefix_from_env is not None:
                self._env_prefix = prefix_from_env
            elif env_prefix is not None:
                self._env_prefix = env_prefix

    def getenv(self, env_name: str) -> Optional[str]:
        if self._env_prefix:
            env_name = self._env_prefix.rstrip('_') + '_' + env_name
        return self.env.get(env_name)

    def relative_to_appdir(self, path: Union[str, Path]) -> Path:
        return path if Path(path).is_absolute() else self.appdir / path

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

    @sections.Common.setting(params.Str, env_name='ENVIRONMENT_VARIABLE_NAME', version_removed='1.0')
    def MY_SETTING(self) -> str:
        ...

    @sections.Common.setting(params.Param[AutodiscoverArg, AutodiscoverArg], default=False)
    def autodiscover(self) -> AutodiscoverArg:
        ...

    @sections.Common.setting(params.Path, env_name='APP_DATADIR', default=DATADIR, related_cli_options={'faust': '--datadir'})
    def datadir(self, path: Union[str, Path]) -> Path:
        ...

    @datadir.on_get_value
    def _prepare_datadir(self, path: Union[str, Path]) -> Path:
        return Path(str(path).format(conf=self))

    @sections.Common.setting(params.Path, default='tables', env_name='APP_TABLEDIR')
    def tabledir(self) -> Path:
        ...

    @tabledir.on_get_value
    def _prepare_tabledir(self, path: Union[str, Path]) -> Path:
        return self.relative_to_appdir(path)

    @sections.Common.setting(params.Bool, env_name='APP_DEBUG', default=False, related_cli_options={'faust': '--debug'})
    def debug(self) -> bool:
        ...

    @sections.Common.setting(params.Str, env_name='APP_ENV_PREFIX', version_introduced='1.11', default=None, ignore_default=True)
    def env_prefix(self) -> Optional[str]:
        ...

    @sections.Common.setting(params.Str, env_name='APP_ID_FORMAT', default='{id}-v{self.version}')
    def id_format(self) -> str:
        ...

    @sections.Common.setting(params.Str, default=None)
    def origin(self) -> Optional[str]:
        ...

    @sections.Common.setting(params.Timezone, version_introduced='1.4', env_name='TIMEZONE', default=timezone.utc)
    def timezone(self) -> tzinfo:
        ...

    @sections.Common.setting(params.Int, env_name='APP_VERSION', default=1, min_value=1)
    def version(self) -> int:
        ...

    @sections.Agent.setting(params.Symbol(Type[SupervisorStrategyT]), env_name='AGENT_SUPERVISOR', default='mode.OneForOneSupervisor')
    def agent_supervisor(self) -> Type[SupervisorStrategyT]:
        ...

    @sections.Common.setting(params.Seconds, env_name='BLOCKING_TIMEOUT', default=None, related_cli_options={'faust': '--blocking-timeout'})
    def blocking_timeout(self) -> Optional[float]:
        ...

    @sections.Common.setting(params.BrokerList, env_name='BROKER_URL')
    def broker(self) -> BrokerArg:
        ...

    @broker.on_set_default
    def _prepare_broker(self) -> BrokerArg:
        return self._url or self.DEFAULT_BROKER_URL

    @sections.Broker.setting(params.BrokerList, version_introduced='1.7', env_name='BROKER_CONSUMER_URL', default_alias='broker')
    def broker_consumer(self) -> BrokerArg:
        ...

    @sections.Broker.setting(params.BrokerList, version_introduced='1.7', env_name='BROKER_PRODUCER_URL', default_alias='broker')
    def broker_producer(self) -> BrokerArg:
        ...

    @sections.Broker.setting(params.Str, version_introduced='1.10', env_name='BROKER_API_VERSION', default='auto')
    def broker_api_version(self) -> str:
        ...

    @sections.Broker.setting(params.Bool, env_name='BROKER_CHECK_CRCS', default=True)
    def broker_check_crcs(self) -> bool:
        ...

    @sections.Broker.setting(params.Str, env_name='BROKER_CLIENT_ID', default=f'faust-{faust_version}')
    def broker_client_id(self) -> str:
        ...

    @sections.Broker.setting(params.UnsignedInt, env_name='BROKER_COMMIT_EVERY', default=10000)
    def broker_commit_every(self) -> int:
        ...

    @sections.Broker.setting(params.Seconds, env_name='BROKER_COMMIT_INTERVAL', default=2.8)
    def broker_commit_interval(self) -> float:
        ...

    @sections.Broker.setting(params.Seconds, env_name='BROKER_COMMIT_LIVELOCK_SOFT_TIMEOUT', default=want_seconds(timedelta(minutes=5)))
    def broker_commit_livelock_soft_timeout(self) -> float:
        ...

    @sections.Common.setting(params.Credentials, version_introduced='1.5', env_name='BROKER_CREDENTIALS', default=None)
    def broker_credentials(self) -> Optional[CredentialsArg]:
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.0.11', env_name='BROKER_HEARTBEAT_INTERVAL', default=3.0)
    def broker_heartbeat_interval(self) -> float:
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.7', env_name='BROKER_MAX_POLL_INTERVAL', default=1000.0)
    def broker_max_poll_interval(self) -> float:
        ...

    @sections.Broker.setting(params.UnsignedInt, version_introduced='1.4', env_name='BROKER_MAX_POLL_RECORDS', default=None, allow_none=True)
    def broker_max_poll_records(self) -> Optional[int]:
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.10', env_name='BROKER_REBALANCE_TIMEOUT', default=60.0)
    def broker_rebalance_timeout(self) -> float:
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.4', env_name='BROKER_REQUEST_TIMEOUT', default=90.0)
    def broker_request_timeout(self) -> float:
        ...

    @sections.Broker.setting(params.Seconds, version_introduced='1.0.11', env_name='BROKER_SESSION_TIMEOUT', default=60.0)
    def broker_session_timeout(self) -> float:
        ...

    @sections.Common.setting(params.SSLContext, default=None)
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        ...

    @sections.Consumer.setting(params.Str, version_introduced='1.10', env_name='CONSUMER_API_VERSION', default_alias='broker_api_version')
    def consumer_api_version(self) -> str:
        ...

    @sections.Consumer.setting(params.UnsignedInt, version_introduced='1.4', env_name='CONSUMER_MAX_FETCH_SIZE', default=1024 ** 2)
    def consumer_max_fetch_size(self) -> int:
        ...

    @sections.Consumer.setting(params.Str, version_introduced='1.5', env_name='CONSUMER_AUTO_OFFSET_RESET', default='earliest')
    def consumer_auto_offset_reset(self) -> str:
        ...

    @sections.Consumer.setting(params.Str, version_introduced='2.1', env_name='CONSUMER_GROUP_INSTANCE_ID', default=None)
    def consumer_group_instance_id(self) -> Optional[str]:
        ...

    @sections.Serialization.setting(params.Codec, env_name='APP_KEY_SERIALIZER', default