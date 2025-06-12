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
        agent_supervisor: Optional[Type[SupervisorStrategyT]] = None,
        broker: Optional[BrokerArg] = None,
        broker_consumer: Optional[BrokerArg] = None,
        broker_producer: Optional[BrokerArg] = None,
        broker_api_version: Optional[str] = None,
        broker_check_crcs: Optional[bool] = None,
        broker_client_id: Optional[str] = None,
        broker_commit_every: Optional[int] = None,
        broker_commit_interval: Optional[Seconds] = None,
        broker_commit_livelock_soft_timeout: Optional[Seconds] = None,
        broker_credentials: Optional[CredentialsArg] = None,
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
        logging_config: Optional[Dict[str, Any]] = None,
        loghandlers: Optional[List[logging.Handler]] = None,
        producer_acks: Optional[int] = None,
        producer_api_version: Optional[str] = None,
        producer_compression_type: Optional[str] = None,
        producer_linger_ms: Optional[int] = None,
        producer_max_batch_size: Optional[int] = None,
        producer_max_request_size: Optional[int] = None,
        producer_partitioner: Optional[PartitionerT] = None,
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
        store: Optional[Union[str, URL]] = None,
        table_cleanup_interval: Optional[Seconds] = None,
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

    def relative_to_appdir(self, path: Path) -> Path:
        """Prepare app directory path."""
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

    # ... (rest of the class methods with appropriate type annotations)
