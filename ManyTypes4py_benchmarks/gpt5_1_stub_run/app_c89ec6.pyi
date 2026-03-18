import abc
from typing import Any, Awaitable, Callable, ClassVar, List, Union

from mode import ServiceT
from .web import View

__all__: List[str] = ...

TaskArg = Union[Callable[['AppT'], Awaitable[Any]], Callable[[], Awaitable[Any]]]


class TracerT(abc.ABC):
    @property
    @abc.abstractmethod
    def default_tracer(self) -> Any: ...

    @abc.abstractmethod
    def trace(self, name: Any, sample_rate: Any = ..., **extra_context: Any) -> Any: ...

    @abc.abstractmethod
    def get_tracer(self, service_name: Any) -> Any: ...


class BootStrategyT:
    enable_kafka: ClassVar[Any]
    enable_kafka_consumer: ClassVar[Any]
    enable_kafka_producer: ClassVar[Any]
    enable_web: ClassVar[Any]
    enable_sensors: ClassVar[Any]

    @abc.abstractmethod
    def __init__(
        self,
        app: Any,
        *,
        enable_web: Any = ...,
        enable_kafka: Any = ...,
        enable_kafka_producer: Any = ...,
        enable_kafka_consumer: Any = ...,
        enable_sensors: Any = ...
    ) -> None: ...

    @abc.abstractmethod
    def server(self) -> Any: ...

    @abc.abstractmethod
    def client_only(self) -> Any: ...

    @abc.abstractmethod
    def producer_only(self) -> Any: ...


class AppT(ServiceT):
    finalized: ClassVar[bool]
    configured: ClassVar[bool]
    rebalancing: ClassVar[bool]
    rebalancing_count: ClassVar[int]
    unassigned: ClassVar[bool]
    in_worker: ClassVar[bool]
    on_configured: ClassVar[Any]
    on_before_configured: ClassVar[Any]
    on_after_configured: ClassVar[Any]
    on_partitions_assigned: ClassVar[Any]
    on_partitions_revoked: ClassVar[Any]
    on_rebalance_complete: ClassVar[Any]
    on_before_shutdown: ClassVar[Any]
    on_worker_init: ClassVar[Any]
    on_produce_message: ClassVar[Any]
    tracer: ClassVar[Any]
    on_startup_finished: Any

    @abc.abstractmethod
    def __init__(self, id: Any, *, monitor: Any, config_source: Any = ..., **options: Any) -> None: ...

    @abc.abstractmethod
    def config_from_object(self, obj: Any, *, silent: Any = ..., force: Any = ...) -> Any: ...

    @abc.abstractmethod
    def finalize(self) -> Any: ...

    @abc.abstractmethod
    def main(self) -> Any: ...

    @abc.abstractmethod
    def worker_init(self) -> Any: ...

    @abc.abstractmethod
    def worker_init_post_autodiscover(self) -> Any: ...

    @abc.abstractmethod
    def discover(self, *extra_modules: Any, categories: Any = ..., ignore: Any = ...) -> Any: ...

    @abc.abstractmethod
    def topic(
        self,
        *topics: Any,
        pattern: Any = ...,
        schema: Any = ...,
        key_type: Any = ...,
        value_type: Any = ...,
        key_serializer: Any = ...,
        value_serializer: Any = ...,
        partitions: Any = ...,
        retention: Any = ...,
        compacting: Any = ...,
        deleting: Any = ...,
        replicas: Any = ...,
        acks: Any = ...,
        internal: Any = ...,
        config: Any = ...,
        maxsize: Any = ...,
        allow_empty: Any = ...,
        has_prefix: Any = ...,
        loop: Any = ...
    ) -> Any: ...

    @abc.abstractmethod
    def channel(self, *, schema: Any = ..., key_type: Any = ..., value_type: Any = ..., maxsize: Any = ..., loop: Any = ...) -> Any: ...

    @abc.abstractmethod
    def agent(
        self,
        channel: Any = ...,
        *,
        name: Any = ...,
        concurrency: Any = ...,
        supervisor_strategy: Any = ...,
        sink: Any = ...,
        isolated_partitions: Any = ...,
        use_reply_headers: Any = ...,
        **kwargs: Any
    ) -> Any: ...

    @abc.abstractmethod
    def task(self, fun: Any, *, on_leader: Any = ..., traced: Any = ...) -> Any: ...

    @abc.abstractmethod
    def timer(self, interval: Any, on_leader: Any = ..., traced: Any = ..., name: Any = ..., max_drift_correction: Any = ...) -> Any: ...

    @abc.abstractmethod
    def crontab(self, cron_format: Any, *, timezone: Any = ..., on_leader: Any = ..., traced: Any = ...) -> Any: ...

    @abc.abstractmethod
    def service(self, cls: Any) -> Any: ...

    @abc.abstractmethod
    def stream(self, channel: Any, beacon: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def Table(self, name: Any, *, default: Any = ..., window: Any = ..., partitions: Any = ..., help: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def GlobalTable(self, name: Any, *, default: Any = ..., window: Any = ..., partitions: Any = ..., help: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def SetTable(self, name: Any, *, window: Any = ..., partitions: Any = ..., start_manager: Any = ..., help: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def SetGlobalTable(self, name: Any, *, window: Any = ..., partitions: Any = ..., start_manager: Any = ..., help: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def page(self, path: Any, *, base: Any = View, cors_options: Any = ..., name: Any = ...) -> Any: ...

    @abc.abstractmethod
    def table_route(
        self,
        table: Any,
        shard_param: Any = ...,
        *,
        query_param: Any = ...,
        match_info: Any = ...,
        exact_key: Any = ...
    ) -> Any: ...

    @abc.abstractmethod
    def command(self, *options: Any, base: Any = ..., **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def create_event(self, key: Any, value: Any, headers: Any, message: Any) -> Any: ...

    @abc.abstractmethod
    async def start_client(self) -> Any: ...

    @abc.abstractmethod
    async def maybe_start_client(self) -> Any: ...

    @abc.abstractmethod
    def trace(self, name: Any, trace_enabled: Any = ..., **extra_context: Any) -> Any: ...

    @abc.abstractmethod
    async def send(
        self,
        channel: Any,
        key: Any = ...,
        value: Any = ...,
        partition: Any = ...,
        timestamp: Any = ...,
        headers: Any = ...,
        schema: Any = ...,
        key_serializer: Any = ...,
        value_serializer: Any = ...,
        callback: Any = ...
    ) -> Any: ...

    @abc.abstractmethod
    def LiveCheck(self, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    async def maybe_start_producer(self) -> Any: ...

    @abc.abstractmethod
    def is_leader(self) -> Any: ...

    @abc.abstractmethod
    def FlowControlQueue(self, maxsize: Any = ..., *, clear_on_resume: Any = ..., loop: Any = ...) -> Any: ...

    @abc.abstractmethod
    def Worker(self, **kwargs: Any) -> Any: ...

    @abc.abstractmethod
    def on_webserver_init(self, web: Any) -> Any: ...

    @abc.abstractmethod
    def on_rebalance_start(self) -> Any: ...

    @abc.abstractmethod
    def on_rebalance_return(self) -> Any: ...

    @abc.abstractmethod
    def on_rebalance_end(self) -> Any: ...

    @property
    def conf(self) -> Any: ...
    @conf.setter
    def conf(self, settings: Any) -> None: ...

    @property
    @abc.abstractmethod
    def transport(self) -> Any: ...
    @transport.setter
    def transport(self, transport: Any) -> None: ...

    @property
    @abc.abstractmethod
    def producer_transport(self) -> Any: ...
    @producer_transport.setter
    def producer_transport(self, transport: Any) -> None: ...

    @property
    @abc.abstractmethod
    def cache(self) -> Any: ...
    @cache.setter
    def cache(self, cache: Any) -> None: ...

    @property
    @abc.abstractmethod
    def producer(self) -> Any: ...

    @property
    @abc.abstractmethod
    def consumer(self) -> Any: ...

    @property
    @abc.abstractmethod
    def tables(self) -> Any: ...

    @property
    @abc.abstractmethod
    def topics(self) -> Any: ...

    @property
    @abc.abstractmethod
    def monitor(self) -> Any: ...
    @monitor.setter
    def monitor(self, value: Any) -> None: ...

    @property
    @abc.abstractmethod
    def flow_control(self) -> Any: ...

    @property
    @abc.abstractmethod
    def http_client(self) -> Any: ...
    @http_client.setter
    def http_client(self, client: Any) -> None: ...

    @property
    @abc.abstractmethod
    def assignor(self) -> Any: ...

    @property
    @abc.abstractmethod
    def router(self) -> Any: ...

    @property
    @abc.abstractmethod
    def serializers(self) -> Any: ...

    @property
    @abc.abstractmethod
    def web(self) -> Any: ...

    @property
    @abc.abstractmethod
    def in_transaction(self) -> Any: ...

    @abc.abstractmethod
    def _span_add_default_tags(self, span: Any) -> Any: ...

    @abc.abstractmethod
    def _start_span_from_rebalancing(self, name: Any) -> Any: ...