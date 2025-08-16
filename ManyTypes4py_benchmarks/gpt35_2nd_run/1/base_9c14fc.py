from faust.types.tables import CollectionT
from faust.types.models import ModelT
from faust.types.stores import StoreT
from faust.types.streams import StreamT
from faust.types.windows import WindowT
from faust.types import AppT

class Collection(Service, CollectionT):
    _data: StoreT = None
    _changelog_compacting: bool = True
    _changelog_deleting: Optional[bool] = None

    def __init__(self, app: AppT, *, name: Optional[str] = None, default: Any = None, store: Optional[str] = None, schema: Optional[SchemaT] = None, key_type: Optional[type] = None, value_type: Optional[type] = None, partitions: Optional[int] = None, window: Optional[WindowT] = None, changelog_topic: Optional[str] = None, help: Optional[str] = None, on_recover: Optional[RecoverCallback] = None, on_changelog_event: Optional[ChangelogEventCallback] = None, recovery_buffer_size: int = 1000, standby_buffer_size: Optional[int] = None, extra_topic_configs: Optional[Mapping[str, Any]] = None, recover_callbacks: Optional[Set[RecoverCallback]] = None, options: Optional[Mapping[str, Any]] = None, use_partitioner: bool = False, on_window_close: Optional[WindowCloseCallback] = None, is_global: bool = False, **kwargs: Any):
        self.app: AppT = app
        self.name: str = cast(str, name)
        self.default: Any = default
        self._store: Optional[URL] = URL(store) if store else None
        self.schema: Optional[SchemaT] = schema
        self.key_type: Optional[type] = key_type
        self.value_type: Optional[type] = value_type
        self.partitions: Optional[int] = partitions
        self.window: Optional[WindowT] = window
        self._changelog_topic: Optional[str] = changelog_topic
        self.extra_topic_configs: Mapping[str, Any] = extra_topic_configs or {}
        self.help: str = help or ''
        self._on_changelog_event: Optional[ChangelogEventCallback] = on_changelog_event
        self.recovery_buffer_size: int = recovery_buffer_size
        self.standby_buffer_size: int = standby_buffer_size or recovery_buffer_size
        self.use_partitioner: bool = use_partitioner
        self._on_window_close: Optional[WindowCloseCallback] = on_window_close
        self.last_closed_window: float = 0.0
        assert self.recovery_buffer_size > 0 and self.standby_buffer_size > 0
        self.options: Optional[Mapping[str, Any]] = options
        self.key_serializer: str = self._serializer_from_type(self.key_type)
        self.value_serializer: str = self._serializer_from_type(self.value_type)
        self._partition_timestamp_keys: MutableMapping[Tuple[int, int], MutableSet[Tuple[Any, Tuple[float, float]]]] = defaultdict(set)
        self._partition_timestamps: MutableMapping[int, List[float]] = defaultdict(list)
        self._partition_latest_timestamp: MutableMapping[int, int] = defaultdict(int)
        self._recover_callbacks: Set[RecoverCallback] = set(recover_callbacks or [])
        if on_recover:
            self.on_recover(on_recover)
        self._sensor_on_get = self.app.sensors.on_table_get
        self._sensor_on_set = self.app.sensors.on_table_set
        self._sensor_on_del = self.app.sensors.on_table_del
