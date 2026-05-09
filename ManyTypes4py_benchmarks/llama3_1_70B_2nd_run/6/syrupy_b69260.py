class HomeAssistantSnapshotSerializer(AmberDataSerializer):
    """Home Assistant snapshot serializer for Syrupy.

    Handles special cases for Home Assistant data structures.
    """

    @classmethod
    def _serialize(cls, data: Any, *, depth: int = 0, exclude: PropertyFilter = None, include: PropertyFilter = None, matcher: PropertyMatcher = None, path: PropertyPath = (), visited: set = None) -> SerializableData:
        """Pre-process data before serializing.

        This allows us to handle specific cases for Home Assistant data structures.
        """
        if isinstance(data, State):
            serializable_data = cls._serializable_state(data)
        elif isinstance(data, ar.AreaEntry):
            serializable_data = cls._serializable_area_registry_entry(data)
        elif isinstance(data, dr.DeviceEntry):
            serializable_data = cls._serializable_device_registry_entry(data)
        elif isinstance(data, er.RegistryEntry):
            serializable_data = cls._serializable_entity_registry_entry(data)
        elif isinstance(data, ir.IssueEntry):
            serializable_data = cls._serializable_issue_registry_entry(data)
        elif isinstance(data, dict) and 'flow_id' in data and ('handler' in data):
            serializable_data = cls._serializable_flow_result(data)
        elif isinstance(data, dict) and set(data) == {'conversation_id', 'response'}:
            serializable_data = cls._serializable_conversation_result(data)
        elif isinstance(data, vol.Schema):
            serializable_data = voluptuous_serialize.convert(data)
        elif isinstance(data, ConfigEntry):
            serializable_data = cls._serializable_config_entry(data)
        elif dataclasses.is_dataclass(type(data)):
            serializable_data = dataclasses.asdict(data)
        elif isinstance(data, IntFlag):
            serializable_data = _IntFlagWrapper(data)
        else:
            serializable_data = data
            with suppress(TypeError):
                if attr.has(type(data)):
                    serializable_data = attrs.asdict(data)
        return super()._serialize(serializable_data, depth=depth, exclude=exclude, include=include, matcher=matcher, path=path, visited=visited)

    @classmethod
    def _serializable_area_registry_entry(cls, data: ar.AreaEntry) -> dict:
        """Prepare a Home Assistant area registry entry for serialization."""
        serialized = AreaRegistryEntrySnapshot(dataclasses.asdict(data) | {'id': ANY})
        serialized.pop('_json_repr')
        serialized.pop('_cache')
        return serialized

    @classmethod
    def _serializable_config_entry(cls, data: ConfigEntry) -> dict:
        """Prepare a Home Assistant config entry for serialization."""
        entry = ConfigEntrySnapshot(data.as_dict() | {'entry_id': ANY})
        return cls._remove_created_and_modified_at(entry)

    @classmethod
    def _serializable_device_registry_entry(cls, data: dr.DeviceEntry) -> dict:
        """Prepare a Home Assistant device registry entry for serialization."""
        serialized = DeviceRegistryEntrySnapshot(attrs.asdict(data) | {'config_entries': ANY, 'config_entries_subentries': ANY, 'id': ANY})
        if serialized['via_device_id'] is not None:
            serialized['via_device_id'] = ANY
        if serialized['primary_config_entry'] is not None:
            serialized['primary_config_entry'] = ANY
        serialized.pop('_cache')
        return cls._remove_created_and_modified_at(serialized)

    @classmethod
    def _remove_created_and_modified_at(cls, data: dict) -> dict:
        """Remove created_at and modified_at from the data."""
        data.pop('created_at', None)
        data.pop('modified_at', None)
        return data

    @classmethod
    def _serializable_entity_registry_entry(cls, data: er.RegistryEntry) -> dict:
        """Prepare a Home Assistant entity registry entry for serialization."""
        serialized = EntityRegistryEntrySnapshot(attrs.asdict(data) | {'config_entry_id': ANY, 'config_subentry_id': ANY, 'device_id': ANY, 'id': ANY, 'options': {k: dict(v) for k, v in data.options.items()}})
        serialized.pop('categories')
        serialized.pop('_cache')
        return cls._remove_created_and_modified_at(serialized)

    @classmethod
    def _serializable_flow_result(cls, data: dict) -> dict:
        """Prepare a Home Assistant flow result for serialization."""
        return FlowResultSnapshot(data | {'flow_id': ANY})

    @classmethod
    def _serializable_conversation_result(cls, data: dict) -> dict:
        """Prepare a Home Assistant conversation result for serialization."""
        return data | {'conversation_id': ANY}

    @classmethod
    def _serializable_issue_registry_entry(cls, data: ir.IssueEntry) -> dict:
        """Prepare a Home Assistant issue registry entry for serialization."""
        return IssueRegistryItemSnapshot(dataclasses.asdict(data) | {'created': ANY})

    @classmethod
    def _serializable_state(cls, data: State) -> dict:
        """Prepare a Home Assistant State for serialization."""
        return StateSnapshot(data.as_dict() | {'context': ANY, 'last_changed': ANY, 'last_reported': ANY, 'last_updated': ANY})

class _IntFlagWrapper:

    def __init__(self, flag: IntFlag):
        self._flag = flag

    def __repr__(self) -> str:
        return f'<{self._flag.__class__.__name__}: {self._flag.value}>'

class HomeAssistantSnapshotExtension(AmberSnapshotExtension):
    """Home Assistant extension for Syrupy."""
    VERSION = '1'
    'Current version of serialization format.\n\n    Need to be bumped when we change the HomeAssistantSnapshotSerializer.\n    '
    serializer_class = HomeAssistantSnapshotSerializer

    @classmethod
    def dirname(cls, *, test_location: PyTestLocation) -> str:
        """Return the directory for the snapshot files.

        Syrupy, by default, uses the `__snapshosts__` directory in the same
        folder as the test file. For Home Assistant, this is changed to just
        `snapshots` in the same folder as the test file, to match our `fixtures`
        folder structure.
        """
        test_dir = Path(test_location.filepath).parent
        return str(test_dir.joinpath('snapshots'))

def _serialize_collections(collections: SnapshotCollections) -> dict:
    return {k: [c.name for c in v] for k, v in collections._snapshot_collections.items()}

def _serialize_report(report: SnapshotReport, collected_items: set, selected_items: dict) -> dict:
    return {'discovered': _serialize_collections(report.discovered), 'created': _serialize_collections(report.created), 'failed': _serialize_collections(report.failed), 'matched': _serialize_collections(report.matched), 'updated': _serialize_collections(report.updated), 'used': _serialize_collections(report.used), '_collected_items': [{'nodeid': c.nodeid, 'name': c.name, 'path': str(c.path), 'modulename': c.obj.__module__, 'methodname': c.obj.__name__} for c in list(collected_items)], '_selected_items': {key: status.value for key, status in selected_items.items()}}

def _merge_serialized_collections(collections: SnapshotCollections, json_data: dict) -> None:
    if not json_data:
        return
    for location, names in json_data.items():
        snapshot_collection = SnapshotCollection(location=location)
        for name in names:
            snapshot_collection.add(Snapshot(name))
        collections.update(snapshot_collection)

def _merge_serialized_report(report: SnapshotReport, json_data: dict) -> None:
    _merge_serialized_collections(report.discovered, json_data['discovered'])
    _merge_serialized_collections(report.created, json_data['created'])
    _merge_serialized_collections(report.failed, json_data['failed'])
    _merge_serialized_collections(report.matched, json_data['matched'])
    _merge_serialized_collections(report.updated, json_data['updated'])
    _merge_serialized_collections(report.used, json_data['used'])
    for collected_item in json_data['_collected_items']:
        custom_item = _FakePytestItem(collected_item)
        if not any((t.nodeid == custom_item.nodeid and t.name == custom_item.nodeid for t in report.collected_items)):
            report.collected_items.add(custom_item)
    for key, selected_item in json_data['_selected_items'].items():
        if key in report.selected_items:
            status = ItemStatus(selected_item)
            if status != ItemStatus.NOT_RUN:
                report.selected_items[key] = status
        else:
            report.selected_items[key] = ItemStatus(selected_item)

def override_syrupy_finish(self) -> int:
    """Override the finish method to allow for custom handling."""
    exitstatus = 0
    self.flush_snapshot_write_queue()
    self.report = SnapshotReport(base_dir=self.pytest_session.config.rootpath, collected_items=self._collected_items, selected_items=self._selected_items, assertions=self._assertions, options=self.pytest_session.config.option)
    needs_xdist_merge = self.update_snapshots or bool(self.pytest_session.config.option.include_snapshot_details)
    if is_xdist_worker():
        if not needs_xdist_merge:
            return exitstatus
        with open('.pytest_syrupy_worker_count', 'w', encoding='utf-8') as f:
            f.write(os.getenv('PYTEST_XDIST_WORKER_COUNT'))
        with open(f'.pytest_syrupy_{os.getenv("PYTEST_XDIST_WORKER")}_result', 'w', encoding='utf-8') as f:
            json.dump(_serialize_report(self.report, self._collected_items, self._selected_items), f, indent=2)
        return exitstatus
    if is_xdist_controller():
        return exitstatus
    if needs_xdist_merge:
        worker_count = None
        try:
            with open('.pytest_syrupy_worker_count', encoding='utf-8') as f:
                worker_count = f.read()
            os.remove('.pytest_syrupy_worker_count')
        except FileNotFoundError:
            pass
        if worker_count:
            for i in range(int(worker_count)):
                with open(f'.pytest_syrupy_gw{i}_result', encoding='utf-8') as f:
                    _merge_serialized_report(self.report, json.load(f))
                os.remove(f'.pytest_syrupy_gw{i}_result')
    if self.report.num_unused:
        if self.update_snapshots:
            self.remove_unused_snapshots(unused_snapshot_collections=self.report.unused, used_snapshot_collections=self.report.used)
        elif not self.warn_unused_snapshots:
            exitstatus |= EXIT_STATUS_FAIL_UNUSED
    return exitstatus
