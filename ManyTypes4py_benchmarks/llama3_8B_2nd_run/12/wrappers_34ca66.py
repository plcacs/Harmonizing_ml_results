class WindowedKeysView(KeysView):
    """The object returned by ``windowed_table.keys()``."""
    def __init__(self, mapping: WindowWrapper, event: Optional[datetime] = None) -> None:
        self._mapping: WindowWrapper = mapping
        self.event: Optional[datetime] = event

    # ... rest of the code ...

class WindowedItemsView(WindowedItemsViewT):
    """The object returned by ``windowed_table.items()``."""
    def __init__(self, mapping: WindowWrapper, event: Optional[datetime] = None) -> None:
        self._mapping: WindowWrapper = mapping
        self.event: Optional[datetime] = event

    # ... rest of the code ...

class WindowedValuesView(WindowedValuesViewT):
    """The object returned by ``windowed_table.values()``."""
    def __init__(self, mapping: WindowWrapper, event: Optional[datetime] = None) -> None:
        self._mapping: WindowWrapper = mapping
        self.event: Optional[datetime] = event

    # ... rest of the code ...

class WindowSet(WindowSetT[KT, VT]):
    """Represents the windows available for table key."""
    def __init__(self, key: KT, table: _Table, wrapper: WindowWrapper, event: Optional[datetime] = None) -> None:
        self.key: KT = key
        self.table: _Table = table
        self.wrapper: WindowWrapper = wrapper
        self.event: Optional[datetime] = event
        self.data: _Table = table

    # ... rest of the code ...

class WindowWrapper(WindowWrapperT):
    """Windowed table wrapper."""
    ValueType: type[WindowSet] = WindowSet
    key_index: bool
    key_index_table: Optional[_Table]

    def __init__(self, table: _Table, *, relative_to: Optional[datetime] = None, key_index: bool = False, key_index_table: Optional[_Table] = None) -> None:
        self.table: _Table = table
        self.key_index: bool = key_index
        self.key_index_table: Optional[_Table] = key_index_table
        if self.key_index and self.key_index_table is None:
            self.key_index_table = self.table.clone(name=f'{self.table.name}-key_index', value_type=int, key_type=table.key_type, window=None)
        self._get_relative_timestamp: Callable[[Optional[datetime]], Optional[datetime]] = self._relative_handler(relative_to)

    # ... rest of the code ...
