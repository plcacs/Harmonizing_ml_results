def _add_delegate_accessors(cls: TypeT, delegate: TypeT, accessors: list[str], typ: str, overwrite: bool = False, accessor_mapping: Callable = lambda x: x, raise_on_missing: bool = True) -> None:
def delegate_names(delegate: object, accessors: list[str], typ: str, overwrite: bool = False, accessor_mapping: Callable = lambda x: x, raise_on_missing: bool = True) -> Callable:
def _register_accessor(name: str, cls: TypeT) -> Callable:
def register_dataframe_accessor(name: str) -> Callable:
def register_series_accessor(name: str) -> Callable:
def register_index_accessor(name: str) -> Callable:
