def _merge_create_stub_map(path_infos: List[PathInfo]) -> Dict[str, PathInfo]:
def _create_stub_map(directory_path_info: PathInfo) -> Dict[str, PathInfo]:
def _get_typeshed_directories(version_info: Tuple[int, int]) -> Generator[PathInfo, None, None]:
def _cache_stub_file_map(version_info: Tuple[int, int]) -> Dict[str, PathInfo]:
def import_module_decorator(func: Callable) -> Callable:
def try_to_load_stub_cached(inference_state, import_names, *args, **kwargs) -> Optional[Union[ValueSet, None]]:
def _try_to_load_stub(inference_state, import_names, python_value_set, parent_module_value, sys_path) -> Optional[Union[StubModuleValue, None]]:
def _load_from_typeshed(inference_state, python_value_set, parent_module_value, import_names) -> Optional[Union[StubModuleValue, None]]:
def _try_to_load_stub_from_file(inference_state, python_value_set, file_io, import_names) -> Optional[Union[StubModuleValue, None]]:
def parse_stub_module(inference_state, file_io) -> Any:
def create_stub_module(inference_state, grammar, python_value_set, stub_module_node, file_io, import_names) -> StubModuleValue:
