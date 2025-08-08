from typing import List, Tuple, Union, Optional

def get_sys_path() -> List[str]:
    return sys.path

def load_module(inference_state, **kwargs) -> Any:
    return access.load_module(inference_state, **kwargs)

def get_compiled_method_return(inference_state, id, attribute, *args, **kwargs) -> Any:
    handle = inference_state.compiled_subprocess.get_access_handle(id)
    return getattr(handle.access, attribute)(*args, **kwargs)

def create_simple_object(inference_state, obj) -> Any:
    return access.create_access_path(inference_state, obj)

def get_module_info(inference_state, sys_path=None, full_name=None, **kwargs) -> Tuple[Union[NamespaceInfo, FileIO, None], Optional[bool]]:
    ...

def get_builtin_module_names(inference_state) -> List[str]:
    return sys.builtin_module_names

def _test_raise_error(inference_state, exception_type) -> None:
    ...

def _test_print(inference_state, stderr=None, stdout=None) -> None:
    ...

def _get_init_path(directory_path) -> Optional[str]:
    ...

def safe_literal_eval(inference_state, value) -> Any:
    return parser_utils.safe_literal_eval(value)

def iter_module_names(*args, **kwargs) -> List[str]:
    return list(_iter_module_names(*args, **kwargs))

def _iter_module_names(inference_state, paths) -> List[str]:
    ...

def _find_module(string, path=None, full_name=None, is_global_search=True) -> Tuple[Union[FileIO, str], bool]:
    ...

def _find_module_py33(string, path=None, loader=None, full_name=None, is_global_search=True) -> Tuple[Union[FileIO, str], bool]:
    ...

def _from_loader(loader, string) -> Tuple[Union[FileIO, str], bool]:
    ...

def _get_source(loader, fullname) -> bytes:
    ...

def _zip_list_subdirectory(zip_path, zip_subdir_path) -> Generator[Tuple[str, bool], None, None]:
    ...
