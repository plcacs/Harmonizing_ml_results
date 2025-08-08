from typing import List, Tuple, Union, Set

class ModuleCache:
    _name_cache: dict

    def __init__(self):
        self._name_cache = {}

    def add(self, string_names: Union[None, Tuple[str, ...]], value_set: ValueSet):
        if string_names is not None:
            self._name_cache[string_names] = value_set

    def get(self, string_names: Union[None, Tuple[str, ...]]) -> Union[None, ValueSet]:
        return self._name_cache.get(string_names)

def infer_import(context, tree_name: str) -> ValueSet:
    ...

def goto_import(context, tree_name: str) -> Set[str]:
    ...

def _prepare_infer_import(module_context, tree_name: str) -> Tuple[Union[None, str], Tuple[str, ...], int, ValueSet]:
    ...

def _add_error(value, name, message):
    ...

def _level_to_base_import_path(project_path, directory, level) -> Tuple[Union[None, List[str]], Union[None, str]]:
    ...

class Importer:
    _inference_state: InferenceState
    level: int
    _module_context: ModuleContext
    _fixed_sys_path: Union[None, List[str]]
    _infer_possible: bool
    import_path: Tuple[str, ...]

    def __init__(self, inference_state: InferenceState, import_path: Tuple[str, ...], module_context: ModuleContext, level: int = 0):
        ...

    @property
    def _str_import_path(self) -> Tuple[str, ...]:
        ...

    def _sys_path_with_modifications(self, is_completion: bool) -> List[str]:
        ...

    def follow(self) -> ValueSet:
        ...

    def _get_module_names(self, search_path: Union[None, List[str]] = None, in_module: Union[None, bool] = None) -> List[ImportName]:
        ...

    def completion_names(self, inference_state: InferenceState, only_modules: bool = False) -> List[ImportName]:
        ...

def import_module_by_names(inference_state: InferenceState, import_names: Tuple[str, ...], sys_path: Union[None, List[str]] = None, module_context: Union[None, ModuleContext] = None, prefer_stubs: bool = True) -> ValueSet:
    ...

def import_module(inference_state: InferenceState, import_names: Tuple[str, ...], parent_module_value: Union[None, ModuleValue], sys_path: List[str]) -> ValueSet:
    ...

def _load_python_module(inference_state: InferenceState, file_io: FileIO, import_names: Union[None, Tuple[str, ...]] = None, is_package: bool = False) -> ModuleValue:
    ...

def _load_builtin_module(inference_state: InferenceState, import_names: Union[None, Tuple[str, ...]] = None, sys_path: Union[None, List[str]] = None) -> Union[None, ModuleValue]:
    ...

def load_module_from_path(inference_state: InferenceState, file_io: FileIO, import_names: Union[None, Tuple[str, ...]] = None, is_package: Union[None, bool] = None) -> Union[ModuleValue, ImplicitNamespaceValue]:
    ...

def load_namespace_from_path(inference_state: InferenceState, folder_io: FolderIO) -> ImplicitNamespaceValue:
    ...

def follow_error_node_imports_if_possible(context: Context, name: str) -> Union[None, ValueSet]:
    ...

def iter_module_names(inference_state: InferenceState, module_context: ModuleContext, search_path: List[str], module_cls: type, add_builtin_modules: bool = True) -> List[ImportName]:
    ...
