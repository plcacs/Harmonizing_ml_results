from typing import Any, Dict, List, Tuple, Union

def normalize(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))

def mock_connection(name: str, state: str = 'open') -> Any:
    conn = mock.MagicMock()
    conn.name = name
    conn.state = state
    return conn

def profile_from_dict(profile: Dict[str, Any], profile_name: str, cli_vars: Union[str, Dict[str, Any]] = '{}') -> Any:
    ...

def project_from_dict(project: Dict[str, Any], profile: Any, packages: Any = None, selectors: Any = None, cli_vars: Union[str, Dict[str, Any]] = '{}') -> Any:
    ...

def config_from_parts_or_dicts(project: Any, profile: Any, packages: Any = None, selectors: Any = None, cli_vars: Dict[str, Any] = {}) -> Any:
    ...

def inject_plugin(plugin: Any) -> None:
    ...

def inject_plugin_for(config: Any) -> Any:
    ...

def inject_adapter(value: Any, plugin: Any) -> None:
    ...

def clear_plugin(plugin: Any) -> None:
    ...

def compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    ...

def assert_from_dict(obj: Any, dct: Dict[str, Any], cls: Any = None) -> None:
    ...

def assert_to_dict(obj: Any, dct: Dict[str, Any]) -> None:
    ...

def assert_symmetric(obj: Any, dct: Dict[str, Any], cls: Any = None) -> None:
    ...

def assert_fails_validation(dct: Dict[str, Any], cls: Any) -> None:
    ...

def generate_name_macros(package: str) -> Any:
    ...

class TestAdapterConversions(TestCase):
    def _get_tester_for(self, column_type: Any) -> Any:
        ...

    def _make_table_of(self, rows: List[Tuple[Any]], column_types: Union[type, List[type]]) -> Any:
        ...

def MockMacro(package: str, name: str = 'my_macro', **kwargs: Any) -> Any:
    ...

def MockMaterialization(package: str, name: str = 'my_materialization', adapter_type: str = None, **kwargs: Any) -> Any:
    ...

def MockGenerateMacro(package: str, component: str = 'some_component', **kwargs: Any) -> Any:
    ...

def MockSource(package: str, source_name: str, name: str, **kwargs: Any) -> Any:
    ...

def MockNode(package: str, name: str, resource_type: Any = None, **kwargs: Any) -> Any:
    ...

def MockDocumentation(package: str, name: str, **kwargs: Any) -> Any:
    ...

def load_internal_manifest_macros(config: Any, macro_hook: Any = lambda m: None) -> Any:
    ...

def dict_replace(dct: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    ...

def replace_config(n: Any, **kwargs: Any) -> Any:
    ...

def make_manifest(nodes: List[Any] = [], sources: List[Any] = [], macros: List[Any] = [], docs: List[Any] = []) -> Any:
    ...
