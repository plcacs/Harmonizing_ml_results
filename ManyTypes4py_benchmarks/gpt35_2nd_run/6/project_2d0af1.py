from typing import List, Union

class Project:
    _environment: Union[None, object] = None

    @staticmethod
    def _get_config_folder_path(base_path: Path) -> Path:
        return base_path.joinpath(_CONFIG_FOLDER)

    @staticmethod
    def _get_json_path(base_path: Path) -> Path:
        return Project._get_config_folder_path(base_path).joinpath('project.json')

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Project':
        ...

    def save(self) -> None:
        ...

    def __init__(self, path: Union[str, Path], *, environment_path: Union[None, str] = None, load_unsafe_extensions: bool = False, sys_path: Union[None, List[str]] = None, added_sys_path: List[str] = [], smart_sys_path: bool = True) -> None:
        ...

    @property
    def path(self) -> Path:
        ...

    @property
    def sys_path(self) -> Union[None, List[str]]:
        ...

    @property
    def smart_sys_path(self) -> bool:
        ...

    @property
    def load_unsafe_extensions(self) -> bool:
        ...

    def _get_base_sys_path(self, inference_state: object) -> List[str]:
        ...

    def _get_sys_path(self, inference_state: object, add_parent_paths: bool = True, add_init_paths: bool = False) -> List[str]:
        ...

    def get_environment(self) -> object:
        ...

    def search(self, string: str, *, all_scopes: bool = False) -> object:
        ...

    def complete_search(self, string: str, **kwargs) -> object:
        ...

    def _search_func(self, string: str, complete: bool = False, all_scopes: bool = False) -> object:
        ...

    def __repr__(self) -> str:
        ...

def _is_potential_project(path: Path) -> bool:
    ...

def _is_django_path(directory: Path) -> bool:
    ...

def get_default_project(path: Union[None, str, Path] = None) -> 'Project':
    ...

def _remove_imports(names: List[object]) -> List[object]:
    ...
