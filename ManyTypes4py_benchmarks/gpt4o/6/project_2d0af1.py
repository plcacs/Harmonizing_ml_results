import json
from pathlib import Path
from typing import Generator, Optional, Union, List, Tuple, Any
from itertools import chain
from jedi import debug
from jedi.api.environment import get_cached_default_environment, create_environment
from jedi.api.exceptions import WrongVersion
from jedi.api.completion import search_in_module
from jedi.api.helpers import split_search_string, get_module_names
from jedi.inference.imports import load_module_from_path, load_namespace_from_path, iter_module_names
from jedi.inference.sys_path import discover_buildout_paths
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.inference.references import recurse_find_python_folders_and_files, search_in_file_ios
from jedi.file_io import FolderIO

_CONFIG_FOLDER = '.jedi'
_CONTAINS_POTENTIAL_PROJECT = ('setup.py', '.git', '.hg', 'requirements.txt', 'MANIFEST.in', 'pyproject.toml')
_SERIALIZER_VERSION = 1

def _try_to_skip_duplicates(func):
    def wrapper(*args: Any, **kwargs: Any) -> Generator:
        found_tree_nodes = []
        found_modules = []
        for definition in func(*args, **kwargs):
            tree_node = definition._name.tree_name
            if tree_node is not None and tree_node in found_tree_nodes:
                continue
            if definition.type == 'module' and definition.module_path is not None:
                if definition.module_path in found_modules:
                    continue
                found_modules.append(definition.module_path)
            yield definition
            found_tree_nodes.append(tree_node)
    return wrapper

def _remove_duplicates_from_path(path: List[str]) -> Generator[str, None, None]:
    used = set()
    for p in path:
        if p in used:
            continue
        used.add(p)
        yield p

class Project:
    _environment: Optional[Any] = None

    @staticmethod
    def _get_config_folder_path(base_path: Path) -> Path:
        return base_path.joinpath(_CONFIG_FOLDER)

    @staticmethod
    def _get_json_path(base_path: Path) -> Path:
        return Project._get_config_folder_path(base_path).joinpath('project.json')

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Project':
        if isinstance(path, str):
            path = Path(path)
        with open(cls._get_json_path(path)) as f:
            version, data = json.load(f)
        if version == 1:
            return cls(**data)
        else:
            raise WrongVersion('The Jedi version of this project seems newer than what we can handle.')

    def save(self) -> None:
        data = dict(self.__dict__)
        data.pop('_environment', None)
        data.pop('_django', None)
        data = {k.lstrip('_'): v for k, v in data.items()}
        data['path'] = str(data['path'])
        self._get_config_folder_path(self._path).mkdir(parents=True, exist_ok=True)
        with open(self._get_json_path(self._path), 'w') as f:
            json.dump((_SERIALIZER_VERSION, data), f)

    def __init__(self, path: Union[str, Path], *, environment_path: Optional[str] = None, load_unsafe_extensions: bool = False, sys_path: Optional[List[str]] = None, added_sys_path: Tuple[str, ...] = (), smart_sys_path: bool = True) -> None:
        if isinstance(path, str):
            path = Path(path).absolute()
        self._path = path
        self._environment_path = environment_path
        if sys_path is not None:
            sys_path = list(map(str, sys_path))
        self._sys_path = sys_path
        self._smart_sys_path = smart_sys_path
        self._load_unsafe_extensions = load_unsafe_extensions
        self._django = False
        self.added_sys_path = list(map(str, added_sys_path))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sys_path(self) -> Optional[List[str]]:
        return self._sys_path

    @property
    def smart_sys_path(self) -> bool:
        return self._smart_sys_path

    @property
    def load_unsafe_extensions(self) -> bool:
        return self._load_unsafe_extensions

    @inference_state_as_method_param_cache()
    def _get_base_sys_path(self, inference_state: Any) -> List[str]:
        sys_path = list(inference_state.environment.get_sys_path())
        try:
            sys_path.remove('')
        except ValueError:
            pass
        return sys_path

    @inference_state_as_method_param_cache()
    def _get_sys_path(self, inference_state: Any, add_parent_paths: bool = True, add_init_paths: bool = False) -> List[str]:
        suffixed = list(self.added_sys_path)
        prefixed = []
        if self._sys_path is None:
            sys_path = list(self._get_base_sys_path(inference_state))
        else:
            sys_path = list(self._sys_path)
        if self._smart_sys_path:
            prefixed.append(str(self._path))
            if inference_state.script_path is not None:
                suffixed += map(str, discover_buildout_paths(inference_state, inference_state.script_path))
                if add_parent_paths:
                    traversed = []
                    for parent_path in inference_state.script_path.parents:
                        if parent_path == self._path or self._path not in parent_path.parents:
                            break
                        if not add_init_paths and parent_path.joinpath('__init__.py').is_file():
                            continue
                        traversed.append(str(parent_path))
                    suffixed += reversed(traversed)
        if self._django:
            prefixed.append(str(self._path))
        path = prefixed + sys_path + suffixed
        return list(_remove_duplicates_from_path(path))

    def get_environment(self) -> Any:
        if self._environment is None:
            if self._environment_path is not None:
                self._environment = create_environment(self._environment_path, safe=False)
            else:
                self._environment = get_cached_default_environment()
        return self._environment

    def search(self, string: str, *, all_scopes: bool = False) -> Generator:
        return self._search_func(string, all_scopes=all_scopes)

    def complete_search(self, string: str, **kwargs: Any) -> Generator:
        return self._search_func(string, complete=True, **kwargs)

    @_try_to_skip_duplicates
    def _search_func(self, string: str, complete: bool = False, all_scopes: bool = False) -> Generator:
        from jedi import Script
        s = Script('', project=self)
        inference_state = s._inference_state
        empty_module_context = s._get_module_context()
        debug.dbg('Search for string %s, complete=%s', string, complete)
        wanted_type, wanted_names = split_search_string(string)
        name = wanted_names[0]
        stub_folder_name = name + '-stubs'
        ios = recurse_find_python_folders_and_files(FolderIO(str(self._path)))
        file_ios = []
        for folder_io, file_io in ios:
            if file_io is None:
                file_name = folder_io.get_base_name()
                if file_name == name or file_name == stub_folder_name:
                    f = folder_io.get_file_io('__init__.py')
                    try:
                        m = load_module_from_path(inference_state, f).as_context()
                    except FileNotFoundError:
                        f = folder_io.get_file_io('__init__.pyi')
                        try:
                            m = load_module_from_path(inference_state, f).as_context()
                        except FileNotFoundError:
                            m = load_namespace_from_path(inference_state, folder_io).as_context()
                else:
                    continue
            else:
                file_ios.append(file_io)
                if Path(file_io.path).name in (name + '.py', name + '.pyi'):
                    m = load_module_from_path(inference_state, file_io).as_context()
                else:
                    continue
            debug.dbg('Search of a specific module %s', m)
            yield from search_in_module(inference_state, m, names=[m.name], wanted_type=wanted_type, wanted_names=wanted_names, complete=complete, convert=True, ignore_imports=True)
        for module_context in search_in_file_ios(inference_state, file_ios, name, complete=complete):
            names = get_module_names(module_context.tree_node, all_scopes=all_scopes)
            names = [module_context.create_name(n) for n in names]
            names = _remove_imports(names)
            yield from search_in_module(inference_state, module_context, names=names, wanted_type=wanted_type, wanted_names=wanted_names, complete=complete, ignore_imports=True)
        sys_path = [p for p in self._get_sys_path(inference_state) if not p.startswith(str(self._path))]
        names = list(iter_module_names(inference_state, empty_module_context, sys_path))
        yield from search_in_module(inference_state, empty_module_context, names=names, wanted_type=wanted_type, wanted_names=wanted_names, complete=complete, convert=True)

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self._path)

def _is_potential_project(path: Path) -> bool:
    for name in _CONTAINS_POTENTIAL_PROJECT:
        try:
            if path.joinpath(name).exists():
                return True
        except OSError:
            continue
    return False

def _is_django_path(directory: Path) -> bool:
    try:
        with open(directory.joinpath('manage.py'), 'rb') as f:
            return b'DJANGO_SETTINGS_MODULE' in f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False

def get_default_project(path: Optional[Union[str, Path]] = None) -> Project:
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)
    check = path.absolute()
    probable_path = None
    first_no_init_file = None
    for dir in chain([check], check.parents):
        try:
            return Project.load(dir)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
        except NotADirectoryError:
            continue
        if first_no_init_file is None:
            if dir.joinpath('__init__.py').exists():
                continue
            elif not dir.is_file():
                first_no_init_file = dir
        if _is_django_path(dir):
            project = Project(dir)
            project._django = True
            return project
        if probable_path is None and _is_potential_project(dir):
            probable_path = dir
    if probable_path is not None:
        return Project(probable_path)
    if first_no_init_file is not None:
        return Project(first_no_init_file)
    curdir = path if path.is_dir() else path.parent
    return Project(curdir)

def _remove_imports(names: List[Any]) -> List[Any]:
    return [n for n in names if n.tree_name is None or n.api_type not in ('module', 'namespace')]
