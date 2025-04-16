"""
Projects are a way to handle Python projects within Jedi. For simpler plugins
you might not want to deal with projects, but if you want to give the user more
flexibility to define sys paths and Python interpreters for a project,
:class:`.Project` is the perfect way to allow for that.

Projects can be saved to disk and loaded again, to allow project definitions to
be used across repositories.
"""
import json
from pathlib import Path
from itertools import chain
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from jedi import debug
from jedi.api.environment import Environment, get_cached_default_environment, create_environment
from jedi.api.exceptions import WrongVersion
from jedi.api.completion import Completion, search_in_module
from jedi.api.helpers import split_search_string, get_module_names
from jedi.inference.imports import (
    load_module_from_path,
    load_namespace_from_path,
    iter_module_names,
)
from jedi.inference.sys_path import discover_buildout_paths
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.inference.references import (
    recurse_find_python_folders_and_files,
    search_in_file_ios,
)
from jedi.file_io import FolderIO

_CONFIG_FOLDER: str = ".jedi"
_CONTAINS_POTENTIAL_PROJECT: Tuple[str, ...] = (
    "setup.py",
    ".git",
    ".hg",
    "requirements.txt",
    "MANIFEST.in",
    "pyproject.toml",
)

_SERIALIZER_VERSION: int = 1

T = Any
V = Any


def _try_to_skip_duplicates(func: Callable[..., Iterator[Any]]) -> Callable[..., Iterator[Any]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
        found_tree_nodes: List[Any] = []
        found_modules: List[str] = []
        for definition in func(*args, **kwargs):
            tree_node = getattr(definition._name, "tree_name", None)
            if tree_node is not None and tree_node in found_tree_nodes:
                continue
            if definition.type == "module" and getattr(definition, "module_path", None) is not None:
                if definition.module_path in found_modules:
                    continue
                found_modules.append(definition.module_path)
            yield definition
            found_tree_nodes.append(tree_node)

    return wrapper


def _remove_duplicates_from_path(path: Iterator[str]) -> Iterator[str]:
    used: set = set()
    for p in path:
        if p in used:
            continue
        used.add(p)
        yield p


class Project:
    """
    Projects are a simple way to manage Python folders and define how Jedi does
    import resolution. It is mostly used as a parameter to :class:`.Script`.
    Additionally there are functions to search a whole project.
    """

    _environment: Optional[Environment] = None

    @staticmethod
    def _get_config_folder_path(base_path: Path) -> Path:
        return base_path.joinpath(_CONFIG_FOLDER)

    @staticmethod
    def _get_json_path(base_path: Path) -> Path:
        return Project._get_config_folder_path(base_path).joinpath("project.json")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Project":
        """
        Loads a project from a specific path. You should not provide the path
        to ``.jedi/project.json``, but rather the path to the project folder.

        :param path: The path of the directory you want to use as a project.
        """
        if isinstance(path, str):
            path = Path(path)
        with open(cls._get_json_path(path), "r", encoding="utf-8") as f:
            version: int
            data: Dict[str, Any]
            version, data = json.load(f)

        if version == 1:
            return cls(**data)
        else:
            raise WrongVersion(
                "The Jedi version of this project seems newer than what we can handle."
            )

    def save(self) -> None:
        """
        Saves the project configuration in the project in ``.jedi/project.json``.
        """
        data: Dict[str, Any] = dict(self.__dict__)
        data.pop("_environment", None)
        data.pop("_django", None)  # TODO make django setting public?
        data = {k.lstrip("_"): v for k, v in data.items()}
        data["path"] = str(data["path"])

        self._get_config_folder_path(self._path).mkdir(parents=True, exist_ok=True)
        with open(self._get_json_path(self._path), "w", encoding="utf-8") as f:
            json.dump((_SERIALIZER_VERSION, data), f)

    def __init__(
        self,
        path: Union[str, Path],
        *,
        environment_path: Optional[str] = None,
        load_unsafe_extensions: bool = False,
        sys_path: Optional[List[str]] = None,
        added_sys_path: Tuple[str, ...] = (),
        smart_sys_path: bool = True,
    ) -> None:
        """
        :param path: The base path for this project.
        :param environment_path: The Python executable path, typically the path
            of a virtual environment.
        :param load_unsafe_extensions: Default False, Loads extensions that are not in the
            sys path and in the local directories. With this option enabled,
            this is potentially unsafe if you clone a git repository and
            analyze it's code, because those compiled extensions will be
            important and therefore have execution privileges.
        :param sys_path: list of str. You can override the sys path if you
            want. By default the ``sys.path.`` is generated by the
            environment (virtualenvs, etc).
        :param added_sys_path: list of str. Adds these paths at the end of the
            sys path.
        :param smart_sys_path: If this is enabled (default), adds paths from
            local directories. Otherwise you will have to rely on your packages
            being properly configured on the ``sys.path``.
        """

        if isinstance(path, str):
            path = Path(path).absolute()
        self._path: Path = path

        self._environment_path: Optional[str] = environment_path
        if sys_path is not None:
            # Remap potential pathlib.Path entries
            sys_path = list(map(str, sys_path))
        self._sys_path: Optional[List[str]] = sys_path
        self._smart_sys_path: bool = smart_sys_path
        self._load_unsafe_extensions: bool = load_unsafe_extensions
        self._django: bool = False
        # Remap potential pathlib.Path entries
        self.added_sys_path: List[str] = list(map(str, added_sys_path))
        """The sys path that is going to be added at the end of the """

    @property
    def path(self) -> Path:
        """
        The base path for this project.
        """
        return self._path

    @property
    def sys_path(self) -> Optional[List[str]]:
        """
        The sys path provided to this project. This can be None and in that
        case will be auto generated.
        """
        return self._sys_path

    @property
    def smart_sys_path(self) -> bool:
        """
        If the sys path is going to be calculated in a smart way, where
        additional paths are added.
        """
        return self._smart_sys_path

    @property
    def load_unsafe_extensions(self) -> bool:
        """
        Whether the project loads unsafe extensions.
        """
        return self._load_unsafe_extensions

    @inference_state_as_method_param_cache()
    def _get_base_sys_path(self, inference_state: Any) -> List[str]:
        # The sys path has not been set explicitly.
        sys_path = list(inference_state.environment.get_sys_path())
        try:
            sys_path.remove("")
        except ValueError:
            pass
        return sys_path

    @inference_state_as_method_param_cache()
    def _get_sys_path(
        self,
        inference_state: Any,
        add_parent_paths: bool = True,
        add_init_paths: bool = False,
    ) -> List[str]:
        """
        Keep this method private for all users of jedi. However internally this
        one is used like a public method.
        """
        suffixed: List[str] = list(self.added_sys_path)
        prefixed: List[str] = []

        if self._sys_path is None:
            sys_path = list(self._get_base_sys_path(inference_state))
        else:
            sys_path = list(self._sys_path)

        if self._smart_sys_path:
            prefixed.append(str(self._path))

            if getattr(inference_state, "script_path", None) is not None:
                suffixed += list(
                    discover_buildout_paths(
                        inference_state,
                        inference_state.script_path,
                    )
                )

                if add_parent_paths:
                    # Collect directories in upward search by:
                    #   1. Skipping directories with __init__.py
                    #   2. Stopping immediately when above self._path
                    traversed: List[str] = []
                    for parent_path in inference_state.script_path.parents:
                        if parent_path == self._path or self._path not in parent_path.parents:
                            break
                        if not add_init_paths and parent_path.joinpath("__init__.py").is_file():
                            continue
                        traversed.append(str(parent_path))

                    # AFAIK some libraries have imports like `foo.foo.bar`, which
                    # leads to the conclusion to by default prefer longer paths
                    # rather than shorter ones by default.
                    suffixed += list(reversed(traversed))

        if self._django:
            prefixed.append(str(self._path))

        path: List[str] = prefixed + sys_path + suffixed
        return list(_remove_duplicates_from_path(iter(path)))

    def get_environment(self) -> Environment:
        if self._environment is None:
            if self._environment_path is not None:
                self._environment = create_environment(self._environment_path, safe=False)
            else:
                self._environment = get_cached_default_environment()
        return self._environment

    def search(self, string: str, *, all_scopes: bool = False) -> Iterator[Any]:
        """
        Searches a name in the whole project. If the project is very big,
        at some point Jedi will stop searching. However it's also very much
        recommended to not exhaust the generator. Just display the first ten
        results to the user.

        There are currently three different search patterns:

        - ``foo`` to search for a definition foo in any file or a file called
          ``foo.py`` or ``foo.pyi``.
        - ``foo.bar`` to search for the ``foo`` and then an attribute ``bar``
          in it.
        - ``class foo.bar.Bar`` or ``def foo.bar.baz`` to search for a specific
          API type.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :yields: :class:`.Name`
        """
        return self._search_func(string, all_scopes=all_scopes)

    def complete_search(self, string: str, **kwargs: Any) -> Iterator[Completion]:
        """
        Like :meth:`.Script.search`, but completes that string. An empty string
        lists all definitions in a project, so be careful with that.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :yields: :class:`.Completion`
        """
        return self._search_func(string, complete=True, **kwargs)

    @_try_to_skip_duplicates
    def _search_func(
        self,
        string: str,
        complete: bool = False,
        all_scopes: bool = False,
    ) -> Iterator[Any]:
        # Using a Script is the easiest way to get an empty module context.
        from jedi import Script

        s: Script = Script("", project=self)
        inference_state = getattr(s, "_inference_state", None)
        empty_module_context = s._get_module_context()

        debug.dbg("Search for string %s, complete=%s", string, complete)
        wanted_type: Optional[str]
        wanted_names: List[str]
        wanted_type, wanted_names = split_search_string(string)
        name: str = wanted_names[0]
        stub_folder_name: str = name + "-stubs"

        ios: Iterator[Tuple[FolderIO, Optional[Path]]] = recurse_find_python_folders_and_files(
            FolderIO(str(self._path))
        )
        file_ios: List[Path] = []

        # 1. Search for modules in the current project
        for folder_io, file_io in ios:
            if file_io is None:
                file_name: str = folder_io.get_base_name()
                if file_name == name or file_name == stub_folder_name:
                    f: Path
                    try:
                        f = folder_io.get_file_io("__init__.py")
                        m = load_module_from_path(inference_state, f).as_context()
                    except FileNotFoundError:
                        try:
                            f = folder_io.get_file_io("__init__.pyi")
                            m = load_module_from_path(inference_state, f).as_context()
                        except FileNotFoundError:
                            m = load_namespace_from_path(inference_state, folder_io).as_context()
                else:
                    continue
            else:
                file_ios.append(Path(file_io.path))
                if Path(file_io.path).name in (f"{name}.py", f"{name}.pyi"):
                    m = load_module_from_path(inference_state, file_io).as_context()
                else:
                    continue

            debug.dbg("Search of a specific module %s", m)
            yield from search_in_module(
                inference_state,
                m,
                names=[m.name],
                wanted_type=wanted_type,
                wanted_names=wanted_names,
                complete=complete,
                convert=True,
                ignore_imports=True,
            )

        # 2. Search for identifiers in the project.
        for module_context in search_in_file_ios(inference_state, file_ios, name, complete=complete):
            names = get_module_names(module_context.tree_node, all_scopes=all_scopes)
            names = [module_context.create_name(n) for n in names]
            names = _remove_imports(names)
            yield from search_in_module(
                inference_state,
                module_context,
                names=names,
                wanted_type=wanted_type,
                wanted_names=wanted_names,
                complete=complete,
                ignore_imports=True,
            )

        # 3. Search for modules on sys.path
        sys_path: List[str] = [
            p
            for p in self._get_sys_path(inference_state)
            # Exclude folders that are handled by recursing of the Python
            # folders.
            if not p.startswith(str(self._path))
        ]
        names = list(iter_module_names(inference_state, empty_module_context, sys_path))
        yield from search_in_module(
            inference_state,
            empty_module_context,
            names=names,
            wanted_type=wanted_type,
            wanted_names=wanted_names,
            complete=complete,
            convert=True,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._path}>"


def _is_potential_project(path: Path) -> bool:
    for name in _CONTAINS_POTENTIAL_PROJECT:
        try:
            if path.joinpath(name).exists():
                return True
        except OSError:
            continue
    return False


def _is_django_path(directory: Path) -> bool:
    """ Detects the path of the very well known Django library (if used) """
    try:
        with open(directory.joinpath("manage.py"), "rb") as f:
            return b"DJANGO_SETTINGS_MODULE" in f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False


def get_default_project(path: Optional[Union[str, Path]] = None) -> Project:
    """
    If a project is not defined by the user, Jedi tries to define a project by
    itself as well as possible. Jedi traverses folders until it finds one of
    the following:

    1. A ``.jedi/config.json``
    2. One of the following files: ``setup.py``, ``.git``, ``.hg``,
       ``requirements.txt`` and ``MANIFEST.in``.
    """
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)

    check: Path = path.absolute()
    probable_path: Optional[Path] = None
    first_no_init_file: Optional[Path] = None
    for dir in chain([check], check.parents):
        try:
            return Project.load(dir)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
        except NotADirectoryError:
            continue

        if first_no_init_file is None:
            if dir.joinpath("__init__.py").exists():
                # In the case that a __init__.py exists, it's in 99% just a
                # Python package and the project sits at least one level above.
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
        # TODO search for setup.py etc
        return Project(probable_path)

    if first_no_init_file is not None:
        return Project(first_no_init_file)

    curdir: Path = path if path.is_dir() else path.parent
    return Project(curdir)


def _remove_imports(names: List[Any]) -> List[Any]:
    return [
        n for n in names
        if getattr(n, "tree_name", None) is None or getattr(n, "api_type", None) not in ("module", "namespace")
    ]
