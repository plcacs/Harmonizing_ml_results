import sys
import os
import inspect
import importlib
import warnings
from pathlib import Path
from zipfile import ZipFile
from zipimport import zipimporter, ZipImportError
from importlib.machinery import all_suffixes
from jedi.inference.compiled import access
from jedi import debug
from jedi import parser_utils
from jedi.file_io import KnownContentFileIO, ZipFileIO
from typing import Any, Tuple, Union, Optional, List, Generator, Iterator, Dict

def get_sys_path() -> List[str]:
    return sys.path

def load_module(inference_state: Any, **kwargs: Any) -> Any:
    return access.load_module(inference_state, **kwargs)

def get_compiled_method_return(
    inference_state: Any, 
    id: Any, 
    attribute: str, 
    *args: Any, 
    **kwargs: Any
) -> Any:
    handle = inference_state.compiled_subprocess.get_access_handle(id)
    return getattr(handle.access, attribute)(*args, **kwargs)

def create_simple_object(inference_state: Any, obj: Any) -> Any:
    return access.create_access_path(inference_state, obj)

def get_module_info(
    inference_state: Any, 
    sys_path: Optional[List[str]] = None, 
    full_name: Optional[str] = None, 
    **kwargs: Any
) -> Tuple[Union[Any, None], Optional[bool]]:
    """
    Returns Tuple[Union[NamespaceInfo, FileIO, None], Optional[bool]]
    """
    if sys_path is not None:
        sys.path, temp = (sys_path, sys.path)
    try:
        return _find_module(full_name=full_name, **kwargs)
    except ImportError:
        return (None, None)
    finally:
        if sys_path is not None:
            sys.path = temp

def get_builtin_module_names(inference_state: Any) -> Tuple[str, ...]:
    return sys.builtin_module_names

def _test_raise_error(inference_state: Any, exception_type: Any) -> None:
    """
    Raise an error to simulate certain problems for unit tests.
    """
    raise exception_type

def _test_print(
    inference_state: Any, 
    stderr: Optional[str] = None, 
    stdout: Optional[str] = None
) -> None:
    """
    Force some prints in the subprocesses. This exists for unit tests.
    """
    if stderr is not None:
        print(stderr, file=sys.stderr)
        sys.stderr.flush()
    if stdout is not None:
        print(stdout)
        sys.stdout.flush()

def _get_init_path(directory_path: str) -> Optional[str]:
    """
    The __init__ file can be searched in a directory. If found return it, else
    None.
    """
    for suffix in all_suffixes():
        path = os.path.join(directory_path, '__init__' + suffix)
        if os.path.exists(path):
            return path
    return None

def safe_literal_eval(inference_state: Any, value: str) -> Any:
    return parser_utils.safe_literal_eval(value)

def iter_module_names(*args: Any, **kwargs: Any) -> List[str]:
    return list(_iter_module_names(*args, **kwargs))

def _iter_module_names(
    inference_state: Any, 
    paths: List[str]
) -> Generator[str, None, None]:
    for path in paths:
        try:
            dir_entries = ((entry.name, entry.is_dir()) for entry in os.scandir(path))
        except OSError:
            try:
                zip_import_info = zipimporter(path)
                dir_entries = _zip_list_subdirectory(zip_import_info.archive, zip_import_info.prefix)
            except ZipImportError:
                debug.warning('Not possible to list directory: %s', path)
                continue
        for name, is_dir in dir_entries:
            if is_dir:
                if name != '__pycache__' and name.isidentifier():
                    yield name
            else:
                if name.endswith('.pyi'):
                    modname = name[:-4]
                else:
                    modname = inspect.getmodulename(name)
                if modname and '.' not in modname:
                    if modname != '__init__':
                        yield modname

def _find_module(
    string: str, 
    path: Optional[List[str]] = None, 
    full_name: Optional[str] = None, 
    is_global_search: bool = True
) -> Tuple[Union[Any, None], bool]:
    """
    Provides information about a module.

    This function isolates the differences in importing libraries introduced with
    python 3.3 on; it gets a module name and optionally a path. It will return a
    tuple containin an open file for the module (if not builtin), the filename
    or the name of the module if it is a builtin one and a boolean indicating
    if the module is contained in a package.
    """
    spec = None
    loader = None
    for finder in sys.meta_path:
        if is_global_search and finder != importlib.machinery.PathFinder:
            p = None
        else:
            p = path
        try:
            find_spec = finder.find_spec
        except AttributeError:
            continue
        spec = find_spec(string, p)
        if spec is not None:
            loader = spec.loader
            if loader is None and (not spec.has_location):
                full_name = string if not path else full_name
                implicit_ns_info = ImplicitNSInfo(full_name, spec.submodule_search_locations._path)
                return (implicit_ns_info, True)
            break
    return _find_module_py33(string, path, loader)

def _find_module_py33(
    string: str, 
    path: Optional[List[str]] = None, 
    loader: Any = None, 
    full_name: Optional[str] = None, 
    is_global_search: bool = True
) -> Tuple[Union[Any, None], bool]:
    loader = loader or importlib.machinery.PathFinder.find_module(string, path)
    if loader is None and path is None:
        try:
            with warnings.catch_warnings(record=True):
                loader = importlib.find_loader(string)
        except ValueError as e:
            raise ImportError('Originally  ' + repr(e))
    if loader is None:
        raise ImportError("Couldn't find a loader for {}".format(string))
    return _from_loader(loader, string)

def _from_loader(loader: Any, string: str) -> Tuple[Union[Any, None], bool]:
    try:
        is_package_method = loader.is_package
    except AttributeError:
        is_package = False
    else:
        is_package = is_package_method(string)
    try:
        get_filename = loader.get_filename
    except AttributeError:
        return (None, is_package)
    else:
        module_path = get_filename(string)
    try:
        f = type(loader).get_source
    except AttributeError:
        raise ImportError('get_source was not defined on loader')
    if f is not importlib.machinery.SourceFileLoader.get_source:
        code = loader.get_source(string)
    else:
        code = _get_source(loader, string)
    if code is None:
        return (None, is_package)
    if isinstance(loader, zipimporter):
        return (ZipFileIO(module_path, code, Path(loader.archive)), is_package)
    return (KnownContentFileIO(module_path, code), is_package)

def _get_source(loader: Any, fullname: str) -> Optional[bytes]:
    """
    This method is here as a replacement for SourceLoader.get_source. That
    method returns unicode, but we prefer bytes.
    """
    path = loader.get_filename(fullname)
    try:
        return loader.get_data(path)
    except OSError:
        raise ImportError('source not available through get_data()', name=fullname)

def _zip_list_subdirectory(
    zip_path: str, 
    zip_subdir_path: str
) -> Generator[Tuple[str, bool], None, None]:
    zip_file = ZipFile(zip_path)
    zip_subdir_path = Path(zip_subdir_path)
    zip_content_file_paths = zip_file.namelist()
    for raw_file_name in zip_content_file_paths:
        file_path = Path(raw_file_name)
        if file_path.parent == zip_subdir_path:
            file_path = file_path.relative_to(zip_subdir_path)
            yield (file_path.name, raw_file_name.endswith('/'))

class ImplicitNSInfo:
    """Stores information returned from an implicit namespace spec"""

    def __init__(self, name: str, paths: List[str]) -> None:
        self.name = name
        self.paths = paths
