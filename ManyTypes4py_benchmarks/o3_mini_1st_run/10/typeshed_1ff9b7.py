#!/usr/bin/env python3
import os
import re
from functools import wraps
from collections import namedtuple
from typing import Dict, Tuple, Generator, List, Optional, Any, Callable, Sequence
from pathlib import Path

from jedi import settings
from jedi.file_io import FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.stub_value import TypingModuleWrapper, StubModuleValue
from jedi.inference.value import ModuleValue

_jedi_path: Path = Path(__file__).parent.parent.parent
TYPESHED_PATH: Path = _jedi_path.joinpath('third_party', 'typeshed')
DJANGO_INIT_PATH: Path = _jedi_path.joinpath('third_party', 'django-stubs', 'django-stubs', '__init__.pyi')
_IMPORT_MAP: Dict[str, str] = dict(_collections='collections', _socket='socket')

PathInfo = namedtuple('PathInfo', 'path is_third_party')
# For type annotation purposes, we consider PathInfo as having:
#   path: Path
#   is_third_party: bool

_version_cache: Dict[Tuple[int, int], Dict[str, PathInfo]] = {}


def _merge_create_stub_map(path_infos: Sequence[PathInfo]) -> Dict[str, PathInfo]:
    map_: Dict[str, PathInfo] = {}
    for directory_path_info in path_infos:
        map_.update(_create_stub_map(directory_path_info))
    return map_


def _create_stub_map(directory_path_info: PathInfo) -> Dict[str, PathInfo]:
    """
    Create a mapping of an importable name in Python to a stub file.
    """
    def generate() -> Generator[Tuple[str, PathInfo], None, None]:
        try:
            listed = os.listdir(directory_path_info.path)
        except (FileNotFoundError, NotADirectoryError):
            return
        for entry in listed:
            path = os.path.join(directory_path_info.path, entry)
            if os.path.isdir(path):
                init = os.path.join(path, '__init__.pyi')
                if os.path.isfile(init):
                    yield (entry, PathInfo(init, directory_path_info.is_third_party))
            elif entry.endswith('.pyi') and os.path.isfile(path):
                name = entry[:-4]
                if name != '__init__':
                    yield (name, PathInfo(path, directory_path_info.is_third_party))
    return dict(generate())


def _get_typeshed_directories(version_info: Any) -> Generator[PathInfo, None, None]:
    check_version_list: List[str] = ['2and3', '3']
    for base in ['stdlib', 'third_party']:
        base_path: Path = TYPESHED_PATH.joinpath(base)
        base_list: List[str] = os.listdir(base_path)
        for base_list_entry in base_list:
            match = re.match(r'(\d+)\.(\d+)$', base_list_entry)
            if match is not None:
                if match.group(1) == '3' and int(match.group(2)) <= version_info.minor:
                    check_version_list.append(base_list_entry)
        for check_version in check_version_list:
            is_third_party: bool = base != 'stdlib'
            yield PathInfo(str(base_path.joinpath(check_version)), is_third_party)


def _cache_stub_file_map(version_info: Any) -> Dict[str, PathInfo]:
    """
    Returns a map of an importable name in Python to a stub file.
    """
    version: Tuple[int, int] = version_info[:2]
    try:
        return _version_cache[version]
    except KeyError:
        pass
    file_set: Dict[str, PathInfo] = _merge_create_stub_map(list(_get_typeshed_directories(version_info)))
    _version_cache[version] = file_set
    return file_set


def import_module_decorator(func: Callable[[Any, Tuple[str, ...], Optional[Any], List[str], bool], ValueSet]) -> Callable[[Any, Tuple[str, ...], Optional[Any], List[str], bool], ValueSet]:
    @wraps(func)
    def wrapper(inference_state: Any, import_names: Tuple[str, ...], parent_module_value: Optional[Any], sys_path: List[str], prefer_stubs: bool) -> ValueSet:
        python_value_set: Optional[ValueSet] = inference_state.module_cache.get(import_names)
        if python_value_set is None:
            if parent_module_value is not None and parent_module_value.is_stub():
                parent_module_values = parent_module_value.non_stub_value_set
            else:
                parent_module_values = [parent_module_value]
            if import_names == ('os', 'path'):
                python_value_set = ValueSet.from_sets(
                    (func(inference_state, (n,), None, sys_path, prefer_stubs) for n in ['posixpath', 'ntpath', 'macpath', 'os2emxpath'])
                )
            else:
                python_value_set = ValueSet.from_sets(
                    (func(inference_state, import_names, p, sys_path, prefer_stubs) for p in parent_module_values)
                )
            inference_state.module_cache.add(import_names, python_value_set)
        if not prefer_stubs or import_names[0] in settings.auto_import_modules:
            return python_value_set
        stub = try_to_load_stub_cached(inference_state, import_names, python_value_set, parent_module_value, sys_path)
        if stub is not None:
            return ValueSet([stub])
        return python_value_set
    return wrapper


def try_to_load_stub_cached(inference_state: Any, import_names: Tuple[str, ...], *args: Any, **kwargs: Any) -> Optional[Any]:
    if import_names is None:
        return None
    try:
        return inference_state.stub_module_cache[import_names]
    except KeyError:
        pass
    inference_state.stub_module_cache[import_names] = None
    result: Optional[Any] = _try_to_load_stub(inference_state, import_names, *args, **kwargs)
    inference_state.stub_module_cache[import_names] = result
    return result


def _try_to_load_stub(inference_state: Any, import_names: Tuple[str, ...], python_value_set: ValueSet, parent_module_value: Optional[Any], sys_path: List[str]) -> Optional[Any]:
    """
    Trying to load a stub for a set of import_names.

    This is modelled to work like "PEP 561 -- Distributing and Packaging Type
    Information", see https://www.python.org/dev/peps/pep-0561.
    """
    if parent_module_value is None and len(import_names) > 1:
        try:
            parent_module_value = try_to_load_stub_cached(inference_state, import_names[:-1], NO_VALUES, parent_module_value=None, sys_path=sys_path)
        except KeyError:
            pass
    if len(import_names) == 1:
        for p in sys_path:
            init: str = os.path.join(p, *import_names) + '-stubs' + os.path.sep + '__init__.pyi'
            m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(init), import_names=import_names)
            if m is not None:
                return m
        if import_names[0] == 'django' and python_value_set:
            return _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(str(DJANGO_INIT_PATH)), import_names=import_names)
    for c in python_value_set:
        try:
            method = c.py__file__
        except AttributeError:
            pass
        else:
            file_path = method()
            file_paths: List[str] = []
            if c.is_namespace():
                file_paths = [os.path.join(p, '__init__.pyi') for p in c.py__path__()]
            elif file_path is not None and file_path.suffix == '.py':
                file_paths = [str(file_path) + 'i']
            for file_path in file_paths:
                m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(file_path), import_names=import_names)
                if m is not None:
                    return m
    m = _load_from_typeshed(inference_state, python_value_set, parent_module_value, import_names)
    if m is not None:
        return m
    if not python_value_set:
        if parent_module_value is not None:
            check_path = parent_module_value.py__path__() or []
            names_for_path = (import_names[-1],)
        else:
            check_path = sys_path
            names_for_path = import_names
        for p in check_path:
            m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(os.path.join(p, *names_for_path) + '.pyi'), import_names=import_names)
            if m is not None:
                return m
    return None


def _load_from_typeshed(inference_state: Any, python_value_set: ValueSet, parent_module_value: Optional[Any], import_names: Tuple[str, ...]) -> Optional[Any]:
    import_name: str = import_names[-1]
    map_: Optional[Dict[str, PathInfo]] = None
    if len(import_names) == 1:
        map_ = _cache_stub_file_map(inference_state.grammar.version_info)
        import_name = _IMPORT_MAP.get(import_name, import_name)
    elif isinstance(parent_module_value, ModuleValue):
        if not parent_module_value.is_package():
            return None
        paths = parent_module_value.py__path__()
        map_ = _merge_create_stub_map([PathInfo(p, is_third_party=False) for p in paths])
    if map_ is not None:
        path_info: Optional[PathInfo] = map_.get(import_name)
        if path_info is not None and (not path_info.is_third_party or python_value_set):
            return _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(path_info.path), import_names=import_names)
    return None


def _try_to_load_stub_from_file(inference_state: Any, python_value_set: ValueSet, file_io: FileIO, import_names: Tuple[str, ...]) -> Optional[Any]:
    try:
        stub_module_node = parse_stub_module(inference_state, file_io)
    except OSError:
        return None
    else:
        return create_stub_module(inference_state, inference_state.latest_grammar, python_value_set, stub_module_node, file_io, import_names)


def parse_stub_module(inference_state: Any, file_io: FileIO) -> Any:
    return inference_state.parse(
        file_io=file_io,
        cache=True,
        diff_cache=settings.fast_parser,
        cache_path=settings.cache_directory,
        use_latest_grammar=True
    )


def create_stub_module(inference_state: Any, grammar: Any, python_value_set: ValueSet, stub_module_node: Any, file_io: FileIO, import_names: Tuple[str, ...]) -> Any:
    if import_names == ('typing',):
        module_cls: Any = TypingModuleWrapper
    else:
        module_cls = StubModuleValue
    file_name: str = os.path.basename(file_io.path)
    stub_module_value: Any = module_cls(
        python_value_set,
        inference_state,
        stub_module_node,
        file_io=file_io,
        string_names=import_names,
        code_lines=get_cached_code_lines(grammar, file_io.path),
        is_package=(file_name == '__init__.pyi')
    )
    return stub_module_value
